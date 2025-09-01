"""
CheXpert training with uncertainty-guided attention (UGA) and ViT backbone.
"""

from __future__ import annotations

import math
import os
import random
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from sklearn.metrics import (
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt 


warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

for var in ("NCCL_BLOCKING_WAIT", "NCCL_ASYNC_ERROR_HANDLING"):
    if var in os.environ:
        os.environ.pop(var, None)

os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

# fine to fail if on cpu / without cuda
try:
    torch.set_float32_matmul_precision("high")
except Exception: 
    pass
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception: 
    pass

torch.backends.cudnn.benchmark = True


# ddp utilities
def is_dist() -> bool:
    """return True if torch.distributed is initialized"""
    return dist.is_available() and dist.is_initialized()


def is_main() -> bool:
    """rank 0 or non-distributed"""
    return (not is_dist()) or dist.get_rank() == 0


def barrier() -> None:
    """process group barrier if distributed"""
    if is_dist():
        dist.barrier()


def init_distributed() -> tuple[int, int]:
    """initialize ddp from torchrun/torch.distributed launch"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if not is_dist():
            dist.init_process_group(
                "nccl",
                init_method="env://",
                timeout=timedelta(minutes=10),
            )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return local_rank, world_size
    return 0, 1


def cleanup_distributed() -> None:
    try:
        if is_dist():
            barrier()
            dist.destroy_process_group()
    except Exception: 
        pass


def all_gather_variable(tensor: torch.Tensor) -> torch.Tensor:
    """
    gather variable-length [N, C] tensors across ranks by padding to max N.

    this avoids needing an allreduce on lengths separately and handles empty shards.
    """
    if not is_dist():
        return tensor
    if tensor.ndim != 2:
        raise ValueError("all_gather_variable expects [N, C].")

    device = tensor.device
    local_n = torch.tensor([tensor.shape[0]], device=device, dtype=torch.long)
    world_size = dist.get_world_size()
    sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)
    if max_n == 0:
        return tensor.new_zeros((0, tensor.shape[1]))

    pad_n = max_n - tensor.shape[0]
    if pad_n > 0:
        pad = torch.zeros((pad_n, tensor.shape[1]), device=device, dtype=tensor.dtype)
        tensor_padded = torch.cat([tensor, pad], dim=0)
    else:
        tensor_padded = tensor

    gathered = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    dist.all_gather(gathered, tensor_padded)

    if any(sizes):
        parts = [gi[:sz] for gi, sz in zip(gathered, sizes) if sz > 0]
        return torch.cat(parts, dim=0)

    return tensor.new_zeros((0, tensor.shape[1]))


def all_gather_1d(vec: torch.Tensor) -> torch.Tensor:
    """gather variable-length [N] or [N,1] vectors across ranks"""
    if vec.ndim == 1:
        vec = vec.unsqueeze(1)
    out2d = all_gather_variable(vec)
    return out2d.squeeze(1)


# configuration / labels
class Config:
    FULL_ROOT = "data/chexpert-full"
    BATCH_DIRS = [
        "CheXpert-v1.0 batch 1 (validate & csv)",
        "CheXpert-v1.0 batch 2 (train 1)",
        "CheXpert-v1.0 batch 3 (train 2)",
        "CheXpert-v1.0 batch 4 (train 3)",
    ]
    TRAIN_CSV = os.path.join(FULL_ROOT, BATCH_DIRS[0], "train.csv")
    VAL_CSV = os.path.join(FULL_ROOT, BATCH_DIRS[0], "valid.csv")
    PREFIX = "CheXpert-v1.0/"
    IMG_DIRS: List[str] = []

    BATCH_SIZE = int(os.environ.get("CHEX_BATCH_SIZE", 64))
    EPOCHS = 50

    LR_BACKBONE = 1e-4
    LR_HEAD = 2e-4
    WARMUP_EPOCHS = 3

    WEIGHT_DECAY = 1e-4
    MAX_NORM = 1.0

    NUM_CLASSES = 14
    IMG_SIZE = 224
    PRETRAINED = True
    VIT_NAME = "vit_base_patch16_224"
    UGA_INJECT_DEPTH = 1

    UNCERTAINTY_STRATEGY = "U-Ones"
    MIN_POSITIVE_SAMPLES = 5

    # uga params
    ALPHA = 1.0
    BETA = 0.5
    MIX_WARMUP_EPOCHS = 15
    LAMBDA_ATTENTION = 0.01

    # loss weights / smoothing
    LAMBDA_UNCERTAINTY = 0.10
    LAMBDA_U_EARLY = 0.02
    LAMBDA_CALIBRATION = 0.05
    LABEL_SMOOTH_EPS = 0.02
    POS_WEIGHT_WARMUP_EPOCHS = 8

    # attention entropy target mixing
    GAMMA = 0.3

    # train strategy
    UNFREEZE_BACKBONE_AT = 3

    NUM_WORKERS = int(os.environ.get("CHEX_NUM_WORKERS", 12))
    PERSISTENT_WORKERS = False
    PREFETCH_FACTOR = 4

    EARLY_STOP_PATIENCE = 5


Config.IMG_DIRS = [os.path.join(Config.FULL_ROOT, d) for d in Config.BATCH_DIRS]

PATHOLOGIES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


# dataset utilities
def _normalize_rel_key(rel: str) -> str:
    rel = rel.replace("\\", "/")
    if rel.startswith(Config.PREFIX):
        rel = rel[len(Config.PREFIX) :]
    if rel.startswith("train/"):
        rel = rel[len("train/") :]
    elif rel.startswith("valid/"):
        rel = rel[len("valid/") :]
    return rel


def _gather_all_images(roots: List[Path]) -> Dict[str, Path]:
    """build a unified index of image files across roots keyed by normalized relative path"""
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    index: Dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix in exts and path.is_file():
                try:
                    rel = path.relative_to(root).as_posix()
                except Exception: 
                    rel = path.name
                key = _normalize_rel_key(rel)
                if key not in index:
                    index[key] = path
    return index


class CheXpertFullDataset(Dataset):
    """chexpert csv + multi-root filesystem dataset with uncertainty processing"""

    def __init__(
        self,
        csv_path: str | Path,
        image_roots: List[str],
        transform: Optional[transforms.Compose] = None,
        uncertainty_strategy: str = "U-Ones",
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.df["Path"] = self.df["Path"].str.replace(Config.PREFIX, "", regex=False)
        self.df["key"] = self.df["Path"].apply(_normalize_rel_key)

        self.roots = [Path(r) for r in image_roots]
        self.transform = transform
        self.uncertainty_strategy = str(uncertainty_strategy)
        self.file_index = _gather_all_images(self.roots)
        if is_main():
            print(
                f"indexing {len(self.file_index):,} images across {len(self.roots)} roots",
                flush=True,
            )
        self.labels, self.u_labels = self.process_labels()
        self.class_weights = self.calculate_class_weights()

    def __len__(self) -> int:
        return len(self.df)

    def process_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """convert raw labels into targets and uncertainty codes"""
        raw = self.df[PATHOLOGIES].values.astype(float).copy()

        # u encodes: 0 known negative, 1 known positive, 2 uncertain, -1 missing/na
        u = np.full_like(raw, -1, dtype=np.float32)
        u[raw == 1.0] = 1.0
        u[raw == 0.0] = 0.0
        u[raw == -1.0] = 2.0
        u[np.isnan(raw)] = -1.0

        labels = raw.copy()
        strategy = self.uncertainty_strategy
        if strategy == "U-Ignore":
            pass
        elif strategy == "U-Zeros":
            labels[labels == -1.0] = 0.0
            labels[np.isnan(labels)] = 0.0
        elif strategy == "U-Ones":
            labels[labels == -1.0] = 1.0
            labels[np.isnan(labels)] = 0.0
        else:
            labels[labels == -1.0] = 0.0
            labels[np.isnan(labels)] = 0.0
        return labels.astype(np.float32), u

    def calculate_class_weights(self) -> torch.Tensor:
        """heuristic pos_weight per class to mitigate imbalance"""
        weights: List[float] = []
        for i in range(Config.NUM_CLASSES):
            labels_i = self.labels[:, i]
            valid = labels_i[~np.isnan(labels_i)]
            if len(valid) == 0:
                weights.append(1.0)
                continue
            pos = (valid == 1).sum()
            neg = (valid == 0).sum()
            total = len(valid)
            if pos == 0 or neg == 0:
                weights.append(1.0)
                continue
            pos_ratio = pos / total
            if pos_ratio < 0.01:
                pos_weight = math.sqrt(neg / pos)
                max_w = 20.0
            elif pos_ratio < 0.1:
                pos_weight = neg / pos
                max_w = 12.0
            else:
                pos_weight = total / (2 * pos)
                max_w = 8.0
            pos_weight = float(min(max(pos_weight, 1.0), max_w))
            weights.append(pos_weight)
        return torch.tensor(weights, dtype=torch.float32)

    def _resolve_image_path(self, idx: int) -> Optional[Path]:
        key = self.df.iloc[idx]["key"]
        path = self.file_index.get(key)
        if path is not None and path.exists():
            return path

        rel = self.df.iloc[idx]["Path"].replace("\\", "/")
        for root in self.roots:
            cand = (root / rel).resolve()
            if cand.exists():
                return cand

        short_rel = _normalize_rel_key(rel)
        for root in self.roots:
            cand = (root / short_rel).resolve()
            if cand.exists():
                return cand
        return None

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self._resolve_image_path(idx)
        if img_path is not None:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception: 
                img = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), "black")
        else:
            img = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), "black")

        if self.transform:
            img = self.transform(img)

        labels = torch.from_numpy(self.labels[idx]).float()
        ulabels = torch.from_numpy(self.u_labels[idx]).float()
        return img, labels, ulabels


def get_transforms(split: str = "train") -> transforms.Compose:
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),
                transforms.RandomCrop(Config.IMG_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# metrics / plots
def calculate_metrics_robust(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Dict[str, object]:
    """compute mean auc/precision/recall/f1 over pathologies with enough positives/negatives"""
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    aucs: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    valid_names: List[str] = []

    for i, name in enumerate(PATHOLOGIES):
        valid_mask = ~np.isnan(targets_np[:, i])
        if valid_mask.sum() < 10:
            continue
        y = targets_np[valid_mask, i]
        s = predictions_np[valid_mask, i]
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        if pos >= Config.MIN_POSITIVE_SAMPLES and neg >= Config.MIN_POSITIVE_SAMPLES:
            try:
                aucs.append(roc_auc_score(y, s))
                valid_names.append(name)
                bin_pred = (s >= 0.5).astype(int)
                precisions.append(precision_score(y, bin_pred, zero_division=0))
                recalls.append(recall_score(y, bin_pred, zero_division=0))
                f1s.append(f1_score(y, bin_pred, zero_division=0))
            except Exception: 
                # silently skip ill-conditioned metrics for stability
                pass

    return {
        "mean_auc": float(np.mean(aucs)) if aucs else 0.5,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "individual_aucs": dict(zip(valid_names, [float(a) for a in aucs])),
        "num_valid_pathologies": int(len(aucs)),
    }


def plot_roc_curves(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    save_path: str = "best_roc_curves_ugavit.png",
) -> float:
    """draw per-class roc curves; returns mean auc for plotted classes"""
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    aucs: List[float] = []

    for i, pathology in enumerate(PATHOLOGIES):
        valid_mask = ~np.isnan(targets_np[:, i])
        if valid_mask.sum() < Config.MIN_POSITIVE_SAMPLES:
            continue
        y = targets_np[valid_mask, i]
        s = predictions_np[valid_mask, i]
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        if pos >= Config.MIN_POSITIVE_SAMPLES and neg >= Config.MIN_POSITIVE_SAMPLES:
            fpr, tpr, _ = roc_curve(y, s)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{pathology}: {roc_auc:.3f}", alpha=0.7, linewidth=2)
            aucs.append(roc_auc)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (Mean AUC: {np.mean(aucs) if aucs else 0.0:.3f})")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if is_main():
        print(f"plotting ROC curves saved to {save_path}")
    return float(np.mean(aucs) if aucs else 0.0)


# uga components
class UncertaintyGuidedAttention(nn.Module):
    """multi-head attention modulated by predicted/label-driven uncertainty"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # maps class-level uncertainties into a per-head modulation signal
        self.unc_proc = nn.Sequential(
            nn.Linear(Config.NUM_CLASSES, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_heads),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, u_scores: Optional[torch.Tensor], uga_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        bsz, num_tokens, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # uncertainty-aware temperature + per-head gain
        if u_scores is not None and uga_scale > 0.0:
            mod = self.unc_proc(u_scores).unsqueeze(-1).unsqueeze(-1)
            attn = attn * (1.0 + (Config.ALPHA * uga_scale) * mod)
            u_mean = u_scores.mean(dim=-1, keepdim=True)
            temperature = math.sqrt(self.head_dim) * (1 + (Config.BETA * uga_scale) * u_mean)
            attn = attn / temperature.unsqueeze(-1).unsqueeze(-1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention entropy per-sample for weak regularization
        p = attn.mean(dim=1) + 1e-8
        ent = -(p * p.log()).sum(-1).mean(-1)  # [B]

        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, ent, math.log(num_tokens)


class UGAResidualBlock(nn.Module):
    """residual block that wraps uga attention + mlp with pathology-aware modulation"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = UncertaintyGuidedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
        self.pathology_mod = nn.Sequential(
            nn.Linear(Config.NUM_CLASSES, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
            nn.LayerNorm(dim),
        )
        self.pathology_gain = 0.10

    def load_from_vit_block(self, vit_block: nn.Module) -> None:
        """copy compatible weights from a timm vit block if available"""
        with torch.no_grad():
            for name in ("norm1", "norm2"):
                try:
                    getattr(self, name).load_state_dict(
                        getattr(vit_block, name).state_dict(), strict=False
                    )
                except Exception: 
                    pass
            try:
                self.attn.qkv.load_state_dict(vit_block.attn.qkv.state_dict(), strict=False)
                self.attn.proj.load_state_dict(vit_block.attn.proj.state_dict(), strict=False)
            except Exception: 
                pass
            try:
                self.mlp[0].weight.copy_(vit_block.mlp.fc1.weight)
                self.mlp[0].bias.copy_(vit_block.mlp.fc1.bias)
                self.mlp[3].weight.copy_(vit_block.mlp.fc2.weight)
                self.mlp[3].bias.copy_(vit_block.mlp.fc2.bias)
            except Exception: 
                pass

    def forward(
        self, x: torch.Tensor, u_scores: Optional[torch.Tensor], uga_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        xn = self.norm1(x)
        if u_scores is not None and self.pathology_gain > 0:
            # modulate token features by class-level uncertainty
            m = self.pathology_mod(u_scores).unsqueeze(1)
            xn = xn * (1.0 + self.pathology_gain * m)
        y, ent, max_ent = self.attn(xn, u_scores, uga_scale)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x, ent, max_ent


class ImprovedUGViT(nn.Module):
    """vit backbone with late-stage uga residual blocks and dual uncertainty heads"""

    def __init__(
        self,
        num_classes: int = 14,
        vit_name: Optional[str] = None,
        pretrained: bool = True,
        inject_depth: int = 1,
    ) -> None:
        super().__init__()
        vit_name = vit_name or Config.VIT_NAME
        self.backbone = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        self.num_classes = num_classes
        self.embed_dim = self.backbone.embed_dim

        total_blocks = len(self.backbone.blocks)
        k = max(1, min(inject_depth, total_blocks))
        self.inject_start = total_blocks - k

        tail_blocks = [self.backbone.blocks[i] for i in range(self.inject_start, total_blocks)]
        self.backbone.blocks = nn.ModuleList(self.backbone.blocks[: self.inject_start])

        sb = self.backbone.blocks[0]
        num_heads = getattr(sb.attn, "num_heads", 12)
        mlp_ratio = float(sb.mlp.fc1.out_features) / float(self.embed_dim)
        try:
            attn_drop = float(sb.attn.attn_drop.p)
        except Exception: 
            attn_drop = 0.0
        try:
            pos_drop = (
                float(self.backbone.pos_drop.p)
                if isinstance(self.backbone.pos_drop, nn.Dropout)
                else 0.0
            )
        except Exception: 
            pos_drop = 0.0

        self.uga_blocks = nn.ModuleList(
            [
                UGAResidualBlock(
                    self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=pos_drop,
                    attn_drop=attn_drop,
                )
                for _ in range(k)
            ]
        )
        # initialize then load from vit tail blocks where compatible
        for i in range(k):
            self.uga_blocks[i].load_state_dict(
                UGAResidualBlock(self.embed_dim, num_heads, mlp_ratio, pos_drop, attn_drop).state_dict(),
                strict=False,
            )
        for i in range(k):
            self.uga_blocks[i].load_from_vit_block(tail_blocks[i])

        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.embed_dim, num_classes))

        self.early_uncertainty = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 255),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(255, num_classes),
            nn.Sigmoid(),
        )
        self.final_uncertainty = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )

    @staticmethod
    def map_u_labels(u_labels: torch.Tensor) -> torch.Tensor:
        """map dataset uncertainty codes to [0,1] scores used by uga"""
        u = torch.zeros_like(u_labels)
        u[u_labels == 2.0] = 1.0
        u[u_labels == -1.0] = 0.5
        return u

    def _tokens_with_pos(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x = self.backbone.patch_embed(x)
        cls = self.backbone.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls, x), dim=1)

        pos = self.backbone.pos_embed
        if pos.shape[1] != x.shape[1]:
            if pos.shape[1] > x.shape[1]:
                pos = pos[:, : x.shape[1], :]
            else:
                pad = torch.zeros(
                    (1, x.shape[1] - pos.shape[1], pos.shape[2]),
                    device=pos.device,
                    dtype=pos.dtype,
                )
                pos = torch.cat([pos, pad], dim=1)
        x = x + pos
        x = self.backbone.pos_drop(x)
        return x

    def forward(  # pylint: disable=too-many-arguments
        self, images: torch.Tensor, u_labels: Optional[torch.Tensor] = None, mix: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float, torch.Tensor]:
        x = self._tokens_with_pos(images)
        pooled = x.mean(dim=1)
        u_pred_early = self.early_uncertainty(pooled)

        if self.training and (u_labels is not None):
            u_from_lbl = self.map_u_labels(u_labels)
            u_for_attn = (1.0 - mix) * u_from_lbl + mix * u_pred_early
        else:
            u_for_attn = u_pred_early

        for block in self.backbone.blocks:
            x = block(x)

        ents: List[torch.Tensor] = []
        for block in self.uga_blocks:
            x, ent, max_ent = block(x, u_for_attn, uga_scale=mix)
            ents.append(ent)
        ent_avg = torch.stack(ents, 0).mean(0) if ents else None

        x = self.norm(x)
        cls = x[:, 0]
        logits = self.classifier(cls)
        u_pred_final = self.final_uncertainty(cls)
        return logits, u_pred_final, ent_avg, math.log(x.shape[1]), u_pred_early


# losses
class SoftCalibrationLoss(nn.Module):
    """
    differentiable, multi-label ece surrogate:
        sum_b (mass_b / N) * (mean_p_b - mean_y_b)^2
    with soft gaussian membership to bins.
    """

    def __init__(self, n_bins: int = 15, sigma: Optional[float] = None, eps: float = 1e-8) -> None:
        super().__init__()
        self.n_bins = int(n_bins)
        self.eps = float(eps)
        self.sigma = float(sigma) if sigma is not None else (1.0 / n_bins) / 2.0

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = probs.device
        p = probs.clamp(1e-6, 1 - 1e-6)
        y = targets
        valid = ~torch.isnan(y)
        if valid.sum() == 0:
            return torch.zeros([], device=device)

        p = p[valid]
        y = y[valid]
        bin_centers = torch.linspace(0.0, 1.0, self.n_bins, device=device).unsqueeze(0)  # [1, B]
        p_col = p.unsqueeze(1)  # [N, 1]

        d2 = (p_col - bin_centers).pow(2)  # [N, B]
        w = torch.exp(-d2 / (2 * (self.sigma**2)))
        w = w / (w.sum(dim=1, keepdim=True) + self.eps)

        mass = w.sum(dim=0) + self.eps
        mean_p = (w * p_col).sum(dim=0) / mass
        mean_y = (w * y.unsqueeze(1)).sum(dim=0) / mass
        err = (mean_p - mean_y).pow(2)

        n = p.numel() + self.eps
        loss = (mass / n) * err
        return loss.sum()


class CompleteLossUG(nn.Module):
    """composite loss for classification + uncertainty + calibration + entropy"""

    def __init__(self, pos_weight: torch.Tensor, label_smooth_eps: float = 0.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight  # [C]
        self.label_smooth_eps = float(label_smooth_eps)
        self.calib = SoftCalibrationLoss(n_bins=15)

    def forward(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        u_pred: torch.Tensor,
        ent_per_sample: Optional[torch.Tensor],
        max_entropy: float,
        u_labels: Optional[torch.Tensor],
        u_pred_early: Optional[torch.Tensor] = None,
        pw_anneal: float = 1.0,
        attn_warmup: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = logits.device
        if self.label_smooth_eps > 0:
            targets = targets * (1.0 - self.label_smooth_eps) + 0.5 * self.label_smooth_eps

        pw_full = self.pos_weight.to(device).view(1, -1)
        pw_eff = 1.0 + pw_anneal * (pw_full - 1.0)

        bce_raw = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = torch.where(targets > 0.5, bce_raw * pw_eff, bce_raw).mean()

        probs = torch.sigmoid(logits)
        unc = nn.functional.mse_loss(u_pred, (probs - targets).abs())

        aux = torch.tensor(0.0, device=device)
        if u_pred_early is not None:
            aux = nn.functional.mse_loss(u_pred_early, (probs - targets).abs())

        attn_reg = torch.tensor(0.0, device=device)
        if (ent_per_sample is not None) and (u_labels is not None):
            u_scores = ImprovedUGViT.map_u_labels(u_labels).to(device)
            u_mean = u_scores.mean(dim=-1)
            target_ent = max_entropy * (Config.GAMMA + (1 - Config.GAMMA) * u_mean)
            attn_reg = (ent_per_sample - target_ent).abs().mean()

        cal = self.calib(probs, targets)

        total = (
            bce
            + Config.LAMBDA_UNCERTAINTY * unc
            + (Config.LAMBDA_ATTENTION * max(0.0, min(1.0, float(attn_warmup)))) * attn_reg
            + Config.LAMBDA_U_EARLY * aux
            + Config.LAMBDA_CALIBRATION * cal
        )
        brier = torch.tensor(0.0, device=device)  # kept for stable returns
        return total, bce, brier, unc, attn_reg, aux, cal


# dataloaders
def build_loaders(
    train_csv: str | Path, val_csv: str | Path, batch_size: int
) -> tuple[DataLoader, DataLoader, CheXpertFullDataset, CheXpertFullDataset, Optional[DistributedSampler], Optional[DistributedSampler]]:
    """construct training and validation dataloaders (ddp-aware)"""
    # torch dataloader workers with spawn ctx play nicer under ddp on some systems
    ctx = torch.multiprocessing.get_context("spawn")
    num_workers = int(Config.NUM_WORKERS)
    worker_args = dict(num_workers=num_workers, pin_memory=True, multiprocessing_context=ctx)
    if num_workers > 0:
        worker_args.update(persistent_workers=bool(Config.PERSISTENT_WORKERS))
        if Config.PREFETCH_FACTOR and Config.PREFETCH_FACTOR > 0:
            worker_args.update(prefetch_factor=int(Config.PREFETCH_FACTOR))

    tr_ds = CheXpertFullDataset(
        train_csv,
        Config.IMG_DIRS,
        get_transforms("train"),
        uncertainty_strategy=Config.UNCERTAINTY_STRATEGY,
    )
    va_ds = CheXpertFullDataset(
        val_csv,
        Config.IMG_DIRS,
        get_transforms("val"),
        uncertainty_strategy=Config.UNCERTAINTY_STRATEGY,
    )

    if is_dist():
        tr_samp = DistributedSampler(tr_ds, shuffle=True, drop_last=True)
        va_samp = DistributedSampler(va_ds, shuffle=False, drop_last=False)
        tr_dl = DataLoader(
            tr_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=tr_samp,
            drop_last=True,
            **worker_args,
        )
        va_dl = DataLoader(
            va_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=va_samp,
            drop_last=False,
            **worker_args,
        )
    else:
        tr_samp = None
        va_samp = None
        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True, **worker_args)
        va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False, **worker_args)
    return tr_dl, va_dl, tr_ds, va_ds, tr_samp, va_samp


# train / eval loops
@torch.no_grad()
def validate(
    model_for_eval: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: CompleteLossUG,
    epoch: int,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """validation with mixing/warmup aligned to training epoch index"""
    model_for_eval.eval()
    loss_sum = 0.0
    preds_list: List[torch.Tensor] = []
    targs_list: List[torch.Tensor] = []

    pbar = tqdm(loader, desc="Validation", disable=not is_main())
    mix = min(1.0, float(epoch + 1) / float(max(1, Config.MIX_WARMUP_EPOCHS)))
    pw_anneal = min(1.0, float(epoch + 1) / float(max(1, Config.POS_WEIGHT_WARMUP_EPOCHS)))

    for images, targets, ulabels in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        ulabels = ulabels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits, u_pred, ent, max_ent, u_pred_early = model_for_eval(
                images, u_labels=ulabels, mix=mix
            )
            loss, *_ = criterion(
                logits,
                targets,
                u_pred,
                ent,
                max_ent,
                ulabels,
                u_pred_early=u_pred_early,
                pw_anneal=pw_anneal,
                attn_warmup=mix,
            )
        loss_sum += float(loss.detach().cpu())
        preds_list.append(torch.sigmoid(logits).detach())
        targs_list.append(targets.detach())
        if is_main():
            pbar.set_postfix({"Loss": f"{float(loss):.4f}"})

    preds_local = (
        torch.cat(preds_list, dim=0) if preds_list else torch.zeros((0, Config.NUM_CLASSES), device=device)
    )
    targs_local = (
        torch.cat(targs_list, dim=0) if targs_list else torch.zeros((0, Config.NUM_CLASSES), device=device)
    )
    preds_all = all_gather_variable(preds_local)
    targs_all = all_gather_variable(targs_local)
    avg_loss = loss_sum / max(len(loader), 1)
    return avg_loss, preds_all, targs_all


def set_backbone_trainable(module: nn.Module, trainable: bool) -> None:
    """freeze or unfreeze timm backbone params"""
    for name, param in module.named_parameters():
        if name.startswith("backbone."):
            param.requires_grad_(trainable)


def init_classifier_bias_from_prevalence(module: nn.Module, prevalence: torch.Tensor) -> None:
    """initialize classifier bias with logit of prevalence to reduce early bias"""
    p = prevalence.clamp(1e-4, 1 - 1e-4)
    b = torch.log(p / (1 - p)).clamp_(-4.0, 4.0)
    if isinstance(module.classifier, nn.Sequential) and isinstance(module.classifier[-1], nn.Linear):
        with torch.no_grad():
            module.classifier[-1].bias.copy_(
                b.to(module.classifier[-1].bias.device, dtype=module.classifier[-1].bias.dtype)
            )


def compute_prevalence(labels_np: np.ndarray) -> torch.Tensor:
    """empirical per-class prevalence ignoring nans"""
    prev: List[float] = []
    for i in range(labels_np.shape[1]):
        li = labels_np[:, i]
        mask = ~np.isnan(li)
        if mask.sum() == 0:
            prev.append(0.01)
            continue
        y = li[mask]
        p = float((y == 1).sum()) / float(mask.sum())
        prev.append(max(1e-3, min(1 - 1e-3, p)))
    return torch.tensor(prev, dtype=torch.float32)


def train_one_epoch(  # pylint: disable=too-many-arguments, too-many-locals
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    criterion: CompleteLossUG,
    epoch: int,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """single-epoch training loop with gradient clipping and amp"""
    model.train()
    loss_sum = 0.0
    preds_list: List[torch.Tensor] = []
    targs_list: List[torch.Tensor] = []

    pbar = tqdm(loader, desc="Training", disable=not is_main())
    mix = min(1.0, float(epoch + 1) / float(max(1, Config.MIX_WARMUP_EPOCHS)))
    pw_anneal = min(1.0, float(epoch + 1) / float(max(1, Config.POS_WEIGHT_WARMUP_EPOCHS)))

    for images, targets, ulabels in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        ulabels = ulabels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits, u_pred, ent, max_ent, u_pred_early = model(images, u_labels=ulabels, mix=mix)
            loss, bce, _, unc, attn, aux, _ = criterion(
                logits,
                targets,
                u_pred,
                ent,
                max_ent,
                ulabels,
                u_pred_early=u_pred_early,
                pw_anneal=pw_anneal,
                attn_warmup=mix,
            )

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.MAX_NORM)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += float(loss.detach().cpu())
        with torch.no_grad():
            preds_list.append(torch.sigmoid(logits).detach())
            targs_list.append(targets.detach())

        if is_main():
            pbar.set_postfix(
                {
                    "Loss": f"{float(loss):.4f}",
                    "BCE": f"{float(bce):.4f}",
                    "Unc": f"{float(unc):.4f}",
                    "Attn": f"{float(attn):.4f}",
                    "AuxU": f"{float(aux):.4f}",
                    "mix": f"{mix:.2f}",
                    "pw": f"{pw_anneal:.2f}",
                }
            )

    preds_local = (
        torch.cat(preds_list, dim=0) if preds_list else torch.zeros((0, Config.NUM_CLASSES), device=device)
    )
    targs_local = (
        torch.cat(targs_list, dim=0) if targs_list else torch.zeros((0, Config.NUM_CLASSES), device=device)
    )
    preds_all = all_gather_variable(preds_local)
    targs_all = all_gather_variable(targs_local)
    avg_loss = loss_sum / max(len(loader), 1)
    return avg_loss, preds_all, targs_all


def train_model(  # pylint: disable=too-many-arguments, too-many-locals
    model: nn.Module,
    model_name: str,
    device: torch.device,
    tr_dl: DataLoader,
    va_dl: DataLoader,
    tr_ds: CheXpertFullDataset,
    epochs: int,
    ckpt_path: str | Path,
) -> None:
    """full training loop with warmup+cosine lr, early stopping, and checkpointing"""
    criterion = CompleteLossUG(
        pos_weight=tr_ds.class_weights.to(device),
        label_smooth_eps=Config.LABEL_SMOOTH_EPS,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # split lr for head vs. backbone
    module = model.module if isinstance(model, DDP) else model
    head_params: List[nn.Parameter] = []
    backbone_params: List[nn.Parameter] = []
    for name, param in module.named_parameters():
        if any(x in name for x in ("classifier", "early_uncertainty", "final_uncertainty", "uga_blocks")):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = (
        [{"params": backbone_params, "lr": Config.LR_BACKBONE}] if backbone_params else []
    ) + (
        [{"params": head_params, "lr": Config.LR_HEAD}]
        if head_params
        else [{"params": module.parameters(), "lr": Config.LR_BACKBONE}]
    )
    optimizer = optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR 

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=Config.WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - Config.WARMUP_EPOCHS), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[Config.WARMUP_EPOCHS])

    prevalence = compute_prevalence(tr_ds.labels)
    init_classifier_bias_from_prevalence(module, prevalence)

    set_backbone_trainable(module, trainable=False)

    best_auc = 0.0
    best_ep = 0
    patience = 0

    for ep in range(epochs):
        if isinstance(tr_dl.sampler, DistributedSampler):
            tr_dl.sampler.set_epoch(ep)
        if is_main():
            print(f"\nEpoch {ep + 1}/{epochs}")

        if ep == Config.UNFREEZE_BACKBONE_AT:
            set_backbone_trainable(module, trainable=True)
            if is_main():
                print("unfroze backbone parameters.")

        train_loss, train_preds_all, train_targs_all = train_one_epoch(
            model, tr_dl, device, optimizer, scaler, criterion, epoch=ep
        )

        # validate using the underlying module (same weights as ddp-wrapped model)
        val_loss, val_preds_all, val_targs_all = validate(module, va_dl, device, criterion, epoch=ep)

        if is_main():
            train_metrics = calculate_metrics_robust(train_preds_all.cpu(), train_targs_all.cpu())
            val_metrics = calculate_metrics_robust(val_preds_all.cpu(), val_targs_all.cpu())
            print(
                f"Train Loss: {train_loss:.4f}, AUC: {train_metrics['mean_auc']:.4f} "
                f"(valid {train_metrics['num_valid_pathologies']})"
            )
            print(
                "Val   Loss: "
                f"{val_loss:.4f}, AUC: {val_metrics['mean_auc']:.4f} "
                f"(valid {val_metrics['num_valid_pathologies']}) | "
                f"val_pred_mean={float(val_preds_all.mean()):.3f}, std={float(val_preds_all.std()):.3f}"
            )

            cur_auc = val_metrics["mean_auc"]
            if cur_auc > best_auc:
                best_auc = cur_auc
                best_ep = ep + 1
                patience = 0
                state = {
                    "epoch": ep + 1,
                    "model_state_dict": module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "config": {
                        "num_classes": Config.NUM_CLASSES,
                        "img_size": Config.IMG_SIZE,
                        "uncertainty_strategy": Config.UNCERTAINTY_STRATEGY,
                    },
                }
                torch.save(state, ckpt_path)
                print(f"  ✓ New best model saved! AUC: {best_auc:.4f}")
                plot_roc_curves(val_preds_all.cpu(), val_targs_all.cpu(), save_path="best_roc_curves_ugavit.png")
            else:
                patience += 1

        # early stop (broadcast flag when ddp)
        stop_flag = torch.tensor([0], device=device)
        if is_main() and patience >= Config.EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {ep + 1}")
            stop_flag[0] = 1
        if is_dist():
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item():
            break

        scheduler.step()
        barrier()

    if is_main():
        print(f"\nBest AUC for {model_name}: {best_auc:.4f} at epoch {best_ep}")


def warmup_pretrained_downloads() -> None:
    if not Config.PRETRAINED:
        return
    if not is_main():
        barrier()
        return
    try:
        _ = timm.create_model(Config.VIT_NAME, pretrained=True, num_classes=Config.NUM_CLASSES)
        del _
    except Exception as exc: 
        print(f"vit pretrained warmup skipped: {exc}")
    barrier()


def build_model_and_wrap(device: torch.device) -> tuple[nn.Module, nn.Module]:
    base = ImprovedUGViT(
        num_classes=Config.NUM_CLASSES,
        vit_name=Config.VIT_NAME,
        pretrained=Config.PRETRAINED,
        inject_depth=Config.UGA_INJECT_DEPTH,
    ).to(device)
    model = (
        DDP(base, device_ids=[int(os.environ.get("LOCAL_RANK", 0))], find_unused_parameters=True)
        if is_dist()
        else base
    )
    return model, base


def run_eval(epochs: Optional[int] = None, batch: Optional[int] = None, workers: Optional[int] = None) -> None:
    local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    assert Path(Config.TRAIN_CSV).exists() and Path(Config.VAL_CSV).exists(), (
        f"train.csv / valid.csv not found under {Config.BATCH_DIRS[0]}"
    )
    for droot in Config.IMG_DIRS:
        assert Path(droot).exists(), f"Image root missing: {droot}"

    warmup_pretrained_downloads()
    if epochs is None:
        epochs = Config.EPOCHS
    if batch is None:
        batch = Config.BATCH_SIZE
    if workers is not None:
        Config.NUM_WORKERS = workers 

    tr_dl, va_dl, tr_ds, _va_ds_unused, _tr_samp, _va_samp = build_loaders(
        Config.TRAIN_CSV, Config.VAL_CSV, batch
    )
    if is_main():
        print(
            f"Using device: {device} | GPUs: {world_size} | batch/GPU: {batch} | workers: {Config.NUM_WORKERS}"
        )

    model, base = build_model_and_wrap(device)
    ckpt = "best_ugvit_full.pth"
    try:
        train_model(model, "ugvit", device, tr_dl, va_dl, tr_ds, epochs, ckpt)
    finally:
        cleanup_distributed()
        del model, base
        torch.cuda.empty_cache()


# entrypoint
def seed_all(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_all(123)
    if is_main():
        print(f"Expecting CSVs at:\n  {Config.TRAIN_CSV}\n  {Config.VAL_CSV}")
        print("Image roots:")
        for droot in Config.IMG_DIRS:
            print("  -", droot)
    run_eval(epochs=Config.EPOCHS, batch=Config.BATCH_SIZE, workers=Config.NUM_WORKERS)
