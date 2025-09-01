"""
benchmarks: CheXpert training for DenseNet, ResNet, and ViT backbones.
"""

from __future__ import annotations

import math
import os
import random
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


matplotlib.use("Agg")
import matplotlib.pyplot as plt



warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# nccl/env defaults
for var in ("NCCL_BLOCKING_WAIT", "NCCL_ASYNC_ERROR_HANDLING"):
    if var in os.environ:
        os.environ.pop(var, None)

os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

# optional acceleration features: safe to skip if unavailable
try:
    torch.set_float32_matmul_precision("high")
except Exception:  # pylint: disable=broad-except
    pass
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:  # pylint: disable=broad-except
    pass

torch.backends.cudnn.benchmark = True


# ddp helpers
def is_dist() -> bool:
    """return True if torch.distributed is initialized"""
    return dist.is_available() and dist.is_initialized()


def is_main() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


def init_distributed() -> Tuple[int, int]:

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
    """destroy process group if initialized"""
    try:
        if is_dist():
            barrier()
            dist.destroy_process_group()
    except Exception:  # pylint: disable=broad-except
        pass


# reproducibility
def seed_all(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# configuration
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
    LR_HEAD = 5e-4
    WEIGHT_DECAY = 1e-4
    MAX_NORM = 1.0

    NUM_CLASSES = 14
    IMG_SIZE = 224
    PRETRAINED = True

    UNCERTAINTY_STRATEGY = "U-Ones"
    MIN_POSITIVE_SAMPLES = 5

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


# dataset helpers
def _normalize_rel_key(rel: str) -> str:
    """normalize a relative path key to be independent of split and prefix"""
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
                except Exception:  # pylint: disable=broad-except
                    rel = path.name
                key = _normalize_rel_key(rel)
                if key not in index:
                    index[key] = path
    return index


class CheXpertFullDataset(Dataset):
    """chexpert csv + multi-root filesystem dataset (labels only, no uncertainty tensor)"""

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
        self.labels, _unused_u = self.process_labels()
        self.class_weights = self.calculate_class_weights()

    def __len__(self) -> int:
        return len(self.df)

    def process_labels(self) -> tuple[np.ndarray, np.ndarray]:
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
                max_w = 50.0
            elif pos_ratio < 0.1:
                pos_weight = neg / pos
                max_w = 20.0
            else:
                pos_weight = total / (2 * pos)
                max_w = 10.0
            weights.append(float(min(max(pos_weight, 1.0), max_w)))
        return torch.tensor(weights, dtype=torch.float32)

    def _resolve_image_path(self, idx: int) -> Optional[Path]:
        key = self.df.iloc[idx]["key"]
        path = self.file_index.get(key)
        if path is not None and path.exists():
            return path

        rel = self.df.iloc[idx]["Path"].replace("\\", "/")
        for root in self.roots:
            candidate = (root / rel).resolve()
            if candidate.exists():
                return candidate

        short_rel = _normalize_rel_key(rel)
        for root in self.roots:
            candidate = (root / short_rel).resolve()
            if candidate.exists():
                return candidate
        return None

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self._resolve_image_path(idx)
        if img_path is not None:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:  # pylint: disable=broad-except
                img = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), "black")
        else:
            img = Image.new("RGB", (Config.IMG_SIZE, Config.IMG_SIZE), "black")

        if self.transform:
            img = self.transform(img)

        labels = torch.from_numpy(self.labels[idx]).float()
        return img, labels


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


# distributed gather for variable-length batches
def all_gather_variable(tensor: torch.Tensor) -> torch.Tensor:
    """
    gather variable-length [N, C] tensors across ranks by padding to max N.

    this avoids needing a separate collective for lengths and handles empty shards.
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


# metrics & plots
def calculate_metrics_robust(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, object]:
    """compute mean auc over pathologies with enough positives/negatives"""
    preds = predictions.detach().cpu().numpy()
    targs = targets.detach().cpu().numpy()
    aucs: List[float] = []
    names: List[str] = []

    for i, name in enumerate(PATHOLOGIES):
        mask = ~np.isnan(targs[:, i])
        if mask.sum() < 10:
            continue
        y = targs[mask, i]
        s = preds[mask, i]
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        if pos >= Config.MIN_POSITIVE_SAMPLES and neg >= Config.MIN_POSITIVE_SAMPLES:
            try:
                aucs.append(roc_auc_score(y, s))
                names.append(name)
            except Exception:  # pylint: disable=broad-except
                # skip ill-conditioned folds for stability
                pass

    return {
        "mean_auc": float(np.mean(aucs)) if aucs else 0.5,
        "individual_aucs": dict(zip(names, [float(a) for a in aucs])),
        "num_valid_pathologies": int(len(aucs)),
    }


def plot_roc_curves(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    save_path: str = "best_roc_curves_benchmark.png",
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


# models
def _tv_densenet121(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import DenseNet121_Weights, densenet121

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        return densenet121(weights=weights)
    except Exception:  # pylint: disable=broad-except
        from torchvision.models import densenet121 as _dn  # type: ignore

        return _dn(pretrained=pretrained)


def _tv_resnet50(pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        return resnet50(weights=weights)
    except Exception:  # pylint: disable=broad-except
        from torchvision.models import resnet50 as _rn  # type: ignore

        return _rn(pretrained=pretrained)


class CheXpertDensenet(nn.Module):
    """densenet121 with linear head"""

    def __init__(self, num_classes: int = 14, pretrained: bool = True) -> None:
        super().__init__()
        net = _tv_densenet121(pretrained=pretrained)
        in_feats = net.classifier.in_features
        net.classifier = nn.Linear(in_feats, num_classes)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CheXpertViT(nn.Module):
    """timm vit wrapper"""

    def __init__(
        self, num_classes: int = 14, model_name: str = "vit_base_patch16_224", pretrained: bool = True
    ) -> None:
        super().__init__()
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CheXpertResNet(nn.Module):
    """resnet50 with linear head"""

    def __init__(self, num_classes: int = 14, pretrained: bool = True) -> None:
        super().__init__()
        net = _tv_resnet50(pretrained=pretrained)
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, num_classes)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def warmup_pretrained_downloads() -> None:
    """preload common pretrained weights so ddp workers don't thundering-herd"""
    if not Config.PRETRAINED:
        return
    if not is_main():
        barrier()
        return
    try:
        _ = _tv_densenet121(pretrained=True)
        del _
    except Exception as exc:  # pylint: disable=broad-except
        print(f"densenet skip: {exc}")
    try:
        _ = _tv_resnet50(pretrained=True)
        del _
    except Exception as exc:  # pylint: disable=broad-except
        print(f"resnet skip: {exc}")
    try:
        _ = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=Config.NUM_CLASSES
        )
        del _
    except Exception as exc:  # pylint: disable=broad-except
        print(f"vit skip: {exc}")
    barrier()


# dataloaders
def build_loaders(
    train_csv: str | Path, val_csv: str | Path, batch_size: int
) -> Tuple[DataLoader, DataLoader, CheXpertFullDataset, CheXpertFullDataset, Optional[DistributedSampler], Optional[DistributedSampler]]:
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
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """validation over a dataloader"""
    model.eval()
    loss_sum = 0.0
    preds_list: List[torch.Tensor] = []
    targs_list: List[torch.Tensor] = []

    pbar = tqdm(loader, desc="Validation", disable=not is_main())
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, targets)
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


def train_one_epoch(  # pylint: disable=too-many-arguments
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    criterion: nn.Module,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """single-epoch training loop with gradient clipping and amp"""
    model.train()
    loss_sum = 0.0
    preds_list: List[torch.Tensor] = []
    targs_list: List[torch.Tensor] = []

    pbar = tqdm(loader, desc="Training", disable=not is_main())
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.MAX_NORM)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += float(loss.detach().cpu())
        with torch.no_grad():
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


def train_model(  # pylint: disable=too-many-arguments, too-many-locals
    model: nn.Module,
    model_name: str,
    device: torch.device,
    tr_dl: DataLoader,
    va_dl: DataLoader,
    tr_ds: CheXpertFullDataset,
    epochs: int,
    artifact_dir: str | Path,
) -> None:
    """full training loop with cosine lr, early stopping, and checkpointing"""
    os.makedirs(artifact_dir, exist_ok=True)
    ckpt_path = os.path.join(artifact_dir, f"best_{model_name}_full.pth")
    roc_path = os.path.join(artifact_dir, f"best_roc_curves_{model_name}.png")

    criterion = nn.BCEWithLogitsLoss(pos_weight=tr_ds.class_weights.to(device))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # split lr for head vs. backbone
    module = model.module if isinstance(model, DDP) else model
    head_params: List[nn.Parameter] = []
    backbone_params: List[nn.Parameter] = []
    for name, param in module.named_parameters():
        if any(x in name for x in ("classifier", "head", "fc", "classif")):
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_auc = 0.0
    best_ep = 0
    patience = 0

    for ep in range(epochs):
        if isinstance(tr_dl.sampler, DistributedSampler):
            tr_dl.sampler.set_epoch(ep)
        if is_main():
            print(f"\nEpoch {ep + 1}/{epochs}")

        train_loss, train_preds_all, train_targs_all = train_one_epoch(
            model, tr_dl, device, optimizer, scaler, criterion
        )
        val_loss, val_preds_all, val_targs_all = validate(model, va_dl, device, criterion)

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
                plot_roc_curves(val_preds_all.cpu(), val_targs_all.cpu(), save_path=roc_path)
            else:
                patience += 1

        # sync early stopping across ranks
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



# orchestrator
def run_eval(  # pylint: disable=too-many-arguments
    epochs: Optional[int] = None, batch: Optional[int] = None, workers: Optional[int] = None
) -> None:
    """main entrypoint: setup, dataloaders, training for each model, teardown"""
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
        Config.NUM_WORKERS = workers  # type: ignore[assignment]

    tr_dl, va_dl, tr_ds, va_ds_unused, tr_samp, va_samp = build_loaders(
        Config.TRAIN_CSV, Config.VAL_CSV, batch
    )
    if is_main():
        print(
            f"Using device: {device} | GPUs: {world_size} | batch/GPU: {batch} | "
            f"workers: {Config.NUM_WORKERS}"
        )

    models = {
        "vit": lambda: CheXpertViT(
            num_classes=Config.NUM_CLASSES, model_name="vit_base_patch16_224", pretrained=Config.PRETRAINED
        ),
        "densenet": lambda: CheXpertDensenet(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED),
        "resnet": lambda: CheXpertResNet(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED),
    }

    for name, ctor in models.items():
        if is_main():
            print("\n" + "=" * 70 + f"\nModel: {name}\n" + "=" * 70)
        base = ctor().to(device)
        model = (
            DDP(base, device_ids=[int(os.environ.get("LOCAL_RANK", 0))]) if is_dist() else base
        )

        artifact_dir = f"artifacts_{name}"
        try:
            train_model(model, name, device, tr_dl, va_dl, tr_ds, epochs, artifact_dir)
        finally:
            del model, base
            torch.cuda.empty_cache()
            barrier()

    cleanup_distributed()


# main
if __name__ == "__main__":
    seed_all(123)
    if is_main():
        print(f"Expecting CSVs at:\n  {Config.TRAIN_CSV}\n  {Config.VAL_CSV}")
        print("Image roots:")
        for droot in Config.IMG_DIRS:
            print("  -", droot)
    run_eval(epochs=Config.EPOCHS, batch=Config.BATCH_SIZE, workers=Config.NUM_WORKERS)
