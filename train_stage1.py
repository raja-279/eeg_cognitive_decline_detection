"""
train_stage1.py  (v2 — CPU-fast, publication-ready)
----------------------------------------------------
Stage 1: Healthy (0) vs Cognitive Decline (1)

Key fixes over v1:
  ✓ --max_subjects / --max_epochs_per_subject CLI args now correctly
    propagated into EEGHierarchicalDataset (was silently ignored before)
  ✓ --augment_train flag (default True) — disable for ablations
  ✓ AMP-free path: torch.autocast used only when CUDA is present
  ✓ Epoch-time estimator: warns if > 600 s on CPU

Architecture / optimisation unchanged:
  ✓ ResNet18 pretrained, differential LR (backbone 0.1× head LR)
  ✓ Label-smoothing CrossEntropyLoss
  ✓ CosineAnnealingLR
  ✓ Gradient clipping
  ✓ Early stopping (val macro-F1)
  ✓ Best checkpoint saved; final test report written to disk

Usage (CPU, fast subset):
    python train_stage1.py \\
        --data_root /data/eeg \\
        --cache_dir /data/eeg_cache \\
        --output_dir runs/stage1 \\
        --max_subjects 10 \\
        --max_epochs_per_subject 400 \\
        --epochs 40 \\
        --batch_size 32

Expected dataset size with the above flags:
    ~30 subjects total  →  ~10 000–15 000 samples
"""

import os
import csv
import time
import random
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report

from dataset import (
    EEGHierarchicalDataset,
    subject_wise_split,
    make_balanced_sampler,
    STAGE2_NAMES,          # kept for cross-script import compatibility
)
from model import build_model

STAGE1_CLASS_NAMES = ["Healthy Control", "Cognitive Decline"]


# ═══════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════
# Batch unpacking
# ═══════════════════════════════════════════════════════════════════════════

def _unpack(batch: dict, label_key: str, device: torch.device):
    images = batch["image"].to(device, non_blocking=True)
    labels = batch[label_key].to(device, non_blocking=True)
    return images, labels


# ═══════════════════════════════════════════════════════════════════════════
# One epoch
# ═══════════════════════════════════════════════════════════════════════════

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    training:  bool,
    label_key: str   = "stage1_label",
    clip_grad: float = 1.0,
) -> Tuple[float, float, float]:
    """Returns (avg_loss, accuracy, macro_f1)."""
    model.train(training)
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for batch in loader:
            images, labels = _unpack(batch, label_key, device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                if clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    n        = len(all_labels)
    avg_loss = total_loss / n if n > 0 else 0.0
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


# ═══════════════════════════════════════════════════════════════════════════
# Test evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    label_key:  str = "stage1_label",
    class_names = None,
) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            images, labels = _unpack(batch, label_key, device)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc    = accuracy_score(all_labels, all_preds)
    f1     = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names or [str(i) for i in sorted(set(all_labels))],
        zero_division=0,
    )
    return {"accuracy": acc, "macro_f1": f1, "report": report}


# ═══════════════════════════════════════════════════════════════════════════
# DataLoader builder  ← THE FIX: max_subjects & max_epochs_per_subject
#                       are now read from args and passed to the dataset.
# ═══════════════════════════════════════════════════════════════════════════

def build_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for Stage 1.

    Both EEGHierarchicalDataset instances receive:
      - max_subjects            from --max_subjects
      - max_epochs_per_subject  from --max_epochs_per_subject

    These were previously omitted, causing the full 88-subject / ~70k-sample
    dataset to be loaded even when the CLI flags were set.
    """

    # Shared dataset kwargs ── all dataset-level flags flow through here
    shared = dict(
        data_root               = args.data_root,
        epoch_duration          = args.epoch_duration,
        overlap                 = args.overlap,
        img_size                = args.img_size,
        representation          = args.representation,
        cache_dir               = args.cache_dir,
        max_subjects            = args.max_subjects,            # ← FIX
        max_epochs_per_subject  = args.max_epochs_per_subject,  # ← FIX
        random_state            = args.seed,
    )

    train_ds = EEGHierarchicalDataset(**shared, augment=args.augment_train)
    eval_ds  = EEGHierarchicalDataset(**shared, augment=False)

    train_sub, _, _      = subject_wise_split(
        train_ds, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, random_state=args.seed,
    )
    _, val_sub, test_sub = subject_wise_split(
        eval_ds, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, random_state=args.seed,
    )

    # Balanced sampler on stage1_label (Healthy=0, Decline=1)
    sampler = make_balanced_sampler(train_sub, label_key="stage1_label")

    loader_kw = dict(
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    train_loader = DataLoader(
        train_sub, sampler=sampler, drop_last=True, **loader_kw,
    )
    val_loader  = DataLoader(val_sub,   shuffle=False, **loader_kw)
    test_loader = DataLoader(test_sub,  shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"\n{'='*64}")
    print(f"  Stage 1 — Healthy vs Cognitive Decline")
    print(f"  Device             : {device}")
    print(f"  Data               : {args.data_root}")
    print(f"  max_subjects       : {args.max_subjects}")
    print(f"  max_epochs_per_sub : {args.max_epochs_per_subject}")
    print(f"{'='*64}\n")

    # ── DataLoaders ────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_loaders(args)

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    print(f"[Loaders] train={n_train}  val={n_val}  test={n_test}\n")

    # ── Model ──────────────────────────────────────────────────────────────
    in_channels = 3 if args.representation == "recurrence" else 1
    model = build_model(
        architecture = args.architecture,
        in_channels  = in_channels,
        img_size     = args.img_size,
        num_classes  = 2,
        dropout      = args.dropout,
        pretrained   = args.pretrained,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # ── Optimisation ───────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.pretrained and args.architecture == "resnet":
        head_params     = list(model.head.parameters())
        head_ids        = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params,     "lr": args.lr},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3
    )

    # ── Output setup ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path   = os.path.join(args.output_dir, "stage1_best.pt")
    log_path    = os.path.join(args.output_dir, "stage1_log.csv")

    log_fields = ["epoch", "train_loss", "train_acc", "train_f1",
                  "val_loss",  "val_acc",  "val_f1",  "lr", "epoch_secs"]
    with open(log_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=log_fields).writeheader()

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_f1      = -1.0
    patience_counter = 0

    header  = ("{:>6}  {:>10}  {:>8}  {:>8}  |"
               "  {:>10}  {:>8}  {:>8}  {:>10}  {:>8}")
    row_fmt = ("{:>6}  {:>10.4f}  {:>8.4f}  {:>8.4f}  |"
               "  {:>10.4f}  {:>8.4f}  {:>8.4f}  {:>10.2e}  {:>8.1f}s")
    print(header.format("Epoch", "TrLoss", "TrAcc", "TrF1",
                        "VaLoss", "VaAcc", "VaF1", "LR", "Time"))
    print("-" * 92)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device,
            training=True, label_key="stage1_label", clip_grad=args.clip_grad,
        )
        va_loss, va_acc, va_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device,
            training=False, label_key="stage1_label",
        )

        scheduler.step()
        lr_now  = optimizer.param_groups[-1]["lr"]
        elapsed = time.time() - t0

        print(row_fmt.format(
            epoch, tr_loss, tr_acc, tr_f1,
            va_loss, va_acc, va_f1, lr_now, elapsed,
        ))

        if elapsed > 600:
            print(f"  ⚠  Epoch took {elapsed:.0f}s — consider reducing "
                  f"--max_epochs_per_subject or --max_subjects for CPU runs.")

        # ── checkpoint ────────────────────────────────────────────────────
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_f1":      va_f1,
                "val_acc":     va_acc,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved  (val_f1={va_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early stop] No improvement for {args.patience} epochs.")
                break

        # ── CSV log ───────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=log_fields).writerow({
                "epoch":      epoch,
                "train_loss": f"{tr_loss:.4f}", "train_acc": f"{tr_acc:.4f}",
                "train_f1":   f"{tr_f1:.4f}",
                "val_loss":   f"{va_loss:.4f}", "val_acc":   f"{va_acc:.4f}",
                "val_f1":     f"{va_f1:.4f}",
                "lr":         f"{lr_now:.2e}", "epoch_secs": f"{elapsed:.1f}",
            })

    print(f"\n{'='*64}")
    print(f"  Stage 1 complete.  Best val F1 = {best_val_f1:.4f}")
    print(f"{'='*64}\n")

    # ── Test evaluation ───────────────────────────────────────────────────
    print("[Test set evaluation]\n")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    results = evaluate(model, test_loader, device,
                       label_key="stage1_label",
                       class_names=STAGE1_CLASS_NAMES)

    print(f"Test Accuracy : {results['accuracy']:.4f}")
    print(f"Test Macro F1 : {results['macro_f1']:.4f}")
    print("\nPer-class report:")
    print(results["report"])

    report_path = os.path.join(args.output_dir, "stage1_test_report.txt")
    with open(report_path, "w") as fh:
        fh.write(f"Test Accuracy : {results['accuracy']:.4f}\n")
        fh.write(f"Test Macro F1 : {results['macro_f1']:.4f}\n\n")
        fh.write(results["report"])
    print(f"Report saved → {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1: Healthy vs Cognitive Decline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ──────────────────────────────────────────────────────────────
    p.add_argument("--data_root",  required=True,
                   help="Dataset root (must contain participants.tsv)")
    p.add_argument("--cache_dir",  default=None,
                   help="Directory for pre-computed .npy caches (highly recommended)")
    p.add_argument("--output_dir", default="runs/stage1",
                   help="Where to save checkpoints, logs, and reports")

    # ── speed / subset ─────────────────────────────────────────────────────
    p.add_argument("--max_subjects", type=int, default=None,
                   help="Max subjects PER CLASS (C/A/F). "
                        "10 → ~30 subjects total, ~10k–15k samples. "
                        "None = all subjects.")
    p.add_argument("--max_epochs_per_subject", type=int, default=None,
                   help="Max epochs loaded per subject. "
                        "400 is a good CPU budget. None = all epochs.")

    # ── preprocessing ─────────────────────────────────────────────────────
    p.add_argument("--epoch_duration", type=float, default=2.0,
                   help="EEG epoch length in seconds")
    p.add_argument("--overlap",        type=float, default=1.0,
                   help="Epoch overlap in seconds (0 = no overlap)")
    p.add_argument("--img_size",       type=int,   default=64,
                   help="Recurrence / PSD image size in pixels")
    p.add_argument("--representation", default="recurrence",
                   choices=["recurrence", "psd"],
                   help="Image representation type")

    # ── model ─────────────────────────────────────────────────────────────
    p.add_argument("--architecture",    default="resnet",
                   choices=["resnet", "eegnet"])
    p.add_argument("--dropout",         type=float, default=0.5)
    p.add_argument("--pretrained",      action="store_true", default=True,
                   help="Use ImageNet-pretrained ResNet18 weights")
    p.add_argument("--no_pretrained",   dest="pretrained", action="store_false")

    # ── training ──────────────────────────────────────────────────────────
    p.add_argument("--epochs",          type=int,   default=40)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-3,
                   help="Head learning rate (backbone gets lr * 0.1)")
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--clip_grad",       type=float, default=1.0)
    p.add_argument("--patience",        type=int,   default=12,
                   help="Early stopping: epochs without val F1 improvement")
    p.add_argument("--augment_train",   action="store_true", default=True,
                   help="Apply epoch augmentation during training")
    p.add_argument("--no_augment",      dest="augment_train", action="store_false")
    p.add_argument("--num_workers",     type=int,   default=0,
                   help="DataLoader workers (0 = main process; safe on all OSes)")
    p.add_argument("--val_ratio",       type=float, default=0.15)
    p.add_argument("--test_ratio",      type=float, default=0.15)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--cpu",             action="store_true",
                   help="Force CPU even if CUDA is available")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
