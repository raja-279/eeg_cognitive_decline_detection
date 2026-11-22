"""
train_stage2.py  (v2 — CPU-fast, publication-ready)
----------------------------------------------------
Stage 2: Alzheimer's Disease (0) vs Frontotemporal Dementia (1)

Operates only on the diseased subset (A + F); healthy samples are
filtered out before training.

Key fixes over v1:
  ✓ --max_subjects / --max_epochs_per_subject CLI args now correctly
    propagated into EEGHierarchicalDataset (was silently ignored before)
  ✓ Epoch-time warning when running on CPU

Unchanged from v1:
  ✓ filter_stage2() used for all three splits
  ✓ Balanced sampler on stage2_label (AD vs FTD)
  ✓ Optional Stage 1 checkpoint warm-start (--stage1_ckpt)
  ✓ Differential LR: backbone 0.1× head LR
  ✓ Label-smoothing, CosineAnnealingLR, gradient clipping, early stopping

Usage (CPU, fast subset):
    python train_stage2.py \\
        --data_root /data/eeg \\
        --cache_dir /data/eeg_cache \\
        --output_dir runs/stage2 \\
        --stage1_ckpt runs/stage1/stage1_best.pt \\
        --max_subjects 10 \\
        --max_epochs_per_subject 400 \\
        --epochs 40 \\
        --batch_size 32
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
    filter_stage2,
    STAGE2_NAMES,
)
from model import build_model

STAGE2_CLASS_NAMES = ["Alzheimer's Disease", "Frontotemporal Dementia"]


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
    label_key: str   = "stage2_label",
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
    label_key:  str = "stage2_label",
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
# Optional Stage 1 → Stage 2 weight transfer
# ═══════════════════════════════════════════════════════════════════════════

def maybe_load_stage1(
    model:      nn.Module,
    ckpt_path:  str | None,
    device:     torch.device,
):
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        print(f"[Stage 2] Stage 1 checkpoint not found: {ckpt_path}  (skipping transfer)")
        return

    ckpt       = torch.load(ckpt_path, map_location=device)
    state_s1   = ckpt["model_state"]
    state_s2   = model.state_dict()

    n_total, n_loaded = 0, 0
    for k, v in state_s1.items():
        n_total += 1
        if k in state_s2 and state_s2[k].shape == v.shape:
            state_s2[k] = v
            n_loaded += 1

    model.load_state_dict(state_s2)
    print(f"[Stage 2] Loaded {n_loaded}/{n_total} layers from Stage 1 checkpoint.")


# ═══════════════════════════════════════════════════════════════════════════
# DataLoader builder  ← THE FIX: max_subjects & max_epochs_per_subject
#                       are now read from args and passed to the dataset.
# ═══════════════════════════════════════════════════════════════════════════

def build_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for Stage 2 (AD vs FTD).

    Both EEGHierarchicalDataset instances receive:
      - max_subjects            from --max_subjects
      - max_epochs_per_subject  from --max_epochs_per_subject

    filter_stage2() is applied to every split to remove Healthy samples.
    """

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

    # ── Filter to AD + FTD only ────────────────────────────────────────────
    print("\n[Stage 2] Filtering to diseased subjects only:")
    train_sub_s2 = filter_stage2(train_sub)
    val_sub_s2   = filter_stage2(val_sub)
    test_sub_s2  = filter_stage2(test_sub)

    if len(train_sub_s2) == 0:
        raise RuntimeError(
            "Stage 2 training set is empty after filtering.\n"
            "Ensure your dataset contains both AD and FTD subjects.\n"
            "If using --max_subjects, try increasing it so both classes "
            "are represented after the 70/15/15 split."
        )

    sampler = make_balanced_sampler(train_sub_s2, label_key="stage2_label")

    loader_kw = dict(
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    train_loader = DataLoader(
        train_sub_s2, sampler=sampler, drop_last=True, **loader_kw,
    )
    val_loader  = DataLoader(val_sub_s2,  shuffle=False, **loader_kw)
    test_loader = DataLoader(test_sub_s2, shuffle=False, **loader_kw)

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
    print(f"  Stage 2 — Alzheimer's Disease vs Frontotemporal Dementia")
    print(f"  Device             : {device}")
    print(f"  Data               : {args.data_root}")
    print(f"  max_subjects       : {args.max_subjects}")
    print(f"  max_epochs_per_sub : {args.max_epochs_per_subject}")
    print(f"{'='*64}")

    # ── DataLoaders ────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_loaders(args)

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    print(f"\n[Loaders] train={n_train}  val={n_val}  test={n_test}\n")

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

    maybe_load_stage1(model, args.stage1_ckpt, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # ── Optimisation ───────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    use_diff_lr = (
        args.architecture == "resnet"
        and (args.pretrained or args.stage1_ckpt)
    )
    if use_diff_lr:
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
    ckpt_path = os.path.join(args.output_dir, "stage2_best.pt")
    log_path  = os.path.join(args.output_dir, "stage2_log.csv")

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
            training=True, label_key="stage2_label", clip_grad=args.clip_grad,
        )
        va_loss, va_acc, va_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device,
            training=False, label_key="stage2_label",
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
    print(f"  Stage 2 complete.  Best val F1 = {best_val_f1:.4f}")
    print(f"{'='*64}\n")

    # ── Test evaluation ───────────────────────────────────────────────────
    print("[Test set evaluation]\n")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    results = evaluate(model, test_loader, device,
                       label_key="stage2_label",
                       class_names=STAGE2_CLASS_NAMES)

    print(f"Test Accuracy : {results['accuracy']:.4f}")
    print(f"Test Macro F1 : {results['macro_f1']:.4f}")
    print("\nPer-class report:")
    print(results["report"])

    report_path = os.path.join(args.output_dir, "stage2_test_report.txt")
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
        description="Stage 2: Alzheimer's Disease vs Frontotemporal Dementia",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ──────────────────────────────────────────────────────────────
    p.add_argument("--data_root",    required=True)
    p.add_argument("--cache_dir",    default=None)
    p.add_argument("--output_dir",   default="runs/stage2")
    p.add_argument("--stage1_ckpt",  default=None,
                   help="Stage 1 checkpoint for encoder weight transfer")

    # ── speed / subset ─────────────────────────────────────────────────────
    p.add_argument("--max_subjects", type=int, default=None,
                   help="Max subjects PER CLASS (C/A/F). None = all.")
    p.add_argument("--max_epochs_per_subject", type=int, default=None,
                   help="Max epochs per subject. None = all.")

    # ── preprocessing ─────────────────────────────────────────────────────
    p.add_argument("--epoch_duration",  type=float, default=2.0)
    p.add_argument("--overlap",         type=float, default=1.0)
    p.add_argument("--img_size",        type=int,   default=64)
    p.add_argument("--representation",  default="recurrence",
                   choices=["recurrence", "psd"])

    # ── model ─────────────────────────────────────────────────────────────
    p.add_argument("--architecture",    default="resnet",
                   choices=["resnet", "eegnet"])
    p.add_argument("--dropout",         type=float, default=0.5)
    p.add_argument("--pretrained",      action="store_true", default=True)
    p.add_argument("--no_pretrained",   dest="pretrained", action="store_false")

    # ── training ──────────────────────────────────────────────────────────
    p.add_argument("--epochs",          type=int,   default=40)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--clip_grad",       type=float, default=1.0)
    p.add_argument("--patience",        type=int,   default=12)
    p.add_argument("--augment_train",   action="store_true", default=True)
    p.add_argument("--no_augment",      dest="augment_train", action="store_false")
    p.add_argument("--num_workers",     type=int,   default=0)
    p.add_argument("--val_ratio",       type=float, default=0.15)
    p.add_argument("--test_ratio",      type=float, default=0.15)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--cpu",             action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
