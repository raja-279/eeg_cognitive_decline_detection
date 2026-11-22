"""
predict.py — Hierarchical EEG Inference (CPU-only, no training code)
---------------------------------------------------------------------
Usage:
    python predict.py --input_npy sample.npy
    python predict.py --input_set subject.set

Hierarchical logic:
    Stage 1: Healthy (0) vs Cognitive Decline (1)
        └─ If Healthy → stop, report Healthy
        └─ If Decline → Stage 2: Alzheimer's (0) vs FTD (1)

Output:
    Stage 1: Diseased  (p=0.82)
    Stage 2: Alzheimer's Disease  (p=0.71)
    Final  : Alzheimer's Disease
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — must match the values used during training
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE       = 64
REPRESENTATION = "recurrence"   # "recurrence" | "psd"
ARCH           = "resnet"        # "resnet"     | "eegnet"
DEVICE         = "cpu"

STAGE1_CKPT    = os.path.join("runs", "stage1", "stage1_best.pt")
STAGE2_CKPT    = os.path.join("runs", "stage2", "stage2_best.pt")

STAGE1_NAMES   = ["Healthy Control", "Cognitive Decline"]
STAGE2_NAMES   = ["Alzheimer's Disease", "Frontotemporal Dementia"]
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, num_classes: int) -> torch.nn.Module:
    """
    Load a checkpoint saved by train_stage1/2.py.

    Handles both save formats:
      - torch.save(model.state_dict(), path)
      - torch.save({"model_state": ..., ...}, path)
      - torch.save({"model_state_dict": ..., ...}, path)
    """
    from model import build_model

    if not os.path.exists(ckpt_path):
        sys.exit(f"[ERROR] Checkpoint not found: {ckpt_path}")

    in_channels = 3 if REPRESENTATION == "recurrence" else 1

    model = build_model(
        architecture = ARCH,
        in_channels  = in_channels,
        img_size     = IMG_SIZE,
        num_classes  = num_classes,
        dropout      = 0.0,       # disabled at inference
        pretrained   = False,     # weights come from checkpoint
    )

    raw = torch.load(ckpt_path, map_location=DEVICE)

    # Unwrap dict if needed
    if isinstance(raw, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in raw:
                raw = raw[key]
                break

    model.load_state_dict(raw)
    model.eval()
    print(f"  Loaded  {ckpt_path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _from_set_file(set_path: str) -> np.ndarray:
    """
    Load a .set file → preprocess → return first epoch as image (C, H, W).
    Requires MNE to be installed.
    """
    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError:
        sys.exit("[ERROR] MNE is required for --input_set. "
                 "Install with:  pip install mne")

    from preprocessing import (load_and_preprocess, extract_epochs,
                                epoch_to_image)

    print(f"  Loading .set file: {set_path}")
    raw    = load_and_preprocess(set_path)
    sfreq  = float(raw.info["sfreq"])
    epochs = extract_epochs(raw, epoch_duration=2.0, overlap=1.0)

    if len(epochs) == 0:
        sys.exit("[ERROR] No epochs extracted from the .set file.")

    # Use the middle epoch — more representative than edge epochs
    ep  = epochs[len(epochs) // 2]
    img = epoch_to_image(ep, representation=REPRESENTATION,
                         img_size=IMG_SIZE, sfreq=sfreq)
    return img   # (C, H, W) float32


def _from_npy(npy_path: str) -> np.ndarray:
    """
    Load a pre-computed image from a .npy file.

    Accepted shapes:
      (C, H, W)          — single image, used directly
      (N, C, H, W)       — batch; the middle sample is used
      (H, W)             — single-channel; expanded to (1, H, W)
    """
    if not os.path.exists(npy_path):
        sys.exit(f"[ERROR] File not found: {npy_path}")

    arr = np.load(npy_path).astype(np.float32)
    print(f"  Loaded  {npy_path}  shape={arr.shape}")

    if arr.ndim == 4:          # (N, C, H, W) — pick middle
        arr = arr[len(arr) // 2]
    elif arr.ndim == 2:        # (H, W)       — add channel dim
        arr = arr[np.newaxis]
    elif arr.ndim != 3:
        sys.exit(f"[ERROR] Unexpected array shape {arr.shape}. "
                 "Expected (C,H,W), (N,C,H,W), or (H,W).")

    return arr   # (C, H, W) float32


def prepare_tensor(img: np.ndarray) -> torch.Tensor:
    """Normalise to [0,1] if needed and convert to (1, C, H, W) tensor."""
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,C,H,W)


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical inference
# ═══════════════════════════════════════════════════════════════════════════

def predict(image_tensor: torch.Tensor,
            model_s1: torch.nn.Module,
            model_s2: torch.nn.Module) -> None:
    """
    Run hierarchical inference and print results.

    Stage 1 → if Healthy: done.
    Stage 1 → if Decline: run Stage 2.
    """
    print()
    print("─" * 48)

    with torch.no_grad():
        # ── Stage 1 ───────────────────────────────────────────────────────
        logits_s1 = model_s1(image_tensor)                   # (1, 2)
        probs_s1  = F.softmax(logits_s1, dim=1)[0]           # (2,)
        pred_s1   = int(probs_s1.argmax())
        conf_s1   = float(probs_s1[pred_s1])
        label_s1  = STAGE1_NAMES[pred_s1]

        print(f"  Stage 1 : {label_s1:<26}  (p={conf_s1:.2f})")

        if pred_s1 == 0:
            # Healthy — no need for Stage 2
            print(f"  Stage 2 : N/A  (subject is Healthy)")
            print(f"  Final   : {label_s1}")
            print("─" * 48)
            return

        # ── Stage 2 ───────────────────────────────────────────────────────
        logits_s2 = model_s2(image_tensor)                   # (1, 2)
        probs_s2  = F.softmax(logits_s2, dim=1)[0]           # (2,)
        pred_s2   = int(probs_s2.argmax())
        conf_s2   = float(probs_s2[pred_s2])
        label_s2  = STAGE2_NAMES[pred_s2]

        print(f"  Stage 2 : {label_s2:<26}  (p={conf_s2:.2f})")
        print(f"  Final   : {label_s2}")
        print("─" * 48)

        # Full probability breakdown
        print()
        print("  Full probability breakdown:")
        for name, p in zip(STAGE1_NAMES, probs_s1.tolist()):
            bar = "█" * int(p * 20)
            print(f"    {name:<28} {p:.3f}  {bar}")
        print()
        for name, p in zip(STAGE2_NAMES, probs_s2.tolist()):
            bar = "█" * int(p * 20)
            print(f"    {name:<28} {p:.3f}  {bar}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Hierarchical EEG prediction (Healthy → AD/FTD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_npy", metavar="PATH",
                       help="Pre-computed image .npy file  (C×H×W or N×C×H×W)")
    group.add_argument("--input_set", metavar="PATH",
                       help="Raw EEG .set file (requires MNE)")

    p.add_argument("--stage1_ckpt", default=STAGE1_CKPT,
                   help="Path to Stage 1 checkpoint")
    p.add_argument("--stage2_ckpt", default=STAGE2_CKPT,
                   help="Path to Stage 2 checkpoint")
    p.add_argument("--img_size",    type=int, default=IMG_SIZE)
    p.add_argument("--representation", default=REPRESENTATION,
                   choices=["recurrence", "psd"])
    p.add_argument("--arch",        default=ARCH,
                   choices=["resnet", "eegnet"])
    return p.parse_args()


def main():
    args = parse_args()

    # Allow CLI overrides of the config block
    global IMG_SIZE, REPRESENTATION, ARCH
    IMG_SIZE       = args.img_size
    REPRESENTATION = args.representation
    ARCH           = args.arch

    print()
    print("=" * 48)
    print("  EEG Hierarchical Inference")
    print(f"  representation : {REPRESENTATION}")
    print(f"  img_size       : {IMG_SIZE}")
    print(f"  architecture   : {ARCH}")
    print(f"  device         : {DEVICE}")
    print("=" * 48)

    # ── Load models ───────────────────────────────────────────────────────
    print("\n[Loading models]")
    model_s1 = load_model(args.stage1_ckpt, num_classes=2)
    model_s2 = load_model(args.stage2_ckpt, num_classes=2)

    # ── Load input ────────────────────────────────────────────────────────
    print("\n[Loading input]")
    if args.input_npy:
        img = _from_npy(args.input_npy)
    else:
        img = _from_set_file(args.input_set)

    tensor = prepare_tensor(img)
    print(f"  Input tensor   : {tuple(tensor.shape)}")

    # ── Inference ─────────────────────────────────────────────────────────
    print("\n[Inference]")
    predict(tensor, model_s1, model_s2)
    print()


if __name__ == "__main__":
    main()
