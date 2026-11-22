"""
app.py — Streamlit demo for hierarchical EEG inference
-------------------------------------------------------
Run with:
    streamlit run app.py

Requirements:
    pip install streamlit torch torchvision numpy

Optional (for .set file upload):
    pip install mne scipy
"""

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — must match training
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE       = 64
REPRESENTATION = "recurrence"
ARCH           = "resnet"
DEVICE         = "cpu"

STAGE1_CKPT    = os.path.join("runs", "stage1", "stage1_best.pt")
STAGE2_CKPT    = os.path.join("runs", "stage2", "stage2_best.pt")

STAGE1_NAMES   = ["Healthy Control", "Cognitive Decline"]
STAGE2_NAMES   = ["Alzheimer's Disease", "Frontotemporal Dementia"]

RESULT_COLORS  = {
    "Healthy Control":         "#2ecc71",
    "Cognitive Decline":       "#e67e22",
    "Alzheimer's Disease":     "#e74c3c",
    "Frontotemporal Dementia": "#9b59b6",
}
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# Cached model loading (runs once per session)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    from model import build_model

    in_ch = 3 if REPRESENTATION == "recurrence" else 1

    def _load(path, n_cls):
        if not os.path.exists(path):
            st.error(f"Checkpoint not found: `{path}`")
            st.stop()
        m = build_model(
            architecture=ARCH, in_channels=in_ch,
            img_size=IMG_SIZE, num_classes=n_cls,
            dropout=0.0, pretrained=False,
        )
        raw = torch.load(path, map_location=DEVICE)
        if isinstance(raw, dict):
            for k in ("model_state", "model_state_dict", "state_dict"):
                if k in raw:
                    raw = raw[k]
                    break
        m.load_state_dict(raw)
        m.eval()
        return m

    return _load(STAGE1_CKPT, 2), _load(STAGE2_CKPT, 2)


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def process_npy(uploaded) -> np.ndarray:
    arr = np.load(uploaded).astype(np.float32)
    if arr.ndim == 4:
        arr = arr[len(arr) // 2]
    elif arr.ndim == 2:
        arr = arr[np.newaxis]
    return arr


def process_set(uploaded) -> np.ndarray:
    try:
        import mne
        mne.set_log_level("WARNING")
        from preprocessing import load_and_preprocess, extract_epochs, epoch_to_image
    except ImportError:
        st.error("MNE is required for .set files. Run: `pip install mne scipy`")
        st.stop()

    with tempfile.NamedTemporaryFile(suffix=".set", delete=False) as f:
        f.write(uploaded.read())
        tmp_path = f.name

    raw    = load_and_preprocess(tmp_path)
    sfreq  = float(raw.info["sfreq"])
    epochs = extract_epochs(raw, epoch_duration=2.0, overlap=1.0)
    os.unlink(tmp_path)

    if len(epochs) == 0:
        st.error("No epochs could be extracted from the .set file.")
        st.stop()

    ep  = epochs[len(epochs) // 2]
    img = epoch_to_image(ep, representation=REPRESENTATION,
                         img_size=IMG_SIZE, sfreq=sfreq)
    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(tensor, model_s1, model_s2):
    with torch.no_grad():
        probs_s1 = F.softmax(model_s1(tensor), dim=1)[0].tolist()
        pred_s1  = int(np.argmax(probs_s1))

        if pred_s1 == 0:
            return {
                "stage1_label": STAGE1_NAMES[pred_s1],
                "stage1_probs": probs_s1,
                "stage1_conf":  probs_s1[pred_s1],
                "stage2_label": None,
                "stage2_probs": None,
                "stage2_conf":  None,
                "final":        STAGE1_NAMES[pred_s1],
            }

        probs_s2 = F.softmax(model_s2(tensor), dim=1)[0].tolist()
        pred_s2  = int(np.argmax(probs_s2))
        return {
            "stage1_label": STAGE1_NAMES[pred_s1],
            "stage1_probs": probs_s1,
            "stage1_conf":  probs_s1[pred_s1],
            "stage2_label": STAGE2_NAMES[pred_s2],
            "stage2_probs": probs_s2,
            "stage2_conf":  probs_s2[pred_s2],
            "final":        STAGE2_NAMES[pred_s2],
        }


# ═══════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════

def render_prob_bar(label: str, prob: float, highlight: bool):
    color  = RESULT_COLORS.get(label, "#555")
    weight = "700" if highlight else "400"
    bar_w  = max(4, int(prob * 100))
    st.markdown(f"""
    <div style="margin:6px 0">
      <div style="display:flex;align-items:center;gap:10px">
        <span style="width:230px;font-weight:{weight};font-size:0.9rem">{label}</span>
        <div style="flex:1;background:#e9ecef;border-radius:4px;height:18px">
          <div style="width:{bar_w}%;background:{color};border-radius:4px;
                      height:18px;transition:width 0.4s ease"></div>
        </div>
        <span style="width:48px;text-align:right;font-weight:{weight};
                     font-size:0.9rem">{prob:.1%}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="EEG Classifier",
        page_icon="🧠",
        layout="centered",
    )

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style="font-size:2rem;font-weight:800;margin-bottom:0">
        🧠 EEG Dementia Classifier
    </h1>
    <p style="color:#666;margin-top:4px">
        Hierarchical neural network · Healthy → Alzheimer's / FTD
    </p>
    <hr style="margin:16px 0">
    """, unsafe_allow_html=True)

    # ── Model loading ─────────────────────────────────────────────────────
    model_s1, model_s2 = load_models()
    st.success("✅ Both models loaded and ready.", icon="✅")

    # ── File upload ───────────────────────────────────────────────────────
    st.markdown("### Upload EEG input")
    col1, col2 = st.columns(2)

    with col1:
        npy_file = st.file_uploader(
            "Pre-computed image (.npy)",
            type=["npy"],
            help="Shape: (C,H,W) or (N,C,H,W). Produced by dataset.py cache.",
        )
    with col2:
        set_file = st.file_uploader(
            "Raw EEG file (.set)  [requires MNE]",
            type=["set"],
            help="EEGLAB .set file. MNE + scipy must be installed.",
        )

    uploaded = npy_file or set_file
    if not uploaded:
        st.info("Upload a `.npy` or `.set` file above, then click **Run Inference**.")
        return

    # ── Run button ────────────────────────────────────────────────────────
    if st.button("▶  Run Inference", type="primary", use_container_width=True):
        with st.spinner("Processing…"):
            img    = process_npy(npy_file) if npy_file else process_set(set_file)
            tensor = to_tensor(img)
            result = run_inference(tensor, model_s1, model_s2)

        # ── Result card ───────────────────────────────────────────────────
        final   = result["final"]
        color   = RESULT_COLORS.get(final, "#333")

        st.markdown(f"""
        <div style="background:{color}18;border:2px solid {color};
                    border-radius:12px;padding:20px 24px;margin:20px 0">
            <div style="font-size:0.8rem;text-transform:uppercase;
                        letter-spacing:2px;color:{color};font-weight:700">
                Final Diagnosis
            </div>
            <div style="font-size:2rem;font-weight:800;color:{color};margin-top:4px">
                {final}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Stage breakdowns ──────────────────────────────────────────────
        st.markdown("#### Stage 1 — Healthy vs Cognitive Decline")
        pred_s1 = result["stage1_label"]
        for name, prob in zip(STAGE1_NAMES, result["stage1_probs"]):
            render_prob_bar(name, prob, highlight=(name == pred_s1))

        if result["stage2_label"]:
            st.markdown("#### Stage 2 — Alzheimer's Disease vs FTD")
            pred_s2 = result["stage2_label"]
            for name, prob in zip(STAGE2_NAMES, result["stage2_probs"]):
                render_prob_bar(name, prob, highlight=(name == pred_s2))
        else:
            st.markdown("#### Stage 2 — Not run (subject classified as Healthy)")

        # ── Raw numbers ───────────────────────────────────────────────────
        with st.expander("Raw confidence scores"):
            st.markdown("**Stage 1**")
            for name, p in zip(STAGE1_NAMES, result["stage1_probs"]):
                st.text(f"  {name:<30} {p:.4f}")
            if result["stage2_probs"]:
                st.markdown("**Stage 2**")
                for name, p in zip(STAGE2_NAMES, result["stage2_probs"]):
                    st.text(f"  {name:<30} {p:.4f}")

        # ── Input preview ─────────────────────────────────────────────────
        with st.expander("Input image preview"):
            img_display = img[0] if img.shape[0] >= 1 else img
            st.image(img_display, caption="Channel 0 of input image",
                     clamp=True, use_container_width=True)


if __name__ == "__main__":
    main()
