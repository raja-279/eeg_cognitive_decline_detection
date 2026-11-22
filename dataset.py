"""
dataset.py  (v5 — speed-optimised)
-----------------------------------
Key changes vs v4:
  ─────────────────────────────────────────────────────────────
  SPEED:
    • max_subjects  – cap the number of subjects loaded per
                      class group (C / A / F) so experiments
                      run on a balanced ~30-subject slice.
                      Set max_subjects=None to use all subjects
                      (original behaviour, fully backwards-compat).
    • max_epochs_per_subject – cap epochs per subject to prevent
                      one long subject from dominating the dataset.
    • num_workers   – passed through to get_dataloaders(); on
                      Linux/Mac set to os.cpu_count()//2 for 2-4×
                      faster data loading. On Windows keep at 0.
    • pin_memory    – enabled automatically when CUDA is available.
    • prefetch_factor – set to 2 when num_workers > 0 for
                      pipeline overlap between CPU and GPU.
    • labels_array caching – property now caches np.array once
                      instead of rebuilding on every call to
                      make_balanced_sampler.

  CORRECTNESS preserved:
    • All cache paths, tags, mmap logic unchanged.
    • Hierarchical labels (stage1 / stage2) unchanged.
    • subject_wise_split / make_balanced_sampler / filter_stage2
      / mixup_collate all unchanged.
    • train_stage1.py / train_stage2.py require zero edits.
  ─────────────────────────────────────────────────────────────

Hierarchical labels:
  stage1_label : 0=Healthy, 1=Cognitive Decline
  stage2_label : 0=Alzheimer's, 1=FTD, -1=N/A (Healthy)

Quick-start (30 subjects, cached):
    ds = EEGHierarchicalDataset(
        data_root  = "/data/eeg",
        cache_dir  = "/data/eeg_cache",
        max_subjects = 10,           # ← 10 per class → ~30 total
    )
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import (Dataset, DataLoader, Subset,
                               WeightedRandomSampler)
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple

from preprocessing import (load_and_preprocess, extract_epochs,
                            epoch_to_image, augment_epoch, N_CHANNELS)

# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP    = {"C": 0, "A": 1, "F": 2}
CLASS_NAMES  = ["Healthy Control", "Alzheimer's Disease",
                "Frontotemporal Dementia"]

STAGE1_MAP   = {0: 0, 1: 1, 2: 1}   # flat → {Healthy=0, Decline=1}
STAGE2_MAP   = {0: -1, 1: 0, 2: 1}  # flat → {N/A=-1, AD=0, FTD=1}
STAGE2_NAMES = ["Alzheimer's Disease", "Frontotemporal Dementia"]
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# participants.tsv loader
# ═══════════════════════════════════════════════════════════════════════════

def load_participants(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    group_col = next((c for c in df.columns if c.lower() == "group"), None)
    if group_col is None:
        raise ValueError("participants.tsv must contain a 'Group' column.")
    df = df.rename(columns={group_col: "Group"})
    df["Group"] = df["Group"].str.strip().str.upper()
    df["label"] = df["Group"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    pid_col = next(
        (c for c in df.columns if "participant" in c.lower()), df.columns[0]
    )
    df = df.rename(columns={pid_col: "participant_id"})
    df["participant_id"] = df["participant_id"].str.strip()
    return df[["participant_id", "Group", "label"]]


def _apply_max_subjects(
    participants: pd.DataFrame,
    max_subjects: Optional[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Limit to at most `max_subjects` participants **per group** (C / A / F).
    Groups with fewer subjects are kept in full.
    Sampling is deterministic via random_state.

    Parameters
    ----------
    participants  : full participants DataFrame
    max_subjects  : cap per group; None → no cap (original behaviour)
    random_state  : seed for reproducible sampling
    """
    if max_subjects is None:
        return participants

    frames = []
    for grp, gdf in participants.groupby("Group"):
        if len(gdf) > max_subjects:
            gdf = gdf.sample(n=max_subjects, random_state=random_state)
        frames.append(gdf)
    result = pd.concat(frames).reset_index(drop=True)
    n_total = len(result)
    print(f"[max_subjects={max_subjects}] {n_total} subjects selected "
          f"({dict(result.groupby('Group').size())})")
    return result


def find_set_file(data_root: str, subject_id: str) -> Optional[str]:
    pattern = os.path.join(data_root, subject_id, "eeg", "*.set")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


# ═══════════════════════════════════════════════════════════════════════════
# Cache helpers  (unchanged from v4)
# ═══════════════════════════════════════════════════════════════════════════

def _cache_tag(sub_id: str, representation: str, img_size: int) -> str:
    return f"{sub_id}_{representation}_{img_size}"


def _cache_paths(cache_dir: str, tag: str) -> Tuple[str, str, str]:
    """Return (img_path, epoch_path, sfreq_path)."""
    return (
        os.path.join(cache_dir, f"{tag}_imgs.npy"),
        os.path.join(cache_dir, f"{tag}_epochs.npy"),
        os.path.join(cache_dir, f"{tag}_sfreq.npy"),
    )


def _subject_cached(cache_dir: str, tag: str) -> bool:
    img_p, ep_p, sf_p = _cache_paths(cache_dir, tag)
    return all(os.path.exists(p) for p in (img_p, ep_p, sf_p))


# ═══════════════════════════════════════════════════════════════════════════
# Precompute helper  ─ run this ONCE before training
# ═══════════════════════════════════════════════════════════════════════════

def precompute_cache(
    data_root:      str,
    cache_dir:      str,
    epoch_duration: float = 2.0,
    overlap:        float = 1.0,
    img_size:       int   = 64,
    representation: str   = "recurrence",
    subjects:       Optional[List[str]] = None,
    max_subjects:   Optional[int] = None,
):
    """
    Pre-generate all images and save to cache_dir.

    Run once:
        from dataset import precompute_cache
        precompute_cache(
            data_root="C:/path/to/dataset",
            cache_dir="C:/eeg_cache",
            max_subjects=10,    # optional: cache only 10 subjects per group
        )

    After this completes, dataset __init__ and __getitem__ are instant.
    """
    os.makedirs(cache_dir, exist_ok=True)

    tsv_path     = os.path.join(data_root, "participants.tsv")
    participants = load_participants(tsv_path)
    if subjects is not None:
        participants = participants[
            participants["participant_id"].isin(subjects)
        ].copy()
    participants = _apply_max_subjects(participants, max_subjects)

    total = len(participants)
    print(f"\n[precompute_cache] {total} subjects  →  {cache_dir}")
    print(f"  representation={representation}  img_size={img_size}\n")

    for i, (_, row) in enumerate(participants.iterrows(), 1):
        sub_id   = row["participant_id"]
        set_file = find_set_file(data_root, sub_id)

        if set_file is None:
            print(f"  [{i:>3}/{total}] {sub_id}  SKIP (no .set file)")
            continue

        tag = _cache_tag(sub_id, representation, img_size)
        if _subject_cached(cache_dir, tag):
            img_p, _, _ = _cache_paths(cache_dir, tag)
            n = np.load(img_p, mmap_mode="r").shape[0]
            print(f"  [{i:>3}/{total}] {sub_id}  already cached  ({n} epochs)")
            continue

        print(f"  [{i:>3}/{total}] {sub_id}  computing …", end="", flush=True)
        try:
            raw    = load_and_preprocess(set_file)
            sfreq  = float(raw.info["sfreq"])
            epochs = extract_epochs(raw,
                                    epoch_duration=epoch_duration,
                                    overlap=overlap)
            if len(epochs) == 0:
                print("  WARN: 0 epochs, skipping.")
                continue

            imgs = np.stack(
                [epoch_to_image(ep,
                                representation=representation,
                                img_size=img_size,
                                sfreq=sfreq)
                 for ep in epochs],
                axis=0,
            )  # (N, C, H, W)  float32

            img_p, ep_p, sf_p = _cache_paths(cache_dir, tag)
            np.save(img_p, imgs)
            np.save(ep_p,  epochs)
            np.save(sf_p,  np.array(sfreq))
            print(f"  done  ({len(epochs)} epochs)")

        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\n[precompute_cache] All done.\n")


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch Dataset  ─ disk-backed, O(1) RAM
# ═══════════════════════════════════════════════════════════════════════════

class EEGHierarchicalDataset(Dataset):
    """
    Fast disk-backed EEG dataset.

    __init__ reads only metadata (file paths, offsets, labels).
    __getitem__ does one mmap slice from the per-subject cache file.

    RAM used: ~few MB regardless of dataset size.

    Returns per __getitem__:
      image        : (C, H, W) float32
      label        : int 0/1/2
      stage1_label : int 0/1
      stage2_label : int 0/1/-1

    Speed parameters
    ----------------
    max_subjects          : int or None
        Cap subjects *per group* (C / A / F).
        E.g. max_subjects=10 → ≤30 subjects total.
        None = use all (original behaviour).
    max_epochs_per_subject : int or None
        Cap epochs loaded per subject.
        Useful when a single subject has thousands of epochs.
        None = use all epochs (original behaviour).
    """

    def __init__(self,
                 data_root:               str,
                 epoch_duration:          float = 2.0,
                 overlap:                 float = 1.0,
                 img_size:                int   = 64,
                 representation:          str   = "recurrence",
                 subjects:                Optional[List[str]] = None,
                 cache_dir:               Optional[str] = None,
                 augment:                 bool  = False,
                 max_subjects:            Optional[int] = None,
                 max_epochs_per_subject:  Optional[int] = None,
                 random_state:            int   = 42):

        self.data_root               = data_root
        self.epoch_duration          = epoch_duration
        self.overlap                 = overlap
        self.img_size                = img_size
        self.representation          = representation
        self.augment                 = augment
        self.cache_dir               = cache_dir
        self.max_epochs_per_subject  = max_epochs_per_subject

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        tsv_path     = os.path.join(data_root, "participants.tsv")
        participants = load_participants(tsv_path)

        if subjects is not None:
            participants = participants[
                participants["participant_id"].isin(subjects)
            ].copy()

        # ── NEW: limit subjects per group ──────────────────────────────────
        participants = _apply_max_subjects(participants, max_subjects,
                                           random_state=random_state)

        # ── lightweight metadata arrays ────────────────────────────────────
        self._labels:        List[int]   = []
        self._stage1_labels: List[int]   = []
        self._stage2_labels: List[int]   = []
        self._img_paths:     List[str]   = []  # per-subject imgs .npy
        self._ep_paths:      List[str]   = []  # per-subject epochs .npy
        self._local_idx:     List[int]   = []  # index within that file
        self._sfreqs:        List[float] = []

        # cached np arrays (built lazily, reset on first access)
        self._labels_arr:   Optional[np.ndarray] = None
        self._s1_arr:       Optional[np.ndarray] = None
        self._s2_arr:       Optional[np.ndarray] = None

        self.subject_indices: Dict[str, List[int]] = {}

        print(f"\n[Dataset] Building index ({len(participants)} subjects) …")

        for _, row in participants.iterrows():
            sub_id   = row["participant_id"]
            label    = int(row["label"])
            set_file = find_set_file(data_root, sub_id)

            if set_file is None:
                print(f"  [WARN] No .set file for {sub_id}, skipping.")
                continue

            result = self._resolve_subject(sub_id, set_file, label, row["Group"])
            if result is None:
                continue

            img_path, ep_path, sfreq, n = result

            # ── NEW: cap epochs per subject ────────────────────────────────
            indices = list(range(n))
            if self.max_epochs_per_subject is not None and n > self.max_epochs_per_subject:
                rng = np.random.default_rng(hash(sub_id) & 0xFFFFFFFF)
                indices = sorted(rng.choice(n, self.max_epochs_per_subject,
                                            replace=False).tolist())
                n = len(indices)

            start = len(self._labels)
            self._img_paths.extend([img_path] * n)
            self._ep_paths.extend([ep_path]  * n)
            self._local_idx.extend(indices)
            self._sfreqs.extend([sfreq] * n)
            self._labels.extend([label] * n)
            self._stage1_labels.extend([STAGE1_MAP[label]] * n)
            self._stage2_labels.extend([STAGE2_MAP[label]] * n)
            self.subject_indices[sub_id] = list(range(start, start + n))

        print(f"\n[Dataset] Total samples: {len(self._labels)}")
        self._print_class_counts()

    # ── subject resolver (unchanged from v4) ─────────────────────────────

    def _resolve_subject(
        self, sub_id: str, set_file: str, label: int, group: str
    ) -> Optional[Tuple[str, str, float, int]]:
        """
        Returns (img_path, ep_path, sfreq, n)  or  None on failure.

        If cache exists → instant return (no EEG loading).
        Otherwise → compute + save (or temp if no cache_dir).
        """
        tag = _cache_tag(sub_id, self.representation, self.img_size)

        # ── fast path: cache hit ───────────────────────────────────────────
        if self.cache_dir and _subject_cached(self.cache_dir, tag):
            img_p, ep_p, sf_p = _cache_paths(self.cache_dir, tag)
            n     = np.load(img_p, mmap_mode="r").shape[0]
            sfreq = float(np.load(sf_p))
            print(f"  {sub_id}  group={group}  label={label}"
                  f"  epochs={n}  [cache ✓]")
            return img_p, ep_p, sfreq, n

        # ── compute path: cache miss ───────────────────────────────────────
        print(f"  {sub_id}  group={group}  label={label}"
              f"  computing …", end="", flush=True)
        try:
            raw    = load_and_preprocess(set_file)
            sfreq  = float(raw.info["sfreq"])
            epochs = extract_epochs(raw,
                                    epoch_duration=self.epoch_duration,
                                    overlap=self.overlap)
            if len(epochs) == 0:
                print("  WARN: 0 epochs, skipping.")
                return None

            imgs = np.stack(
                [epoch_to_image(ep,
                                representation=self.representation,
                                img_size=self.img_size,
                                sfreq=sfreq)
                 for ep in epochs],
                axis=0,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            return None

        n = len(imgs)
        if self.cache_dir:
            img_p, ep_p, sf_p = _cache_paths(self.cache_dir, tag)
            np.save(img_p, imgs)
            np.save(ep_p,  epochs)
            np.save(sf_p,  np.array(sfreq))
            print(f"  done ({n} epochs) [saved]")
            return img_p, ep_p, sfreq, n
        else:
            import tempfile, atexit, shutil
            tmp   = tempfile.mkdtemp(prefix="eeg_tmp_")
            atexit.register(shutil.rmtree, tmp, ignore_errors=True)
            img_p = os.path.join(tmp, "imgs.npy")
            ep_p  = os.path.join(tmp, "eps.npy")
            np.save(img_p, imgs)
            np.save(ep_p,  epochs)
            print(f"  done ({n} epochs) [no cache_dir]")
            return img_p, ep_p, sfreq, n

    # ── label arrays: cached on first access (no rebuild overhead) ────────

    @property
    def labels_array(self) -> np.ndarray:
        if self._labels_arr is None:
            self._labels_arr = np.array(self._labels, dtype=np.int64)
        return self._labels_arr

    @property
    def stage1_labels_array(self) -> np.ndarray:
        if self._s1_arr is None:
            self._s1_arr = np.array(self._stage1_labels, dtype=np.int64)
        return self._s1_arr

    @property
    def stage2_labels_array(self) -> np.ndarray:
        if self._s2_arr is None:
            self._s2_arr = np.array(self._stage2_labels, dtype=np.int64)
        return self._s2_arr

    def _print_class_counts(self):
        lbl = self.labels_array
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {(lbl == i).sum()} samples")

    def __len__(self) -> int:
        return len(self._labels)

    # ── FAST __getitem__ (unchanged from v4) ─────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        img_path = self._img_paths[idx]
        ep_path  = self._ep_paths[idx]
        local_i  = self._local_idx[idx]
        sfreq    = self._sfreqs[idx]

        # mmap_mode="r" reads only the requested row, not the whole file
        image = np.load(img_path, mmap_mode="r")[local_i].copy()  # (C,H,W)

        if self.augment:
            raw_ep = np.load(ep_path, mmap_mode="r")[local_i].copy()
            raw_ep = augment_epoch(raw_ep, sfreq=sfreq)
            image  = epoch_to_image(raw_ep,
                                    representation=self.representation,
                                    img_size=self.img_size,
                                    sfreq=sfreq)

        return {
            "image":        torch.tensor(image,                    dtype=torch.float32),
            "label":        torch.tensor(self._labels[idx],        dtype=torch.long),
            "stage1_label": torch.tensor(self._stage1_labels[idx], dtype=torch.long),
            "stage2_label": torch.tensor(self._stage2_labels[idx], dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Subject-wise split  (unchanged from v4)
# ═══════════════════════════════════════════════════════════════════════════

def subject_wise_split(
    dataset:      EEGHierarchicalDataset,
    val_ratio:    float = 0.15,
    test_ratio:   float = 0.15,
    random_state: int   = 42,
) -> Tuple[Subset, Subset, Subset]:
    all_subs  = list(dataset.subject_indices.keys())
    test_size = max(1, int(len(all_subs) * test_ratio))
    val_size  = max(1, int(len(all_subs) * val_ratio))

    train_subs, test_subs = train_test_split(
        all_subs, test_size=test_size, random_state=random_state
    )
    train_subs, val_subs = train_test_split(
        train_subs, test_size=val_size, random_state=random_state
    )

    def idx_for(subs):
        out = []
        for s in subs:
            out.extend(dataset.subject_indices[s])
        return out

    ti, vi, si = idx_for(train_subs), idx_for(val_subs), idx_for(test_subs)
    print(f"\n[Split] Subjects  train={len(train_subs)}"
          f"  val={len(val_subs)}  test={len(test_subs)}")
    print(f"[Split] Epochs    train={len(ti)}  val={len(vi)}  test={len(si)}\n")
    return Subset(dataset, ti), Subset(dataset, vi), Subset(dataset, si)


# ═══════════════════════════════════════════════════════════════════════════
# Balanced sampler  (unchanged from v4)
# ═══════════════════════════════════════════════════════════════════════════

def make_balanced_sampler(subset: Subset,
                          label_key: str = "label") -> WeightedRandomSampler:
    ds      = subset.dataset
    indices = np.array(subset.indices)

    arr_map = {
        "label":        ds.labels_array,
        "stage1_label": ds.stage1_labels_array,
        "stage2_label": ds.stage2_labels_array,
    }
    if label_key not in arr_map:
        raise ValueError(f"label_key must be one of {list(arr_map)}.")

    labels      = arr_map[label_key][indices]
    classes     = np.unique(labels[labels >= 0])
    counts      = np.array([(labels == c).sum() for c in classes], dtype=float)
    weight_map  = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
    sample_w    = np.array(
        [weight_map.get(int(l), 0.0) for l in labels], dtype=np.float32
    )

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_w),
        num_samples=len(sample_w),
        replacement=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Stage-2 filter  (unchanged from v4)
# ═══════════════════════════════════════════════════════════════════════════

def filter_stage2(subset: Subset) -> Subset:
    ds      = subset.dataset
    indices = np.array(subset.indices)
    s2      = ds.stage2_labels_array[indices]
    keep    = indices[s2 >= 0].tolist()
    print(f"  Stage-2 filter: {len(indices)} → {len(keep)} samples (AD+FTD)")
    return Subset(ds, keep)


# ═══════════════════════════════════════════════════════════════════════════
# Mixup collate  (unchanged from v4)
# ═══════════════════════════════════════════════════════════════════════════

def mixup_collate(alpha: float = 0.2):
    def _collate(batch):
        images = torch.stack([b["image"]        for b in batch])
        labels = torch.stack([b["label"]        for b in batch])
        s1     = torch.stack([b["stage1_label"] for b in batch])
        s2     = torch.stack([b["stage2_label"] for b in batch])
        if alpha > 0:
            lam    = float(np.random.beta(alpha, alpha))
            perm   = torch.randperm(len(images))
            images = lam * images + (1.0 - lam) * images[perm]
        return {"image": images, "label": labels,
                "stage1_label": s1, "stage2_label": s2}
    return _collate


# ═══════════════════════════════════════════════════════════════════════════
# DataLoader factory  — updated with speed params
# ═══════════════════════════════════════════════════════════════════════════

def get_dataloaders(
    data_root:               str,
    batch_size:              int   = 64,       # ← raised default (was 32)
    epoch_duration:          float = 2.0,
    overlap:                 float = 1.0,
    img_size:                int   = 64,
    representation:          str   = "recurrence",
    num_workers:             int   = 0,
    cache_dir:               Optional[str] = None,
    val_ratio:               float = 0.15,
    test_ratio:              float = 0.15,
    use_mixup:               bool  = False,
    mixup_alpha:             float = 0.2,
    random_state:            int   = 42,
    max_subjects:            Optional[int] = None,
    max_epochs_per_subject:  Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, "EEGHierarchicalDataset"]:
    """
    Speed-tuned DataLoader factory.

    New parameters vs v4
    --------------------
    max_subjects            : int or None — subjects per group cap
    max_epochs_per_subject  : int or None — epochs per subject cap

    pin_memory is auto-enabled when CUDA is available.
    prefetch_factor=2 is used when num_workers > 0.
    """
    cuda_available = torch.cuda.is_available()
    pin_memory     = cuda_available
    prefetch       = 2 if num_workers > 0 else None

    shared_kwargs = dict(
        data_root               = data_root,
        epoch_duration          = epoch_duration,
        overlap                 = overlap,
        img_size                = img_size,
        representation          = representation,
        cache_dir               = cache_dir,
        max_subjects            = max_subjects,
        max_epochs_per_subject  = max_epochs_per_subject,
        random_state            = random_state,
    )

    train_ds = EEGHierarchicalDataset(**shared_kwargs, augment=True)
    eval_ds  = EEGHierarchicalDataset(**shared_kwargs, augment=False)

    train_sub, _, _      = subject_wise_split(
        train_ds, val_ratio=val_ratio,
        test_ratio=test_ratio, random_state=random_state,
    )
    _, val_sub, test_sub = subject_wise_split(
        eval_ds, val_ratio=val_ratio,
        test_ratio=test_ratio, random_state=random_state,
    )

    sampler    = make_balanced_sampler(train_sub, label_key="label")
    collate_fn = mixup_collate(mixup_alpha) if use_mixup else None

    loader_kw = dict(
        batch_size      = batch_size,
        num_workers     = num_workers,
        pin_memory      = pin_memory,
        prefetch_factor = prefetch,
    )

    train_loader = DataLoader(
        train_sub, sampler=sampler, drop_last=True,
        collate_fn=collate_fn, **loader_kw,
    )
    val_loader = DataLoader(val_sub,   shuffle=False, **loader_kw)
    test_loader = DataLoader(test_sub, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, train_ds


# ═══════════════════════════════════════════════════════════════════════════
# CLI: python dataset.py --data_root ... --cache_dir ...
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Precompute EEG image cache")
    p.add_argument("--data_root",      required=True)
    p.add_argument("--cache_dir",      required=True)
    p.add_argument("--epoch_duration", type=float, default=2.0)
    p.add_argument("--overlap",        type=float, default=1.0)
    p.add_argument("--img_size",       type=int,   default=64)
    p.add_argument("--representation", default="recurrence",
                   choices=["recurrence", "psd"])
    p.add_argument("--max_subjects",   type=int,   default=None,
                   help="Cap subjects per group (C/A/F). Default: all subjects.")
    args = p.parse_args()

    precompute_cache(
        data_root      = args.data_root,
        cache_dir      = args.cache_dir,
        epoch_duration = args.epoch_duration,
        overlap        = args.overlap,
        img_size       = args.img_size,
        representation = args.representation,
        max_subjects   = args.max_subjects,
    )
