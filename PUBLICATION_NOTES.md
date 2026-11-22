# Publication notes

## Source code: unchanged
`preprocessing.py`, `dataset.py`, `model.py`, `train_stage1.py`, `train_stage2.py`, `predict.py`, `app.py` and `imgs.py` are byte-for-byte identical to the original project files. No file was renamed, moved or deleted; no methodology, preprocessing, model, training, evaluation or result was changed.

## Added
| File | Purpose |
|---|---|
| `README.md` | Replaces the old README, which described a different earlier layout (`train.py`, a 3-class EEGNet, a `MultiViewEEGNet` and `sswt`/`coherence` options that are not in the code) |
| `requirements.txt` | Original list plus `streamlit` (imported by `app.py` but missing before) |
| `.gitignore` | Keeps EEG data, caches, checkpoints, outputs, environments and IDE files out of Git |
| `DATASET_CITATION.md` | Dataset identification, citation, licence, items still to confirm |
| `PUBLICATION_NOTES.md`, `PUBLICATION_CHECKLIST.md` | This file; checklist and Git commands |

## Reorganised
`sample_1.png`–`sample_3.png` were placed in `docs/images/` (used by the README). `imgs.py` still writes new samples to the current folder, as before.

## Safety check result
Checked: e-mail addresses, keys/tokens/passwords, absolute paths, data files, large files. Nothing else was found; all files are under 35 KB; no EEG data, caches or checkpoints are present.
**One item to be aware of:** `imgs.py` line 6 contains an absolute local path (`C:\Users\Admin\Downloads\...`). It exposes only a generic Windows folder name, so it was left unchanged as requested. If you prefer, change that line to a placeholder such as `r"path/to/eeg_cache/sub-XXX_recurrence_64_imgs.npy"` before committing. That is a one-line edit; tell me and I will do it.

## Placeholders you must fill in
- `README.md` and `DATASET_CITATION.md`: dataset version (and confirmation that the accession is ds004504).
- `README.md`: `<your-username>` in the clone URL.
- No `LICENSE` file was added. Without one, the code is "all rights reserved" by default. Add one (e.g. MIT) if you want others to reuse it.

## Optional future improvements (NOT applied; they would change the implementation)
These came from reading the code. None was applied.
1. `predict.py` / `app.py` rescale each input image to [0, 1] with min-max, while training used the raw cached values. The two preprocessing steps differ, which may affect demo predictions.
2. `dataset.py` seeds the per-subject epoch cap with Python's `hash()`, which changes between runs, so `--max_epochs_per_subject` runs are not exactly reproducible (`zlib.crc32` would fix this).
3. Recurrence plots keep 64 samples of each 1000-sample epoch without low-pass filtering; with 2 s at 500 Hz the effective Nyquist is about 15.8 Hz, so part of the beta band (13–30 Hz) is aliased. `--img_size 128` would be one experiment (needs a new cache and retraining).
4. `mixup_collate` in `dataset.py` mixes images but not labels; it is unused by the training scripts.
5. Subject-level (per-patient) aggregation of epoch predictions, and measuring the full two-stage cascade, are not implemented.
6. `train_stage2.py` uses `str | None` and therefore needs Python 3.10+ (documented in the README).
7. The warm start in `train_stage2.py` does not check that Stage 1 used the same subject split.

A fuller version with some of these addressed (tests, subject-level evaluation script) was produced earlier in this conversation; it is optional and not part of this package.
