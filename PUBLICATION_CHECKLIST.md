# GitHub publication checklist

- [ ] Confirm dataset accession and version (`dataset_description.json`, `CHANGES`) and fill in `README.md` / `DATASET_CITATION.md`.
- [ ] Replace `<your-username>` in `README.md`.
- [ ] (Optional) Replace the absolute path in `imgs.py` line 6; add a `LICENSE`.
- [ ] Copy your original local `.git`-free project folder contents only; do NOT copy `eeg_cache/`, dataset folders, `runs/` or `*.pt` files into this folder.
- [ ] Run the checks below and confirm the output is as described.
- [ ] Create an **empty** repository named `eeg-cognitive-decline-detection` on GitHub (no README/licence/.gitignore), then push.

## Checks (run from the project folder after `git init` and `git add .`)
```bash
git status --short                                   # only the files listed in README "Repository structure"
git ls-files | xargs -d '\n' ls -lh | sort -k5 -h | tail -5   # nothing large
git ls-files | grep -Ei "\.(set|fdt|edf|npy|npz|pt|pth|ckpt|tsv)$"   # prints nothing
git ls-files -ci --exclude-standard                  # prints nothing
```

## Publish
```bash
git init -b main
git add .
git commit -m "Initial commit: EEG cognitive decline detection (academic research prototype)"
git remote add origin https://github.com/<your-username>/eeg-cognitive-decline-detection.git
git push -u origin main
```
If `git remote add` says `origin` already exists, or the GitHub repository is not empty, stop and check before pushing: do not force-push and do not overwrite existing history.
