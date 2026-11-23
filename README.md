# EEG-Based Cognitive Decline Detection Using Recurrence Plot Representation and Hierarchical Deep Learning (ResNet)

This is my B.Tech (Computer Science and Engineering) final year major project.

> **This is an academic research prototype. It is not a clinically validated diagnostic system.** It must not be used to diagnose Alzheimer's disease, frontotemporal dementia or any other condition.

## 1. Overview
The project takes resting state EEG recordings, turns them into recurrence plot images, and classifies them with a modified ResNet18. The classification happens in two stages:

```
EEG (.set) → bandpass + notch filter → 2 s epochs → 3 channel recurrence plot (64×64)
   Stage 1: Healthy Control  vs  Cognitive Decline
      └─ if Decline → Stage 2: Alzheimer's Disease (AD)  vs  Frontotemporal Dementia (FTD)
```

## 2. Problem statement
It can be hard to tell dementia types apart clinically. EEG is cheap and noninvasive, so I wanted to see whether deep learning on images made from EEG signals could do two things: separate healthy people from people with cognitive decline, and then tell Alzheimer's disease apart from frontotemporal dementia.

## 3. Research objectives
1. Convert multichannel EEG epochs into images that a CNN can work with.
2. Use a ResNet18 with transfer learning for the two binary tasks.
3. Split the data by subject, so epochs from the same person never end up in both training and testing.

I first tried a graph based approach. I converted 19 channel EEG into about 35,254 graphs and tried GNN and GAT models, but the performance was not reliable enough. I then moved to the recurrence plot approach that this repository contains. The graph code is not included here.

## 4. Dataset
| | |
|---|---|
| Source | OpenNeuro (EEG from healthy controls, Alzheimer's disease and frontotemporal dementia; groups `C` / `A` / `F` in `participants.tsv`) |
| Accession | **ds004504 (inferred, still to be confirmed, see [DATASET_CITATION.md](DATASET_CITATION.md))** |
| Version used | **[PLACEHOLDER: fill in from your local dataset's `dataset_description.json` / `CHANGES`]** |

**The EEG recordings are not included in this repository.** Please download them from OpenNeuro, follow the dataset's terms, and cite it as described in [DATASET_CITATION.md](DATASET_CITATION.md).

## 5. EEG preprocessing (`preprocessing.py`)
* The `.set` files are loaded with MNE (EEGLAB format).
* An FIR bandpass filter (0.5 to 45 Hz, Hamming window) and a 50 Hz notch filter are applied.
* Each recording is cut into 2 second epochs with 1 second overlap.

## 6. Recurrence plot generation (`preprocessing.py`)
For every epoch, channel and frequency band (theta 4 to 8 Hz, alpha 8 to 13 Hz, beta 13 to 30 Hz), the steps are:

1. Bandpass the signal using the FFT.
2. Reduce it to 64 points.
3. Build a pairwise distance matrix and divide it by its maximum value.
4. Combine a soft similarity map with a thresholded map (threshold 0.2).

The plots from all channels are averaged within each band, which gives one 3 × 64 × 64 image per epoch. The code also has a single channel PSD spectrogram option (`--representation psd`).

## 7. Modified ResNet18 (`model.py`)
I used ResNet18 with ImageNet weights. Since the images are only 64×64, I replaced the first convolution with a 3×3 convolution with stride 1 and removed the max pool layer. The classification head is Dropout, Linear(512, 256), ReLU, Dropout, Linear(256, 2). A lightweight CNN is also included as an alternative (`--architecture eegnet`).

## 8. Hierarchical classification
* **Stage 1:** Healthy Control (0) vs Cognitive Decline (1).
* **Stage 2:** Alzheimer's Disease (0) vs Frontotemporal Dementia (1). It is trained only on AD and FTD subjects and can optionally start from the Stage 1 checkpoint.

During inference (`predict.py` and `app.py`), Stage 1 runs first. Stage 2 only runs if Stage 1 predicts Cognitive Decline.

## 9. Training methodology (`train_stage1.py`, `train_stage2.py`)
Training uses AdamW (the backbone learning rate is 0.1 times the head learning rate), a cosine annealing schedule, gradient clipping, label smoothing of 0.1 and dropout of 0.5. Classes are balanced during sampling, and epochs are augmented with noise and a time shift. Early stopping and checkpoint selection are based on validation macro F1.

Defaults: 40 epochs, batch size 32, learning rate 1e-3 for Stage 1 and 5e-4 for Stage 2, seed 42.

## 10. Evaluation methodology
Subjects are divided into training, validation and test sets (70 / 15 / 15 by default, seed 42) before epochs are assigned. This means all epochs of one subject stay in the same set. Accuracy and macro F1 are reported on the held out test subjects, along with a per class report. These numbers are calculated over individual EEG epochs.

## 11. Results
*These are approximate ranges that I reported from my experiments and review presentations. The training logs and test reports are not part of this repository, so I have not re-checked the numbers here. They are ranges, not the output of one documented run.*

| Metric | Stage 1: Healthy vs Cognitive Decline | Stage 2: AD vs FTD |
|---|---|---|
| Accuracy | about 59 to 62 % | about 70 to 73 % |
| Macro F1 | about 0.59 to 0.61 | about 0.45 to 0.48 |

Stage 1 was fairly consistent. Stage 2 has higher accuracy, but its macro F1 is lower, which means AD and FTD were not classified in a balanced way. The results are modest and do not show clinical grade performance.

## 12. Installation
Python 3.10 or newer is needed, because `train_stage2.py` uses the `X | None` type hint syntax.

```bash
git clone https://github.com/<your-username>/eeg-cognitive-decline-detection.git
cd eeg-cognitive-decline-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 13. Dependencies
`requirements.txt` includes MNE, PyTorch, torchvision, NumPy, SciPy, pandas, scikit-learn, matplotlib (used by `imgs.py`) and Streamlit (used by `app.py`).

## 14. How to run
The code expects the dataset to be laid out like this:

```
dataset_root/
  participants.tsv        # columns: participant_id, Group (C / A / F)
  sub-001/eeg/*.set
  sub-002/eeg/*.set
  ...
```

```bash
# 1. Precompute and cache recurrence plot images (only needed once)
python dataset.py --data_root /path/to/dataset --cache_dir /path/to/eeg_cache

# 2. Train Stage 1, then Stage 2
python train_stage1.py --data_root /path/to/dataset --cache_dir /path/to/eeg_cache --output_dir runs/stage1
python train_stage2.py --data_root /path/to/dataset --cache_dir /path/to/eeg_cache --output_dir runs/stage2 \
       --stage1_ckpt runs/stage1/stage1_best.pt

# 3. Run inference on one cached image array or one raw .set file
python predict.py --input_npy input.npy
python predict.py --input_set subject.set
```

For quicker runs on a CPU, add `--max_subjects 10 --max_epochs_per_subject 400`. Use the same seed and subject limits for both stages. Checkpoints are not included in the repository, so you need to train first. `predict.py` and `app.py` look for `runs/stage1/stage1_best.pt` and `runs/stage2/stage2_best.pt`. Both of them classify a single 2 second epoch (the middle one) of the input.

## 15. Streamlit application
```bash
streamlit run app.py
```
Run this from the repository root after both checkpoints exist. You can upload a precomputed `.npy` image array or a raw `.set` file and click **Run Inference**. The app then shows the Stage 1 and Stage 2 probabilities. It is only a demo of the research prototype and not a diagnostic tool.

## 16. Repository structure
```
├── preprocessing.py          EEG loading, filtering, epoching, image conversion
├── dataset.py                Cached dataset, subject wise split, cache precomputation
├── model.py                  Modified ResNet18 and lightweight CNN
├── train_stage1.py           Stage 1 training and test report
├── train_stage2.py           Stage 2 training and test report
├── predict.py                Command line hierarchical inference
├── app.py                    Streamlit demo
├── imgs.py                   Saves recurrence plot images from a cached .npy file
├── DATASET_CITATION.md
├── PUBLICATION_NOTES.md
├── PUBLICATION_CHECKLIST.md
├── requirements.txt
└── .gitignore
```

## 17. Limitations
* This is an academic prototype. It is not clinically validated and not meant for diagnosis.
* The number of subjects is small, so the test set has very few people and results can change with the split.
* Metrics are calculated per EEG epoch, there is no external validation dataset, and the split is not stratified by class.
* Performance is modest, especially the macro F1 for AD vs FTD.
* The reported figures are approximate ranges without logs (see Section 11).

## 18. References and dataset citation
* Miltiadous, A., Tzimourta, K. D., Afrantou, T., Ioannidis, P., Grigoriadis, N., Tsalikakis, D. G., Angelidis, P., Tsipouras, M. G., Glavas, E., Giannakeas, N., & Tzallas, A. T. (2023). A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG. *Data*, 8(6), 95. (The dataset identification still needs to be confirmed, see `DATASET_CITATION.md`.)
* Markiewicz, C. J. *et al.* (2021). The OpenNeuro resource for sharing of neuroscience data. *eLife*, 10, e71774.
* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770 to 778.
* Eckmann, J.-P., Kamphorst, S. O., & Ruelle, D. (1987). Recurrence plots of dynamical systems. *Europhysics Letters*, 4(9), 973 to 977.
* Gramfort, A. *et al.* (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.
