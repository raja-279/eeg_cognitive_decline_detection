# Dataset identification and citation

## Status: identified by inference; **needs your confirmation**

None of the project files states an accession number. The dataset is identified from the evidence below. Confirm it against your local copy before publishing.

### Evidence in the project files
| Evidence | Source in project |
|---|---|
| Group labels `C` / `A` / `F` in `participants.tsv` | `dataset.py` (`LABEL_MAP`, `load_participants`) |
| Task `eyesclosed`, layout `sub-XXX/eeg/*.set` | original README |
| "88-subject" dataset | comment in `train_stage1.py` |
| 19 EEG channels, obtained from OpenNeuro | your project description |

### Evidence from published sources (found by web search)
These all describe OpenNeuro **ds004504** and agree with the evidence above:
- 88 participants: 36 Alzheimer's disease (AD), 23 frontotemporal dementia (FTD), 29 healthy controls (CN); resting state, **eyes closed**; 19 scalp electrodes; 500 Hz sampling.
- Data descriptor: Miltiadous et al., *Data* 8(6), 95, published 27 May 2023, dataset licence **CC0**. The descriptor cites snapshot DOI `10.18112/openneuro.ds004504.v1.0.2`.
- Later snapshots are cited elsewhere (v1.0.6, 1.0.7, 1.0.8, 1.0.9), so several versions exist; I could not determine the newest one or which one you downloaded.
- A *different* dataset from the same cohort exists: **ds006036** (eyes-open photic stimulation; Ntetska et al., *Data* 2025). Your `eyesclosed` task name points to ds004504, not ds006036.

### What I could NOT verify
- I could not open the official OpenNeuro page (`openneuro.org/datasets/ds004504`); the checks above come from the data descriptor and papers that cite the dataset, not from the OpenNeuro record itself.
- Which snapshot/version you used.
- Whether your `data_root` pointed at the dataset root (raw `.set` files) or at its `derivatives/` tree (artifact-cleaned data). This changes how preprocessing should be described.
- The licence on the specific snapshot (verify on the OpenNeuro page; the data descriptor states CC0).

### To finish verification
Open your local copy of the dataset and read `dataset_description.json` (fields `Name`, `BIDSVersion`, `License`, `Authors`, `DatasetDOI`) and the `CHANGES` file (latest version number). Then compare with <https://openneuro.org/datasets/ds004504>. Fill in the version in `README.md` and below.

## Citation

**Dataset paper (please cite):**
Miltiadous, A., Tzimourta, K. D., Afrantou, T., Ioannidis, P., Grigoriadis, N., Tsalikakis, D. G., Angelidis, P., Tsipouras, M. G., Glavas, E., Giannakeas, N., & Tzallas, A. T. (2023). A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG. *Data*, 8(6), 95. https://www.mdpi.com/2306-5729/8/6/95
(DOI is expected to be `10.3390/data8060095` from the MDPI URL pattern; confirm on the article page.)

**Dataset record:**
Miltiadous, A. *et al.* *A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects.* OpenNeuro, ds004504, version **[FILL IN]**. DOI: `10.18112/openneuro.ds004504.v[FILL IN]`. https://openneuro.org/datasets/ds004504

**Platform:**
Markiewicz, C. J. *et al.* (2021). The OpenNeuro resource for sharing of neuroscience data. *eLife*, 10, e71774. https://doi.org/10.7554/eLife.71774

## Obtaining the data
Download it yourself from the OpenNeuro dataset page (web download, the OpenNeuro command-line tool, or DataLad). Example with the OpenNeuro CLI: `openneuro download --snapshot <version> ds004504 ds004504-download/`. **No EEG recordings are included in this repository** and none should be committed.

## Licence and usage
- The data descriptor states **CC0**: no legal restriction on use or redistribution, but citing the dataset is expected scholarly practice.
- Secondary sources describe the data as de-identified. Do not attempt to re-identify participants.
- This repository nevertheless does not redistribute the recordings or the cached arrays; derived arrays/checkpoints are left out to keep the repository small and to avoid presenting them as validated clinical tools.
