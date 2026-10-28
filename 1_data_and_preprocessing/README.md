# 1. Data and preprocessing

BRISC and BraTS loaders, normalization, label handling, and validation live in
this stage's `src/data/`. Local datasets live under this stage's `data/`.

1. Place the data in `1_data_and_preprocessing/data/BRISC/brisc2025/` and
   `1_data_and_preprocessing/data/BraTS/training_data_additional/` (or set `BRISC_ROOT` and
   `BRATS_ROOT`).
2. From the repository root, run `python 1_data_and_preprocessing/scripts/validate_data.py --all`.
3. Read the reports in `1_data_and_preprocessing/outputs/data_validation/` before training.

See [`../docs/DATASETS.md`](../docs/DATASETS.md) for label ontology and split
caveats. Data and validation outputs stay local and are git-ignored.
