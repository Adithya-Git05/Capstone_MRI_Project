# Local datasets

Dataset contents are intentionally excluded from Git. Expected layouts:

```text
1_data_and_preprocessing/data/
  BRISC/brisc2025/                 # official BRISC classification/segmentation files
  BraTS/training_data_additional/   # BraTS cases with four modalities and seg
  BraTS/demo_cases/                # small local inference examples
```

To use another location, set `BRISC_ROOT` and `BRATS_ROOT`; see
[`../../docs/DATASETS.md`](../../docs/DATASETS.md) for formats, labels, and
known provenance/split limitations. Do not commit scans, masks, manifests
containing identifiers, checkpoints, or generated volumes.
