# BRISC image-hash and split audit

This audit concerns exact image-file hashes, not patient identity. BRISC's
manifests do not provide verified patient identifiers; no patient-level
leakage claim is justified.

## Classification/segmentation tree overlap

Hashing the BRISC image files listed by
[`manifest.csv`](../../1_data_and_preprocessing/data/BRISC/brisc2025/manifest.csv) found **4,760 distinct
shared image hashes across 4,883 matching file pairs** between classification
and segmentation image trees:

| Classification split | Segmentation split | Shared distinct hashes | Matching file pairs |
|---|---|---:|---:|
| train | train | 3,904 | 3,997 |
| test | test | 856 | 868 |
| train | test | 7 | 9 |
| test | train | 7 | 9 |

The tree-to-tree overlap is expected for a multi-task dataset using the same
MRI images. Different membership across tasks is important if experiments are
combined or one model's output is used to evaluate a model trained on the same
image in another task; it is not, by itself, evidence that either standalone
model trained on its own test samples. The seven cross-split hashes are:

```text
02799f7863d49de182357e72179f3cc4e65e70099355f2d5aa2e39d356e492d9
08690b76457b2b44d58682cece62ec951ef6d30d935ce07df852e97d44434680
39f4f7b252d7809d8d3d2e4521607de2195bf0f66c0592587b1cebd632aac640
3e802dd22f483ce4d442fe2c7f1890c1c810ec8c1131b364fd4e4593bf0f268a
960155b05ea8f0ddcbcd3a5e29905c2a4b55832f03e5b7d6491c73955960d278
b5e17076d1427e41a618f92bc72100c1be4df26608223512aac51c911ac1eb88
f3d00bbb942117c2fb62466414d3c466ca283b678200fe87424e4128b80b2dba
```

## Classification experiment

The official classification train/test folders share **7 distinct hashes**,
with **9 train-side file copies**. The primary custom-CNN run explicitly
excluded those matching train copies from its internal fit/validation/
calibration partitions while retaining the official 1,000-image test set.
Therefore that run did not fit on the identified train-side duplicate copies,
but the source official test split is still not fully independent. Evidence:
the run's [`metadata.json`](../../4_2d_classification/outputs/classification/20261019_104128_seed2026/metadata.json)
and [`split_manifest.csv`](../../4_2d_classification/outputs/classification/20261019_104128_seed2026/split_manifest.csv).

## Attention-segmentation experiment

The saved attention U-Net split is sample-level. Reconstructing it from the
current file set and split code found **6 distinct exact-hash groups crossing
fit and validation**, comprising **7 fit-side files and 6 validation-side
files**. This may inflate validation-based checkpoint selection. It is a
deterministic reconstruction; no persisted split-manifest hash count was
found. Evidence: run
[`metadata.json`](../../3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/metadata.json)
and split creation
[`engine_attention.py`](../../3_2d_segmentation/src/segmentation2d/engine_attention.py#L351).
An exact copy example: `brisc2025_train_01276_me_ax_t1.jpg` (fit) and
`brisc2025_train_01277_me_ax_t1.jpg` (validation), hash
`09b9031e30e757eb6e648686f0ca412357e86895ebbc52d300d2d0a9305fd417`.

## Segmentation official train/test overlap

Direct SHA-256 comparison of the official segmentation image folders found
**7 distinct hashes shared between segmentation train and test**, with **9
train-file copies and 7 test-file copies** (9 matching train/test file pairs).
No saved segmentation hash-overlap manifest was found; this is a direct
comparison of the files enumerated by
[`segmentation_records`](../../1_data_and_preprocessing/src/data/brisc.py#L50) under
`segmentation_task/{train,test}/images`. The 2D attention run used the
segmentation train tree for its 3,343 fit and 590 validation images. The
assignment was reconstructed from the sorted train records and seed 2026
because there is no persisted hash-to-split manifest. Among the seven
train/test-shared hashes, **6 hashes (6 test copies) have a train-side copy in
fit**, **2 hashes (2 test copies) have a train-side copy in validation**, and
**1 hash is in both** because its train-side copies split across partitions.
That shared hash has two fit-side files and one validation-side file. Thus all
7 test-image copies overlap a train copy assigned to fit, validation, or both;
**6 of the 7 test copies have a matching fit-side image**.
This directly affects the Attention U-Net evaluation on the official-folder
test set: its reported test metrics include exact image duplicates of
training data. The split assignment is reconstructed, not a persisted run
field. Do not describe the supplied test split as independent for this run.

These are duplicate-image findings, not patient-level leakage findings.
Patient IDs are unavailable; patient-level train/test disjointness cannot be
established.
