# Experiment protocols and commands

## BRISC classification (2D)

`4_2d_classification/src/classification/model.py` is a small grayscale CNN. `1_data_and_preprocessing/src/data/brisc.py`
loads class-folder labels in fixed order: glioma, meningioma, no_tumor,
pituitary. The classifier architecture is retained. A seeded split of the
official training folder groups identical image bytes by SHA-256 into separate
model-fit, validation, and calibration partitions. Validation cross-entropy
selects the checkpoint; the calibration subset is reserved for temperature
scaling and is not used for fitting or checkpoint selection. The official test
folder is used only to audit exact-hash overlap: matching train-side copies are
explicitly excluded from every internal partition, while official test
membership is not modified.

```powershell
python 4_2d_classification\scripts\train_classification.py --config 4_2d_classification\configs\classification.yaml
python 4_2d_classification\scripts\evaluate_classification.py --checkpoint 4_2d_classification\outputs\classification\<run>\best.pt --split test
python 4_2d_classification\scripts\predict_classification.py --checkpoint 4_2d_classification\outputs\classification\<run>\best.pt --input path\to\image.jpg
```

Training pads grayscale images to 160 × 160 by default to preserve aspect
ratio. Mild horizontal flips, small rotations, and brightness/contrast changes
are configurable and applied only to model-fit images. AdamW is retained; the
optional cosine scheduler is enabled in the checked-in config. Evaluation
records accuracy, balanced accuracy, macro/weighted precision, recall and F1,
confusion matrices, one-vs-rest per-class AUROC/AUPRC where mathematically
defined, cross-entropy, negative log-likelihood, multiclass Brier score, and
15-bin top-label ECE. Raw and temperature-scaled reports, ROC/PR curves,
reliability diagrams, training history/curves, split membership, manifest
checksums, calibration details, and software/device metadata are saved with
each run. Raw `metrics.json` is uncalibrated;
`calibrated_metrics.json` contains the temperature-scaled report.

For Colab with the project and BRISC data at
`/content/brain-tumor-analysis` and
`/content/brain-tumor-analysis/1_data_and_preprocessing/data/BRISC/brisc2025`, run this cell to train
and then evaluate the printed run's checkpoint exactly once:

```bash
%%bash
set -o pipefail
cd /content/brain-tumor-analysis
python -m pip install -e .
RUN_DIR="$(python 4_2d_classification/scripts/train_classification.py --config 4_2d_classification/configs/classification.yaml | tail -n 1)"
printf 'Evaluating checkpoint: %s/best.pt\n' "$RUN_DIR"
python 4_2d_classification/scripts/evaluate_classification.py --checkpoint "$RUN_DIR/best.pt" --split test
```

Evaluation creates a new output directory and refuses to overwrite an existing
one. Test metrics must not be used to select model settings or calibrate. The
current dataset has seven exact image hashes in common across its official
train/test folders. Matching train-side files are excluded and listed in the
split manifest; because official test membership is preserved, test results
are not a fully independent estimate.

## BRISC segmentation (2D)

`3_2d_segmentation/src/segmentation2d/model.py` is a configurable U-Net-like binary segmenter.
Training uses the official train segmentation pairs only, makes a seeded
sample-level validation split, and applies paired random flips. BRISC does not
provide a verifiable patient ID, so the split cannot claim patient-level
separation. Input images are resized to 256 × 256; masks are thresholded at
128 before nearest-neighbor resizing. The validation foreground Dice selects
the checkpoint.

```powershell
python 3_2d_segmentation\scripts\train_segmentation2d.py --config 3_2d_segmentation\configs\segmentation2d.yaml
python 3_2d_segmentation\scripts\evaluate_segmentation2d.py --checkpoint 3_2d_segmentation\outputs\segmentation2d\<run>\best.pt --split test
python 3_2d_segmentation\scripts\predict_segmentation2d.py --checkpoint 3_2d_segmentation\outputs\segmentation2d\<run>\best.pt --input path\to\images
```

Metrics include global pixel foreground Dice/IoU, mean per-image Dice/IoU,
nonempty-target Dice, and per-image CSV records. Sample MRI/ground-truth/
prediction overlays and training curves are saved as PNG/PDF.

## BraTS 3D MONAI (SegResNet and SwinUNETR)

The generic `BraTSVolumeDataset` validates the four expected modalities and
their geometry, maps them in configured order, normalizes nonzero voxels,
and independently constructs four overlapping binary masks from the raw
label volume in the fixed output-channel order **WT, TC, ET, RC**:

```text
WT = [1, 2, 3]   TC = [1, 3]   ET = [3]   RC = [4]
```

The raw-label interpretation is provisional for this local data because its
exact BraTS release/subset provenance is unresolved. Do not use these names
as verified official labels in a paper until provenance is confirmed.
SegResNet, SwinUNETR, the residual 3D U-Net baseline, and the EfficientNet-style
MBConv 3D U-Net baseline use four output channels and the same
BCE-plus-soft-Dice training objective; they do not use a mutually exclusive
softmax. The two baselines are trained from scratch and share the same
preprocessing, patch size, target ontology, and grouped split. The EfficientNet
variant is a compact MBConv encoder/decoder, not a pretrained torchvision
EfficientNet-B0. Splits use the filename prefix before the final hyphen as an
inferred grouping heuristic and do not mix these groups across
training/validation. Actual patient-level independence is not verified. The
validation set is internal to the local training data; it is not an official
test set.

New runs use a timestamped, non-reused directory. MONAI evaluation and
prediction also refuse nonempty output directories. Checkpoints record the
ontology ID, mapping, and channel order. A legacy checkpoint with absent or
different ontology metadata is rejected: old `L1`–`L4` weights must only be
used with their historical target semantics, not loaded as WT/TC/ET/RC.

```powershell
python 2_3d_segmentation\scripts\train_segmentation3d.py --model segresnet --config 2_3d_segmentation\configs\segmentation3d.yaml
python 2_3d_segmentation\scripts\train_segmentation3d.py --model swinunetr --config 2_3d_segmentation\configs\segmentation3d.yaml
python 2_3d_segmentation\scripts\train_segmentation3d.py --model resnet3d --config 2_3d_segmentation\configs\segmentation3d_resnet.yaml
python 2_3d_segmentation\scripts\train_segmentation3d.py --model efficientnet3d --config 2_3d_segmentation\configs\segmentation3d_efficientnet.yaml
python 2_3d_segmentation\scripts\evaluate_segmentation3d.py --checkpoint path\to\best.pt --output-dir 2_3d_segmentation\outputs\segmentation3d\evaluation
python 2_3d_segmentation\scripts\predict_segmentation3d.py --checkpoint path\to\best.pt --case-dir path\to\case --output-dir 2_3d_segmentation\outputs\segmentation3d\predictions
```

The 96³ patch and batch size 1 defaults are conservative starting points, not
benchmarked memory guarantees. SwinUNETR and full-volume sliding-window
validation may still need a recent GPU with ample memory. CPU inference works
in principle but 3D training will be very slow. Reduce patch size/feature size
or use a GPU if out of memory; do not change the held-out group split to tune
against the official test data (none is available here).

Outputs include best checkpoint, run metadata (including ontology and
channel-order details), loss/validation curves, per-case/per-region CSV,
JSON summary, 3D slice overlays, and WT/TC/ET/RC NIfTI region predictions for
validation and explicit inference under `region_masks/`. Combined overlays
are explicitly labeled as region unions; inspect individual channel NIfTI
files for separate masks. Region
Dice/IoU are reported separately;
HD95 is in millimeters when both target and prediction contain that region.
HD95 is `null` and excluded from its mean when either set is empty.

## nnU-Net v2 adapter

The preserved original notebooks use external Colab/Drive and `/home/...`
paths. The adapter below creates a separate nnU-Net dataset from the **271
actual local cases**, keeps modality order consistent, and creates grouped
cross-validation folds. It copies roughly 5 GiB of NIfTI data before nnU-Net
preprocessing; ensure substantial additional free disk space.

```powershell
python -m pip install -e ".[nnunet]"
python 2_3d_segmentation\scripts\nnunet_brats.py prepare --config 2_3d_segmentation\configs\segmentation3d.yaml --work-root 2_3d_segmentation\outputs\segmentation3d\nnunet_work
python 2_3d_segmentation\scripts\nnunet_brats.py preprocess --config 2_3d_segmentation\configs\segmentation3d.yaml --work-root 2_3d_segmentation\outputs\segmentation3d\nnunet_work
python 2_3d_segmentation\scripts\nnunet_brats.py train --work-root 2_3d_segmentation\outputs\segmentation3d\nnunet_work --fold 0
python 2_3d_segmentation\scripts\nnunet_brats.py evaluate-validation --config 2_3d_segmentation\configs\segmentation3d.yaml --work-root 2_3d_segmentation\outputs\segmentation3d\nnunet_work --fold 0 --output-dir 2_3d_segmentation\outputs\segmentation3d\nnunet_fold0_validation
```

Repeat the last command for folds 1–4 only when resources permit. `prepare`
does not overwrite an existing dataset directory or experiment metadata. Its
generated `dataset.json` records the actual count (271), the approved
WT/TC/ET/RC raw-label unions, and explicit region order; it does not modify
the legacy config. nnU-Net region training represents overlapping targets,
but its standard integer NIfTI prediction is a lossy serialization of those
regions. `evaluate-validation` requests saved probabilities and thresholds
each channel independently for metrics; it fails if the required probability
archive is missing and does not treat the integer labelmap as equivalent. A
fair comparison requires training nnU-Net using the generated region
configuration. The generated grouped split file keeps shared-prefix cases
together. Evaluation stages only the selected fold's held-out cases and
writes per-region Dice/IoU/HD95 metrics, named NIfTI region masks, and
overlays. It copies validation
modalities into the work directory, so allow additional disk space. No
official independent test set is present in the local folder.

## Preserved legacy experiments

The original notebooks, `swinunetr_train.py`, nnU-Net configuration, and
fold progress images are retained in `2_3d_segmentation/legacy/BraTS_Challenge/`. The legacy
dataset manifest and historical outputs use generic `L1`–`L4` regions; the
legacy README's WT/TC/ET/RC statements are not evidence that those old target
sets had the approved meanings. Do not reinterpret old checkpoints or scores.
They are reference artifacts, not called implicitly by the new pipelines.
Notable legacy limitations include external/absolute data paths, a custom
MONAI loader that duplicates T1N and omits T2W, mismatched case counts, and
unclear semantic label naming. No saved 3D checkpoint was present in the
workspace at inspection time.
