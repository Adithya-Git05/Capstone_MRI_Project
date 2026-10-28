# Attention-gated 2D BRISC U-Net

**Status: trained with a checkpoint and official-folder test metrics; imported
run provenance is partially verified.** Model source:
[`model_attention.py`](../../../3_2d_segmentation/src/segmentation2d/model_attention.py#L10).
The encoder widths are 16/32/64, the bottleneck is 128 channels at 32x32 for
256x256 input, and additive gates modulate three skip connections. The final
head emits one full-resolution foreground logit. Checkpoint
[`best.pt`](../../../3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/best.pt)
is 2,000,897 bytes, SHA-256
`282f1499be25cef06bc20d828317ab2cef17771e26891652466e0b207f1dccab`, with 56
state tensors and no optimizer/scheduler state.

## Data, training, and evaluation

The 2D BRISC dataset has 3,933 training pairs and 860 official test pairs.
Images are grayscale, bilinearly resized to 256x256 and normalized to [-1,1].
Masks are thresholded at >=128 before nearest-neighbor resize. Images/masks are
paired by filename stem and mismatch raises an error. Training uses paired
horizontal/vertical flips at p=0.5. No patient IDs are verified; the training
split is sample-level.

Reconstructing the split found six distinct exact-hash groups (7 fit-side
files/6 validation-side files) crossing training and validation. This can
inflate validation-based checkpoint selection. The count is reconstructed
from the split code/current files rather than recorded in a persisted
split-manifest. No patient-level overlap claim is supported. See
[`data_split_audit.md`](../data_split_audit.md).

Separately, 7 exact hashes (9 train-side copies) overlap the official
segmentation train/test trees. Deterministic reconstruction finds that 6
hashes (6 test copies) have a train-side copy in fit, 2 hashes (2 test copies)
have a train-side copy in validation, and 1 hash is in both because it has two
fit-side copies and one validation-side copy. All 7 test copies therefore
overlap a fit or validation copy; 6 have a matching fit-side image. The test metrics include exact
duplicates of training images and should not be described as independent.
These are image hashes, not patient identifiers. Full counts and method are in
[`data_split_audit.md`](../data_split_audit.md).

The generic standalone CLI inference path uses `BRISCClassificationDataset`
with `resize_mode=stretch`; it shares grayscale conversion and `[-1,1]`
normalization with training/evaluation but uses a different dataset/resize
path. See [`engine.py`](../../../3_2d_segmentation/src/segmentation2d/engine.py#L350) and
[`brisc.py`](../../../1_data_and_preprocessing/src/data/brisc.py#L165). The integrated pipeline README's
`[0,1]` statement should not replace this source-level BRISC contract.

Recorded split: 3,343 training, 590 validation; 40 epochs completed, best
epoch 39. Optimizer: AdamW (lr 0.0003, weight decay 0.0001), batch 8, patience
8, no scheduler. The loss is exactly 0.45 BCE-with-logits + 0.45 soft Dice +
0.10 boundary-gradient L1. Validation mean per-image Dice selects the
checkpoint. The 40-row epoch history and run log are available at
[`training_history.csv`](../../../3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/training_history.csv)
and the 40-row copy
[`attention_unet_training_history.csv`](./attention_unet_training_history.csv).
The run config is
[`config.yaml`](../../../3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/config.yaml);
the training CLI is
[`train_segmentation2d_attention.py`](../../../3_2d_segmentation/scripts/train_segmentation2d_attention.py).
The training log is
[`run.log`](../../../3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/run.log).

Official-folder test metrics (n=860): global foreground Dice 0.875253, global
IoU 0.778178, mean per-image Dice 0.856058, mean per-image IoU 0.781571, and
loss 0.076798. All 860 targets are nonempty. Global scores aggregate pixel
counts across images; per-image values are arithmetic means; both-empty
per-image scores are 1. Prediction threshold is sigmoid >=0.5. No HD95,
confusion matrix, or postprocessing is reported. Source:
[`test_metrics.json`](../../../3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/evaluation_test_20261019_121848_909528/test_metrics.json).
Validation per-image rows:
[`per_image_metrics.csv`](../../../3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/per_image_metrics.csv);
test per-image rows:
[`per_image_metrics.csv`](../../../3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/evaluation_test_20261019_121848_909528/per_image_metrics.csv).
Also present are
[`representative_test_overlays.pdf`](../../../3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/evaluation_test_20261019_121848_909528/representative_test_overlays.pdf),
and
[`test_metric_distributions.pdf`](../../../3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/evaluation_test_20261019_121848_909528/test_metric_distributions.pdf)
are present alongside the evaluation; they are segmentation overlays, not
Grad-CAM.

The historical metadata/log point to `/content/project/...`; the local
checkpoint hash is tied to the later test evaluation metadata. Training
metadata reports Linux/Python 3.13.15/PyTorch 2.11.0+cu130/CUDA (GPU model
unknown), while test evaluation reports Windows/Python 3.12.1/PyTorch
2.13.0+cpu.

## Grad-CAM

Recommended target `model.bottleneck.layers[3]` (second bottleneck
convolution, before GroupNorm/SiLU), observed `[B,128,32,32]`. It is a semantic
pre-gate encoder feature. Define a scalar foreground-logit target over a fixed
ROI or one pixel. `model.att3` returns a gated skip tensor, not the attention
coefficient map; `model.att3.score` is the gate-map candidate. No Grad-CAM
implementation or successful attribution exists.
