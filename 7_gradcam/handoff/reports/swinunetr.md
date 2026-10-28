# MONAI SwinUNETR

**Status: model construction and configuration exist; no SwinUNETR checkpoint,
run-level metrics, or saved output were found.** Do not claim a trained or
validated SwinUNETR experiment.

The configured model has four input and four output channels,
`img_size=(96,96,96)`, `feature_size=48`, and activation checkpointing. It
shares the current 3D pipeline's four-modality order
`[t2f,t1n,t1c,t2w]`, RAS orientation, nonzero-voxel per-modality z-score,
96³ crop/ROI, zero-padding, and 3D flips. The configured output regions are
WT/TC/ET/RC with raw labels `[1,2,3]`, `[1,3]`, `[3]`, `[4]`, but this
ontology is **PROVISIONAL — ONTOLOGY VERIFICATION REQUIRED** because the local
release is unverified.

The base config proposes 100 epochs, batch 1, AdamW lr 1e-4/weight decay 1e-5,
equal BCE-with-logits and soft-Dice, no scheduler in the current engine,
patience 30, and best mean per-case Dice selection. Those are configured
settings, not training results. There are no SwinUNETR Dice, IoU, HD95,
confusion-style results, or test metrics. The separate comparison table is
unattributed and must not be represented as this model's result.

## Grad-CAM

No exact target path or activation shape is validated. The model construction
is in [`models.py`](../../../2_3d_segmentation/src/segmentation3d/models.py#L28), but no
SwinUNETR checkpoint or gradient-hook forward was inspected. Before
implementing attribution, instantiate the exact configured MONAI model,
inspect its module tree, hook a candidate final encoder feature, and record
the tensor layout/shape on the 96³ input. Then define a selected region-logit
scalar over a declared 3D ROI, calculate a 3D CAM, and preserve the RAS affine.
Do not use the SegResNet target or metrics for this model.

The synthetic forward was not attempted: only 1.75 GB RAM was free during the
audit, making a gradient-enabled 96³ SwinUNETR pass an unsafe memory risk.
