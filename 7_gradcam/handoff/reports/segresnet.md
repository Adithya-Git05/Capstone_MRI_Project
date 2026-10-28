# MONAI SegResNet

**Status: implemented, checkpointed, and run-level validation completed for one
epoch only.** No held-out official test set is available. The sole checkpoint
[`best.pt`](../../../2_3d_segmentation/outputs/segmentation3d/segresnet/20261019_151137_seed2026/best.pt)
is 18,836,959 bytes, SHA-256
`92A9F71D4E733E158DF5A1666606419CAA2E1B1E767E4B229A78E7F0C2E2CD72`. It is a
safe-loaded dictionary with 83 state keys and config/ontology metadata; it
does not contain optimizer, scheduler, or RNG state.

## Inputs and protocol

The local tree contains 271 case folders and 1,355 NIfTI files (four images
and one segmentation per case). Local release provenance is not established.
Input order is `[t2f, t1n, t1c, t2w]`. The loader checks matching shapes and
affines, converts float32 arrays to RAS, and z-scores each modality over
nonzero voxels while retaining zero background. No resampling is performed;
the validated local spacing is 1 mm isotropic. Training uses a random 96³
crop, zero-padding if needed, and independent flips on each spatial axis at
p=0.5. Inference uses whole-volume sliding windows with ROI 96³, batch 1,
overlap 0.25, and MONAI's default constant blending.

The model uses four input and four output channels, spatial_dims=3 and
init_filters=16. The base YAML specifies 100 epochs and patience 30, but this
run overrides both to one. It used seed 2026, AdamW lr 1e-4/weight decay
1e-5, batch 1, equal BCE-with-logits plus soft Dice, no scheduler, and selected
by mean per-case Dice. Grouping uses folder-name prefix before the final
hyphen, not verified patient IDs; 215 cases/108 groups trained and 56
cases/10 groups validated. Source run metadata:
[`metadata.json`](../../../2_3d_segmentation/outputs/segmentation3d/segresnet/20261019_151137_seed2026/metadata.json).
The one-epoch loss/Dice row is also copied to
[`segresnet_training_history.csv`](./segresnet_training_history.csv).

## Ontology and metrics

The configured order is WT, TC, ET, RC with raw labels `[1,2,3]`, `[1,3]`,
`[3]`, `[4]`. The checkpoint metadata explicitly says release provenance is
not verified. These are independent, potentially overlapping binary masks:
**PROVISIONAL — ONTOLOGY VERIFICATION REQUIRED**. Legacy generic L1-L4
semantics and historical metrics must not be silently relabeled as these
regions.

One-epoch validation over 56 grouped cases:

| Region | Mean Dice | Mean IoU | Mean HD95 (mm) | HD95-valid |
|---|---:|---:|---:|---:|
| WT | 0.500850 | 0.370024 | 61.128216 | 56/56 |
| TC | 0.244497 | 0.161615 | 75.468664 | 53/56 |
| ET | 0.156379 | 0.093527 | 77.252061 | 53/56 |
| RC | 0.071082 | 0.044500 | 77.500272 | 45/56 |

Validation BCE+soft-Dice loss is 0.748115. Dice/IoU are per-case; both-empty
regions score 1. HD95 is null if either mask is empty; otherwise it is the
95th percentile of symmetric surface distances. Metrics:
[`metrics.json`](../../../2_3d_segmentation/outputs/segmentation3d/segresnet/20261019_151137_seed2026/metrics.json).
Training curves and representative 3D slice outputs are available at
[`training_curves.pdf`](../../../2_3d_segmentation/outputs/segmentation3d/segresnet/20261019_151137_seed2026/training_curves.pdf)
and
[`representative_3d_slices.pdf`](../../../2_3d_segmentation/outputs/segmentation3d/segresnet/20261019_151137_seed2026/representative_3d_slices.pdf).
The separate paper comparison CSV has no cohort/split/sample count or matching
run; it is not reproducible from available artifacts.

## Grad-CAM and geometry

Recommended target `model.down_layers[3][4].conv2.conv`, observed
`[B,128,12,12,12]` for a 96³ input. Derive a scalar from one selected region's
logits over a fixed 3D ROI. The inference output is thresholded at 0.5,
written as uint8 with the reoriented RAS affine, with no output interpolation.
No inverse transform to the original LAS voxel-array order is implemented.
No Grad-CAM code or saved CAM exists. Full sliding-window gradient attribution
is memory intensive; patch-wise CAM must be labeled as such.
