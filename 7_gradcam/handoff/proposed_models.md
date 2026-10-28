# Incomplete-model specifications

The 2D encoder-decoder models and both MONAI 3D models have implementation
evidence; the missing items are complete, reproducible experiments and/or
validated Grad-CAM paths. The classifier EfficientNetB0 is implemented but
untrained. Specs below are modest next-run protocols, not reported results.

## Proposed — ResNet50 U-Net on BRISC

**PROPOSED — NOT YET TRAINED/VALIDATED.** Architecture is implemented at
`3_2d_segmentation/src/segmentation2d/model.py`; the matching config exists. Use the existing
single-channel 256×256 BRISC grayscale path and paired masks, threshold masks
at 128 before nearest-neighbor resizing. Use one shared deterministic internal
split manifest for both
encoder variants (seed 2026; 15% validation) and group byte-identical image
hashes across those internal partitions. The dataset does not expose verified
patient IDs, so this split cannot honestly be called patient-independent.
The official segmentation train/test trees share seven exact hashes; preserve
the official membership for reporting, but use an additional genuinely
disjoint holdout if claiming independent test performance.
Use batch 8, AdamW (`lr=3e-4`, `weight_decay=1e-4`), up to
40 epochs, patience 8, and select the best validation Dice. Use the generic
engine's BCE + soft-Dice objective (equal weights) unless explicitly
experimenting with a separately declared loss. Report foreground Dice/IoU as
global pixel-aggregated and mean-per-image values, including empty-target
handling; report the 0.5 sigmoid threshold and per-image scores. Save resolved
config, split manifest and source hashes, epoch history, best state dict,
optimizer/scheduler state if available, environment versions, predictions,
per-image metrics, summary metrics, and overlays. Do not reuse the attention
model's test metrics.

## Proposed — EfficientNetB0 U-Net on BRISC

**PROPOSED — NOT YET TRAINED/VALIDATED.** Use the implemented grayscale
EfficientNetB0 encoder and U-Net-style decoder. Follow the same shared
hash-grouped sample split and evaluation contract as the ResNet50 U-Net so the
two variants are comparable. BRISC has no verified patient identifier, so
this split is not patient-independent. The official segmentation train/test
trees share seven exact hashes; preserve the official membership for reporting
and use an additional genuinely disjoint holdout if claiming independent
test performance. Keep the official train/test folders and
report known duplicate-hash overlap rather than silently moving official test
samples. Use 256×256, batch 8, seed 2026, AdamW (`3e-4`, `1e-4`), up to 40
epochs, patience 8, BCE + soft-Dice, 0.5 output threshold, and the same metric
and artifact checklist as the ResNet50 variant. Do not reuse attention-model
metrics.

## Proposed — EfficientNetB0 classifier run

**PROPOSED — NOT YET TRAINED/VALIDATED.** The classifier implementation/config
exists, but no run is present. Use the custom CNN's hash-aware BRISC split,
160×160 grayscale centered padding and `[-1,1]` normalization, single-channel
stem, four-class logits, cross-entropy, AdamW (`1e-3`, `1e-4`), batch 32,
cosine schedule, and up to 30 epochs with validation-loss checkpointing and
patience 7. Preserve the official test membership and known train/test hash
overlaps. Save the full run config, history, state dict, class order, manifest
hashes, calibration fit provenance, evaluation metadata, per-class metrics,
confusion matrix, ROC/PR only if actually computed, and environment/hardware.
This is a proposed run plan, not an experiment result.

## Proposed — completed ResNet50 classifier experiment

**PROPOSED — COMPLETE/VALIDATED EXPERIMENT NOT AVAILABLE LOCALLY.** Continue
from an identified checkpoint only if its run provenance is recovered; do not
silently treat the current two-epoch checkpoint as the full model. Otherwise
train from the configured non-pretrained initialization under the same split,
preprocessing, optimizer/scheduler and evaluation protocol as the custom CNN
and proposed EfficientNetB0 classifier. Save test metrics and confusion
matrix independently for this run.

## Proposed — full 3D training and SwinUNETR Grad-CAM validation

**PROPOSED — NOT YET TRAINED/VALIDATED FOR SWINUNETR.** Before training,
confirm the dataset release and raw-label ontology. Retain the current input
order `[t2f, t1n, t1c, t2w]`, RAS orientation, nonzero-voxel per-modality
z-score normalization, 96³ training crop, AdamW, equal BCE-with-logits and
soft-Dice weights, 0.5 inference threshold, and four-output channel order only
if the provisional mapping is validated. Use a grouped train/validation split;
do not call filename-derived groups verified patients. Train a realistic
budgeted run (for example 10–20 epochs with patience 5, subject to available
memory/compute) with checkpoint-selection criterion documented, save full config,
per-epoch history, model/optimizer state, versions/hardware, per-case region
Dice/IoU/HD95, predictions, and original/RAS affine metadata. For SwinUNETR,
inspect the instantiated module tree and run a gradient-enabled forward hook
to record candidate activations before claiming a target path.

For SegResNet, recover or repeat a completed run rather than treating the
existing one-epoch validation as a final experiment. A modest 10–20 epoch run
with early stopping (patience 5) and the same grouped split is a reasonable
capstone baseline if compute permits; keep the checkpoint-selection rule and
all run artifacts explicit. Neither schedule is an existing result.
