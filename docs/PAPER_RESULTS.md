# Paper-results guide

This guide defines what to report; it contains no invented scores. Only report
results present in run artifacts produced by actually executing an experiment.

## Required run records

Each experiment folder in `outputs/` should be archived with its `config.yaml`,
`metadata.json`, `run.log`, checkpoint, sample counts, split description, and
metrics. Record the source data root, software versions, seed, model
configuration, checkpoint selection criterion, and whether evaluation used
the official test split or a validation partition. Keep training/validation
curves and representative figures beside those records. Classification runs
also include manifest checksums, exact SHA-256-based internal partition
membership, calibration subset and fitted temperature, scheduler and learning
rate history, and an official train/test hash-overlap report.

## Classification table and plots

Use `metrics.json` and `per_class_metrics.csv` for:

- Sample count and split (`validation` or official `test`).
- Accuracy, balanced accuracy, macro/weighted precision, recall, and F1.
- Confusion matrix and class support.
- Per-class one-vs-rest AUROC/AUPRC only where both class-positive and
  class-negative examples exist.
- Uncalibrated and temperature-scaled negative log-likelihood, multiclass
  Brier score, and 15-bin top-label expected calibration error (ECE), with
  reliability diagram(s). Fit temperature using only the reserved official
  train calibration partition.

Include `roc.png`/`.pdf`, `precision_recall.png`/`.pdf`,
`confusion_matrix.png`/`.pdf`, calibrated counterparts, reliability plots, and
training curves. Do not compare metrics computed on different sample
populations without stating the split/count.

## BRISC classification data limitation and attribution

The local classification manifest contains seven distinct image hashes that
occur in both official train and test splits, plus duplicates within train.
Matching train-side copies of official test images are explicitly excluded
from all internal partitions, and the remaining train duplicates are grouped
by hash. Official test membership is deliberately preserved. Do not describe
the supplied official test set as fully independent; report this known overlap
alongside test metrics. Patient-level independence is also unverifiable
because patient IDs are not provided.

Cite the BRISC dataset paper (Fateh et al., 2025, arXiv:2506.14318) when
reporting results. The project does not bundle the original BRISC model code;
verify applicable dataset terms with the publisher before redistribution or
other uses requiring permission. Padding, augmentation, scheduling, and
temperature scaling are established techniques; do not claim methodological
novelty solely from these changes or from renaming/combining them.

## 2D segmentation table and figures

Report sample count/split, mask threshold and resize policy, global foreground
Dice/IoU, mean per-image Dice/IoU, and mean Dice for nonempty targets. Report
empty-target counts to contextualize the all-image averages. Include the
training curves and representative image/ground-truth/prediction overlays.

## 3D segmentation table and figures

Report case count, patient-group count, exact group split, input modality
order, orientation, region definitions, and checkpoint. Report per-region
Dice and IoU plus HD95 in mm with the number of cases for which HD95 was
defined. HD95 is omitted for cases with empty target or prediction, not
replaced with an arbitrary distance. Include the training/validation curves,
representative slice overlays, and generated NIfTI masks as supplementary
artifacts when permitted.

The active project protocol defines overlapping binary outputs in channel
order WT, TC, ET, RC using raw label sets `[1,2,3]`, `[1,3]`, `[3]`, and `[4]`,
respectively. This is an approved, provisional interpretation for experiments,
not confirmation of the local dataset's official release or ontology. The
local release provenance remains unknown; the legacy manifest declares 1,351
cases while 271 folders are available. State this limitation explicitly and
do not attribute official BraTS provenance or clinical semantics to the local
data without authoritative confirmation. Historical `L1`–`L4` artifacts use
different mappings and must not be pooled or relabeled as WT/TC/ET/RC.

The filename-prefix grouping rule is a heuristic, not a verified patient ID.
Patient-level independence is not established. The nnU-Net model trains with
region-based overlapping targets, but its ordinary integer segmentation
export cannot retain overlaps; report nnU-Net metrics only when derived from
the independently thresholded region probability channels.

## Current evidence / limitations

- No new training or test evaluation has been executed as part of project
  setup.
- No model weights were found at migration time.
- Legacy `Validation/progress_fold*.png` are preserved but are not independently
  validated run metrics in this project.
- BRISC's README sample total differs from its present segmentation files;
  BRISC has no patient identifier in its manifest.
- The BraTS config's declared `numTraining=1351` differs from 271 local cases.
  No official test split is included in the available BraTS folder.
- The current environment has CPU-only PyTorch; CUDA training performance,
  memory use, and results remain untested.
