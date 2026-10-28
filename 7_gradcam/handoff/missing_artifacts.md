# Missing artifacts and recovery actions

## Classification

- **ResNet50:** recover the complete training history, final run metadata,
  checkpoint-selection evidence, and official-test evaluation (metrics,
  per-class report, and confusion matrix) from the run environment if the
  experiment was continued. The local checkpoint is only a two-epoch partial
  run. Its metadata says `checkpoint: null` and `metrics: null`, despite a
  `best.pt` file.
- **EfficientNetB0:** no trained checkpoint, run metadata, logs, or evaluation
  were found. If this model is required, run the defined baseline and save a
  complete run directory. Do not report results from architecture/config files.
- The earlier custom-CNN checkpoint is from a sample-level protocol and has
  validation results only. Prefer the later hash-aware run and disclose seven
  exact image hashes shared between official train/test. Patient-level
  separation is not established.
- Recover the original `manifest.csv`/`manifest.json` versions if exact
  historical dataset provenance or a rerun against the same manifest is needed.

## 2D segmentation

- The attention run's 40-row history, training log, validation metrics,
  per-image validation metrics, test metrics, per-image test metrics, and
  overlays are present in
  `3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/`.
  The run metadata points to a historical `/content/project/...` training
  location, while the evaluated local checkpoint is separately bound by the
  test `evaluation_metadata.json`. Recover the original Colab run bundle or
  provenance manifest if the source of the imported model must be established.
- Seven exact hashes (nine train-side copies) overlap official segmentation
  train/test images. Reconstruction shows all seven test copies also have
  train-side copies assigned to fit, validation, or both, and six test copies
  have a fit-side duplicate. Additionally, six exact-hash groups cross the
  attention run's fit/validation split. The test results are not independent.
  Recover a persisted hash-grouped split manifest and evaluate on a genuinely
  disjoint holdout before making independent-test claims.
- The `3_2d_segmentation/outputs/segmentation2d/20261018_222743_seed2026/best.pt` file exists,
  but its metadata has null checkpoint/metrics fields. Recover the originating
  run's config, checkpoint provenance, training history, and evaluation before
  using or attributing it.
- ResNet50- and EfficientNetB0-decoder variants have code and configs, but no
  matching training runs, checkpoints, or metrics.
- Patient IDs are not supplied in BRISC manifests; sample-level split
  membership cannot be described as patient-independent.

## 3D segmentation

- The SegResNet checkpoint contains inference weights/config/ontology only.
  Optimizer, scheduler, and RNG state are absent, so exact training resumption
  is not supported. The available run is one epoch with 56 grouped-validation
  cases and no official test split.
- Recover a full training run (history, best-checkpoint rule, environment,
  validation/test predictions and per-case metrics) before presenting
  performance as a finished experiment.
- No SwinUNETR run/checkpoint/metrics or hook-activation evidence was found.
  Recover a completed run and inspect its exact installed MONAI module tree
  before choosing a Grad-CAM layer.
- The table
  `2_3d_segmentation/outputs/segmentation3d/segmentation_paper_figures/reported_metrics.csv`
  has no documented cohort/split/sample count or matching run artifact. Obtain
  its source experiment and per-case predictions before reporting those
  values as reproducible results.
- Verify the exact local BraTS release and ontology. The current WT/TC/ET/RC
  mapping declares `release_provenance_verified: false`; historical generic
  L1-L4 results are not semantically interchangeable.

## General

- Checkpoint discovery found six first-party `.pt` files described in
  `checkpoint_manifest.csv`. No first-party model `.pth`, `.ckpt`, or
  `.safetensors` weights were found; `.venv` package files are not model
  artifacts.
- No Grad-CAM source code or saved Grad-CAM result was found for any of the
  listed models. All targets in this package are recommendations backed by
  architecture inspection, not successful attribution tests.
- `Capstone_Technical_Dossier/05_models_and_training_configuration.md` says
  ResNet50/EfficientNetB0 implementations were not located. That statement is
  inconsistent with the source/config and partial ResNet50 checkpoint:
  ResNet50 is implemented with a partial run, EfficientNetB0 is implemented
  without a run.
- Hardware identifiers are missing for the recorded CUDA attention-model
  training. The classification and 3D run records report CPU environments.
- The local Python/library versions are not necessarily the training
  environment. Preserve the environment metadata with recovered runs.

## Additional legacy code (not part of the eight-model inventory)

- `4_2d_classification/legacy/legacy_brats_tumor_classifier.py` implements a separate 2D
  BraTS-slice LGG/HGG classifier using ImageNet-pretrained EfficientNet-B4,
  with two outputs. Its source config describes a seeded 80/20 random split,
  batch 4, 224x224, and 10 epochs, but no corresponding
  `efficientnet_tumor_classifier.pth` or result was found. This is not the
  four-class BRISC classifier and does not evidence training of the requested
  BRISC EfficientNetB0.
- `2_3d_segmentation/legacy/segmentation.py` contains a separate 3D BraTS
  Attention U-Net, with a four-channel 128-cube training contract and code to
  save `best_model.pth` including model/optimizer state. No checkpoint was
  found under that legacy tree. It is not the requested MONAI SegResNet or
  SwinUNETR and is not included as a model row.
- These are additional legacy code definitions, not evidence of trained
  model artifacts or evaluation results. The 3D surgical-risk mapper itself
  was not audited.
