# Grad-CAM model handoff

This evidence inventory was prepared before the Grad-CAM integration. Its run
and checkpoint findings remain historical; the current implementation is in
`7_gradcam/gradcam.py` and the integrated app displays it for compatible custom
BRISC classifier checkpoints. No attribution outputs are bundled.

This package is an evidence-based handoff for classification and segmentation
only. It does not modify models, train models, or include/copy model weights.
All model/checkpoint references are paths relative to the repository root.

## Important scope notes

- The request says “seven model entries,” but its lists name **eight** distinct
  variants: 3 classifiers, 3 2D segmenters, and 2 3D segmenters. This handoff
  includes every named variant.
- A model definition, YAML config, or paper table is not evidence of a
  completed run. Statuses distinguish complete runs, partial runs, code-only
  variants, and missing artifacts.
- Only the custom 2D CNN has a completed classification test evaluation.
  ResNet50 has a partial two-epoch checkpoint. EfficientNetB0 has no run.
- The attention-gated BRISC segmenter has a 40-epoch training record and
  official-folder test metrics, but exact image-hash reconstruction shows six
  of seven matching test images also occur in its fit partition. The
  ResNet50/EfficientNetB0 2D segmenters are implemented but do not have
  matching runs or checkpoints.
- SegResNet has a one-epoch grouped-validation run, not a held-out test result.
  SwinUNETR has implementation/configuration but no run or checkpoint.
- All eight named architectures have an implementation/configuration in the
  repository. Thus none is "proposed only" or absent; incomplete models have
  proposed next-run specifications. Using the four requested evidence
  categories: the custom CNN, Attention U-Net, and one-epoch SegResNet fall in
  category 1 (SegResNet is validation-only and undertrained); the partial
  ResNet50 classifier and the untrained EfficientNetB0 classifier,
  ResNet50/EfficientNetB0 segmenters, and SwinUNETR fall in category 3.
  Category 2 is not separately supported: imported Colab-like Attention U-Net
  artifacts, including checkpoint/history/evaluation, are present locally.
- The BraTS local release and WT/TC/ET/RC label semantics remain unverified.
  Preserve the label mapping as provisional.
- The source artifacts and history CSVs remain in their original output
  directories. The two 2D completed-run histories and the one-epoch 3D history
  are also included as small CSV copies for direct review; checkpoints are not
  copied.
- No Grad-CAM outputs were found in the audited experiment artifacts.

## Contents

- [model_inventory.csv](./model_inventory.csv) — all eight requested variants.
- [checkpoint_manifest.csv](./checkpoint_manifest.csv) — discovered checkpoint
  paths, sizes, hashes, and inspection status; no weights are copied.
- [data_split_audit.md](./data_split_audit.md) — byte-identical BRISC
  classification/segmentation images and the affected internal split.
- [configs/](./configs/) — one JSON record per model, with observed values and
  explicitly marked unknowns.
- [reports/](./reports/) — one evidence report per model.
- [gradcam_targets.md](./gradcam_targets.md) — target choices, scalar targets,
  computation recipe, geometry, and caveats.
- [missing_artifacts.md](./missing_artifacts.md) — artifacts to recover and
  provenance/reproducibility limits.
- [proposed_models.md](./proposed_models.md) — modest training/evaluation
  specifications for incomplete models.

## Reproducibility

Use the repository's existing loaders and checkpoint loaders; do not silently
substitute preprocessing. For every checkpoint, verify the recorded
`model_config`, class/ontology metadata, and preprocessing contract before
inference. For Grad-CAM, run a normal gradient-enabled forward pass
(`model.eval()` is appropriate; `torch.no_grad()` and `torch.inference_mode()`
are not), capture the chosen activation and its gradient, and document the
input, target scalar, layer, and display transform used.

The reports identify source artifacts and line references. Histories and
per-image metrics are linked in place so the original epoch rows remain the
source of truth.

## Downstream interfaces (scope-limited)

The integrated research prototype consumes segmentation outputs as follows:
the 3D adapter consumes the four provisional WT/TC/ET/RC masks, uses the WT
prediction for downstream case/report context, and checks its geometry against
the MRI. The optional risk stage still requires a separate aligned SynthSeg
label volume; this handoff does not audit or validate that stage. The
classification adapter exposes BRISC class probabilities/prediction and the
2D segmentation adapter exposes a BRISC foreground mask; both are used in the
integrated report, and MRI-slice use is explicitly out-of-distribution and
unvalidated. Source: `integrated_pipeline/README.md` (component integration
table and data-flow description).

The integration README describes one 2D adapter preprocessing path as `[0,1]`
and says the existing CLI predictor uses a different normalization. The BRISC
training loader and generic CLI prediction source both normalize images to
`[-1,1]`; their resize paths differ. Verify the current integrated adapter
against the actual training tensor contract before using it. This note
identifies an interface/documentation mismatch only; no surgical-risk stage
was audited.
