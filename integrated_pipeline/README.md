# Brain MRI Analysis Application

The Streamlit app is a research prototype with one four-modality study intake.
Upload co-registered BraTS-style `t2f`, `t1n`, `t1c`, and `t2w` NIfTI volumes
together; the selected sequence supplies the axial slice for the current 2D
models, while the 3D model requires all four channels. Outputs are
experimental, and use of BRISC models on other MRI domains is not validated.

## Launch

From the repository root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m pip install -e ".[brats,integrated]"
.\venv\Scripts\python.exe -m streamlit run app.py
```

The root `app.py` delegates to `integrated_pipeline/app.py`. Uploads are
processed locally by the app process, held temporarily or in Streamlit session
state, and are not written as persistent case files. Do not expose this
prototype publicly or upload identifiable data without appropriate
authorization.

Use the same Python environment for installation and launch. The active
SegResNet checkpoint was strictly tested with PyTorch 2.14.1+cpu and MONAI
1.6.1; the UI also performs a strict compatibility check on the local weights.
The 3D forward path additionally requires `einops`, included in the `brats` and
`integrated` extras.

The project’s `.streamlit/config.toml` sets the readable light-theme default.
Use the sidebar **Dark mode** toggle to switch the app palette. The UI groups
the workflow into Overview, MRI & 2D Analysis,
Explainability, Spatial & Surgical, Chat / Ask MRI Assistant, and Final Report.
Model paths and configuration fields are inside a collapsed advanced
configuration panel.

## Selected checkpoints and evidence

`model_registry.json` contains portable, project-relative paths and evidence
for the selected candidates:

| Task | Checkpoint | Selection evidence |
|---|---|---|
| 2D segmentation | `3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/best.pt` (`BRISCAttentionUNet`) | Official-folder test: 860 cases, global foreground Dice 0.875253, IoU 0.778178, mean per-image Dice 0.856058. |
| 2D classification | `4_2d_classification/outputs/classification/20261018_214516_seed2026/best.pt` (`BRISCClassifier`) | User-supplied earlier 30-epoch checkpoint; validation-only evidence and older sample-level split protocol. No test metrics are attributed to this checkpoint. |
| 3D segmentation | `2_3d_segmentation/outputs/segmentation3d/SegResNet_BraTS2021/model.pt` (`MONAI SegResNet`, 4→3 channels) | User-supplied checkpoint; strict state-dict compatibility and one-case inference are checked separately from clinical validation. |

The 2D classifier path above follows the user-supplied local checkpoint; it is
not the newer test-evaluated checkpoint described in the technical dossier.
Do not attribute the newer checkpoint's test metrics to this earlier file.
Dataset provenance and patient-level independence are not established.
Classifier outputs are the BRISC folder classes `glioma`, `meningioma`,
`no_tumor`, and `pituitary`; they are not pathology-confirmed subtypes.

The 2D checkpoint files are local artifacts; the supplied 3D SegResNet checkpoint
is retained at its registered path. When present, the app checks the recorded
size and SHA-256 before inference; another checkout must supply matching files.
No predictions are produced without compatible weights. See the checkpoint manifest and
[`06_segmentation_results.md`](../Capstone_Technical_Dossier/06_segmentation_results.md)
and
[`07_classification_results.md`](../Capstone_Technical_Dossier/07_classification_results.md)
for provenance and limitations.

## Workflow and limitations

- MRI input validation checks all four supported NIfTI sequences for
  dimensions, non-empty and finite data, and matching spatial metadata. The 2D
  models receive one grayscale slice from the selected sequence;
  model-specific resizing and normalization follow the checkpoint pipeline.
  NIfTI uses an explicit per-slice 1st-99th-percentile intensity window before
  the same model normalization; that conversion is a domain adaptation, not
  part of BRISC training.
- Segmentation presents a binary foreground mask, overlay, pixel count, and
  physical in-plane area only when NIfTI geometry is available. It does not
  estimate 3D tumor volume.
- Classification displays raw logits and uncalibrated softmax output. These
  values are not calibrated clinical probabilities.
- Grad-CAM is available for a compatible classifier checkpoint. A heatmap is
  generated only after its model hooks succeed.
- Spatial-risk and corridor analysis require compatible 3D outputs and
  anatomical inputs; they remain blocked for this 2D-only workflow.
- The active 3D checkpoint is the user-supplied raw MONAI SegResNet state
  dictionary. Inspection found 83 tensors, a 4-channel input projection, and a
  3-channel output head; strict loading passed with PyTorch 2.14.1+cpu and
  MONAI 1.6.1. The user-provided channel contract is input
  `[T1ce, T1, T2, FLAIR]`, output `[TC, WT, ET]`. Channel semantics and training
  provenance are not embedded in the checkpoint, so this contract is recorded
  as supplied, not independently proven by tensor inspection.
- 3D inference requires four finite, co-registered NIfTI inputs with 1 mm
  isotropic spacing. It preserves native voxel axes and affine, does not
  resample, and applies the project's per-channel nonzero normalization; a
  match to this checkpoint's original training preprocessing is unverified.
  Runtime inference uses a 64-cube sliding window, overlap 0.25, and batch size
  1, with sigmoid probabilities thresholded at 0.5. Those are tested runtime
  settings, not checkpoint metadata. TC, WT, and ET masks are exported; RC is
  unavailable and is never synthesized. CPU/RAM preflight can block inference.
  The integrated CPU smoke run on one BraTS2021 case completed in 60.12 s and
  measured Dice 0.673707/0.814576/0.550974 for TC/WT/ET against its supplied
  reference mask. This is a one-case smoke comparison, not clinical validation
  or a generalization estimate. A separate BraTS-GLI case completed inference
  but yielded only 2 WT voxels and no TC/ET voxels; performance across dataset
  variants is therefore not established.
- The original SwinUNETR checkpoint remains on disk but is not the active
  registered model. SegResNet's MONAI forward path imports `einops`; it is
  included in the `brats` and `integrated` optional dependencies.
- The V8 spatial risk mapper may run after 3D segmentation when an aligned
  SynthSeg volume and its configured template are available. Its scores and
  corridor candidates are experimental, not calibrated injury probabilities
  or surgical recommendations. DTI and tract inputs are optional; when absent,
  V8 skips the tract component and renormalizes the weights of its available
  components rather than requiring a fabricated placeholder volume. Outputs
  include the spatial-risk map and any generated corridor candidates. This
  SegResNet checkpoint predicts only TC/WT/ET; it cannot produce an RC
  (resection-cavity) mask. A verified model or a separately supplied registered
  cavity label would be required for that output. Standalone corridor planning
  is not implemented.
- RAG is optional. Structured 2D and 3D segmentation, classification, spatial
  risk/corridor results, and modality labels/geometry are sent with the question
  only when the user explicitly enables structured context. General chat
  questions contain no patient context. Raw MRI pixels, voxel masks, and files
  are never sent. The chat checks the local API, literature chunk index, and
  Gemini API-key configuration; missing prerequisites are shown with setup
  instructions. Gemini receives the structured context, so do not send
  identifiable clinical information.
  Retrieved passages/citations are displayed separately from generated answer
  text. The local RAG setup guide is
  [`6_rag_treatment_planning/README.md`](../6_rag_treatment_planning/README.md).
- Markdown and PDF reports are generated from structured stage outputs and
  identify unavailable stages. When 3D inference has run, the PDF includes the
  model-generated ET/TC/WT overlay visualization, native-space geometry and
  mask voxel counts. Unrun or blocked stages are reported as such.

Run focused application tests with:

```powershell
python -m unittest integrated_pipeline.tests.test_application integrated_pipeline.tests.test_pipeline
```

The real 3D inference smoke test is opt-in because it requires a complete
four-modality study. Set `MRI_3D_SMOKE_CASE_DIR` to a folder containing
`<case-id>-t1n.nii.gz`, `-t1c.nii.gz`, `-t2w.nii.gz`, and `-t2f.nii.gz`;
optionally set `MRI_3D_SMOKE_CASE_ID` if the folder name differs from the file
prefix. Then run:

```powershell
python -m unittest 2_3d_segmentation.tests.test_segmentation3d_ontology.ModelCardInferenceSmokeTests
```

The test performs real checkpoint loading and full sliding-window inference,
then verifies output channels, finite probabilities, binary ET/TC/WT masks,
native geometry, and the saved visualization. It is skipped when no sample
directory is configured; a skipped test is not a successful inference
validation.
