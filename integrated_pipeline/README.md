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

Use the same Python environment for installation and launch. The registered
SwinUNETR checkpoint was strictly tested with PyTorch 2.14.1+cpu and MONAI
1.6.1; a different PyTorch release is not treated as verified by the UI. The
3D forward path additionally requires `einops`, included in the `brats` and
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

The 2D classifier path above follows the user-supplied local checkpoint; it is
not the newer test-evaluated checkpoint described in the technical dossier.
Do not attribute the newer checkpoint's test metrics to this earlier file.
Dataset provenance and patient-level independence are not established.
Classifier outputs are the BRISC folder classes `glioma`, `meningioma`,
`no_tumor`, and `pituitary`; they are not pathology-confirmed subtypes.

The selected checkpoint files are local artifacts and are not version-controlled.
When present at the registry paths, the app checks the recorded size and SHA-256
before inference; another checkout must supply the matching files or configure
local paths in the sidebar. No predictions are produced without compatible
weights. See the checkpoint manifest and
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
- The registered SwinUNETR checkpoint is specified by its supplied model-card
  contract as four inputs in `[T1, T1ce, T2, FLAIR]` order and three outputs in
  `[ET, TC, WT]` order. The local checkpoint's tensor shapes and strict state
  dictionary load were verified against PyTorch 2.14.1+cpu and MONAI 1.6.1;
  that proves architectural compatibility, not training provenance or clinical
  performance. Model-card details were supplied by the user because the remote
  model card could not be independently retrieved. The app verifies the
  checkpoint manifest and runtime versions before enabling inference.
- 3D inference requires all four finite, registered NIfTI volumes and the
  user's confirmation that they are from the same skull-stripped study. It
  preserves native voxel axes/affine, performs channel-wise nonzero
  normalization, and uses a 64-cube sliding window with batch size 1 for the
  CPU-constrained app. The supplied model-card ROI is 128-cube; using a smaller
  ROI is an explicit runtime memory adjustment, not a change to the checkpoint
  or whole-volume preprocessing. It produces ET/TC/WT masks only; RC is
  unavailable and is never synthesized. CPU/RAM preflight can block inference.
  A successful 64-cube checkpoint forward pass verifies runtime compatibility,
  not a patient case. A complete real four-modality smoke test and clinical
  performance remain unvalidated until a suitable study sample is run.
- SwinUNETR's MONAI forward path imports `einops`; it is included in the
  `brats` and `integrated` optional dependencies so installing
  `.[brats,integrated]` supplies the runtime dependency.
- The V8 spatial risk mapper may run after 3D segmentation when an aligned
  SynthSeg volume and its configured template are available. Its scores and
  corridor candidates are experimental, not calibrated injury probabilities
  or surgical recommendations. Standalone corridor planning is not
  implemented.
- RAG is optional. Only structured analysis outputs and the question are sent
  when the user explicitly enables structured context. General chat questions
  contain no patient context. Raw MRI pixels and files are never sent. The chat
  checks the local API, literature chunk index, Ollama service, and configured
  Ollama model; missing prerequisites are shown with setup instructions.
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
