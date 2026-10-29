# Colab demo implementation audit

Audit performed against the repository contents and the three checkpoint paths supplied in the task. The notebook is a research demonstration with prerequisite gating; it does not represent a completed clinical workflow.

## Component findings

| Stage | Actual implementation and artifacts | Finding / notebook decision |
|---|---|---|
| 2D segmentation | `3_2d_segmentation/src/segmentation2d/model_attention.py`, `engine_attention.py`, `1_data_and_preprocessing/src/data/brisc.py`; checkpoint `3_2d_segmentation/outputs/segmentation2d/02_attention_boundary_unet_import_20261019_1209/best.pt` | One-channel BRISC Attention U-Net, 256 square, grayscale bilinear resize, values normalized to [-1,1], foreground is sigmoid >=0.5. Checkpoint stores `model_state`, `model_config`, `image_size`, `mask_threshold`. Binary foreground only; no tumor subregion ontology. The run metadata says the split is sample-level and patient IDs unavailable. The demo uses an explicitly selected T1 slice and labels it as a BRISC-domain model applied to MRI; output is experimental and not clinical performance evidence. |
| 2D classification | `4_2d_classification/src/classification/model.py`, `engine.py`; checkpoint `4_2d_classification/outputs/classification/20261019_104128_seed2026/best.pt` | Custom BRISC CNN, four labels ordered `glioma, meningioma, no_tumor, pituitary`; 160x160, grayscale, pad resize, [-1,1], temperature-scaled softmax. Scores are model probabilities, not patient-level diagnosis or validated clinical likelihood. Official test split has seven exact image hashes overlapping train; patient separation is unavailable. This model's image domain is BRISC 2D imagery; MRI slice inference is out of distribution and must be explicitly disclosed. |
| 3D segmentation | `2_3d_segmentation/src/segmentation3d/models.py`, `monai_engine.py`, `1_data_and_preprocessing/src/data/brats.py`; local `2_3d_segmentation/outputs/segmentation3d/SwinUNETR/best.pth` | Repository training/inference pipeline expects its own `WT,TC,ET,RC` ontology and loader config order `t2f,t1n,t1c,t2w`. Its `_load_compatible_checkpoint` requires an ontology/config payload. Supplied local checkpoint contains `epoch`, `state_dict`, `optimizer_state_dict`, `best_acc`, so it is incompatible with that engine. Notebook uses the supplied Hugging Face model-card contract (`T1,T1ce,T2,FLAIR`; `ET,TC,WT`; feature size 48; nonzero channel-wise normalization; sigmoid; 0.5; sliding window 128 cube, overlap .7) separately and only proceeds after a strict MONAI state-dict load. The contract is user-supplied provenance; the local checkpoint itself does not encode the ontology. |
| Grad-CAM | `7_gradcam/gradcam.py`, `integrated_pipeline/modules/model_adapters.py`, `7_gradcam/handoff/gradcam_targets.md` | Notebook invokes the existing classifier Grad-CAM for predicted class and documented `BRISCClassifier.features[14]` target. It is a single-slice logit explanation. The handoff explicitly says no verified SwinUNETR target; no 3D heatmap is generated. |
| SynthSeg | `5_surgical_risk_mapping/pipeline/run_synthseg.py`, `scripts/run_synthseg.py` | Wrapper executes `freesurfer/freesurfer:8.2.0` through Docker, requires a separately supplied `license.txt`/`FS_LICENSE`, and is not a Colab-ready installation. No license or patient input supplied. |
| Registration and risk | `5_surgical_risk_mapping/pipeline/run_v8_pipeline.py`, `v7_risk_components.py`, `configs/config_v7.json` | V8 has strict geometry QC and expects MRI, tumor mask, SynthSeg labels, and optional DTI/tract inputs already in common patient space. It does not provide a general arbitrary-volume registration stage or a configurable V8 risk-map template for this upload workflow. Existing config points to a specific local case and cannot be reused as patient truth. Notebook gates risk until genuine aligned inputs and QC exist. |
| Corridor planning | `5_surgical_risk_mapping/pipeline/run_v8_pipeline.py` (A* path search, candidate ranking, Pareto handling) and legacy `corridor_planner.py` | V8 corridor method exists, but requires successful V8 patient-specific risk/anatomy inputs. It contains explicit fallback risk assumptions for missing structures in its code; absence of DTI/functional/vascular evidence must be reported. No corridor is described as safe. |
| RAG | `6_rag_treatment_planning/src/{ingestion,chunking,hybrid_retriever,vector_retriever,reranker,verification,rag_pipeline,llm}.py` | PDF extraction/chunking, BM25 + vector RRF, optional sentence-transformers embedding and cross-encoder with deterministic lexical fallbacks, evidence filters/verifiers, optional Ollama JSON generation. `knowledge_base/chunks.json` is absent in this checkout (`vector_db` only has `.gitkeep`); docs identify 40 downloaded papers in a local corpus, not included. Notebook skips unless a real built corpus exists. No API credential is embedded. |
| PDF/report | `integrated_pipeline/modules/report_generator.py`, `6_rag_treatment_planning/report_assets/` | Existing reports belong to separate stages; no general report compiler consumes outputs from all stages. Notebook adds a ReportLab summary generated from actual available stage outputs and status metadata. |

## Dependencies, compatibility and blockers

- Root `pyproject.toml` requires Python >=3.10 and torch >=2.1; `.[brats]` adds MONAI, nibabel, scipy. Project environment metadata records MONAI 1.4.0 with torch 2.11.0 for the 2D run and torch 2.13.0 CPU for classification. Colab setup leaves its preinstalled torch in place and pins MONAI 1.4.0 rather than reinstalling torch.
- Local checkpoints exist at the three paths above. The Swin checkpoint is large and the notebook uses a strict state load. Its provenance URL is the user-supplied Hugging Face model card; the model weights are not packaged in the source snapshot.
- No genuine uploaded four-modality patient study is available in repository data. The data README says datasets are local-only; file audit found no NIfTI case inputs under the preprocessing data folder. No end-to-end patient inference or Colab GPU run is claimed.
- The local `best.pth` container has an optimizer state and NumPy 2 pickle globals. Notebook uses an explicit safe-global allowlist with `weights_only=True`; this was verified on the local checkpoint. Never accept an untrusted `.pth` file.
- The explicit model-card ROI is 128 cubed; CPU memory/time may be substantial. OOM/incompatibility sets 3D to Failed with the exception and gates dependent stages.
- No Swin Grad-CAM target is verified; no functional, tractography or vascular input was supplied. DICOM is not supported by these project paths; required input is NIfTI.

## Deliverables and remaining gaps

- `notebooks/Capstone_Full_Demo.ipynb`: executable Colab-oriented flow, explicit modality selection/geometry validation, strict checkpoint load, stage status reporting, and PDF/JSON/ZIP artifacts.
- `notebooks/colab_demo_helpers.py`: affine/input validation, hashes, checkpoint-container checks, JSON and ZIP generation.
- Notebook still has `REPO_URL` placeholder requiring the owner's actual public repository URL or uploading the provided source snapshot. Classification inference on a selected NIfTI slice is out-of-domain and explicitly labeled. Risk/SynthSeg remain correctly gated because licensing/registration/data are unavailable.
- Checkpoints the user must provide to a fresh Colab session: 2D segmentation `best.pt`, 2D classification `best.pt`, SwinUNETR `best.pth` (or a verified download URL and checksum). They are excluded from source snapshot.
- External artifacts needed to unlock optional stages: one study with four explicitly mapped, already co-registered NIfTI modalities; optional ground-truth labels in a declared ontology; FreeSurfer license and working SynthSeg container; risk stage's compatible config/registration transforms; RAG `knowledge_base/chunks.json` or user-authorized corpus build.

## Test report

Commands executed in this implementation environment:

- `python notebooks/build_colab_notebook.py` — passed; generated 34-cell notebook.
- `python notebooks/validate_notebook.py` — passed; valid nbformat 4 JSON and all Python cells parse after removing Colab `%pip` magics.
- `python notebooks/create_source_snapshot.py` — passed; ZIP integrity verified, 1.47 MB source snapshot.
- `python -m unittest discover -s tests -p test_colab_demo_helpers.py -v` — 5 passed (missing modality, geometry/affine preservation, affine mismatch, PDF validity signature, ZIP integrity).
- Local Swin checkpoint safe container load (`torch.load(..., weights_only=True)` with the notebook's explicit globals) — passed; keys `epoch,state_dict,optimizer_state_dict,best_acc`, 159 state tensors. Strict MONAI wrapper load — passed with 0 missing and 0 unexpected keys. No forward inference was run because no genuine four-modality study is available.
- `python -m pytest ...` — not run: pytest is not installed in the local environment. No genuine study, SynthSeg license/runtime, or Colab GPU was available. Thus no clinical inference, risk map, corridor run, or Colab execution is claimed.

No clinical accuracy test can be inferred from software smoke checks.
