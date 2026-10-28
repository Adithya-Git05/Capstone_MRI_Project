# 02 — Architecture and dataflow

## Frontend and backend

- **Integrated UI:** Streamlit in `integrated_pipeline/app.py`. It accepts four uploaded NIfTI modalities, checkpoint paths, optional SynthSeg labels, slice settings, and an optional RAG query. UI and `run_pipeline` execute in the same Python application process; this is not a separate integrated REST backend.
- **Local orchestrator:** `integrated_pipeline/pipeline.py:run_pipeline` validates and temporarily stores uploads, runs the current 3D predictor, extracts a `t1c` axial slice, and calls optional 2D segmentation/classification adapters. Optional V8 risk mapping requires a separately supplied anatomical SynthSeg volume.
- **RAG service:** Separate FastAPI application in `Capstone_MRI_Project/rag-chatbot/api/main.py`; serves a static HTML frontend and JSON endpoints. The Streamlit app posts a structured `PatientContext` to `/api/rag/query` only after the user checks an explicit authorization control. `rag_adapter.py` sends no raw MRI.
- **Report output:** `integrated_pipeline/modules/report_generator.py:generate_report` creates a PDF in memory from structured run results. Streamlit exposes a download button; no sample PDF is persisted in the workspace.

## Stage flow

1. Validate exactly the four `t2f`, `t1n`, `t1c`, `t2w` volumes for 3D shape/affine match, finite/nonzero voxel arrays and registered geometry (`mri_io.validate_study`).
2. Run `src.segmentation3d.monai_engine.predict_case` with a checkpoint carrying matching ontology metadata; output is RAS-oriented and thresholded at 0.5.
3. Load predicted WT/TC/ET/RC masks, check their shape/affine against source volume, choose an automatic WT-centre slice or user-selected slice, and extract `t1c` (`slice_extraction`).
4. Optionally run `BRISCUNet` segmentation and `BRISCClassifier` on the extracted slice. These are model outputs, not confirmed diagnosis/segmentation, and the MRI-to-BRISC domain transfer is explicitly unvalidated.
5. Risk mapping remains blocked unless aligned SynthSeg anatomical labels are uploaded. Integrated adapter rejects geometry/label problems, reorients MRI/SynthSeg to RAS, and calls legacy V8. It supplies placeholder missing DTI/tract paths; risk is therefore structural/anatomical only in this adapter, not the DTI-backed saved legacy case.
6. Optionally send structured model results and a question to the separate RAG service after explicit authorization. The raw MRI volume is not sent.
7. Display stage states, RAS slice overlays, optional risk/corridor output and RAG JSON, and prepare a downloadable research PDF.

## Integration status

Implemented interfaces do not prove a complete successful end-to-end run. 3D segmentation is the required configured stage; 2D tasks are optional; risk requires a compatible uploaded SynthSeg volume; RAG requires a separately available FastAPI/Ollama service and user authorization. The integration README documents domain/provenance warnings and no clinical validation. No saved end-to-end output or application screenshot was found.

The Mermaid diagram at `FIGURES/architecture/system_architecture.mmd` is a source-grounded dossier diagram. It distinguishes current modules from reused legacy services and optional stages. No original class, activity or use-case diagram was found.
