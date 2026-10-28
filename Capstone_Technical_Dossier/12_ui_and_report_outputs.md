# 12 — UI and report outputs

## Integrated Streamlit UI

`integrated_pipeline/app.py` presents upload controls for `t2f`, `t1n`, `t1c`, `t2w`; optional SynthSeg labels; checkpoint paths; automatic tumor-centre/manual axial slice; toggles for optional 2D models; stage status; 3D/2D overlays; optional classification and risk/corridor tables; and an optional RAG query. A changed upload/configuration signature invalidates cached results. The RAG request requires an explicit authorization checkbox; the UI warns that raw uploaded MRI is never sent.

No application screenshot was found. The documentation describes implemented UI controls, not a demonstrated successful UI workflow.

## Research PDF output

`generate_report` builds a PDF in memory via Matplotlib `PdfPages`. It includes UTC generation time, study/model/config/shape/orientation/spacing/affine/slice/device/software metadata, stage statuses and reasons, prediction disclaimer, classification output, 3D WT and 2D segmentation overlays, optional risk/corridor overlay and risk provenance, and optional RAG answer/citations/warnings. It has no ground truth for uploads and labels masks predictions. Streamlit exposes a download button; no sample generated PDF was found or included.

The standalone RAG project includes report-generation scripts, but no example generated report artifact was located. Do not imply reports were reviewed or validated.
