# 01 — Project overview

## Scope

The workspace contains several related research pipelines under **AI-Driven Framework for Brain Tumor Analysis from MRI**, but the modules use different data, tasks and levels of evidence. Current `New` source code is authoritative for BRISC 2D classification/segmentation and local BraTS-format 3D segmentation. A separate `integrated_pipeline` Streamlit application orchestrates those current modules and calls selected older Capstone risk/RAG components. Older code remains relevant for spatial-risk mapping, corridor optimization and the standalone RAG service; it is not evidence that all modules have been validated together.

## Implemented vs. evaluated vs. sample-only

| Component | Implemented | Experimentally evaluated in saved outputs | Integration status |
|---|---|---|---|
| BRISC classifier | Custom `BRISCClassifier`; four-class training/evaluation/inference | One validation result and one official-folder test result, with per-class metrics, exact confusion matrix, ROC/PR/calibration plots | Optional adapter on one `t1c` slice; domain compatibility to BRISC is unvalidated |
| BRISC binary segmentation | `BRISCUNet` and attention-gated `BRISCAttentionUNet`; training/evaluation | Attention-boundary model trained 40 epochs, best epoch 39, and evaluated on 860 test images; visual overlays | Optional adapter; applies BRISC mask model to MRI slice out of domain |
| 3D segmentation | MONAI SegResNet/SwinUNETR paths and ontology-aware loaders/predictor | One SegResNet one-epoch grouped-validation run on 56 cases; no official test and no current Swin run found | Required first stage of Streamlit app when a compatible checkpoint is provided |
| Spatial risk | Legacy V8 geometric/anatomical/tract/consequence model, strict QC, tables/maps | Saved V8 one-example outputs; DTI is available in saved legacy example | Optional only with user-uploaded, registered SynthSeg labels; integrated path has no DTI/tract inputs |
| Corridor optimization | A* candidate route search and multiple Pareto evaluators | Saved one-example candidate tables/plots | Invoked by V8 risk pipeline when risk inputs pass; no clinically validated route |
| RAG | Standalone FastAPI service, retrieval, generation, citation normalization and consistency checks | Corpus audit only; all 25 evaluation questions pending annotation | Separate service called over HTTP after explicit Streamlit authorization |
| PDF report | Current integration report generator | No saved generated sample report | Download generated on demand from in-memory stage results |

## Main evidence gaps

The local BraTS-format release provenance and label ontology are unverified. The run-level SegResNet result is weak and only one epoch, whereas the supplied comparison table reports much higher values without evaluation provenance. The distinction must remain explicit. RAG quality has not been evaluated. The integration test files are present but no test-run or end-to-end application evidence was saved. There is no clinical validation.

Sources: `README.md`; `docs/DATASETS.md`; `docs/PAPER_RESULTS.md`; `integrated_pipeline/README.md`; module inventory in `03_module_inventory.csv`.
