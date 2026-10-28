# Source inventory

## Current project (`New`)

- `README.md`, `docs/DATASETS.md`, `docs/EXPERIMENTS.md`, `docs/PAPER_RESULTS.md`
- `configs/classification.yaml`, `configs/segmentation2d.yaml`, `configs/segmentation3d.yaml`
- `src/classification/model.py`, `src/classification/engine.py`
- `src/segmentation2d/model.py`, `model_attention.py`, `engine.py`, `engine_attention.py`
- `src/data/brisc.py`, `src/data/brats.py`, `src/segmentation3d/models.py`, `metrics.py`, `monai_engine.py`
- classification `metrics.json`, calibrated metrics, evaluation metadata and plots
- 2D segmentation test/validation metric JSONs and plots
- SegResNet run `metrics.json`, sanitized metadata facts, and config
- `integrated_pipeline/app.py`, `pipeline.py`, adapter modules, `README.md`, tests

## Older project components reused or compared

- `Capstone_MRI_Project/pipeline/v7_risk_components.py`, `run_v8_pipeline.py`, `label_mapping.py`, `corridor_planner.py`, `pareto_frontier_analysis.py`
- `Capstone_MRI_Project/config_v7.json`
- `Capstone_MRI_Project/outputs/V7_VALIDATED/` risk, QC, tract and ablation summaries
- three separately saved risk-ablation table versions
- `Capstone_MRI_Project/outputs/V8_CORRIDORS/` candidate/Pareto CSVs and figures
- `Capstone_MRI_Project/rag-chatbot/` README, API, RAG pipeline, chunking, retrievers, reranker, generation, verification, schemas, test sources and corpus audit

## Exclusions

No raw MRI/NIfTI data, individual-case metric rows, identifiers, checkpoint/weight files, full run logs, secrets, APIs, model caches, executable installers or literature PDFs are copied. The available 3D case-level visualization was excluded because visible case identifiers were printed on the figure. Three other case-specific figure title bands were cropped in packaged copies; their underlying plotted image data were otherwise retained.
