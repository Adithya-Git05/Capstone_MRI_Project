# Capstone Technical Dossier — start here

> **Evidence snapshot, not the current implementation guide.** This dossier
> preserves the report evidence and code inventory from its audit date. Some
> source paths and implementation-status statements predate the numbered stage
> layout and the ResNet/EfficientNet baseline implementations. Use the root
> [`README.md`](../README.md) and each numbered stage guide for current code,
> commands, and configuration paths. Historical metrics remain subject to the
> limitations recorded below and in the result chapters.

This directory is an evidence-backed dossier for the report and IEEE-paper writing stage. It is not a report draft. The ZIP contains this directory at its top level.

## Reading order

1. `01_project_overview.md` and `02_architecture_and_dataflow.md`
2. `04_datasets_and_preprocessing.md` and `dataset_inventory.csv`
3. `05_models_and_training_configuration.md`
4. `06_segmentation_results.md`, `07_classification_results.md`
5. `08_spatial_risk_methodology.md`, `09_corridor_optimization.md`
6. `10_rag_methodology_and_results.md`, `11_integration_and_testing.md`, `12_ui_and_report_outputs.md`
7. `14_results_provenance.md`, `15_known_limitations_and_open_questions.md`, `16_claims_evidence_matrix.csv`
8. `17_references_and_related_work.md`, `18_report_and_paper_writing_guide.md`

CSV and JSON files preserve metric rows, units, sources and caveats. `FIGURES/` holds selected original outputs; three legacy figures have only the case-ID title band cropped. Empty figure subfolders indicate that no actual artifact was located.

## Evidence rules

- Separate 2D from 3D, validation from test, run-level results from supplied aggregates, and measured values from derived/configured values.
- The actual current-code SegResNet output is one-epoch grouped validation on 56 cases; it is not a test result. A separate supplied SegResNet/SwinUNETR summary has substantially different values and no cohort/split provenance.
- Exact-image duplicates are found across BRISC official train/test folders, but training code excludes matching train-file copies before fit. Patient-level independence is still unverified.
- Spatial risk values are relative research estimates. Corridor outputs are computational candidates. No clinical validation or clinical safety claim is supported.
- RAG's 25 evaluation questions are pending annotation; retrieval/claim-quality scores are unavailable.

No raw datasets, checkpoints, run logs, credentials, patient identifiers, or sample patient reports are included. No training or expensive experiment was run for this dossier.
