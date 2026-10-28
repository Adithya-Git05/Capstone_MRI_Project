# 14 — Results provenance and discrepancies

## Metric evidence

- **Classification validation:** `outputs/classification/20261018_214516_seed2026/metrics.json`, n=750. Directly recorded validation metrics.
- **Classification test:** `outputs/classification/20261019_104128_seed2026/evaluation_test_20261019_112654/metrics.json`, per-class CSV/plots and `evaluation_metadata.json`, n=1,000. Directly recorded official-folder test. The fit pipeline explicitly excluded nine train-file copies of seven image hashes found in test, but patient-level independence is not known.
- **2D segmentation:** `.../evaluation_test.../test_metrics.json` and adjacent metadata. Directly recorded test metrics, n=860, model `BRISCAttentionUNet`. Validation is a separate n=590 artifact. Neither is a 3D metric.
- **3D segmentation:** `outputs/segmentation3d/segresnet/20261019_151137_seed2026/metrics.json` is a current-code one-epoch grouped-validation output (56 cases). Directly recorded per-case means; not a test set.
- **Separate 3D comparison:** `outputs/segmentation3d/segmentation_paper_figures/reported_metrics.csv` with README/generator. Dice and HD95 are supplied; IoU is derived. No cohort/split/case count or current-run correspondence is recorded.
- **Spatial-risk example:** `Capstone_MRI_Project/outputs/V7_VALIDATED/patient_summary.json`, tract/QC CSV/JSON, and three ablation output versions. Single-example reported measurements and configurations.
- **Corridors:** `Capstone_MRI_Project/outputs/V8_CORRIDORS/corridor_results.csv`, Pareto tables and figures. Direct saved candidate values; baseline/Pareto variants are separate scripts.
- **RAG:** corpus audit and 25-question set. Counts are direct; quality metrics unavailable.

## 3D discrepancy

The current run-level SegResNet grouped-validation Dice values are approximately 0.501 WT, 0.244 TC, 0.156 ET, 0.071 RC, with HD95 approximately 61–78 mm on valid cases. The separate supplied SegResNet table reports 86.94%, 81.61%, 80.18%, 84.77% Dice and 5.21–7.39 mm HD95. This is a substantial discrepancy. The supplied table has no cohort, split, sample count, run config or checkpoint provenance; its IoU is calculated from supplied Dice. There is no evidence to reconcile the values or declare them from the same evaluation. Current root code and run metadata make the one-epoch grouped-validation record the traceable current-code result; retain the supplied table as a distinct unverified aggregate. SwinUNETR only appears in the supplied table; no current run output found.

## Risk ablation discrepancies

`spatial_case_results.csv` preserves the legacy ablation summary, V7_VALIDATED table and V8_ablation table. Values for identically named A–F configurations vary, including metrics and weights. Version labels and code context differ. The dossier does not choose one as a correction or combine the metrics.

## Corridor discrepancies

Candidate values are repeated in `corridor_results.csv`/`pareto_results.csv` with consistent measurements. However, V8 source `pareto_analysis` minimizes length/mean/tract and maximizes coverage; separate Pareto code includes max-risk 4D objectives. The naive baseline uses a third hard-coded cost formula and coverage value, not the candidate composite formula. Pareto booleans and baseline cost should therefore be attributed to their source output, not treated as one comparable universal ranking.

All source paths are relative to the original workspace. Patient identifiers and literal hash values are intentionally not reproduced.
