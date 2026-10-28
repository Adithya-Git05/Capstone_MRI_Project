# 13 — Figure and table catalog

## Existing figures included

| Filename | Description / module | Provenance | Suggested section | Caveat |
|---|---|---|---|---|
| `FIGURES/classification/confusion_matrix_test.png` | Test confusion matrix | Original output | Results | Exact hashes overlap across source folders; fit copies excluded, patient-level identity unknown |
| `FIGURES/classification/roc_test_uncalibrated.png` | Uncalibrated ROC | Original output | Results | Macro AUROC in table is derived from class-wise values |
| `FIGURES/classification/precision_recall_test_uncalibrated.png` | Uncalibrated precision-recall | Original output | Results | Macro AUPRC is derived |
| `FIGURES/classification/reliability_test_uncalibrated.png` | Calibration plot | Original output | Results | Calibration does not improve saved ECE/Brier/NLL |
| `FIGURES/classification/*temperature_scaled.png` | Calibrated score plots | Original output | Results | Shows fitted temperature; test probability metrics worsened |
| `FIGURES/segmentation/representative_2d_test_overlays.png` | MRI/reference/prediction | Original qualitative output | Results | BRISC slice-level qualitative samples |
| `FIGURES/segmentation/2d_test_metric_distributions.png` | Per-image metric distribution | Original output | Results | n=860 test set |
| `FIGURES/segmentation/3d_supplied_dice_comparison.png` | Supplied aggregate Dice | Existing generated plot | Results | Supplied values, no documented cohort/split |
| `FIGURES/segmentation/3d_derived_iou_comparison.png` | IoU derived from supplied Dice | Existing generated plot | Results | Derived, not an independent metric |
| `FIGURES/segmentation/3d_supplied_hd95_comparison.png` | Supplied aggregate HD95 | Existing generated plot | Results | No run-level provenance |
| `FIGURES/risk_mapping/risk_map_comparison_anonymized.png` | Baseline/enhanced/delta | Existing figure; title band cropped | Method/results | Relative risk score; one example |
| `FIGURES/risk_mapping/component_maps_anonymized.png` | Risk components | Existing figure; title band cropped | Method/results | Example case; no clinical validity |
| `FIGURES/corridor_optimization/candidate_corridor_overlay.png` | Candidate path contours | Existing figure | Results | Candidate only |
| `FIGURES/corridor_optimization/pareto_tradeoff_frontier_anonymized.png` | Candidate trade-offs | Existing figure; title band cropped | Results | Pareto objectives vary by code path |
| `FIGURES/architecture/system_architecture.mmd` | Source-grounded architecture | Dossier diagram, generated from code relationships | System design | Not evidence of execution |

Figure copies with `_anonymized` in filename differ only by removal of visible case-ID title band; data panels remain original. The dossier does not include a case-level 3D overlay because the available image visibly prints case identifiers.

## Tables

See `03_module_inventory.csv`, `dataset_inventory.csv`, `label_mapping.csv`, `segmentation_metrics.csv`, `classification_metrics.csv`, `confusion_matrix.csv`, `risk_component_weights.csv`, `spatial_case_results.csv`, `corridor_results.csv`, `rag_configuration.csv`, `rag_evaluation.csv`, `test_inventory.csv`, and `16_claims_evidence_matrix.csv`. Every table has source paths and/or caveats in its rows or corresponding module narrative.

## Not found

No Grad-CAM visualization, class diagram, activity diagram, use-case diagram, app screenshot, sample generated PDF, or verified case-level 3D prediction overlay was found. These categories are not fabricated.
