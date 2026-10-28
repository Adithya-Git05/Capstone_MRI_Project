# 06 — Segmentation results

## 2D BRISC attention-boundary model

The saved test evaluation is n=860, official-folder `test`. Global foreground Dice is 0.875253 and IoU 0.778178. Mean per-image Dice is 0.856058 and mean per-image IoU 0.781571. Boundary BCE + soft-Dice loss is 0.076798. All 860 targets are nonempty; the source reports empty_target_count=0. The evaluation was run on a test split, but patient independence is not established. The model is `BRISCAttentionUNet`; it is distinct from the plain `BRISCUNet` and all 3D models. No 2D HD95 is recorded.

The earlier validation artifact has n=590, global Dice 0.863298, global IoU 0.759477, mean image Dice 0.859982, mean IoU 0.784476. It is validation, not test.

## 3D SegResNet actual current-code validation

One run trained for one epoch and evaluated 56 grouped-validation cases. Per-region means are:

| Region | Mean per-case Dice | Mean per-case IoU | Mean HD95 (mm) among valid cases | HD95 valid cases / 56 |
|---|---:|---:|---:|---:|
| WT | 0.500850 | 0.370024 | 61.1282 | 56 |
| TC | 0.244497 | 0.161615 | 75.4687 | 53 |
| ET | 0.156379 | 0.093527 | 77.2521 | 53 |
| RC | 0.071082 | 0.044500 | 77.5003 | 45 |

Aggregate validation BCE+soft-Dice loss is 0.748115. These are per-case arithmetic means in `metrics.json`, not test metrics. `region_metrics` assigns Dice/IoU=1 when both masks are empty; HD95 is null when either mask is empty. The reported HD95 mean uses only valid pairs, with counts shown above. The run is one epoch and the labels/release provenance are unverified; do not overstate performance.

## Separate supplied 3D comparison table and discrepancy

Another CSV supplies much higher SegResNet values (Dice 80.18–86.94%; HD95 5.21–7.39 mm) and SwinUNETR values (Dice 81.59–87.42%; HD95 6.02–7.21 mm), with IoU derived from Dice. It documents no cohort, split, sample count or corresponding run metadata. These values conflict substantially with the current run-level SegResNet grouped-validation values above. The source README says values were supplied and figures do not establish individual-case or independent evaluation. Do not average, select, or reconcile the two sets without provenance.

No current SwinUNETR run-level result was found. No 3D case-level overlay is packaged because the available figure visibly embeds case identifiers; aggregate plots are included and identified as supplied/derived.

All rows, units and source types are in `segmentation_metrics.csv` and `.json`.
