# 09 — Surgical corridor optimization

## V8 route generation

The current legacy V8 route implementation is `pipeline/run_v8_pipeline.py:plan_corridors_v8`, with its own `astar_3d`; an older `pipeline/corridor_planner.py` has a different route cost multiplier and step limit. These implementations must not be conflated.

- **Target:** rounded tumor-mask centroid voxel.
- **Entry generation:** one middle candidate voxel is selected from brain-mask boundary voxels by axis-coordinate predicates labelled lateral frontal, superior frontal, anterior frontal and pterional/temporal. If fewer than two candidate classes are found, code adds fallback surface points. This is geometric heuristic selection, not neurosurgical anatomy/clinical approval.
- **A* neighbors:** 18 face/edge-connected neighbors (6 axial plus 12 two-axis diagonal); no three-axis corner moves. Search terminates when Euclidean voxel-coordinate distance to target is <=1.5, or fails at 150,000 steps.
- **Edge/path cost:** each physical step distance is multiplied by `(1 + risk_at_next_voxel*4)`. Heuristic is physical Euclidean distance times 2.5. Because it is weighted, it can be non-admissible for the implemented edge cost and does not guarantee globally optimal paths.
- **Corridor mask:** path voxels are binary-dilated for two iterations on the voxel grid; the result is not inherently a fixed 2mm radius under anisotropic spacing.
- **Path length:** sum of physical distances between consecutive path voxels.
- **Component metrics:** mean/max risk over dilated corridor; means of consequence and tract maps when present. Vascular risk is hard-coded 0 because imaging unavailable; ventricular risk is hard-coded 0 because penetration is not modeled.
- **Coverage:** dilated corridor/tumor voxel intersection divided by tumor voxel count.

Composite cost in V8 code is:

`0.15*(length_mm/100) + 0.25*eloquent + 0.30*tract + 0.15*vascular + 0.15*ventricular - 0.20*coverage` (weights can be configured).

Terms are not normalized to a common statistical scale. No clinically validated interpretation or calibration exists. Because vessel and ventricle terms are zeros by implementation, their contribution is not measured evidence.

## Pareto definitions and saved outputs

`run_v8_pipeline.py:pareto_analysis` minimizes mean risk, tract risk and path length and maximizes coverage. A separate `pareto_frontier_analysis.py:evaluate_pareto_front` marks a 2D (length/mean risk) frontier and 4D dominance on length, mean risk, max risk and tract risk. Saved CSV/HTML outputs report all four named candidate trajectories as Pareto-efficient and a baseline as not efficient; no run metadata proves which implementation generated every dashboard artifact. Preserve the saved flags but state that code files contain different objective sets.

The baseline cost in the separate Pareto script uses a different hard-coded expression (length, tract and mean risk) and a fixed coverage value, unlike candidate V8 composite cost. Baseline and candidate composite costs therefore are not directly comparable. Corridor CSV variants otherwise retain the four candidate metric values; see `corridor_results.csv`.

## Example output and interpretation

Four saved candidates have lengths 29.97–56.11 mm, mean risk 0.2547–0.3647, max risk 0.4591–0.6938, tract risk 0.0735–0.111 and reported costs 0.0736–0.1287. Coverage values 0.0049–0.0063 are preserved as dimensionless source values; do not reinterpret as percent because the reporting semantics are not validated. The Pareto table assigns strategy descriptions such as shortest path or lowest mean/peak risk.

These are computational candidate trajectories for one example. They are not clinically validated plans, a recommendation, or evidence of surgical safety.
