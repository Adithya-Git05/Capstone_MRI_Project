# 18 — Report and paper writing guide

This is a factual map for drafting against the supplied BMSCE/IEEE templates; it is not report prose.

## BMSCE chapter mapping

1. **Introduction:** problem context and research prototype scope from `01_project_overview.md`.
2. **Problem Statement:** define separate tasks and supported claims; avoid clinical outcome claims.
3. **Detailed Survey:** use independently verified works only. Candidate local RAG records in `references.bib` are not verified.
4. **Survey Summary Table:** compile only after bibliographic verification; no fabricated survey comparisons.
5. **System Requirement Specification:** use actual modality/checkpoint/label prerequisites in architecture and module inventory.
6. **System Design:** Mermaid source-grounded architecture/dataflow. Mark optional risk/RAG stages and missing inputs.
7. **Implementation:** module methods and config; distinguish current `New` modules from reused legacy risk/RAG/corridor components.
8. **Results:** use `classification_metrics.csv`, `confusion_matrix.csv`, `segmentation_metrics.csv`, `spatial_case_results.csv`, and `corridor_results.csv`. Keep validations, test metrics and supplied aggregates separate.
9. **Testing:** list source-defined tests from `test_inventory.csv`; state no saved execution result, unless a separately verified test run is later provided.
10. **Conclusion and Future Enhancements:** state only observed software/evaluation; future work from `15_known_limitations_and_open_questions.md`.
11. **Bibliography/appendix:** verify candidate references and cite source datasets/models before finalizing.

## IEEE paper mapping

- **Abstract/Introduction:** name modules and contributions without clinical utility claims.
- **Related Work:** use primary-source verified citations; local corpus metadata is provisional.
- **Methodology:** architecture, datasets, preprocessing, exact model configs, current risk equations, corridor algorithm and RAG workflow.
- **Experimental setup:** state separate cohorts, sample counts, split definitions, one-epoch 3D limitation and image-hash duplicate handling.
- **Results:** show actual classification test and 2D test; actual 3D grouped validation; supplied 3D aggregates in a clearly distinct/qualified table; single-example spatial/corridor measurements; RAG unavailable scores.
- **Discussion:** discuss model/run disparities, dataset/provenance uncertainties, threshold/weight limitations and the lack of end-to-end/clinical validation.
- **Conclusion:** report demonstrated engineering implementation and narrow quantitative evidence only.

## Recommended tables/figures

- System architecture Mermaid: System Design / Methodology.
- Dataset and label CSVs: Experimental Setup / Methods.
- Confusion matrix and class curves: Classification Results.
- 2D test overlay: qualitative 2D Results.
- 3D metrics table: Results with grouped-validation label; show supplied aggregates separately and discuss discrepancy.
- Risk components/tract table and redacted component map: Spatial-risk Methodology/Results, with nonprobability disclaimer.
- Corridor overlay/Pareto plot: Results as candidate analysis only.
- RAG table: report corpus counts and pending evaluation status, not quality scores.

The dossier contains no clinical images generated synthetically, no fake metrics, and no final prose.
