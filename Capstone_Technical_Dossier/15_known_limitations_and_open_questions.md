# 15 — Known limitations and open questions

## Data and split provenance

- BRISC source/release citation is not preserved in the inspected project docs. Filenames are not verified patient identifiers; image-level duplicate grouping is not patient-level independence.
- 3D local data have 271 folders versus a preserved legacy 1,351 count; source release and official label ontology are unverified. Current region mapping is provisional.
- Segmentation and classification use separate BRISC tasks and partitions; do not treat them as the same cohort or split.
- One spatial-risk output is a single example; original cohort/publication source is not established in inspected artifacts.

## Models and evaluation

- No verified ResNet50 or EfficientNetB0 implementation/evaluation found.
- Actual 3D SegResNet result is one epoch, CPU, 56 grouped validation cases; there is no official test split. No current SwinUNETR run found.
- Supplied 3D metrics differ substantially from current SegResNet results; supplied cohort/protocol is unknown.
- Dice/IoU aggregation and empty-mask handling differ by evaluator. Current 3D code defines empty-empty Dice/IoU as 1 and omits HD95 if either mask is empty; supplied values do not document aggregation.
- 2D segmentation test metrics are binary foreground; no HD95 recorded. BRISC patient-level independence unavailable.
- Class-wise classification metrics are available; macro AUROC/AUPRC in dossier are derived arithmetic means, and calibrated probability-quality metrics are worse than uncalibrated values.

## Spatial risk and route planning

- DTI/tract provenance, thresholding and registration to T1 are not independently established. Saved code warns tract overlap may be invasion or partial volume/registration error. Functional fMRI and MRA/CTA absent in saved example.
- Risk threshold labels are unvalidated visualization bins. Risk is not calibrated injury probability and no clinical outcome validation exists.
- Integration risk path has no DTI/tract inputs; weights differ from saved DTI-backed example.
- Corridor planner has different A* and Pareto variants across source files. Composite terms have inconsistent scales, vascular/ventricular terms are hard-coded zero, and Pareto outputs are not clinical recommendations.

## RAG and integration

- 40-paper corpus has 25 missing licenses; evidence labels and citations are not manually reviewed.
- All 25 RAG questions pending; no Recall@K, Precision@K, citation quality, claim support or numerical consistency evaluation results.
- Claim support is lexical overlap, not semantic entailment; citation ID verification is not semantic citation verification.
- No integrated end-to-end run summary, screenshots or test pass output. Defined source test counts do not imply execution.
- No verified consolidated methods bibliography; local RAG candidate references remain unverified.

## Report-writing cautions

Do not claim clinical validation, diagnostic effectiveness, patient safety, clinically recommended route, independent patient-level test results, statistical significance or unreported sample counts. Do not merge 2D and 3D metrics or reconcile inconsistent aggregates by averaging.
