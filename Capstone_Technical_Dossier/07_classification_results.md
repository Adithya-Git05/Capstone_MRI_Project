# 07 — Classification results

## Dataset/classes and split

The four folder classes are glioma, meningioma, no_tumor and pituitary. The documented official-folder class counts are 1,147/1,329/1,067/1,457 for train and 254/306/140/300 for test. One run reports a 750-image validation split with accuracy 0.936000, balanced accuracy 0.938141, macro-F1 0.936198 and weighted-F1 0.935676. A later run records a 1,000-image official-folder test evaluation.

The training engine groups exact SHA-256 duplicate images when making train/validation/calibration partitions. In the later run, 5,000 official train images were audited; 9 train-file copies matching 7 unique official test image hashes were excluded before model fitting. The fit/validation/calibration counts recorded were 3,742/749/500. Test membership was not used for checkpoint selection. This protects model fitting from those exact duplicates but does not establish patient-level independence, rule out near-duplicates, or replace patient IDs.

## Test performance

Uncalibrated: accuracy 0.866000; balanced accuracy 0.882199; macro precision 0.855946; macro recall 0.882199; macro-F1 0.861831; weighted-F1 0.865206; NLL 0.335593; multiclass Brier 0.196397; ECE 0.018711. Per-class AUROC/AUPRC are recorded in `classification_metrics.csv`; macro AUROC 0.977906 and macro AUPRC 0.943957 are derived as the unweighted arithmetic means of four one-vs-rest class values, not direct aggregate fields.

Temperature-scaled test output uses scalar T=0.788178 fitted on the 500-image official-train calibration subset. Accuracy/F1 do not change; NLL is 0.347729, Brier 0.200279 and ECE 0.044589 in the calibrated JSON. This snapshot's probability-quality metrics are worse than the uncalibrated test values; do not claim that calibration improved them.

The exact recorded test confusion matrix (actual rows, predicted columns, order glioma/meningioma/no_tumor/pituitary) is:

```text
[[211, 38,  0,  5],
 [ 11,226, 53, 16],
 [  0,  1,139,  0],
 [  3,  5,  2,290]]
```

These are observed count cells, not inferred values. `confusion_matrix.csv` is the long-form equivalent. The plotted ROC, precision-recall, confusion and reliability outputs are existing artifacts.

## Model limitation

The current code implements a custom `BRISCClassifier`; it is not ResNet50 or EfficientNetB0. No verified evaluation of those named architectures was found. Do not attribute these scores to them.
