# Custom four-stage CNN classifier

**Status: trained and test-evaluated; checkpoint available.** The primary
checkpoint is
[`best.pt`](../../../4_2d_classification/outputs/classification/20261019_104128_seed2026/best.pt)
(1,563,901 bytes; SHA-256
`CE8D857CFB2048DA1771B2F1E7A4F9988B0635EAB224E578526F7278FC564B58).
It safe-loads as a state-dictionary/configuration record, not a full model or
optimizer checkpoint.

The model has four Conv–GroupNorm–SiLU–MaxPool stages with widths 32, 64, 128,
256, then adaptive average pooling, dropout 0.25, and a four-logit linear head.
Each convolution is 3x3, stride 1, padding 1, bias false; GroupNorm uses 8
groups. The 389,348-parameter count is verified. Code:
[`model.py`](../../../4_2d_classification/src/classification/model.py#L52).

## Training and evaluation

The seed-2026 run completed 30 epochs and selected epoch 29 by minimum
validation cross-entropy. It used cross-entropy, AdamW (lr 0.001, weight decay
0.0001), batch 32, cosine annealing (eta_min 1e-6), and patience 7. The
hash-aware internal partition contains 3,742 fit, 749 validation, and 500
calibration images; 9 train files matching 7 official-test hashes were
excluded from internal partitions. The official test contains 1,000 images.
Seven exact image hashes are still shared across official train and test;
patient-level independence cannot be established because BRISC provides no
patient identifier. The run excluded the 9 matching train-side files from its
internal partitions, so no such copy was in the identified fit set. Cross-task
hash overlap counts are documented in
[`data_split_audit.md`](../data_split_audit.md). Source: run
[`metadata.json`](../../../4_2d_classification/outputs/classification/20261019_104128_seed2026/metadata.json)
and
[`evaluation_metadata.json`](../../../4_2d_classification/outputs/classification/20261019_104128_seed2026/evaluation_test_20261019_112654/evaluation_metadata.json).

The epoch-level record is preserved in
[`training_history.csv`](../../../4_2d_classification/outputs/classification/20261019_104128_seed2026/training_history.csv)
(30 rows with train loss, validation loss/accuracy/balanced accuracy/macro-F1
and learning rate). A compact copy of epoch, train loss, validation loss, and
validation accuracy is included as
[`custom_cnn_training_history.csv`](./custom_cnn_training_history.csv). The completed run log is
[`run.log`](../../../4_2d_classification/outputs/classification/20261019_104128_seed2026/run.log).

Raw official-test metrics: accuracy 0.866, balanced accuracy 0.8822, macro
precision 0.8559, macro recall 0.8822, macro F1 0.8618, weighted F1 0.8652,
NLL 0.3356, Brier 0.1964, ECE 0.0187. Class order is `glioma`, `meningioma`,
`no_tumor`, `pituitary`; this order is established in
[`brisc.py`](../../../1_data_and_preprocessing/src/data/brisc.py#L14).

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| glioma | 0.937778 | 0.830709 | 0.881002 | 254 |
| meningioma | 0.837037 | 0.738562 | 0.784722 | 306 |
| no_tumor | 0.716495 | 0.992857 | 0.832335 | 140 |
| pituitary | 0.932476 | 0.966667 | 0.949264 | 300 |

Confusion matrix (actual rows, predicted columns):

```text
[[211,  38,   0,   5],
 [ 11, 226,  53,  16],
 [  0,   1, 139,   0],
 [  3,   5,   2, 290]]
```

One-vs-rest AUCs are stored per class in the test JSON. Derived using class
supports, raw macro/support-weighted ROC-AUC are 0.977906/0.974425 and raw
macro/support-weighted PR-AUC are 0.943957/0.937034. These aggregate values
are calculated from per-class values, not directly saved. Temperature
`0.7881783159` was fit on the reserved 500-example calibration split; raw
metrics use T=1. Calibrated probability-quality values differ, but positive
temperature scaling preserves argmax classification. Full artifacts:
[`metrics.json`](../../../4_2d_classification/outputs/classification/20261019_104128_seed2026/evaluation_test_20261019_112654/metrics.json)
and
[`calibrated_metrics.json`](../../../4_2d_classification/outputs/classification/20261019_104128_seed2026/evaluation_test_20261019_112654/calibrated_metrics.json).
Training curves, confusion matrix, ROC, and precision-recall figures are
available in the run folder:
[`training_curves.pdf`](../../../4_2d_classification/outputs/classification/20261019_104128_seed2026/training_curves.pdf),
[`confusion_matrix.pdf`](../../../4_2d_classification/outputs/classification/20261019_104128_seed2026/confusion_matrix.pdf),
[`roc.pdf`](../../../4_2d_classification/outputs/classification/20261019_104128_seed2026/roc.pdf),
and
[`precision_recall.pdf`](../../../4_2d_classification/outputs/classification/20261019_104128_seed2026/precision_recall.pdf).
No inference-overlay or Grad-CAM example was found for this classifier.

## Grad-CAM

Recommended target: `model.features[14]`, observed `[B,256,20,20]` for a
160x160 input. Score the predicted class raw logit (optionally repeat for the
ground-truth logit). No Grad-CAM source or saved visualization exists. Overlay
on the same padded/resized grayscale 160x160 input used by inference.

## Provenance discrepancy

The technical dossier says no ResNet50/EfficientNetB0 implementation was
located; code/configs and a partial ResNet50 checkpoint exist. That statement
should be corrected to distinguish this completed CNN result from the partial
ResNet50 and untrained EfficientNetB0 variants. The earlier custom-CNN run
(`20261018_214516_seed2026`) completed 30 epochs but has validation-only
metrics and a less careful sample-level split; it is not the primary test
result.
