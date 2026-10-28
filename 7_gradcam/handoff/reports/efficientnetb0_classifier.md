# EfficientNetB0 classifier

**Status: implemented/configured; no training run, checkpoint, or evaluation
was found.** This must not be reported as a trained model.

The implementation wraps torchvision EfficientNetB0 with `weights=None`.
`network.features[0][0]` is replaced by a single-channel
`Conv2d(1,32,3,stride=2,padding=1,bias=false)`. The nine-stage feature trunk
ends with a 320-to-1280 1x1 Conv/BatchNorm/SiLU block; the classifier is
Dropout(0.2) then Linear(1280,4). The instantiated parameter count is
4,012,096. Source: [`model.py`](../../../4_2d_classification/src/classification/model.py#L28).
Class order is `glioma`, `meningioma`, `no_tumor`, `pituitary` per
[`brisc.py`](../../../1_data_and_preprocessing/src/data/brisc.py#L14).

The existing config selects a 160x160 grayscale input, seed 2026, batch 32,
up to 30 epochs, AdamW (lr 0.001, weight decay 0.0001), cosine scheduler,
patience 7, and no pretrained weights. Reuse the hash-aware classification
protocol, but save an independent run and test evaluation. No test metrics,
confusion matrix, ROC-AUC, or PR-AUC are available.

Grad-CAM target: `model.network.features[8]`, observed `[B,1280,5,5]` for
160x160. Score a raw class logit and overlay on the exact model grayscale
input. This is an architecture forward-hook check, not a trained-checkpoint
CAM validation. No Grad-CAM implementation exists in the repository.

Configuration: [`classification_efficientnet_b0.yaml`](../../../4_2d_classification/configs/classification_efficientnet_b0.yaml).
