# ResNet50 encoder U-Net

**Status: architecture/config implemented, but no matching training run,
checkpoint, or metric artifact was found.** It is therefore an implemented
but untrained/evaluation-unverified model, not a proposed-only architecture.

The grayscale-compatible ResNet50 uses `weights=None`, replaces the first
convolution for one input channel, and connects encoder features from stem,
layer1, layer2, and layer3 to decoder stages `dec0` through `dec3`; `layer4`
forms the deep center and `head` produces one binary foreground logit.
Architecture source [`model.py`](../../../3_2d_segmentation/src/segmentation2d/model.py#L69) and
config [`segmentation2d_resnet50.yaml`](../../../3_2d_segmentation/configs/segmentation2d_resnet50.yaml).

The provided baseline config specifies 256x256, batch 8, seed 2026, up to 40
epochs, AdamW lr 0.0003/weight decay 0.0001, validation fraction 0.15,
patience 8, and no pretrained weights. The generic engine uses equal BCE and
soft-Dice components, sigmoid threshold 0.5, and best validation Dice.
These are configured/proposed run settings, not measured training outcomes.
BRISC has no verified patient IDs; a hash-grouped sample split is the best
available leakage precaution, but cannot be called patient-independent.

**No model-specific metrics.** Do not reuse the Attention U-Net scores.
Recommended Grad-CAM target is `model.layer4`, expected `[B,2048,8,8]` for
256x256. This shape is derived from source architecture, not locally forward
observed because torchvision was unavailable. The scalar target should be the
mean foreground logit over a fixed ROI or a selected pixel. No CAM test exists.

Recovery: run this architecture with a saved shared split manifest, history,
checkpoint, per-image predictions/metrics, test metrics, confusion-like pixel
counts, environment, and overlay artifacts. Report global and mean-per-image
Dice/IoU separately.
