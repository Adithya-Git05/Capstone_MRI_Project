# EfficientNetB0 encoder U-Net

**Status: architecture/config implemented, but no matching training run,
checkpoint, or metric artifact was found.** The implementation is not an
evaluated model.

The grayscale EfficientNetB0 encoder is configured with `weights=None` and a
one-channel stem. The decoder uses `features[8]` as the center and skips from
`features[7]`, `[5]`, `[3]`, `[2]`, and `[1]` through `dec4` to `dec0`; the
head produces one binary foreground logit. Source:
[`model.py`](../../../3_2d_segmentation/src/segmentation2d/model.py#L115); config:
[`segmentation2d_efficientnet_b0.yaml`](../../../3_2d_segmentation/configs/segmentation2d_efficientnet_b0.yaml).

The baseline config specifies 256x256, batch 8, seed 2026, up to 40 epochs,
AdamW lr 0.0003/weight decay 0.0001, validation fraction 0.15, patience 8,
and no pretrained weights. Generic engine behavior is equal BCE and
soft-Dice, sigmoid threshold 0.5, and best validation Dice. No run result is
available. BRISC lacks verified patient IDs; use a documented hash-grouped
split and state that patient independence cannot be assured.

Recommended Grad-CAM target is `model.features[8]`, expected
`[B,1280,8,8]` for 256 input. It is source-derived, not locally forward
observed because torchvision was unavailable. Select a fixed ROI/pixel and
differentiate its foreground logit. Do not reuse Attention U-Net test metrics;
no Grad-CAM implementation/test exists.
