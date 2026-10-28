# Grad-CAM targets and implementation plan

## Status and target matrix

The original audit found no Grad-CAM implementation or saved attribution
output. The implementation was added later in `7_gradcam/gradcam.py` and the
integrated app currently displays Grad-CAM for compatible custom BRISC
classifier checkpoints. This handoff's remaining module paths are
architecture-inspected
recommendations; the SegResNet and BRISCAttentionUNet paths/shapes were
forward-hook inspected, while ResNet/EfficientNet segmentation shapes are
source-derived because local torchvision is unavailable. SwinUNETR has no
validated target path.

| Model | Recommended module | Spatial activation | Why / caveat |
|---|---|---|---|
| Custom four-stage CNN classifier | `model.features[14]` | `[B,256,20,20]` for 160×160 | Last convolutional stage before adaptive pooling. |
| ResNet50 classifier | `model.network.layer4` | `[B,2048,5,5]` for 160×160 | Final residual feature stage before global pooling/head. |
| EfficientNetB0 classifier | `model.network.features[8]` | `[B,1280,5,5]` for 160×160 | Final spatial feature stage before classifier pooling. |
| Attention-gated U-Net | `model.bottleneck.layers[3]` | `[B,128,32,32]` for 256×256 | Last bottleneck convolution, before GroupNorm/SiLU; a semantic feature target not itself a mask. |
| ResNet50 encoder U-Net | `model.layer4` | Expected `[B,2048,8,8]` at 256×256 | Final encoder stage; source-derived, not forward-observed locally. `model.dec3` is a higher-resolution decoder alternative, not the current recommendation. |
| EfficientNetB0 encoder U-Net | `model.features[8]` | Expected `[B,1280,8,8]` at 256×256 | Final encoder stage; source-derived, not forward-observed locally. `model.dec4` is a higher-resolution decoder alternative. |
| MONAI SegResNet | `model.down_layers[3][4].conv2.conv` | `[B,128,12,12,12]` for 96³ | Deep 3D encoder feature; observed with a synthetic forward. |
| MONAI SwinUNETR | **Not yet identified** | Not verified | No forward-hook layout/shape was inspected. Instantiate the configured model and inspect its module tree/output before recording a path. Never copy the SegResNet target. |

The attention gate's `model.att3` output is a gated skip feature
(`[B,64,64,64]` at 256 input), not the scalar attention-coefficient map. To
visualize the gate coefficient itself, inspect `model.att3.score` (observed
layout `[B,1,64,64]`); that is an attention diagnostic, not Grad-CAM. The
pre-gate bottleneck is recommended for class/foreground attribution because it
represents the encoder context before the skip gate modulates a specific
feature.

## Scalar targets

### Classification

Run one image at a time (or retain batch indices explicitly). For the predicted
class, select `logits[batch_index, predicted_class]`; optionally repeat using
the ground-truth class. Use the raw selected class **logit**, not its softmax
probability, to avoid the softmax coupling/saturation changing the target.
These classifiers use four-way argmax, not a probability threshold. Temperature
scaling is a calibration output and should not be used to change the raw-logit
target.

### 2D binary segmentation

A segmentation model returns one foreground logit per pixel, so select a
scalar from those logits:

- Whole predicted foreground: mean foreground logit over a fixed predicted
  foreground ROI. This explains evidence supporting the selected predicted
  region as a whole; it does not explain an image-level class.
- Reference-region explanation: mean logit over a supplied ground-truth ROI.
  Keep the ROI fixed during the gradient calculation; the reference mask
  chooses the target region but is not differentiated.
- Local explanation: one pixel's foreground logit, or the mean over a small
  user-selected ROI. This is spatially local and can be noisier.

If the predicted mask is empty, do not silently substitute a different target:
ask for a reference ROI or an explicitly selected pixel. The CAM explains the
selected logits and is **not** a segmentation mask; use the model's sigmoid
and 0.5 threshold output as the actual prediction.

### 3D region segmentation

Select one output channel (provisional order WT, TC, ET, RC) and define a scalar
as the mean logit over a supplied 3D ROI, a fixed predicted region, or a
ground-truth region. A single voxel logit is a local alternative. Specify
region, ROI source, coordinates, and empty-mask handling with every saved
result. Because target channels may overlap, one region's explanation is not
exclusive of another.

## Hook and computation recipe

Use the model in evaluation mode to disable training-time behavior, but do not
use `torch.no_grad()` or `torch.inference_mode()`. A forward hook can capture
the chosen tensor; `torch.autograd.grad` then obtains its gradient with respect
to the scalar target. Example for one image:

```python
captured = {}

def save_activation(module, inputs, output):
    captured["activation"] = output

handle = target_module.register_forward_hook(save_activation)
try:
    model.eval()
    logits = model(input_tensor)
    score = select_scalar_target(logits)
    activation = captured["activation"]
    gradients = torch.autograd.grad(score, activation)[0]
finally:
    handle.remove()
```

For 2D activations `A` and gradients `dS/dA` shaped `[B,C,H,W]`, compute
channel weights as
`alpha[k] = mean(dS/dA[k,:,:], spatial axes H,W)`, then
`CAM = ReLU(sum_k(alpha[k] * A[k,:,:]))`.
For 3D tensors `[B,C,D,H,W]`, average gradients over `D,H,W` and combine each
channel with its activation the same way. Keep the requested sample/channel
indices explicit. Interpolate the nonnegative map to the model-input size
(bilinear for 2D, trilinear for 3D), normalize per sample using min/max with a
small epsilon, and retain the unnormalized map for reproducibility.

Prefer `torch.autograd.grad` over deprecated module backward hooks. A forward
hook should store the exact activation tensor from that pass; remove the hook
afterward. Do not retain activations across batches or reuse a hook capture
from an earlier forward. In-place ReLU/SiLU may modify a hooked tensor; if that
causes autograd/version-counter errors, hook the preceding convolution output
or use a non-in-place activation in a controlled attribution-only copy of the
model, then verify its logits match the original. Do not silently alter the
model's trained inference behavior.

## Overlay and geometry

- **Classification:** overlay the resized 20×20 or 5×5 map on the exact
  grayscale 160×160 tensor image used for inference, not a separately loaded
  or differently cropped image. Undo the model normalization for display
  (`[-1,1]` to `[0,1]`) while preserving the same pad/crop/resize frame.
- **2D segmentation:** overlay on the exact 256×256 grayscale model input.
  The input is normalized to `[-1,1]`; convert only the display copy. Apply the
  identical grayscale resize and alignment transforms to the CAM and displayed
  slice. Show the predicted mask separately and label it as segmentation.
- **3D:** make a 3D CAM volume and resample it to the input tensor's voxel grid.
  Show axial, coronal, and sagittal slices over the matching RAS-oriented MRI
  channels/slices. Preserve the NIfTI affine. The loader reorients data to RAS
  and writes outputs with updated affine; it does not explicitly restore
  original LAS voxel-array order. If displaying in the original array
  orientation, apply the corresponding inverse orientation transform to the
  CAM and use the original affine.

## Memory and volume interpretation

A 3D activation/gradient volume is much larger than an individual 2D map.
SegResNet's candidate activation is 128×12×12×12 per sample; SwinUNETR memory
must be measured after target-layer inspection. Full-volume differentiable
sliding-window attribution may retain a large graph across windows. A practical
alternative is patch-wise attribution over a declared 96³ ROI, stitching
attributions with the same window positions and weights as inference; label
that result **patch-wise** and do not claim it equals a full-volume gradient.
Attribution computed independently per 2D slice is not 3D volumetric
Grad-CAM.

## Source evidence

- Classifier architecture and hooks/shapes: `4_2d_classification/src/classification/model.py:7-77`;
  input pipeline `1_data_and_preprocessing/src/data/brisc.py:208-227`.
- Attention architecture/target: `3_2d_segmentation/src/segmentation2d/model_attention.py:10-73`;
  common convolution blocks `3_2d_segmentation/src/segmentation2d/model.py:8-24`.
- Encoder-decoder variants: `3_2d_segmentation/src/segmentation2d/model.py:69-158`.
- 3D model source: `2_3d_segmentation/src/segmentation3d/models.py:8-34`; loader/output geometry:
  `1_data_and_preprocessing/src/data/brats.py:174-223`; inference sliding windows:
  `2_3d_segmentation/src/segmentation3d/monai_engine.py:163-175`.
