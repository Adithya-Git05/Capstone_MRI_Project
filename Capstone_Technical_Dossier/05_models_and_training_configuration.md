# 05 — Models and training configuration

## Current BRISC classifier

`src/classification/model.py:BRISCClassifier` is a custom four-stage 2D CNN, not ResNet50 or EfficientNetB0: 1-channel input; Conv2d / GroupNorm / SiLU / 2x max-pool stages with widths 32, 64, 128, 256; adaptive average pool; dropout 0.25; linear four-class output. Checked-in `configs/classification.yaml` specifies 160x160, batch 32, up to 30 epochs, AdamW lr=0.001 and weight decay 0.0001, cross-entropy, cosine annealing to 1e-6, and early stopping patience 7. Checkpoint selection is minimum validation cross-entropy. Run metadata has a source config path but does not serialize the exact model config as a named architecture. No ResNet50/EfficientNetB0 implementation or result was located.

## BRISC 2D segmentation

`BRISCUNet` is a 1-input/1-logit-output encoder-decoder with four convolutional channel widths (base 16 through 128), GroupNorm/SiLU double-convolution blocks, max pooling and skip connections. `BRISCAttentionUNet` adds attention gates over three skip levels and the boundary loss described below. It is not interchangeable with the plain U-Net.

The recorded imported attention-boundary run is `BRISCAttentionUNet`, seed 2026, CUDA, trained 40 epochs with best epoch 39. The config path in its metadata points to the 2D config; model input size 256, mask threshold 128, batch size 8, AdamW lr=0.0003, weight decay 0.0001, validation fraction 0.15 and patience 8. Loss is `(1-boundary_weight)*(0.5*BCE + 0.5*soft-Dice) + boundary_weight*boundary_L1`, with default boundary weight 0.10. Image/mask flips use p=0.5 each for horizontal and vertical axes. Checkpoint selection is highest mean per-image validation Dice. Patient-level grouping is unavailable.

The plain `BRISCUNet` training/evaluation route is separately implemented with the same 256/batch8/AdamW defaults; its loss is 0.5 BCE + 0.5 soft-Dice. Do not attribute the attention-run metrics to the plain model.

## 3D MONAI models

`src/segmentation3d/models.py:build_monai_model` constructs 3D MONAI SegResNet or SwinUNETR. Each expects 4 input channels and emits 4 independent region logits. SegResNet `init_filters=16`; SwinUNETR `feature_size=48`, `img_size=96^3`, gradient checkpointing enabled. The checked-in config specifies input order `[t2f,t1n,t1c,t2w]`, RAS, random 96^3 crop, batch 1, AdamW lr=1e-4/weight decay=1e-5, BCE+soft-Dice, max 100 epochs, early-stop patience 30, validation fraction .2 and seed 2026. Training augments by independent random flips of the three spatial axes (p=.5 each); no rotation is implemented in the loader.

The actual current SegResNet run config overrides the default max epochs and patience to 1; metadata records one completed epoch, best epoch 1, CPU, 215 train and 56 grouped-validation cases. Checkpoint selection is maximum mean per-case Dice across WT/TC/ET/RC. The default config's `folds=5` is not used by the single split path. Current code supports SwinUNETR, but no corresponding run result was found.

Inference in `src/segmentation3d/monai_engine.py:_predict_logits` uses MONAI sliding-window inference with 96^3 ROI, batch 1, overlap 0.25; sigmoid probability threshold 0.5; outputs preserve RAS affine and four binary masks. Checkpoints are rejected if their ontology metadata differs from the current overlapping WT/TC/ET/RC definition.

## Spatial risk and RAG

Risk model parameters and actual/effective weights are in `risk_component_weights.csv`. RAG default retriever/generator settings and interfaces are in `rag_configuration.csv`. These are software configurations, not model-performance claims.
