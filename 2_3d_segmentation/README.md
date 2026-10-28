# 2. 3D segmentation

BraTS models, training, evaluation, and inference are in this stage's
`src/segmentation3d/` and `scripts/`. Training supports **SegResNet** and
**SwinUNETR**, with 3D **ResNet** and **EfficientNet-style MBConv** baselines.
The baselines are encoder/decoder segmentation models initialized from
scratch; they use the same data, target channels, patch size, loss, and split.

```powershell
python 2_3d_segmentation/scripts/train_segmentation3d.py --model segresnet --config 2_3d_segmentation/configs/segmentation3d.yaml
python 2_3d_segmentation/scripts/train_segmentation3d.py --model swinunetr --config 2_3d_segmentation/configs/segmentation3d.yaml
python 2_3d_segmentation/scripts/train_segmentation3d.py --model resnet3d --config 2_3d_segmentation/configs/segmentation3d_resnet.yaml
python 2_3d_segmentation/scripts/train_segmentation3d.py --model efficientnet3d --config 2_3d_segmentation/configs/segmentation3d_efficientnet.yaml
python 2_3d_segmentation/scripts/evaluate_segmentation3d.py --checkpoint path/to/best.pt --output-dir 2_3d_segmentation/outputs/segmentation3d/evaluation
```

Run each model independently from the repository root and retain its timestamped config, checkpoint,
and evaluation metrics for comparison. The EfficientNet-style baseline uses
3D MBConv blocks with squeeze/excitation; it is not a pretrained or exact
torchvision EfficientNet-B0 port. Label mapping and other protocol caveats are in
[`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).
