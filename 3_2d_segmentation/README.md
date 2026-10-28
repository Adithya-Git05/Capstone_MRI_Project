# 3. 2D segmentation

BRISC paired images/masks are handled by this stage's `src/segmentation2d/`. The custom
U-Net and ResNet50/EfficientNet-B0 encoder variants use separate configs.

```powershell
python 3_2d_segmentation/scripts/train_segmentation2d.py --config 3_2d_segmentation/configs/segmentation2d.yaml
python 3_2d_segmentation/scripts/train_segmentation2d.py --config 3_2d_segmentation/configs/segmentation2d_resnet50.yaml
python 3_2d_segmentation/scripts/train_segmentation2d.py --config 3_2d_segmentation/configs/segmentation2d_efficientnet_b0.yaml
python 3_2d_segmentation/scripts/evaluate_segmentation2d.py --checkpoint path/to/best.pt --split test
```

Each training run writes a new timestamped output folder. Follow the BRISC
split and mask-threshold notes in [`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).
