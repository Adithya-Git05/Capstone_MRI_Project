# 4. 2D classification

The BRISC classifier and ResNet50/EfficientNet-B0 baselines are implemented
under this stage's `src/classification/` with independent configs.

```powershell
python 4_2d_classification/scripts/train_classification.py --config 4_2d_classification/configs/classification.yaml
python 4_2d_classification/scripts/train_classification.py --config 4_2d_classification/configs/classification_resnet50.yaml
python 4_2d_classification/scripts/train_classification.py --config 4_2d_classification/configs/classification_efficientnet_b0.yaml
python 4_2d_classification/scripts/evaluate_classification.py --checkpoint path/to/best.pt --split test
```

The BRISC labels are image-folder categories, not pathology-confirmed tumour
subtypes. Test-split duplicate caveats and the evaluation protocol are in
[`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).
