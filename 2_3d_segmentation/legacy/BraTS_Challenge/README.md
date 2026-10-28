# Historical BraTS Segmentation Artifacts

> Historical README claims below were not verified as provenance for the
> local 271-case data. The local release/subset remains unverified. The old
> `L1`–`L4` mappings are not the project's current WT/TC/ET/RC outputs, and
> existing legacy checkpoints/results must not be reinterpreted using the new
> ontology.

This historical material describes an implementation that was labelled as
BraTS-2024; that claim is not established for the data currently available in
this workspace.

The old dataset manifest defined these generic regions:

```text
L1 = [1, 2, 3, 4]
L2 = [1, 2, 3]
L3 = [1, 3]
L4 = [3]
```

Those historical mappings are not the current project ontology:

```text
WT = [1, 2, 3]
TC = [1, 3]
ET = [3]
RC = [4]
```

Do not rename historical L1-L4 results as WT/TC/ET/RC or compare them
directly with results from the current target semantics without retraining.

The remaining statements in this file are preserved legacy descriptions and
are not independently verified results.
The legacy documentation described CNN-based (nnU-Net, SegResNet) and
Transformer-based (SwinUNETR) models; their implementation and results are
not independently validated by this project.

---

## Models Implemented
- **nnUNet**
  - Self-configuring UNet-based architecture.
  - Supports **2D and 3D convolutions**.
  - Features: deep supervision, ensembling, extensive data augmentation.
  - Hyperparameter details stored in `nnUNet_hyperparameters`.

- **SegResNet**
  - Residual network tailored for medical image segmentation.
  - Efficient in capturing hierarchical spatial features.
  - Implemented in `SegResNet_Train.ipynb`.

- **SwinUNETR**
  - Transformer-based model using shifted windows for global context.
  - Strong performance on volumetric medical imaging.
  - Implemented in `swinunetr_train.py`.

---

## Historical Region Claims

The original README named WT, TC, ET, and RC, but the checked-in legacy
manifest used generic `L1`–`L4` target sets shown above. The available
evidence does not establish that those historical model targets matched the
named regions or that the claimed experiments completed.

---

## Repository Contents
- **Training Notebooks & Scripts**
  - `nnUNet_train.ipynb` – Training pipeline for nnUNet.
  - `SegResNet_Train.ipynb` – Training pipeline for SegResNet.
  - `swinunetr_train.py` – Training pipeline for SwinUNETR.
- **Results & Visualization**
  - `nnUNet_results.ipynb` – Qualitative and quantitative results.
  - `Validation/` – Fold-wise loss/accuracy curves (`progress_fold0.png` → `progress_fold4.png`).
- **Configurations**
  - `nnUNet_hyperparameters/` – Hyperparameter details for nnUNet.
  - `dataset.json`, `dataset_fingerprint.json`, `plans.json` – Dataset and preprocessing configs.

---

## Historical Results & Visualizations
- The legacy files include plots labelled as fold-wise training/validation
  results; their provenance and association with the local data/checkpoints
  have not been verified.
- Legacy visualizations include notebook output and progress plots:
  - `progress_fold0.png` → `progress_fold4.png`.
---

## Objectives & Insights
- The original stated objectives were comparing multiple segmenters and
  using cross-validation. Completion and split independence are unverified.
- The historical stated objective was to compare segmentation models. No
  clinical validation or suitability for diagnosis, treatment planning, or
  surgical risk mapping is established by these artifacts.

---

## About This Project
Historical descriptions of nnUNet, SegResNet, and SwinUNETR experiments.
Dataset provenance, completed training, metric validity, and region semantics
remain unverified for the current local data.

---

## Topics
`Deep Learning` `Medical Imaging` `Brain Tumor Segmentation` `MRI` `nnUNet` `SegResNet` `SwinUNETR` `Computer Vision` `PyTorch`
