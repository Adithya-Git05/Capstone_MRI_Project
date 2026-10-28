# Datasets and validation

## BRISC 2025

**Path:** `1_data_and_preprocessing/data/BRISC/brisc2025/` (upstream name `brisc2025`).

| Task / split | Layout | Count observed |
| --- | --- | ---: |
| Classification train | `classification_task/train/<class>/*.jpg` | 5,000 |
| Classification test | `classification_task/test/<class>/*.jpg` | 1,000 |
| Segmentation train | `segmentation_task/train/{images,masks}/` | 3,933 pairs |
| Segmentation test | `segmentation_task/test/{images,masks}/` | 860 pairs |

Classification labels come from the folder names `glioma`, `meningioma`,
`no_tumor`, and `pituitary`. Observed train/test counts by class:

| Class | Train | Test |
| --- | ---: | ---: |
| glioma | 1,147 | 254 |
| meningioma | 1,329 | 306 |
| no_tumor | 1,067 | 140 |
| pituitary | 1,457 | 300 |

Segmentation images are JPG; masks are grayscale PNG with identical stems and
matching dimensions for all 4,793 checked pairs. Dimensions are variable:
most are 512 × 512, but some are as small as 174 × 230 and as large as
1,365 × 1,365. Raw masks include values near black (`0..10`) and near white
(`246..255`) due to antialiased edges; they are not strictly binary files.
The training loader thresholds masks at 128 and resizes masks with nearest
neighbor interpolation while images use bilinear interpolation.

The official test folder is kept separate from training and validation. The
training scripts select a deterministic validation subset only from official
train data. Manifest file names and image indices are not verified patient
identifiers, so patient-level train/test disjointness cannot be established
from these files.

The upstream README says 6,000 segmentation samples, but this workspace has
4,793 pairs; the manifest has 15,586 image/mask/classification-file entries
and agrees with the files counted on disk. No data have been synthesized to
match the upstream prose.

## Local BraTS-format 3D segmentation data

**Path:** `1_data_and_preprocessing/data/BraTS/training_data_additional/`.

The local release/subset provenance is **unverified**. The 271 folders and
their raw label values do not establish that this is a particular BraTS
challenge release. The preserved legacy nnU-Net `dataset.json` declares
`numTraining: 1351`, while only 271 case folders are present locally. Project
documentation records this discrepancy; the generated nnU-Net dataset
metadata records the observed local count and does not modify the legacy
manifest or invent cases.

The inventory contains **271 case directories** and 1,355 NIfTI files. Each
case has `-t1n`, `-t1c`, `-t2w`, `-t2f`, and `-seg` volumes. All were readable;
all checked modalities in each case matched shape and affine. Every file has
shape 182 × 218 × 182, 1 mm spacing, and LAS orientation. The loader reorients
to RAS before model input/prediction and normalizes each image channel over
nonzero voxels.

The YAML input channel order is `[t2f, t1n, t1c, t2w]`, following the channel
names in the preserved nnU-Net `dataset.json` (FLAIR, T1, T1GD, T2). This
corrects the duplicate `t1n`/omitted `t2w` list in the legacy custom MONAI
dataset code; original files are unchanged.

All observed segmentation values belong to `{0,1,2,3,4}`. Per-case observed
label-value-set frequencies are: `(0,1,2,3)` 31 cases; `(0,1,2,3,4)` 110;
`(0,2,3,4)` 79; `(0,2,4)` 32; `(0,2,3)` 12; `(0,2)` 4; `(0,3,4)` 1; `(0,4)`
2. This confirms labels 1 and 4 are absent in some volumes; no class is
invented for those cases.

The active 3D configuration uses four independent, overlapping composite
binary target masks, in fixed model-channel order:

```text
WT = [1, 2, 3]
TC = [1, 3]
ET = [3]
RC = [4]
```

Each mask is derived independently from the original segmentation volume;
WT, TC, and ET consequently overlap. These are derived unions of raw labels,
not mutually exclusive classes. They follow the provisional raw-label
interpretation approved for this project, but must not be described as
verified official ontology for the local data until release provenance and
the authoritative label specification are established. Historical `L1`–`L4`
checkpoints and outputs used different target sets and are not compatible
with the new channel semantics, even though their output tensor shape is the
same.

The folder names share a prefix before the final hyphen (118 unique prefixes
among 271 folders; 75 prefixes have multiple folders). Splits keep these
prefix groups together to reduce likely repeated-acquisition leakage. This
grouping is inferred from filenames and is not cross-checked against a
patient registry; patient-level independence is therefore not verified.
There is no documented official local validation/test split; the MONAI
workflow makes a deterministic grouped validation partition and reports it as
such, not as test performance. No official test split is included in this
local folder.

The nnU-Net adapter trains with region-based targets, which can encode these
overlapping masks. nnU-Net's ordinary integer NIfTI segmentation export
serializes overlapping channels into one label per voxel and loses overlap
information. Adapter evaluation therefore uses the saved region probability
archive (`--save_probabilities`), thresholds its four channels independently,
and never decodes the serialized labelmap as if it were a set of exclusive
classes.

## Validation command and outputs

```powershell
python 1_data_and_preprocessing\scripts\validate_data.py --all
python 1_data_and_preprocessing\scripts\validate_data.py --brisc-only
python 1_data_and_preprocessing\scripts\validate_data.py --brats-only
```

Reports are saved as `1_data_and_preprocessing/outputs/data_validation/brisc.json` and `brats.json`.
The full check decodes all BRISC images/masks, checks pairing/dimensions and
manifest/checksum metadata, and reads all BraTS NIfTI headers and label arrays.
Warnings such as the BraTS 271-versus-1,351 count do not cause data to be
silently changed. Missing, unreadable, mispaired, shape-inconsistent, or
affine-inconsistent files are errors.
