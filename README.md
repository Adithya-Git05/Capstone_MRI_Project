# Brain Tumour MRI Capstone

All maintained project code and local project inputs are inside this repository.
The numbered folders contain each stage's implementation, scripts, configs,
and tests. `src/` and `pipeline/` at the repository root are import-compatibility
shims only.

## Stages

| Stage | Implementation | Run guide |
|---|---|---|
| 1. Data and preprocessing | `1_data_and_preprocessing/src/`, `scripts/`, `data/` | [Stage 1](1_data_and_preprocessing/README.md) |
| 2. 3D segmentation | `2_3d_segmentation/src/`, `scripts/`, `configs/` | [Stage 2](2_3d_segmentation/README.md) |
| 3. 2D segmentation | `3_2d_segmentation/src/`, `scripts/`, `configs/` | [Stage 3](3_2d_segmentation/README.md) |
| 4. 2D classification | `4_2d_classification/src/`, `scripts/`, `configs/` | [Stage 4](4_2d_classification/README.md) |
| 5. Surgical-risk mapping | `5_surgical_risk_mapping/pipeline/`, `scripts/`, `configs/`, `data/` | [Stage 5](5_surgical_risk_mapping/README.md) |
| 6. Literature RAG assistant | `6_rag_treatment_planning/api/`, `src/`, `scripts/`, `frontend/` | [Stage 6](6_rag_treatment_planning/README.md) |
| 7. Grad-CAM | `7_gradcam/gradcam.py` | [Stage 7](7_gradcam/README.md) |

`integrated_pipeline/` contains the optional Streamlit app that connects the
stages. Research methods and data caveats are in [`docs/`](docs/) and the
[`technical dossier`](Capstone_Technical_Dossier/README_FIRST.md).

## Run the Streamlit application

Install the integration and 3D inference dependencies with
`python -m pip install -e ".[brats,integrated]"`, then launch from the
repository root with `python -m streamlit run app.py` using that same Python
environment. The 3D SwinUNETR path uses CPU sliding-window inference with a
64-cube ROI and batch size 1; four aligned MRI modalities and adequate free
RAM are required. The selected checkpoint contract, runtime adjustments, and
validation limits are documented in
[`integrated_pipeline/README.md`](integrated_pipeline/README.md).

## Main environment

Use Python 3.10 or newer. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[brats,integrated]"
```

The RAG stage has separate dependencies and a local model service; follow its
stage guide. A CUDA GPU is recommended for 3D training.

## Run the model stages

Put BRISC data in `1_data_and_preprocessing/data/BRISC/brisc2025/` and BraTS
data in `1_data_and_preprocessing/data/BraTS/training_data_additional/`, or set
`BRISC_ROOT` and `BRATS_ROOT`. Validate both before training:

```powershell
python 1_data_and_preprocessing/scripts/validate_data.py --all
```

Training and evaluation commands are in stages 2–4. The 3D trainer supports
SegResNet, SwinUNETR, a residual 3D encoder/decoder, and an EfficientNet-style
MBConv 3D encoder/decoder. Both 2D tasks include custom, ResNet50, and
EfficientNet-B0 configurations. The 3D baseline models start from random
weights; no new comparison results are claimed until the experiments have
been run and reviewed.

## Run the risk and RAG stages

The risk stage requires registered MRI, tumour mask, and SynthSeg labels. From
the repository root, run `python -m pipeline.run_v8_pipeline`; see stage 5 for
input requirements and configuration. Its scores are research outputs, not
injury probabilities or a surgical plan.

Run the RAG API from its own directory, as described in stage 6. It needs a
locally built literature index and an available Ollama service. It provides
literature assistance, not autonomous treatment decisions.

## Data and outputs

Datasets, patient scans, downloaded paper text, checkpoints, generated
predictions, and run outputs are local-only and ignored by Git. Each stage
documents its expected data paths. Generated outputs are kept inside their
own stage folders. Compact, reviewed results needed for the report belong in
the technical dossier.
