# 08 — Spatial risk methodology

## Core components and equation

The legacy V8 pipeline uses a normalized component-weighted composite over the brain mask:

`R(x) = clip(sum_i(w_i * component_i(x)) / sum_i(w_i for available components), 0, 1)`.

`combine_active_risk_components` includes non-`None` arrays, renormalizes their weights by the available-weight sum, sets outside-brain voxels to zero, and clips to [0,1]. A zero total falls back to divisor 1. This is a relative research score, not a calibrated probability of injury.

- **G(x), geometric proximity:** `exp(-d(x)/sigma_G)` with tumor voxels forced to 1; config `sigma_G_mm=5`.
- **S(x), anatomical importance:** categorical importance from SynthSeg/FreeSurfer ID mapping. `label_mapping.py` maps background 0, general structures (for example cerebral white matter) around .3, important structures .7, and critical ventricles/brainstem 1.0; unknown IDs have fallback mapping. Cortical IDs 1000–2999 map to generic cortical parcels at .7.
- **T(x), DTI tract risk in V8:** tract proximity `P_i=exp(-d_i/sigma_T)`, overlap `O_i=max(tumor_overlap_fraction, tract_overlap_fraction)`, and `T_i=(alpha*P_i+beta*O_i)*tract_importance_i`, with alpha=.7, beta=.3, sigma_T=10mm. A voxel field decays from tract ROI using a configured 2mm buffer and combines by voxelwise maximum. The stored legacy example describes DTI/FA-threshold atlas ROI masks. The current integrated adapter supplies no DTI/tract data; its risk output cannot be described as DTI-backed.
- **C(x), consequence:** assigns configured severity to associated parcellated anatomy and tract ROIs; overlaps combined by maximum. Functional consequence categories include motor, language, visual, interhemispheric, sensory, coordination and brainstem.
- **U(x), uncertainty:** designed for boundary/segmentation uncertainty when a legitimate source exists. Saved case reports empirical uncertainty unavailable (one consensus model/no ensemble); do not claim an uncertainty map was measured.
- **V(x), vascular proximity:** only supported with true vessel imaging. Saved case reports no MRA/CTA; no measured vascular evidence.

Base config weights: G .35, S .15, T .20, C .20, U .10, V 0.00. The saved example reports effective available weights G .388889, S .166667, T .222222, C .222222 after renormalization; those are different from the base weights. In the Streamlit integration DTI/tracts are not provided, so the applicable weights differ from that saved example.

## Spatial geometry and data provenance

V8 uses physical Euclidean distance transforms with voxel-spacing sampling. Strict QC compares MRI/tumor/parcellation shapes, spacing and affine; optional DTI/tract checks are conditional. In the legacy result, saved QC states 1mm spacing, shape 240x240x155, orientation LPS, and matched shape/spacing/affines. The integrated adapter first checks uploaded SynthSeg shape/affine against T1c, reorients both to RAS, checks alignment again, verifies integer/allowed label IDs and nonempty tumor. It refuses to substitute BraTS tumor-region masks for anatomical SynthSeg labels.

V8 label names/importance are configured code mappings, not proof that the uploaded atlas is correctly registered or clinically meaningful. The saved single example has T1/T1CE/T2/FLAIR and DTI-derived tract data; source dataset provenance is not documented in the inspected outputs. The integrated adapter has no DTI FA or tract-mask upload path and explicitly marks those missing.

## Thresholds and outputs

Prototype display intervals are LOW [0,.20), MODERATE [.20,.40), ELEVATED [.40,.60), HIGH [.60,.80), VERY_HIGH [.80,1.00]. Source labels them nonvalidated visualization thresholds. V8 writes component/risk NIfTIs, per-structure/tract tables, evidence status, QC JSON, comparison and ablation outputs, figures and candidate corridors. Risk ablation values in `spatial_case_results.csv` retain three source versions because same-named configurations differ.

## Example case results

Tumor volume 41,919 mm^3; fMRI/vascular imaging and ensemble uncertainty unavailable. Tract overlaps/distance fields are tabulated without case ID. The source warns nonzero overlap may reflect invasion or partial-volume/registration error; it requests registration verification before clinical interpretation. These are measurements from one research example, not clinical risk evidence.
