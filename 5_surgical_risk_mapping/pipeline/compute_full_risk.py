import os
import json
import csv
import numpy as np
import nibabel as nib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def compute_combined_risk(config_path='configs/config.json'):
    """Combine base risk with modality-specific risk maps using configured weights.

    The function reads the base risk map generated in Stage 3 and the optional
    modality risk maps produced by run_fmri, run_vascular and run_uncertainty.
    If an optional modality (like fMRI) is absent, its weight is automatically
    and proportionally redistributed across the active modalities so that the
    total weights always sum to 1.0.
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    out_dir = config['paths']['output_dir']
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(project_root, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    combined_path = os.path.join(out_dir, 'combined_risk.nii.gz')

    # Load base risk map
    base_risk_path = config['paths'].get('risk_map_nii')
    if not base_risk_path:
        raise FileNotFoundError('Base risk map path missing from config.')
    if not os.path.isabs(base_risk_path):
        base_risk_path = os.path.join(project_root, base_risk_path)
    if not os.path.exists(base_risk_path):
        raise FileNotFoundError(f'Base risk map not found at {base_risk_path}; cannot compute combined risk.')

    base_nii = nib.load(base_risk_path)
    base_data = base_nii.get_fdata()

    # Helper to load optional modality maps
    def load_optional(path_key):
        p = config['paths'].get(path_key)
        if not p:
            return None
        if not os.path.isabs(p):
            p = os.path.join(project_root, p)
        if os.path.exists(p):
            return nib.load(p).get_fdata()
        return None

    fmri_data = load_optional('fmri_risk')
    vascular_data = load_optional('vascular_risk')
    uncertainty_data = load_optional('uncertainty_map')

    # DTI tract overlap CSV check
    dti_csv = os.path.join(out_dir, 'dti_tract_overlap.csv')
    mean_dti_risk = None
    if os.path.exists(dti_csv):
        try:
            risks = []
            with open(dti_csv, newline='', encoding='utf-8') as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    risks.append(float(row['mean_risk']))
            if risks:
                mean_dti_risk = float(np.mean(risks))
        except Exception as e:
            print(f'[DTI] Warning: could not parse DTI overlap CSV ({e})')

    # Retrieve configured modality weights
    weights = config.get('modalities', {})
    cfg_fmri = weights.get('weight_fmri', 0.0)
    cfg_vasc = weights.get('weight_vascular', 0.0)
    cfg_unc = weights.get('weight_uncertainty', 0.0)
    cfg_dti = weights.get('weight_dti', 0.0)

    # Base weight takes the remaining fraction from configuration
    configured_base = max(0.1, 1.0 - (cfg_fmri + cfg_vasc + cfg_unc + cfg_dti))

    # Identify active modalities
    raw_active = {'base': configured_base}
    if fmri_data is not None and cfg_fmri > 0:
        raw_active['fmri'] = cfg_fmri
    if vascular_data is not None and cfg_vasc > 0:
        raw_active['vasc'] = cfg_vasc
    if uncertainty_data is not None and cfg_unc > 0:
        raw_active['unc'] = cfg_unc
    if mean_dti_risk is not None and cfg_dti > 0:
        raw_active['dti'] = cfg_dti

    # Dynamically renormalize active weights so they sum to 1.0
    total_raw = sum(raw_active.values())
    norm_w = {k: v / total_raw for k, v in raw_active.items()}

    weight_summary = ", ".join(f"{k}: {v*100:.1f}%" for k, v in norm_w.items())
    print(f"[COMBINED] Dynamic active modality weights -> {weight_summary}")

    # Compute weighted sum
    combined = norm_w['base'] * base_data
    if 'fmri' in norm_w:
        combined += norm_w['fmri'] * fmri_data
    if 'vasc' in norm_w:
        combined += norm_w['vasc'] * vascular_data
    if 'unc' in norm_w:
        combined += norm_w['unc'] * uncertainty_data
    if 'dti' in norm_w:
        combined += norm_w['dti'] * mean_dti_risk

    combined = np.clip(combined, 0.0, 1.0)
    combined_nii = nib.Nifti1Image(combined.astype(np.float32), affine=base_nii.affine, header=base_nii.header)
    nib.save(combined_nii, combined_path)
    print(f'[COMBINED] Combined risk map saved to {combined_path}')
    return combined_path
