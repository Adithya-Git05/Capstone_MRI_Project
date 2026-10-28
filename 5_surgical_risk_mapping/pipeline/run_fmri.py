import os
import json
import numpy as np
import nibabel as nib


def run_fmri(config_path='configs/config.json'):
    """Process fMRI activation data and compute risk within activated voxels.

    Expected config entries:
        paths['fmri_activation'] – path to fMRI activation NIfTI file.
        paths['risk_map_nii'] – path to the base risk map NIfTI.
        paths['output_dir'] – directory where outputs are stored.

    The function writes a NIfTI file ``fmri_risk.nii.gz`` containing the risk values
    limited to the activation mask, and prints the mean risk within the activation.
    """
    with open(config_path) as f:
        config = json.load(f)

    fmri_path = config['paths'].get('fmri_activation')
    risk_path = config['paths'].get('risk_map_nii')
    out_dir = config['paths'].get('output_dir')
    out_path = os.path.join(out_dir, 'fmri_risk.nii.gz')
    os.makedirs(out_dir, exist_ok=True)

    if not fmri_path or not os.path.exists(fmri_path):
        print('[SKIP] fMRI activation file missing; skipping fMRI analysis.')
        return None
    if not risk_path or not os.path.exists(risk_path):
        print('[SKIP] Base risk map missing; cannot compute fMRI risk.')
        return None

    # Load images
    fmri_nii = nib.load(fmri_path)
    fmri_data = fmri_nii.get_fdata()
    risk_nii = nib.load(risk_path)
    risk_data = risk_nii.get_fdata()

    # Simple threshold: any non-zero activation is considered active
    activation_mask = fmri_data != 0
    if not np.any(activation_mask):
        print('[WARN] No activated voxels found in fMRI image.')
        mean_risk = 0.0
    else:
        mean_risk = float(np.mean(risk_data[activation_mask]))
    print(f'[fMRI] Mean risk within activation voxels: {mean_risk:.4f}')

    # Create risk map limited to activation mask
    fmri_risk = np.zeros_like(risk_data)
    fmri_risk[activation_mask] = risk_data[activation_mask]
    fmri_risk_nii = nib.Nifti1Image(fmri_risk.astype(np.float32), affine=risk_nii.affine, header=risk_nii.header)
    nib.save(fmri_risk_nii, out_path)
    print(f'[fMRI] Activation-restricted risk map saved to {out_path}')
    return out_path
