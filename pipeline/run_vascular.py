import os
import json
import numpy as np
import nibabel as nib


def run_vascular(config_path='config.json'):
    """Process vascular mask and compute mean risk within vessels.

    Expected config entries:
        paths['vessel_mask'] – path to binary vessel mask NIfTI file.
        paths['risk_map_nii'] – path to the base risk map NIfTI.
        paths['output_dir'] – directory where outputs are stored.
    """
    with open(config_path) as f:
        config = json.load(f)

    vessel_path = config['paths'].get('vessel_mask')
    risk_path = config['paths'].get('risk_map_nii')
    out_dir = config['paths'].get('output_dir')
    out_path = os.path.join(out_dir, 'vascular_risk.nii.gz')
    os.makedirs(out_dir, exist_ok=True)

    if not vessel_path or not os.path.exists(vessel_path):
        print('[SKIP] Vessel mask file missing; skipping vascular analysis.')
        return None
    if not risk_path or not os.path.exists(risk_path):
        print('[SKIP] Base risk map missing; cannot compute vascular risk.')
        return None

    vessel_nii = nib.load(vessel_path)
    vessel_data = vessel_nii.get_fdata()
    risk_nii = nib.load(risk_path)
    risk_data = risk_nii.get_fdata()

    vessel_mask = vessel_data != 0
    if not np.any(vessel_mask):
        print('[WARN] No vessel voxels found in mask.')
        mean_risk = 0.0
    else:
        mean_risk = float(np.mean(risk_data[vessel_mask]))
    print(f'[VASC] Mean risk within vessels: {mean_risk:.4f}')

    vascular_risk = np.zeros_like(risk_data)
    vascular_risk[vessel_mask] = risk_data[vessel_mask]
    vascular_nii = nib.Nifti1Image(vascular_risk.astype(np.float32), affine=risk_nii.affine, header=risk_nii.header)
    nib.save(vascular_nii, out_path)
    print(f'[VASC] Vessel-restricted risk map saved to {out_path}')
    return out_path
