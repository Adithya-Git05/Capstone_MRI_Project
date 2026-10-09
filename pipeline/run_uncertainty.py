import os
import json
import numpy as np
import nibabel as nib


def run_uncertainty(config_path='config.json'):
    """Generate a simple uncertainty map based on the base risk map.

    This placeholder implementation defines uncertainty as the complement of the
    risk map (i.e., `uncertainty = 1 - risk`). More sophisticated approaches could
    incorporate model variance, Monte-Carlo dropout, or atlas-based confidence.

    Expected config entries:
        paths['risk_map_nii'] – path to the base risk map NIfTI.
        paths['output_dir'] – directory where outputs are stored.
    """
    with open(config_path) as f:
        config = json.load(f)

    risk_path = config['paths'].get('risk_map_nii')
    out_dir = config['paths'].get('output_dir')
    out_path = os.path.join(out_dir, 'uncertainty_map.nii.gz')
    os.makedirs(out_dir, exist_ok=True)

    if not risk_path or not os.path.exists(risk_path):
        print('[SKIP] Base risk map missing; cannot compute uncertainty.')
        return None

    risk_nii = nib.load(risk_path)
    risk_data = risk_nii.get_fdata()

    # Simple complement uncertainty
    uncertainty_data = 1.0 - np.clip(risk_data, 0.0, 1.0)
    uncertainty_nii = nib.Nifti1Image(uncertainty_data.astype(np.float32), affine=risk_nii.affine, header=risk_nii.header)
    nib.save(uncertainty_nii, out_path)
    print(f'[UNCERTAINTY] Uncertainty map saved to {out_path}')
    return out_path
