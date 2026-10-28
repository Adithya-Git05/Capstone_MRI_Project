import os
import json
import numpy as np
import nibabel as nib
import csv

def run_dti(config_path='configs/config.json'):
    """Process DTI data and compute tract-risk overlap.
    Expected config entry (optional):
        paths['dti_fa'] – path to a FA (Fractional Anisotropy) NIfTI file.
        paths['tract_masks'] – directory containing binary tract mask NIfTI files.
    The function writes a CSV with tract name and mean risk within each tract.
    If the required inputs are missing, the step is skipped with a warning.
    """
    with open(config_path) as f:
        config = json.load(f)
    dti_path = config['paths'].get('dti_fa')
    tract_dir = config['paths'].get('tract_masks')
    risk_path = config['paths'].get('risk_map_nii')
    out_csv = os.path.join(config['paths']['output_dir'], 'dti_tract_overlap.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not dti_path or not os.path.exists(dti_path):
        print('[SKIP] DTI FA file not provided or missing; skipping DTI analysis.')
        return
    if not tract_dir or not os.path.isdir(tract_dir):
        print('[SKIP] Tract mask directory not provided or missing; skipping DTI analysis.')
        return
    # Load risk map
    risk_nii = nib.load(risk_path)
    risk_data = risk_nii.get_fdata()
    # Iterate over tract mask files
    rows = []
    for fname in os.listdir(tract_dir):
        if not fname.lower().endswith('.nii') and not fname.lower().endswith('.nii.gz'):
            continue
        tract_name = os.path.splitext(os.path.splitext(fname)[0])[0]  # strip .nii.gz
        tract_path = os.path.join(tract_dir, fname)
        tract_nii = nib.load(tract_path)
        tract_data = tract_nii.get_fdata()
        mask = tract_data > 0
        if np.any(mask):
            mean_risk = float(np.mean(risk_data[mask]))
        else:
            mean_risk = 0.0
        rows.append({'tract': tract_name, 'mean_risk': round(mean_risk, 4)})
    # Write CSV
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['tract', 'mean_risk'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'[DTI] Tract-risk overlap written to {out_csv}')
    return out_csv
