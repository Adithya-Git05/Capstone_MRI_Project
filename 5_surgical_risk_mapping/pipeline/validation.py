import os
import nibabel as nib
import numpy as np

def validate_zero_overlap(risk_path: str, label_path: str) -> bool:
    """Check that there are no voxels where risk > 0 but label == 0 (zero-mm anatomical overlap)."""
    if not os.path.exists(risk_path):
        raise FileNotFoundError(f"Risk map not found: {risk_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label map not found: {label_path}")
    risk_data = nib.load(risk_path).get_fdata()
    label_data = nib.load(label_path).get_fdata()
    overlap = np.logical_and(risk_data > 0, label_data == 0)
    if np.any(overlap):
        print("[FAIL] Zero-mm anatomical overlap detected.")
        return False
    print("[PASS] No zero-mm anatomical overlap.")
    return True

def validate_visualization(png_path: str) -> bool:
    """Check that the risk-map visualization PNG exists and is non-empty."""
    if not os.path.exists(png_path):
        print(f"[FAIL] Visualization file missing: {png_path}")
        return False
    if os.path.getsize(png_path) == 0:
        print(f"[FAIL] Visualization file is empty: {png_path}")
        return False
    print("[PASS] Visualization file present and non-empty.")
    return True
