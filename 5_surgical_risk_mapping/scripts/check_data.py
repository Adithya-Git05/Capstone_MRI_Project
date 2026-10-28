import os
import json
import nibabel as nib
import numpy as np

def check_patient_data(config_path="configs/config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    mri_path = config["paths"]["mri_image"]
    seg_path = config["paths"]["tumor_mask"]

    print("=" * 60)
    print("STAGE 0 & 1: Patient Data Setup & MRI Preprocessing Check")
    print("=" * 60)
    print(f"Patient ID: {config['patient_id']}")
    print(f"MRI Path:   {mri_path}")
    print(f"Tumor Mask: {seg_path}")

    if not os.path.exists(mri_path):
        raise FileNotFoundError(f"MRI file not found: {mri_path}")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Tumor mask file not found: {seg_path}")

    # Load NIfTI files
    mri_nii = nib.load(mri_path)
    seg_nii = nib.load(seg_path)

    mri_data = mri_nii.get_fdata()
    seg_data = seg_nii.get_fdata()

    print("\n[1] Dimensionality & Voxel Spacing:")
    print(f"  MRI Shape:          {mri_data.shape}")
    print(f"  Tumor Mask Shape:   {seg_data.shape}")
    mri_zooms = tuple(float(z) for z in mri_nii.header.get_zooms()[:3])
    seg_zooms = tuple(float(z) for z in seg_nii.header.get_zooms()[:3])
    print(f"  MRI Voxel Spacing:  {mri_zooms} mm")
    print(f"  Seg Voxel Spacing:  {seg_zooms} mm")

    shapes_match = mri_data.shape == seg_data.shape
    zooms_match = np.allclose(mri_zooms, seg_zooms)
    print(f"  Shapes Match:       {'PASS' if shapes_match else 'FAIL'}")
    print(f"  Voxel Spacing Match:{'PASS' if zooms_match else 'FAIL'}")

    print("\n[2] Affine & Coordinate Orientation:")
    affine_match = np.allclose(mri_nii.affine, seg_nii.affine, atol=1e-3)
    print(f"  Affine Matrix Match:{'PASS' if affine_match else 'FAIL'}")
    mri_orient = nib.aff2axcodes(mri_nii.affine)
    seg_orient = nib.aff2axcodes(seg_nii.affine)
    print(f"  MRI Orientation:    {mri_orient}")
    print(f"  Seg Orientation:    {seg_orient}")

    print("\n[3] Signal & Mask Integrity:")
    print(f"  MRI Value Range:    [{mri_data.min():.2f}, {mri_data.max():.2f}] (dtype: {mri_data.dtype})")
    unique_labels = np.unique(seg_data)
    print(f"  Tumor Mask Labels:  {unique_labels.tolist()}")
    tumor_voxels = int(np.sum(seg_data > 0))
    voxel_vol_mm3 = np.prod(mri_zooms)
    tumor_vol_mm3 = tumor_voxels * voxel_vol_mm3
    print(f"  Tumor Volume:       {tumor_vol_mm3:.2f} mm^3 ({tumor_voxels} voxels)")

    if not shapes_match or not zooms_match or not affine_match:
        print("\n[FAIL] CHECKPOINT FAILED: MRI and tumor mask are not in the same spatial grid!")
        return False
    elif tumor_voxels == 0:
        print("\n[FAIL] CHECKPOINT FAILED: Tumor mask is empty!")
        return False
    else:
        print("\n[PASS] CHECKPOINT PASSED: MRI and tumor mask belong to the same space and are valid for Stage 2.")
        return True

if __name__ == "__main__":
    check_patient_data()
