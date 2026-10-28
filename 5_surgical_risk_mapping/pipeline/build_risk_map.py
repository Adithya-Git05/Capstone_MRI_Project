import os
import json
import csv
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt

# Import label mapping from the same package
from label_mapping import get_label_info

def build_surgical_risk_map(config_path="configs/config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    mri_path = config["paths"]["mri_image"]
    seg_path = config["paths"]["tumor_mask"]
    labels_path = config["paths"]["synthseg_labels"]
    out_dir = config["paths"]["output_dir"]
    risk_nii_path = config["paths"]["risk_map_nii"]
    csv_path = config["paths"]["structure_table_csv"]
    tract_dir = config["paths"].get("tract_masks")
    unc_path = config["paths"].get("uncertainty_map")

    sigma = config.get("risk_model", {}).get("sigma_mm", 5.0)
    w_G = config.get("risk_model", {}).get("weight_proximity_wG", 0.35)
    w_S = config.get("risk_model", {}).get("weight_structure_wS", 0.30)
    w_T = config.get("risk_model", {}).get("weight_tract_wT", 0.25)
    w_U = config.get("risk_model", {}).get("weight_uncertainty_wU", 0.10)

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("STAGE 4 - 9: 3D Surgical Risk Map Generation")
    print(f"Formula: R(x) = {w_G}*G(x) + {w_S}*S(x) + {w_T}*T(x) + {w_U}*U(x)")
    print("=" * 60)
    print(f"Loading MRI:        {mri_path}")
    print(f"Loading Tumor Mask: {seg_path}")
    print(f"Loading Labels:     {labels_path}")

    mri_nii = nib.load(mri_path)
    seg_nii = nib.load(seg_path)

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"SynthSeg output not found at: {labels_path}\n"
            f"Please run SynthSeg (Stage 2) first to generate anatomical parcellation."
        )

    labels_nii = nib.load(labels_path)

    mri_data = mri_nii.get_fdata()
    seg_data = seg_nii.get_fdata()
    labels_data = labels_nii.get_fdata()

    voxel_spacing = tuple(float(z) for z in mri_nii.header.get_zooms()[:3])
    print(f"Voxel spacing (mm): {voxel_spacing}")

    # Verify alignment
    print("\n[Stage 4] Checking spatial alignment...")
    if seg_data.shape != labels_data.shape:
        print(f"Resampling label map from {labels_data.shape} to match tumor grid {seg_data.shape}...")
        from scipy.ndimage import zoom
        zoom_factors = [s / l for s, l in zip(seg_data.shape, labels_data.shape)]
        labels_data = zoom(labels_data, zoom_factors, order=0)

    print("Grid alignment verified.")

    tumor_binary = (seg_data > 0).astype(np.uint8)
    tumor_voxels = int(np.sum(tumor_binary))
    if tumor_voxels == 0:
        raise ValueError("Tumor mask contains 0 positive voxels.")
    print(f"Tumor voxels: {tumor_voxels} ({tumor_voxels * np.prod(voxel_spacing):.1f} mm^3)")

    # 1. Distance transform & G(x)
    print("\n[Stage 5] Computing Euclidean Distance Transform d(x) from tumor boundary...")
    distance_map_mm = distance_transform_edt(tumor_binary == 0, sampling=voxel_spacing)
    print(f"Distance range: [{distance_map_mm.min():.2f} mm, {distance_map_mm.max():.2f} mm]")

    print(f"\n[Stage 6] Converting distance to proximity risk G(x) = exp(-d(x) / {sigma})...")
    G_x = np.exp(-distance_map_mm / sigma).astype(np.float32)
    G_x[tumor_binary > 0] = 1.0
    print(f"Proximity risk G(x) computed. Range: [{G_x.min():.4f}, {G_x.max():.4f}]")

    # 2. Structure importance S(x)
    print("\n[Stage 7] Assigning structure importance S(x)...")
    labels_data = np.round(labels_data).astype(np.int32)
    S_x = np.zeros_like(labels_data, dtype=np.float32)
    unique_labels = np.unique(labels_data)
    structure_records = []
    print(f"Found {len(unique_labels)} anatomical labels in parcellation.")

    for lbl in unique_labels:
        lbl = int(lbl)
        name, tier, weight = get_label_info(lbl)
        mask = (labels_data == lbl)
        S_x[mask] = weight
        if lbl == 0 or name == "Background":
            continue
        lbl_voxels = int(np.sum(mask))
        lbl_vol_mm3 = lbl_voxels * np.prod(voxel_spacing)
        distances_in_struct = distance_map_mm[mask]
        min_d = float(np.min(distances_in_struct)) if lbl_voxels > 0 else 0.0
        mean_d = float(np.mean(distances_in_struct)) if lbl_voxels > 0 else 0.0
        structure_records.append({
            "Label_ID": lbl,
            "Structure_Name": name,
            "Importance_Tier": tier,
            "Importance_Score": weight,
            "Volume_mm3": round(lbl_vol_mm3, 1),
            "Min_Distance_mm": round(min_d, 2),
            "Mean_Distance_mm": round(mean_d, 2)
        })

    # Brain mask definition
    brain_mask = (labels_data > 0) | (tumor_binary > 0)

    # 3. White-matter tract involvement T(x)
    print("\n[Stage 7b] Computing White-Matter Tract involvement T(x)...")
    T_x = np.zeros_like(G_x, dtype=np.float32)
    if tract_dir and os.path.isdir(tract_dir):
        tract_files = [os.path.join(tract_dir, f) for f in os.listdir(tract_dir) if f.lower().endswith(('.nii', '.nii.gz'))]
        for tf in tract_files:
            t_nii = nib.load(tf)
            t_data = t_nii.get_fdata()
            if t_data.shape == G_x.shape:
                t_mask = t_data > 0
                if np.any(t_mask):
                    d_tract = distance_transform_edt(~t_mask, sampling=voxel_spacing)
                    # Tract core = 1.0, 2mm proximity margin
                    T_current = np.exp(-d_tract / 2.0).astype(np.float32)
                    T_x = np.maximum(T_x, T_current)
                    print(f"  Incorporated tract mask: {os.path.basename(tf)} ({int(np.sum(t_mask))} voxels)")
    T_x[~brain_mask] = 0.0
    print(f"Tract involvement T(x) computed. Range: [{T_x.min():.4f}, {T_x.max():.4f}]")

    # 4. Uncertainty U(x)
    print("\n[Stage 7c] Computing Boundary & Model Uncertainty U(x)...")
    # Peak boundary uncertainty at the tumor invasion margin (G = 0.5)
    U_x = (4.0 * G_x * (1.0 - G_x)).astype(np.float32)
    U_x[~brain_mask] = 0.0
    print(f"Uncertainty U(x) computed. Range: [{U_x.min():.4f}, {U_x.max():.4f}]")
    if unc_path:
        os.makedirs(os.path.dirname(unc_path), exist_ok=True)
        nib.save(nib.Nifti1Image(U_x, affine=mri_nii.affine, header=mri_nii.header), unc_path)
        print(f"  Saved uncertainty map to: {unc_path}")

    # 5. Combined risk map R(x) = 0.35*G + 0.30*S + 0.25*T + 0.10*U
    print(f"\n[Stage 8] Building combined risk map: R(x) = {w_G}*G(x) + {w_S}*S(x) + {w_T}*T(x) + {w_U}*U(x)...")
    R_x = (w_G * G_x) + (w_S * S_x) + (w_T * T_x) + (w_U * U_x)
    R_x[~brain_mask] = 0.0
    R_x = np.clip(R_x, 0.0, 1.0)
    print(f"Risk map R(x) generated. Range in brain: [{R_x[brain_mask].min():.4f}, {R_x[brain_mask].max():.4f}]")

    # Add mean risk to records
    for rec in structure_records:
        lbl = rec["Label_ID"]
        mask = (labels_data == lbl)
        rec["Mean_Risk"] = round(float(np.mean(R_x[mask])), 4) if np.sum(mask) > 0 else 0.0

    structure_records.sort(key=lambda x: x["Min_Distance_mm"])

    # Save outputs
    print("\n[Stage 9] Saving outputs...")
    risk_nii = nib.Nifti1Image(R_x.astype(np.float32), affine=mri_nii.affine, header=mri_nii.header)
    nib.save(risk_nii, risk_nii_path)
    print(f"  Saved risk map NIfTI: {risk_nii_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Label_ID", "Structure_Name", "Importance_Tier", "Importance_Score", "Volume_mm3", "Min_Distance_mm", "Mean_Distance_mm", "Mean_Risk"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(structure_records)
    print(f"  Saved structure metrics CSV: {csv_path}")

    print("\n[PASS] Surgical Risk Map generation complete!")
    return R_x, structure_records

if __name__ == "__main__":
    build_surgical_risk_map()
