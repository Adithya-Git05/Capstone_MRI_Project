"""
v6_baseline.py
==============
Preserved V6 Baseline Surgical Risk Model.
Formula:
    R_base(x) = 0.5 * G(x) + 0.5 * S(x)
where:
    G(x) = exp(-d(x) / 5.0)  [Tumor proximity risk]
    S(x) = anatomical structure importance score

Saves outputs to: outputs/V6_baseline/
"""
import os
import json
import csv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from label_mapping import get_label_info

def run_v6_baseline(config_path="config_v7.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    mri_path = cfg["paths"]["mri_image"]
    seg_path = cfg["paths"]["tumor_mask"]
    labels_path = cfg["paths"]["synthseg_labels"]
    out_dir = cfg["paths"].get("output_dir_v6", "outputs/V6_baseline")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("  RUNNING V6 BASELINE (Unchanged 2-Component Model)")
    print("  Formula: R_base(x) = 0.5 * G(x) + 0.5 * S(x)")
    print("=" * 65)

    mri_nii = nib.load(mri_path)
    seg_nii = nib.load(seg_path)
    labels_nii = nib.load(labels_path)

    mri_data = mri_nii.get_fdata()
    seg_data = seg_nii.get_fdata()
    labels_data = labels_nii.get_fdata()
    voxel_spacing = tuple(float(z) for z in mri_nii.header.get_zooms()[:3])

    # Alignment check
    if seg_data.shape != labels_data.shape:
        from scipy.ndimage import zoom
        zoom_factors = [s / l for s, l in zip(seg_data.shape, labels_data.shape)]
        labels_data = zoom(labels_data, zoom_factors, order=0)

    tumor_binary = (seg_data > 0).astype(np.uint8)
    distance_map_mm = distance_transform_edt(tumor_binary == 0, sampling=voxel_spacing)

    # G(x) = exp(-d / 5.0)
    G_x = np.exp(-distance_map_mm / 5.0).astype(np.float32)
    G_x[tumor_binary > 0] = 1.0

    # S(x)
    labels_data = np.round(labels_data).astype(np.int32)
    S_x = np.zeros_like(labels_data, dtype=np.float32)
    unique_labels = np.unique(labels_data)
    structure_records = []

    for lbl in unique_labels:
        lbl = int(lbl)
        name, tier, weight = get_label_info(lbl)
        mask = (labels_data == lbl)
        S_x[mask] = weight
        if lbl == 0 or name == "Background":
            continue
        lbl_voxels = int(np.sum(mask))
        lbl_vol_mm3 = lbl_voxels * np.prod(voxel_spacing)
        d_struct = distance_map_mm[mask]
        min_d = float(np.min(d_struct)) if lbl_voxels > 0 else 0.0
        mean_d = float(np.mean(d_struct)) if lbl_voxels > 0 else 0.0
        structure_records.append({
            "Label_ID": lbl,
            "Structure_Name": name,
            "Importance_Tier": tier,
            "Importance_Score": weight,
            "Volume_mm3": round(lbl_vol_mm3, 1),
            "Min_Distance_mm": round(min_d, 2),
            "Mean_Distance_mm": round(mean_d, 2)
        })

    # R_base(x)
    R_base = (0.5 * G_x) + (0.5 * S_x)
    brain_mask = (labels_data > 0) | (tumor_binary > 0)
    R_base[~brain_mask] = 0.0
    R_base = np.clip(R_base, 0.0, 1.0)

    for rec in structure_records:
        mask = (labels_data == rec["Label_ID"])
        rec["Mean_Risk"] = round(float(np.mean(R_base[mask])), 4) if np.sum(mask) > 0 else 0.0

    # Save NIfTI
    out_nii = os.path.join(out_dir, "baseline_risk_map.nii.gz")
    nib.save(nib.Nifti1Image(R_base.astype(np.float32), mri_nii.affine, mri_nii.header), out_nii)

    # Save CSV
    out_csv = os.path.join(out_dir, "structure_table.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Label_ID", "Structure_Name", "Importance_Tier", "Importance_Score", "Volume_mm3", "Min_Distance_mm", "Mean_Distance_mm", "Mean_Risk"])
        writer.writeheader()
        writer.writerows(structure_records)

    # Save Slice Visualization
    tumor_coords = np.argwhere(tumor_binary > 0)
    cx, cy, cz = np.round(np.mean(tumor_coords, axis=0)).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0e1117")
    planes = [
        ("Sagittal (X = %d)" % cx, np.rot90(mri_data[cx, :, :]), np.rot90(R_base[cx, :, :]), np.rot90(tumor_binary[cx, :, :])),
        ("Coronal (Y = %d)" % cy, np.rot90(mri_data[:, cy, :]), np.rot90(R_base[:, cy, :]), np.rot90(tumor_binary[:, cy, :])),
        ("Axial (Z = %d)" % cz, np.rot90(mri_data[:, :, cz]), np.rot90(R_base[:, :, cz]), np.rot90(tumor_binary[:, :, cz]))
    ]
    im = None
    for idx, (title, mri_s, risk_s, seg_s) in enumerate(planes):
        ax = axes[idx]
        ax.set_facecolor("#0e1117")
        ax.set_title(title, color="white", fontsize=13, weight="bold")
        ax.imshow(mri_s, cmap="gray")
        risk_m = np.ma.masked_where(risk_s <= 0.05, risk_s)
        im = ax.imshow(risk_m, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
        ax.contour(seg_s, levels=[0.5], colors=["#00FF66"], linewidths=1.8, linestyles="dashed")
        ax.axis("off")
    cbar_ax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("V6 Baseline Risk Score R_base(x)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    fig.suptitle(f"Patient: {cfg['patient_id']} | V6 Baseline Surgical Risk Map", color="white", fontsize=14, weight="bold")
    out_png = os.path.join(out_dir, "risk_map_slices.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"  [V6 BASELINE DONE] Outputs saved to {out_dir}")
    return R_base, out_nii
