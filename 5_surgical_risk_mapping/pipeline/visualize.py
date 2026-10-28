import os
import sys
import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def visualize_risk_map(config_path="configs/config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    mri_path = config["paths"]["mri_image"]
    seg_path = config["paths"]["tumor_mask"]
    risk_path = config["paths"]["risk_map_nii"]
    out_png = config["paths"]["visualization_png"]

    print("=" * 60)
    print("STAGE 10: 3D Surgical Risk Map Multi-Planar Visualization")
    print("=" * 60)

    if not os.path.exists(risk_path):
        raise FileNotFoundError(f"Risk map file not found: {risk_path}")

    mri_data = nib.load(mri_path).get_fdata()
    seg_data = nib.load(seg_path).get_fdata()
    risk_data = nib.load(risk_path).get_fdata()

    # Find tumor centroid to display representative slice
    tumor_coords = np.argwhere(seg_data > 0)
    if len(tumor_coords) > 0:
        c_x, c_y, c_z = np.round(np.mean(tumor_coords, axis=0)).astype(int)
    else:
        c_x, c_y, c_z = [s // 2 for s in mri_data.shape]

    print(f"Tumor Centroid Slices: Sagittal={c_x}, Coronal={c_y}, Axial={c_z}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0e1117")
    planes = [
        ("Sagittal (X = %d)" % c_x, np.rot90(mri_data[c_x, :, :]), np.rot90(risk_data[c_x, :, :]), np.rot90(seg_data[c_x, :, :])),
        ("Coronal (Y = %d)" % c_y, np.rot90(mri_data[:, c_y, :]), np.rot90(risk_data[:, c_y, :]), np.rot90(seg_data[:, c_y, :])),
        ("Axial (Z = %d)" % c_z, np.rot90(mri_data[:, :, c_z]), np.rot90(risk_data[:, :, c_z]), np.rot90(seg_data[:, :, c_z]))
    ]

    im_risk = None
    for idx, (title, mri_slice, risk_slice, seg_slice) in enumerate(planes):
        ax = axes[idx]
        ax.set_facecolor("#0e1117")
        ax.set_title(title, color="white", fontsize=14, pad=10, weight="bold")
        ax.imshow(mri_slice, cmap="gray", interpolation="bilinear")
        risk_masked = np.ma.masked_where(risk_slice <= 0.05, risk_slice)
        im_risk = ax.imshow(risk_masked, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0, interpolation="bilinear")
        tumor_mask = (seg_slice > 0).astype(float)
        if np.sum(tumor_mask) > 0:
            ax.contour(tumor_mask, levels=[0.5], colors=["#00FF66"], linewidths=1.8, linestyles="dashed")
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
    cbar = fig.colorbar(im_risk, cax=cbar_ax)
    cbar.set_label("Surgical Risk Score R(x) [0 = Safe, 1 = Critical]", color="white", fontsize=12, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    fig.suptitle(
        f"Patient: {config['patient_id']} | 3D Surgical Risk Map Overlay\n"
        f"[Green Dashed Contour: Tumor Boundary | Heatmap: Combined Anatomical Risk R(x)]",
        color="white", fontsize=15, weight="bold", y=0.98
    )

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved visualization to: {out_png}")
    print("[PASS] Multi-planar visualization generated successfully.")

if __name__ == "__main__":
    visualize_risk_map()
