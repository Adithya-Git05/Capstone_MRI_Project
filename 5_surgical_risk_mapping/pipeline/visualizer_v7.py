"""
visualizer_v7.py
================
Generates publication-quality visualizations for the V7 Enhanced Surgical Risk Model:
  1. enhanced_risk_slices.png: Multi-planar view with fixed [0, 1] scale and scientific labels
  2. risk_map_comparison.png: Side-by-side comparison (V6 Baseline vs V7 Enhanced vs Difference)
  3. component_maps.png: Individual component maps (G, S, T, C)
"""
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def generate_v7_visualizations(mri_data, tumor_binary, R_base, R_enhanced, components, patient_id, out_dir_v7, out_dir_comp):
    os.makedirs(out_dir_v7, exist_ok=True)
    os.makedirs(out_dir_comp, exist_ok=True)

    tumor_coords = np.argwhere(tumor_binary > 0)
    cx, cy, cz = np.round(np.mean(tumor_coords, axis=0)).astype(int)

    # 1. Enhanced Risk Multi-Planar Slice View
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0e1117")
    planes = [
        ("Sagittal (X = %d)" % cx, np.rot90(mri_data[cx, :, :]), np.rot90(R_enhanced[cx, :, :]), np.rot90(tumor_binary[cx, :, :])),
        ("Coronal (Y = %d)" % cy, np.rot90(mri_data[:, cy, :]), np.rot90(R_enhanced[:, cy, :]), np.rot90(tumor_binary[:, cy, :])),
        ("Axial (Z = %d)" % cz, np.rot90(mri_data[:, :, cz]), np.rot90(R_enhanced[:, :, cz]), np.rot90(tumor_binary[:, :, cz]))
    ]

    im = None
    for idx, (title, mri_s, risk_s, seg_s) in enumerate(planes):
        ax = axes[idx]
        ax.set_facecolor("#0e1117")
        ax.set_title(title, color="white", fontsize=13, weight="bold")
        ax.imshow(mri_s, cmap="gray", interpolation="bilinear")
        risk_m = np.ma.masked_where(risk_s <= 0.05, risk_s)
        im = ax.imshow(risk_m, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0, interpolation="bilinear")
        if np.sum(seg_s) > 0:
            ax.contour(seg_s, levels=[0.5], colors=["#00FF66"], linewidths=1.8, linestyles="dashed")
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Model-derived surgical risk score [0 = Low, 1 = Very High]", color="white", fontsize=11, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    fig.suptitle(
        f"Patient: {patient_id} | V7 Enhanced Multi-Modal Surgical Risk Map\n"
        f"[Green Dashed Contour: Tumor Boundary | Fixed Scale [0.0, 1.0]]",
        color="white", fontsize=15, weight="bold", y=0.98
    )

    out_png_v7 = os.path.join(out_dir_v7, "enhanced_risk_slices.png")
    plt.savefig(out_png_v7, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # 2. Baseline vs Enhanced Comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor="#0e1117")
    diff_map = R_enhanced - R_base
    mri_ax = np.rot90(mri_data[:, :, cz])
    tumor_ax = np.rot90(tumor_binary[:, :, cz])
    r_base_ax = np.rot90(R_base[:, :, cz])
    r_enh_ax = np.rot90(R_enhanced[:, :, cz])
    diff_ax = np.rot90(diff_map[:, :, cz])

    # A: Baseline
    ax0 = axes[0]
    ax0.set_facecolor("#0e1117")
    ax0.set_title("V6 Baseline Risk R_base\n[0.5*G + 0.5*S]", color="white", fontsize=13, weight="bold")
    ax0.imshow(mri_ax, cmap="gray")
    ax0.imshow(np.ma.masked_where(r_base_ax <= 0.05, r_base_ax), cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
    ax0.contour(tumor_ax, levels=[0.5], colors=["#00FF66"], linewidths=1.5, linestyles="dashed")
    ax0.axis("off")

    # B: Enhanced
    ax1 = axes[1]
    ax1.set_facecolor("#0e1117")
    ax1.set_title("V7 Enhanced Risk R_enhanced\n[Active: wG*G + wS*S + wT*T + wC*C]", color="white", fontsize=13, weight="bold")
    ax1.imshow(mri_ax, cmap="gray")
    im_enh = ax1.imshow(np.ma.masked_where(r_enh_ax <= 0.05, r_enh_ax), cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
    ax1.contour(tumor_ax, levels=[0.5], colors=["#00FF66"], linewidths=1.5, linestyles="dashed")
    ax1.axis("off")

    # C: Difference Map (Enhanced - Baseline)
    ax2 = axes[2]
    ax2.set_facecolor("#0e1117")
    ax2.set_title("Difference Map: Delta R\n[Enhanced - Baseline]", color="white", fontsize=13, weight="bold")
    ax2.imshow(mri_ax, cmap="gray")
    im_diff = ax2.imshow(np.ma.masked_where(np.abs(diff_ax) < 0.02, diff_ax), cmap="coolwarm", alpha=0.70, vmin=-0.4, vmax=0.4)
    ax2.contour(tumor_ax, levels=[0.5], colors=["#00FF66"], linewidths=1.5, linestyles="dashed")
    ax2.axis("off")

    cbar_ax2 = fig.add_axes([0.92, 0.20, 0.015, 0.60])
    cbar2 = fig.colorbar(im_diff, cax=cbar_ax2)
    cbar2.set_label("Delta Risk Score (Red: Enhanced > Base, Blue: Base > Enhanced)", color="white", fontsize=10)
    cbar2.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar2.ax.axes, "yticklabels"), color="white")

    fig.suptitle(f"Patient: {patient_id} | V6 Baseline vs V7 Enhanced Comparison (Axial Z = {cz})", color="white", fontsize=15, weight="bold")
    out_png_comp = os.path.join(out_dir_comp, "risk_map_comparison.png")
    plt.savefig(out_png_comp, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # 3. Component Maps Grid (G, S, T, C)
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor="#0e1117")
    comp_titles = [
        ("G(x): Proximity Risk", components.get("G")),
        ("S(x): Anatomy Importance", components.get("S")),
        ("T(x): White-Matter Tract Risk", components.get("T")),
        ("C(x): Functional Consequence", components.get("C"))
    ]

    for idx, (title, comp_arr) in enumerate(comp_titles):
        ax = axes[idx]
        ax.set_facecolor("#0e1117")
        ax.set_title(title, color="white", fontsize=12, weight="bold")
        ax.imshow(mri_ax, cmap="gray")
        if comp_arr is not None:
            c_slice = np.rot90(comp_arr[:, :, cz])
            ax.imshow(np.ma.masked_where(c_slice <= 0.01, c_slice), cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
        else:
            ax.text(0.5, 0.5, "UNAVAILABLE", color="yellow", fontsize=14, ha="center", va="center", transform=ax.transAxes)
        ax.contour(tumor_ax, levels=[0.5], colors=["#00FF66"], linewidths=1.2, linestyles="dashed")
        ax.axis("off")

    fig.suptitle(f"Patient: {patient_id} | V7 Individual Active Component Maps (Axial Z = {cz})", color="white", fontsize=14, weight="bold")
    out_png_comp_maps = os.path.join(out_dir_comp, "component_maps.png")
    plt.savefig(out_png_comp_maps, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"  [VISUALIZER] Generated enhanced slices, baseline comparison, and component maps.")
    return out_png_v7, out_png_comp, out_png_comp_maps
