"""
ablation_study.py
=================
Executes the 8 required component ablation configurations:
    A. G only
    B. S only
    C. G + S
    D. G + S + T
    E. G + S + C
    F. G + S + T + C
    G. Full model with uncertainty
    H. Full model with vascular information

For each configuration saves:
  - risk map NIfTI
  - mean risk, max risk, tumor-adjacent mean risk
  - summary CSV and JSON
  - multi-panel ablation comparison figure
"""
import os
import json
import csv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def run_ablation_study(G_x, S_x, T_x, C_x, U_sim, V_sim, tumor_binary, brain_mask, distance_tumor_mm, affine, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    peritumoral_mask = brain_mask & (distance_tumor_mm <= 10.0)

    configs = {
        "A (G only)":           {"G": 1.0},
        "B (S only)":           {"S": 1.0},
        "C (G + S)":            {"G": 0.50, "S": 0.50},
        "D (G + S + T)":        {"G": 0.45, "S": 0.25, "T": 0.30},
        "E (G + S + C)":        {"G": 0.45, "S": 0.25, "C": 0.30},
        "F (G + S + T + C)":    {"G": 0.35, "S": 0.15, "T": 0.25, "C": 0.25},
        "G (Full + Uncertainty)": {"G": 0.35, "S": 0.15, "T": 0.20, "C": 0.20, "U": 0.10},
        "H (Full + Vascular)":    {"G": 0.30, "S": 0.15, "T": 0.20, "C": 0.20, "U": 0.10, "V": 0.05}
    }

    comp_dict = {
        "G": G_x, "S": S_x, "T": T_x if T_x is not None else np.zeros_like(G_x),
        "C": C_x if C_x is not None else np.zeros_like(G_x),
        "U": U_sim if U_sim is not None else np.zeros_like(G_x),
        "V": V_sim if V_sim is not None else np.zeros_like(G_x)
    }

    ablation_results = []
    slice_z = 105  # Representative axial slice

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="#0e1117")
    axes = axes.flatten()

    for idx, (cname, w_dict) in enumerate(configs.items()):
        total_w = sum(w_dict.values())
        norm_w = {k: v / total_w for k, v in w_dict.items()}

        R_abl = np.zeros_like(G_x, dtype=np.float32)
        for k, weight in norm_w.items():
            R_abl += weight * comp_dict[k]

        R_abl[~brain_mask] = 0.0
        R_abl = np.clip(R_abl, 0.0, 1.0)

        mean_r = float(np.mean(R_abl[brain_mask])) if np.sum(brain_mask) > 0 else 0.0
        max_r = float(np.max(R_abl[brain_mask])) if np.sum(brain_mask) > 0 else 0.0
        peri_mean_r = float(np.mean(R_abl[peritumoral_mask])) if np.sum(peritumoral_mask) > 0 else 0.0

        tag = cname.split()[0]
        nii_name = f"ablation_{tag}_risk_map.nii.gz"
        nib.save(nib.Nifti1Image(R_abl, affine), os.path.join(out_dir, nii_name))

        ablation_results.append({
            "configuration": cname,
            "weights_used": w_dict,
            "mean_risk_brain": round(mean_r, 4),
            "max_risk_brain": round(max_r, 4),
            "tumor_adjacent_mean_risk_10mm": round(peri_mean_r, 4),
            "nii_file": nii_name
        })

        # Plot slice
        ax = axes[idx]
        ax.set_facecolor("#0e1117")
        ax.set_title(cname, color="white", fontsize=11, weight="bold")
        ax.imshow(np.rot90(brain_mask[:, :, slice_z]), cmap="gray", alpha=0.3)
        r_slice = np.rot90(R_abl[:, :, slice_z])
        im = ax.imshow(np.ma.masked_where(r_slice <= 0.05, r_slice), cmap="jet", alpha=0.65, vmin=0.0, vmax=1.0)
        ax.contour(np.rot90(tumor_binary[:, :, slice_z]), levels=[0.5], colors=["#00FF66"], linewidths=1.2, linestyles="dashed")
        ax.axis("off")

    fig.suptitle("Component Ablation Study: Progression from G-only to Full Multi-Modal Model\n[Axial Z=105 | Jet Colormap [0, 1]]", color="white", fontsize=15, weight="bold")
    png_path = os.path.join(out_dir, "ablation_comparison_slices.png")
    plt.savefig(png_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # Save summary CSV & JSON
    csv_path = os.path.join(out_dir, "ablation_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["configuration", "weights_used", "mean_risk_brain", "max_risk_brain", "tumor_adjacent_mean_risk_10mm", "nii_file"])
        writer.writeheader()
        writer.writerows(ablation_results)

    json_path = os.path.join(out_dir, "ablation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print(f"  [ABLATION] Completed 8 configurations. Saved to {out_dir}")
    return ablation_results
