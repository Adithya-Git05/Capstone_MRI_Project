"""
grade_adaptive_risk.py
======================
Grade-Adaptive Surgical Risk Modeling:
Connecting Preoperative Tumor Classification (LGG vs. HGG) to Dynamic 3D Surgical Risk Margins.

Clinical Rationale:
-------------------
- High-Grade Gliomas (HGG / Glioblastoma, WHO Grade III/IV):
  Highly invasive. Neoplastic cells aggressively infiltrate white-matter fascicles 
  beyond the visible gross tumor volume (GTV / T1ce enhancing rim).
  -> Expands proximity risk spread (sigma_G: 5.0 mm -> 8.5 mm)
  -> Injects an infiltrative microscopic safety halo (+5.0 mm dilation)
  -> Increases white matter tract safety weight (w_T: 0.20 -> 0.25)
  
- Low-Grade Gliomas (LGG, WHO Grade II):
  Typically slower growing with distinct infiltration kinetics.
  Prioritizes functional cortex preservation.
  -> Standard compact margin (sigma_G: 5.0 mm)
  -> Higher cortical anatomical weighting (w_S: 0.20)

Outputs:
--------
- outputs/V7_enhanced/adaptive_risk_map_HGG.nii.gz
- outputs/V7_enhanced/grade_adaptive_comparison.png
- outputs/V7_enhanced/config_adaptive_grade.json
- outputs/V7_enhanced/grade_adaptive_summary.txt
"""

import os
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_dilation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def classify_tumor_phenotype(mri_path, tumor_path, flair_path=None):
    """
    Evaluates tumor morphometry and radiomic intensity patterns 
    to classify glioma grade phenotype (LGG vs HGG) and quantify infiltration.
    """
    print("  Evaluating tumor imaging biomarkers...")
    tumor_nii = nib.load(tumor_path)
    tumor_data = tumor_nii.get_fdata() > 0
    zooms = tumor_nii.header.get_zooms()[:3]
    voxel_vol_cm3 = np.prod(zooms) / 1000.0
    
    total_tumor_voxels = np.sum(tumor_data)
    tumor_vol_cm3 = float(total_tumor_voxels * voxel_vol_cm3)
    
    # Assess necrotic / enhancing heterogeneity
    mri_data = nib.load(mri_path).get_fdata().astype(np.float32)
    tumor_intensities = mri_data[tumor_data]
    
    if len(tumor_intensities) > 0:
        p10, p90 = np.percentile(tumor_intensities, [10, 90])
        heterogeneity_ratio = float((p90 - p10) / (np.mean(tumor_intensities) + 1e-5))
    else:
        heterogeneity_ratio = 0.5
        
    # Check for edema / infiltrative rim if FLAIR exists
    edema_vol_cm3 = 0.0
    edema_to_core_ratio = 0.0
    if flair_path and os.path.exists(flair_path):
        flair_data = nib.load(flair_path).get_fdata().astype(np.float32)
        flair_brain = flair_data[flair_data > 0]
        flair_p75 = np.percentile(flair_brain, 75) if len(flair_brain) > 0 else 1.0
        # High FLAIR peri-tumoral voxels
        dilated_tumor = binary_dilation(tumor_data, iterations=5)
        peritumor_zone = dilated_tumor & (~tumor_data)
        high_flair_peritumor = np.sum((flair_data > flair_p75) & peritumor_zone)
        edema_vol_cm3 = float(high_flair_peritumor * voxel_vol_cm3)
        edema_to_core_ratio = float(edema_vol_cm3 / max(tumor_vol_cm3, 1.0))
        
    # Classification logic based on volumetric & heterogeneity biomarkers
    # High grade: larger volume (>20 cm3), high heterogeneity, or substantial peritumoral edema
    hgg_score = 0.0
    if tumor_vol_cm3 > 25.0:
        hgg_score += 0.35
    elif tumor_vol_cm3 > 12.0:
        hgg_score += 0.20
        
    if heterogeneity_ratio > 0.8:
        hgg_score += 0.35
    elif heterogeneity_ratio > 0.5:
        hgg_score += 0.20
        
    if edema_to_core_ratio > 0.4 or edema_vol_cm3 > 15.0:
        hgg_score += 0.30
    else:
        hgg_score += 0.15
        
    hgg_confidence = min(0.98, max(0.05, hgg_score))
    predicted_grade = "HGG" if hgg_confidence >= 0.50 else "LGG"
    
    classification_report = {
        "predicted_grade": predicted_grade,
        "confidence": round(hgg_confidence if predicted_grade == "HGG" else (1.0 - hgg_confidence), 3),
        "tumor_volume_cm3": round(tumor_vol_cm3, 2),
        "heterogeneity_ratio": round(heterogeneity_ratio, 3),
        "edema_volume_cm3": round(edema_vol_cm3, 2),
        "edema_to_core_ratio": round(edema_to_core_ratio, 3),
        "pathological_phenotype": "Infiltrative Glioblastoma Phenotype (WHO Grade IV)" if predicted_grade == "HGG" else "Circumscribed / Low-Grade Astrocytoma (WHO Grade II)"
    }
    return classification_report, tumor_data, zooms

def get_grade_adaptive_parameters(classification_report):
    """
    Returns biologically modulated parameters based on tumor grade.
    """
    grade = classification_report["predicted_grade"]
    if grade == "HGG":
        params = {
            "grade": "HGG",
            "sigma_G_mm": 8.5,           # Expanded proximity decay
            "infiltrative_halo_mm": 5.0, # Added microscopic invasion margin
            "halo_baseline_risk": 0.35,  # Minimum risk floor in peritumoral halo
            "weights": {
                "wG": 0.35,
                "wS": 0.12,
                "wT": 0.25,              # Heightened tract preservation
                "wC": 0.20,
                "wU": 0.08,
                "wV": 0.00
            },
            "clinical_rationale": "High-Grade Glioma exhibits microscopic infiltration along white matter tracts. Proximity spread is expanded to 8.5 mm and an infiltrative safety perimeter of 5.0 mm is enforced."
        }
    else:
        params = {
            "grade": "LGG",
            "sigma_G_mm": 5.0,           # Standard compact margin
            "infiltrative_halo_mm": 0.0,
            "halo_baseline_risk": 0.00,
            "weights": {
                "wG": 0.30,
                "wS": 0.22,              # Elevated cortical gray matter preservation
                "wT": 0.18,
                "wC": 0.20,
                "wU": 0.10,
                "wV": 0.00
            },
            "clinical_rationale": "Low-Grade Glioma presents more indolent borders. Standard 5.0 mm proximity margin is preserved with increased weight on preserving eloquent cortex."
        }
    return params

def compute_grade_adaptive_risk_map(project_root, adaptive_params, tumor_data, zooms):
    """
    Computes the Grade-Adaptive 3D Risk Map incorporating microscopic infiltration margins.
    """
    v7_out = os.path.join(project_root, "outputs", "V7_enhanced")
    
    # Load existing V7 components
    S_x = nib.load(os.path.join(v7_out, "anatomy_importance_S.nii.gz")).get_fdata().astype(np.float32)
    T_x = nib.load(os.path.join(v7_out, "tract_risk_T.nii.gz")).get_fdata().astype(np.float32)
    C_x = nib.load(os.path.join(v7_out, "functional_consequence_C.nii.gz")).get_fdata().astype(np.float32)
    std_risk_nii = nib.load(os.path.join(v7_out, "enhanced_risk_map.nii.gz"))
    affine = std_risk_nii.affine
    standard_risk = std_risk_nii.get_fdata().astype(np.float32)
    
    # Brain mask: strictly restrict all calculations to intracranial parenchyma
    # Combines standard risk footprint, anatomical labels, and tumor volume
    brain_mask = (standard_risk > 0.0) | (S_x > 0.0) | (tumor_data > 0)
    
    # Compute adaptive G_adaptive(x)
    sigma_G = adaptive_params["sigma_G_mm"]
    halo_mm = adaptive_params["infiltrative_halo_mm"]
    halo_floor = adaptive_params["halo_baseline_risk"]
    
    print(f"  Computing adaptive proximity field (sigma_G={sigma_G} mm, halo={halo_mm} mm)...")
    dist_mm = distance_transform_edt(tumor_data == 0, sampling=zooms).astype(np.float32)
    
    # Adaptive proximity decay
    G_adaptive = np.exp(-dist_mm / sigma_G).astype(np.float32)
    G_adaptive[tumor_data > 0] = 1.0
    
    # Apply microscopic infiltration halo for HGG
    if halo_mm > 0.0:
        halo_mask = (dist_mm > 0.0) & (dist_mm <= halo_mm)
        G_adaptive[halo_mask] = np.maximum(G_adaptive[halo_mask], halo_floor)
        
    # Strictly mask proximity field to intracranial parenchyma (zero out extracranial air/skull)
    G_adaptive[~brain_mask] = 0.0
        
    # Uncertainty component U(x)
    U_x = (G_adaptive > 0.20) & (G_adaptive < 0.80) & brain_mask
    U_x = U_x.astype(np.float32) * 0.25
    
    # Weighted composite adaptive risk R_adaptive(x)
    w = adaptive_params["weights"]
    R_adaptive = (
        w["wG"] * G_adaptive +
        w["wS"] * S_x +
        w["wT"] * T_x +
        w["wC"] * C_x +
        w["wU"] * U_x
    )
    # Strictly zero out non-brain tissue and clamp
    R_adaptive[~brain_mask] = 0.0
    R_adaptive = np.clip(R_adaptive, 0.0, 1.0).astype(np.float32)
    
    # Save adaptive risk NIfTI
    out_nii_path = os.path.join(v7_out, f"adaptive_risk_map_{adaptive_params['grade']}.nii.gz")
    nib.save(nib.Nifti1Image(R_adaptive, affine), out_nii_path)
    print(f"  [ADAPTIVE] Saved adaptive risk NIfTI to: {out_nii_path}")
    
    return standard_risk, R_adaptive, G_adaptive, dist_mm, affine

def plot_adaptive_comparison(project_root, standard_risk, adaptive_risk, tumor_data, classification_report, adaptive_params, out_png_path):
    """
    Renders a multi-view visual comparison between Standard Static V7 vs Grade-Adaptive V7.
    """
    mri_path = os.path.join(project_root, "data", "UCSF-PDGM-0004", "t1.nii.gz")
    mri_data = nib.load(mri_path).get_fdata().astype(np.float32)
    
    # Find tumor centroid
    tumor_vox = np.argwhere(tumor_data > 0)
    cx, cy, cz = np.round(np.mean(tumor_vox, axis=0)).astype(int)
    
    # Prepare MRI background slice
    mri_slice = np.rot90(mri_data[:, :, cz])
    p2, p98 = np.percentile(mri_slice[mri_slice > 0], [2, 98]) if np.any(mri_slice > 0) else (0, 1)
    mri_slice = np.clip((mri_slice - p2) / max(p98 - p2, 1e-5), 0.0, 1.0)
    
    std_slice = np.rot90(standard_risk[:, :, cz])
    adapt_slice = np.rot90(adaptive_risk[:, :, cz])
    diff_slice = adapt_slice - std_slice
    tumor_contour = np.rot90(tumor_data[:, :, cz])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0e1117")
    
    titles = [
        "A. Standard V7 Risk Map\n(Static sigma_G = 5.0 mm)",
        f"B. Grade-Adaptive V7 Risk Map ({adaptive_params['grade']})\n(Adaptive sigma_G = {adaptive_params['sigma_G_mm']} mm + Infiltrative Halo)",
        "C. Differential Safety Margin (Delta Risk)\n(Elevated Hazard Buffer for Glioblastoma Infiltration)"
    ]
    
    # Plot 1: Standard V7
    axes[0].imshow(mri_slice, cmap="gray", vmin=0, vmax=1)
    std_masked = np.ma.masked_where(std_slice < 0.05, std_slice)
    im0 = axes[0].imshow(std_masked, cmap="turbo", alpha=0.65, vmin=0.0, vmax=1.0)
    axes[0].contour(tumor_contour, levels=[0.5], colors=["#00FF66"], linewidths=1.8, linestyles="dashed")
    axes[0].axis("off")
    axes[0].set_title(titles[0], color="white", fontsize=11, weight="bold", pad=10)
    
    # Plot 2: Adaptive V7
    axes[1].imshow(mri_slice, cmap="gray", vmin=0, vmax=1)
    adapt_masked = np.ma.masked_where(adapt_slice < 0.05, adapt_slice)
    im1 = axes[1].imshow(adapt_masked, cmap="turbo", alpha=0.65, vmin=0.0, vmax=1.0)
    axes[1].contour(tumor_contour, levels=[0.5], colors=["#00FF66"], linewidths=1.8, linestyles="dashed")
    axes[1].axis("off")
    axes[1].set_title(titles[1], color="#33ccff", fontsize=11, weight="bold", pad=10)
    
    # Plot 3: Differential Risk (Delta)
    axes[2].imshow(mri_slice, cmap="gray", vmin=0, vmax=1)
    diff_masked = np.ma.masked_where(diff_slice <= 0.01, diff_slice)
    im2 = axes[2].imshow(diff_masked, cmap="YlOrRd", alpha=0.75, vmin=0.0, vmax=0.35)
    axes[2].contour(tumor_contour, levels=[0.5], colors=["#00FF66"], linewidths=1.8, linestyles="dashed")
    axes[2].axis("off")
    axes[2].set_title(titles[2], color="#ff7b72", fontsize=11, weight="bold", pad=10)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
    cb = fig.colorbar(im1, cax=cbar_ax)
    cb.set_label("Surgical Risk Score [0.0 - 1.0]", color="white", fontsize=10, weight="bold")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="white")
    
    fig.suptitle(
        f"Grade-Adaptive Surgical Risk Modeling | Patient: UCSF-PDGM-0004\n"
        f"Histopathological Phenotype: {classification_report['pathological_phenotype']} (Confidence: {classification_report['confidence']*100:.1f}%)",
        color="white", fontsize=13, weight="bold", y=0.98
    )
    
    plt.savefig(out_png_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [ADAPTIVE] Comparative visual figure saved to: {out_png_path}")

def run_grade_adaptive_pipeline():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("\n" + "=" * 70)
    print("  RUNNING GRADE-ADAPTIVE SURGICAL RISK MODELING (SUGGESTION 3)")
    print("=" * 70)
    
    patient_dir = os.path.join(project_root, "data", "UCSF-PDGM-0004")
    mri_path = os.path.join(patient_dir, "t1.nii.gz")
    tumor_path = os.path.join(patient_dir, "tumor_mask.nii.gz")
    flair_path = os.path.join(patient_dir, "flair.nii.gz")
    
    # Step 1: Preoperative Tumor Phenotype Classification
    classification_report, tumor_data, zooms = classify_tumor_phenotype(mri_path, tumor_path, flair_path)
    print(f"  Classification Result: {classification_report['predicted_grade']} ({classification_report['confidence']*100:.1f}% confidence)")
    print(f"  Tumor Volume: {classification_report['tumor_volume_cm3']} cm3 | Edema: {classification_report['edema_volume_cm3']} cm3")
    print(f"  Phenotype: {classification_report['pathological_phenotype']}")
    
    # Step 2: Biological Parameter Adaptation
    adaptive_params = get_grade_adaptive_parameters(classification_report)
    print(f"  Adapted Parameters: sigma_G={adaptive_params['sigma_G_mm']} mm, Infiltrative Halo={adaptive_params['infiltrative_halo_mm']} mm")
    print(f"  Weights: {adaptive_params['weights']}")
    
    # Step 3: Compute Adaptive Risk Volume
    std_risk, adapt_risk, G_adapt, dist_mm, affine = compute_grade_adaptive_risk_map(
        project_root, adaptive_params, tumor_data, zooms
    )
    
    # Step 4: Visual Comparison
    out_png = os.path.join(project_root, "outputs", "V7_enhanced", "grade_adaptive_comparison.png")
    plot_adaptive_comparison(project_root, std_risk, adapt_risk, tumor_data, classification_report, adaptive_params, out_png)
    
    # Step 5: Save Config & Summary Records
    v7_out = os.path.join(project_root, "outputs", "V7_enhanced")
    config_out = os.path.join(v7_out, "config_adaptive_grade.json")
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump({
            "classification_report": classification_report,
            "adaptive_parameters": adaptive_params
        }, f, indent=2)
    print(f"  [ADAPTIVE] Config JSON saved to: {config_out}")
    
    summary_txt = os.path.join(v7_out, "grade_adaptive_summary.txt")
    summary_content = f"""================================================================================
PATIENT GRADE-ADAPTIVE SURGICAL RISK SUMMARY
Patient: UCSF-PDGM-0004 | Target: Right Frontal Glioblastoma
================================================================================

1. PREOPERATIVE HISTOPATHOLOGICAL PHENOTYPING:
   - Predicted Grade: {classification_report['predicted_grade']}
   - Confidence: {classification_report['confidence']*100:.1f}%
   - Pathological Phenotype: {classification_report['pathological_phenotype']}
   - Gross Tumor Volume: {classification_report['tumor_volume_cm3']} cm3
   - Infiltrative Edema Zone: {classification_report['edema_volume_cm3']} cm3
   - Edema-to-Core Ratio: {classification_report['edema_to_core_ratio']}
   - Intratumoral Heterogeneity Ratio: {classification_report['heterogeneity_ratio']}

2. BIOLOGICALLY-ADAPTED RISK PARAMETERS:
   - Proximity Decay Spread (sigma_G): {adaptive_params['sigma_G_mm']} mm (Standard: 5.0 mm)
   - Microscopic Infiltrative Halo: +{adaptive_params['infiltrative_halo_mm']} mm boundary expansion
   - Halo Minimum Risk Floor: {adaptive_params['halo_baseline_risk']}
   - Adjusted Weights:
       wG (Proximity): {adaptive_params['weights']['wG']}
       wS (Anatomy): {adaptive_params['weights']['wS']}
       wT (Tracts): {adaptive_params['weights']['wT']} (Elevated for WM invasion)
       wC (Functional): {adaptive_params['weights']['wC']}
       wU (Uncertainty): {adaptive_params['weights']['wU']}

3. NEUROSURGICAL IMPLICATIONS:
   {adaptive_params['clinical_rationale']}
   The model expands the high-hazard surgical zone beyond the contrast-enhancing
   tumor border into white matter pathways, warning the surgeon against aggressive
   resection trajectories that compromise infiltrated peritumoral tracts.

================================================================================
"""
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(summary_content)
    print(f"  [ADAPTIVE] Clinical summary saved to: {summary_txt}")
    
    print("\n" + "=" * 70)
    print("  GRADE-ADAPTIVE RISK MODELING COMPLETE!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_grade_adaptive_pipeline()
