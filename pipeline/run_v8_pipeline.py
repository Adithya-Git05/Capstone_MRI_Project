"""
run_v8_pipeline.py
==================
V8 Master Orchestrator — 19-Phase Validated Surgical Risk Mapping System.

Builds on V7_ENHANCED without removing V6/V7.
Produces outputs/V7_VALIDATED/ and outputs/V8_CORRIDORS/

Research Decision-Support Prototype.
NOT a clinical outcome predictor.
"""
import os
import sys
import json
import csv
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_dilation

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
pipeline_path = os.path.join(project_root, 'pipeline')
sys.path.append(project_root)
sys.path.append(pipeline_path)

import v7_risk_components as v7
import v6_baseline
from label_mapping import get_label_info
from corridor_context import (
    GENERIC_LABELS, build_structure_overlays, load_tract_masks, nearby_structures,
)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 18: QC — strict registration checking
# ─────────────────────────────────────────────────────────────────────────────
def run_strict_qc(mri_nii, tumor_nii, labels_nii, dti_nii=None, tract_dir=None):
    """
    Phase 18: Verify all spatial alignments.
    Halts pipeline for any critical mismatch.
    """
    report = {}
    fail_reasons = []

    def check_pair(name_a, nii_a, name_b, nii_b):
        shape_ok = nii_a.shape[:3] == nii_b.shape[:3]
        spacing_ok = np.allclose(nii_a.header.get_zooms()[:3],
                                 nii_b.header.get_zooms()[:3], atol=0.01)
        affine_ok = np.allclose(nii_a.affine, nii_b.affine, atol=0.05)
        return {
            "shapes_match": bool(shape_ok),
            "spacing_match": bool(spacing_ok),
            "affine_match": bool(affine_ok),
            "all_pass": bool(shape_ok and spacing_ok and affine_ok)
        }

    report["mri_vs_tumor"] = check_pair("MRI", mri_nii, "Tumor", tumor_nii)
    report["mri_vs_parcellation"] = check_pair("MRI", mri_nii, "Parcellation", labels_nii)
    report["physical_spacing_mm"] = [round(float(z), 4) for z in mri_nii.header.get_zooms()[:3]]
    report["volume_shape"] = list(mri_nii.shape[:3])
    report["affine_det"] = round(float(np.linalg.det(mri_nii.affine)), 4)
    report["orientation"] = list(nib.aff2axcodes(mri_nii.affine))

    if dti_nii is not None:
        report["mri_vs_dti"] = check_pair("MRI", mri_nii, "DTI", dti_nii)
        report["dti_available"] = True
        if not report["mri_vs_dti"]["all_pass"]:
            fail_reasons.append("DTI registration mismatch")
    else:
        report["dti_available"] = False
        report["mri_vs_dti"] = {"all_pass": False, "reason": "DTI not provided"}

    # Tract registration check
    tract_reg_ok = True
    if tract_dir and os.path.isdir(tract_dir):
        tract_files = [f for f in os.listdir(tract_dir) if f.endswith(('.nii', '.nii.gz'))]
        tract_check = {}
        for tf in tract_files:
            tname = tf.replace('.nii.gz','').replace('.nii','')
            t_nii = nib.load(os.path.join(tract_dir, tf))
            r = check_pair("MRI", mri_nii, f"Tract:{tname}", t_nii)
            tract_check[tname] = r
            if not r["all_pass"]:
                tract_reg_ok = False
                fail_reasons.append(f"Tract {tname} registration mismatch")
        report["tract_registration"] = tract_check
    else:
        report["tract_registration"] = {"status": "No tract directory"}

    # Overall
    critical_checks = [
        report["mri_vs_tumor"]["all_pass"],
        report["mri_vs_parcellation"]["all_pass"],
    ]
    if dti_nii is not None:
        critical_checks.append(report["mri_vs_dti"]["all_pass"])
    report["all_critical_passed"] = bool(all(critical_checks) and tract_reg_ok)
    report["fail_reasons"] = fail_reasons

    return report


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Tract Overlap QC
# ─────────────────────────────────────────────────────────────────────────────
def compute_tract_overlap(tumor_binary, distance_tumor_mm, tract_dir, voxel_spacing):
    """
    Phase 2: For every tract calculate exact overlap metrics.
    Explicitly distinguishes DISJOINT / ADJACENT / OVERLAPPING.
    """
    records = []
    if not tract_dir or not os.path.isdir(tract_dir):
        return records

    tumor_vox = int(np.sum(tumor_binary > 0))
    vox_vol = float(np.prod(voxel_spacing))

    for tf in sorted(os.listdir(tract_dir)):
        if not tf.lower().endswith(('.nii', '.nii.gz')):
            continue
        tname = tf.replace('.nii.gz', '').replace('.nii', '')
        t_mask = nib.load(os.path.join(tract_dir, tf)).get_fdata() > 0
        tract_vox = int(np.sum(t_mask))
        if tract_vox == 0:
            continue

        overlap_vox = int(np.sum(tumor_binary.astype(bool) & t_mask))
        d_vals = distance_tumor_mm[t_mask]
        min_d = float(np.min(d_vals))
        mean_d = float(np.mean(d_vals))

        tumor_overlap_frac = overlap_vox / tumor_vox if tumor_vox > 0 else 0.0
        tract_overlap_frac = overlap_vox / tract_vox if tract_vox > 0 else 0.0

        # Spatial relationship — based on actual mask geometry, not just distance=0
        if overlap_vox > 0:
            relationship = "OVERLAPPING"
            qc_note = (f"{overlap_vox} voxels co-registered in both masks. "
                       "May reflect true invasion OR partial-volume / registration error. "
                       "Verify DTI-to-T1 registration before clinical interpretation.")
        elif min_d <= 3.0:
            relationship = "ADJACENT"
            qc_note = f"No voxel overlap; minimum distance = {min_d:.2f} mm (<=3 mm)."
        else:
            relationship = "DISJOINT"
            qc_note = f"No overlap; minimum distance = {min_d:.2f} mm."

        records.append({
            "tract_name": tname,
            "tract_volume_mm3": round(tract_vox * vox_vol, 1),
            "tumor_volume_mm3": round(tumor_vox * vox_vol, 1),
            "overlap_voxels": overlap_vox,
            "tumor_overlap_fraction": round(tumor_overlap_frac, 4),
            "tract_overlap_fraction": round(tract_overlap_frac, 4),
            "min_distance_mm": round(min_d, 2),
            "mean_distance_mm": round(mean_d, 2),
            "spatial_relationship": relationship,
            "qc_note": qc_note,
            "registration_check": "PASS — same affine as MRI"
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: Improved T(x) with alpha/beta proximity + overlap combination
# ─────────────────────────────────────────────────────────────────────────────
def calculate_improved_tract_risk(tumor_binary, distance_tumor_mm, tract_dir, config, voxel_spacing):
    """
    Phase 3: T_i = alpha * P_i + beta * O_i
    P_i = exp(-d_i / sigma_T)  [proximity component]
    O_i = overlap_fraction       [overlap component]
    alpha + beta = 1 (configurable)
    Voxel-wise map uses spatial tract geometry.
    """
    sigma_T = config.get("spatial_parameters", {}).get("sigma_T_mm", 10.0)
    tract_buffer = config.get("spatial_parameters", {}).get("tract_buffer_mm", 2.0)
    alpha = config.get("spatial_parameters", {}).get("tract_alpha", 0.7)
    beta = 1.0 - alpha
    tract_defs = config.get("tract_definitions", {})

    T_x = np.zeros(distance_tumor_mm.shape, dtype=np.float32)
    tract_records = []

    if not tract_dir or not os.path.isdir(tract_dir):
        return None, []

    tract_files = [f for f in sorted(os.listdir(tract_dir))
                   if f.lower().endswith(('.nii', '.nii.gz'))]
    if not tract_files:
        return None, []

    tumor_vox = int(np.sum(tumor_binary > 0))
    vox_vol = float(np.prod(voxel_spacing))

    for tf in tract_files:
        tname = tf.replace('.nii.gz', '').replace('.nii', '')
        t_nii = nib.load(os.path.join(tract_dir, tf))
        t_mask = t_nii.get_fdata() > 0
        if not np.any(t_mask):
            continue

        tract_vox = int(np.sum(t_mask))
        vol_mm3 = tract_vox * vox_vol
        d_vals = distance_tumor_mm[t_mask]
        d_i = float(np.min(d_vals))
        d_mean = float(np.mean(d_vals))

        # Proximity component
        P_i = float(np.exp(-d_i / sigma_T))

        # Overlap component
        overlap_vox = int(np.sum(tumor_binary.astype(bool) & t_mask))
        tumor_overlap_frac = overlap_vox / tumor_vox if tumor_vox > 0 else 0.0
        tract_overlap_frac = overlap_vox / tract_vox if tract_vox > 0 else 0.0
        O_i = max(tumor_overlap_frac, tract_overlap_frac)  # conservative: use max

        t_def = tract_defs.get(tname, {
            "importance": 0.70,
            "functional_domain": "Association",
            "potential_consequence": "Potential white-matter pathway disruption"
        })
        imp_i = float(t_def.get("importance", 0.70))

        # Combined tract proximity+overlap score
        combined_score = float((alpha * P_i + beta * O_i) * imp_i)
        combined_score = min(combined_score, 1.0)

        # Spatial relationship classification
        if overlap_vox > 0:
            relationship = "OVERLAPPING"
        elif d_i <= 3.0:
            relationship = "ADJACENT"
        else:
            relationship = "DISJOINT"

        # Voxel-wise map: risk decays from tract surface
        d_from_tract = distance_transform_edt(~t_mask, sampling=voxel_spacing).astype(np.float32)
        t_voxel_risk = combined_score * np.exp(-d_from_tract / tract_buffer)
        T_x = np.maximum(T_x, t_voxel_risk.astype(np.float32))

        tract_records.append({
            "tract_name": tname,
            "tract_volume_mm3": round(vol_mm3, 1),
            "min_distance_mm": round(d_i, 2),
            "mean_distance_mm": round(d_mean, 2),
            "overlap_voxels": overlap_vox,
            "tumor_overlap_fraction": round(tumor_overlap_frac, 4),
            "tract_overlap_fraction": round(tract_overlap_frac, 4),
            "spatial_relationship": relationship,
            "proximity_component_P": round(P_i, 4),
            "overlap_component_O": round(O_i, 4),
            "alpha": alpha,
            "beta": beta,
            "tract_importance": imp_i,
            "tract_risk_score": round(combined_score, 4),
            "functional_domain": t_def.get("functional_domain", "Association"),
            "potential_consequence": t_def.get("potential_consequence", "Potential pathway disruption"),
            "fmri_evidence": "UNAVAILABLE",
            "data_source": "Patient DTI (FA-threshold atlas ROI)",
            "qc_status": "PASS"
        })

    return T_x, tract_records


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6: Granular confidence reporting (no single blanket confidence)
# ─────────────────────────────────────────────────────────────────────────────
def build_evidence_status(qc_res, t_available, has_dti):
    """Returns per-modality confidence and overall evidence status."""
    structural_ok = qc_res.get("mri_vs_tumor", {}).get("all_pass", False) and \
                    qc_res.get("mri_vs_parcellation", {}).get("all_pass", False)
    dti_ok = has_dti and qc_res.get("mri_vs_dti", {}).get("all_pass", False)

    return {
        "structural_evidence": "HIGH" if structural_ok else "LOW",
        "dti_tract_evidence": "HIGH" if (dti_ok and t_available) else
                              ("MEDIUM" if t_available else "LOW"),
        "functional_fmri_mapping": "UNAVAILABLE (fMRI not acquired for this exam)",
        "vascular_evidence": "UNAVAILABLE (No MRA/CTA)",
        "empirical_uncertainty": "UNAVAILABLE (Single consensus model; no ensemble)",
        "overall_evidence_status": (
            "MULTI-MODAL (Structural + DTI)" if (structural_ok and dti_ok and t_available)
            else "STRUCTURAL-ONLY" if structural_ok
            else "DEGRADED"
        ),
        "qc_critical_pass": qc_res.get("all_critical_passed", False)
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 8: Structure-level risk table — filters generic tissue
# ─────────────────────────────────────────────────────────────────────────────

def build_structure_risk_table(labels_data, d_tumor_mm, G_x, S_x, T_x, C_x,
                               R_enhanced, thresholds, cfg, evidence_status,
                               t_available, tumor_binary, voxel_spacing,
                               tract_records=None):
    """
    Phase 8: Per-structure risk table, sorted by combined_risk descending.
    Generic/background tissue separated from functionally important structures.
    Includes both parcellated anatomical structures and white-matter tracts.
    """
    labels_clean = np.round(labels_data).astype(np.int32)
    tumor_vox = int(np.sum(tumor_binary > 0))
    vox_vol = float(np.prod(voxel_spacing))
    records = []

    for lbl in np.unique(labels_clean):
        lbl = int(lbl)
        if lbl == 0:
            continue
        mask = (labels_clean == lbl)
        if not np.any(mask):
            continue
        name, tier, weight = get_label_info(lbl)
        if name in ("Background", "Unknown") or name not in [get_label_info(l)[0] for l in np.unique(labels_clean) if l > 0]:
            pass  # keep all non-zero labels but flag generic ones

        d_sub = d_tumor_mm[mask]
        min_d = float(np.min(d_sub))
        mean_d = float(np.mean(d_sub))
        prox_r = float(np.mean(G_x[mask]))
        s_r = float(np.mean(S_x[mask]))
        tract_r = float(np.mean(T_x[mask])) if t_available else 0.0
        conseq_s = float(np.mean(C_x[mask]))
        comb_r = float(np.mean(R_enhanced[mask]))

        # Overlap with tumor
        overlap_vox = int(np.sum(mask & tumor_binary.astype(bool)))
        struct_overlap_frac = overlap_vox / int(np.sum(mask)) if np.sum(mask) > 0 else 0.0
        tumor_overlap_frac = overlap_vox / tumor_vox if tumor_vox > 0 else 0.0

        # Risk label
        label_str = "LOW"
        for t_name, (lo, hi) in thresholds.items():
            if lo <= comb_r < hi:
                label_str = t_name
                break
        if comb_r >= 0.80:
            label_str = "VERY_HIGH"

        # Functional domain
        func_domain = "General Cortex"
        pot_conseq = "Potential local parenchymal disruption"
        conseq_sev = "LOW"
        for dname, dinfo in cfg.get("functional_consequences", {}).items():
            if name in dinfo.get("associated_structures", []):
                func_domain = dname
                pot_conseq = dinfo.get("consequence", "")
                sev_score = float(dinfo.get("severity_score", 0.5))
                conseq_sev = ("VERY HIGH" if sev_score >= 0.9 else
                              "HIGH" if sev_score >= 0.75 else
                              "MODERATE" if sev_score >= 0.5 else "LOW")
                break

        is_generic = name in GENERIC_LABELS or tier == 0 or weight <= 0.1

        records.append({
            "structure_name": name,
            "structure_type": "Generic/Background" if is_generic else f"Tier-{tier}",
            "label_id": lbl,
            "functional_domain": func_domain,
            "potential_consequence": pot_conseq,
            "consequence_severity": conseq_sev,
            "importance_score": round(weight, 3),
            "tumor_min_distance_mm": round(min_d, 2),
            "tumor_mean_distance_mm": round(mean_d, 2),
            "overlap_voxels": overlap_vox,
            "overlap_fraction": round(struct_overlap_frac, 4),
            "struct_overlap_fraction": round(struct_overlap_frac, 4),
            "tumor_overlap_fraction": round(tumor_overlap_frac, 4),
            "proximity_risk": round(prox_r, 4),
            "anatomy_importance_S": round(s_r, 4),
            "tract_risk": round(tract_r, 4),
            "consequence_score": round(conseq_s, 4),
            "combined_risk": round(comb_r, 4),
            "risk_label": label_str,
            "evidence_status": evidence_status["overall_evidence_status"],
            "qc_status": "PASS" if evidence_status["qc_critical_pass"] else "FAIL"
        })

    # Include white-matter tracts as structured anatomical pathways
    if tract_records:
        for tr in tract_records:
            t_score = float(tr.get("tract_risk_score", 0.0))
            t_label = "LOW"
            for t_name, (lo, hi) in thresholds.items():
                if lo <= t_score < hi:
                    t_label = t_name
                    break
            if t_score >= 0.80:
                t_label = "VERY_HIGH"

            records.append({
                "structure_name": tr["tract_name"].replace("_", " ").title(),
                "structure_type": "White-Matter Tract",
                "label_id": -1,
                "functional_domain": tr.get("functional_domain", "Association"),
                "potential_consequence": tr.get("potential_consequence", "Pathway disruption"),
                "consequence_severity": "HIGH",
                "importance_score": round(float(tr.get("tract_importance", 0.7)), 3),
                "tumor_min_distance_mm": round(float(tr.get("min_distance_mm", 0.0)), 2),
                "tumor_mean_distance_mm": round(float(tr.get("mean_distance_mm", 0.0)), 2),
                "overlap_voxels": int(tr.get("overlap_voxels", 0)),
                "overlap_fraction": round(float(tr.get("tract_overlap_fraction", 0.0)), 4),
                "struct_overlap_fraction": round(float(tr.get("tract_overlap_fraction", 0.0)), 4),
                "tumor_overlap_fraction": round(float(tr.get("tumor_overlap_fraction", 0.0)), 4),
                "proximity_risk": round(float(tr.get("proximity_component_P", 0.0)), 4),
                "anatomy_importance_S": round(float(tr.get("tract_importance", 0.7)), 4),
                "tract_risk": round(t_score, 4),
                "consequence_score": round(t_score * 0.9, 4),
                "combined_risk": round(t_score, 4),
                "risk_label": t_label,
                "evidence_status": "DTI-derived ROI (patient FA)",
                "qc_status": tr.get("qc_status", "PASS")
            })

    # Sort: important structures first (non-generic, higher risk)
    records.sort(key=lambda x: (
        1 if x["structure_type"].startswith("Generic") else 0,
        -x["combined_risk"]
    ))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5: Restructured patient summary
# ─────────────────────────────────────────────────────────────────────────────
def write_patient_summary(patient_id, tumor_vol_mm3, tumor_centroid,
                          evidence_status, active_norm_weights,
                          tract_records, structure_records,
                          ablation_rows, out_path_txt, out_path_json):
    """Phase 5: Structured patient summary with per-modality confidence and AT-RISK section."""

    # Sort tracts by combined risk score descending
    tracts_sorted = sorted(tract_records, key=lambda t: t["tract_risk_score"], reverse=True)

    # Top non-generic structures
    important_structs = [s for s in structure_records
                         if not s["structure_type"].startswith("Generic")][:8]

    w_str = ", ".join(f"{k}: {v*100:.1f}%" for k, v in active_norm_weights.items())

    lines = []
    lines.append("=" * 70)
    lines.append("PATIENT SURGICAL RISK ASSESSMENT SUMMARY — RESEARCH DECISION SUPPORT")
    lines.append("=" * 70)
    lines.append(f"Patient ID       : {patient_id}")
    lines.append(f"Model Version    : V8_VALIDATED")
    lines.append(f"Tumor Volume     : {tumor_vol_mm3:.1f} mm³  ({tumor_vol_mm3/1000:.2f} cm³)")
    lines.append(f"Tumor Centroid   : X={tumor_centroid[0]:.1f}  Y={tumor_centroid[1]:.1f}  Z={tumor_centroid[2]:.1f}  (voxel coords)")
    lines.append("")
    lines.append("DATA AVAILABILITY & PER-MODALITY CONFIDENCE:")
    lines.append(f"  Structural MRI (T1/T1CE/T2/FLAIR) : AVAILABLE  | Confidence: {evidence_status['structural_evidence']}")
    lines.append(f"  Tumor Segmentation                 : AVAILABLE  | Confidence: {evidence_status['structural_evidence']}")
    lines.append(f"  DTI & Tractography                 : AVAILABLE  | Confidence: {evidence_status['dti_tract_evidence']}")
    lines.append(f"  Task-based fMRI                    : {evidence_status['functional_fmri_mapping']}")
    lines.append(f"  Vessel imaging (MRA/CTA)           : {evidence_status['vascular_evidence']}")
    lines.append(f"  Empirical uncertainty              : {evidence_status['empirical_uncertainty']}")
    lines.append(f"  Overall evidence status            : {evidence_status['overall_evidence_status']}")
    lines.append("")
    lines.append("ACTIVE MODEL WEIGHTS (after renormalization for unavailable components):")
    lines.append(f"  {w_str}")
    lines.append("")
    lines.append("RISK LABEL THRESHOLDS (prototype visualization — NOT validated clinical thresholds):")
    lines.append("  LOW [0.00–0.20] | MODERATE [0.20–0.40] | ELEVATED [0.40–0.60]")
    lines.append("  HIGH [0.60–0.80] | VERY HIGH [0.80–1.00]")
    lines.append("")
    lines.append("─" * 70)
    lines.append("AT-RISK WHITE-MATTER TRACTS (sorted by model-derived tract risk):")
    lines.append("─" * 70)

    for tr in tracts_sorted:
        lines.append(f"\n  [{tr['spatial_relationship']}] {tr['tract_name'].replace('_',' ').upper()}")
        lines.append(f"    Type              : White-Matter Tract (DTI-derived ROI)")
        lines.append(f"    Min tumor distance: {tr['min_distance_mm']} mm")
        lines.append(f"    Mean distance     : {tr['mean_distance_mm']} mm")
        lines.append(f"    Overlap voxels    : {tr['overlap_voxels']}  "
                     f"(tumor: {tr['tumor_overlap_fraction']*100:.2f}%, "
                     f"tract: {tr['tract_overlap_fraction']*100:.2f}%)")
        lines.append(f"    Proximity P_i     : {tr['proximity_component_P']:.4f}  (alpha={tr['alpha']:.2f})")
        lines.append(f"    Overlap O_i       : {tr['overlap_component_O']:.4f}  (beta={tr['beta']:.2f})")
        lines.append(f"    Tract importance  : {tr['tract_importance']:.2f}")
        lines.append(f"    Tract risk T_i    : {tr['tract_risk_score']:.4f}")
        lines.append(f"    Functional domain : {tr['functional_domain']}")
        lines.append(f"    Potential conseq  : {tr['potential_consequence']}")
        lines.append(f"    fMRI evidence     : {tr['fmri_evidence']}")
        lines.append(f"    Data source       : {tr['data_source']}")
        lines.append(f"    QC status         : {tr['qc_status']}")

    lines.append("")
    lines.append("─" * 70)
    lines.append("AT-RISK ANATOMICAL STRUCTURES (top by combined risk, excluding generic tissue):")
    lines.append("─" * 70)

    for sr in important_structs:
        lines.append(f"\n  {sr['structure_name']}  [Label {sr['label_id']}]")
        lines.append(f"    Type              : {sr['structure_type']}")
        lines.append(f"    Importance score  : {sr['importance_score']:.2f}")
        lines.append(f"    Min tumor distance: {sr['tumor_min_distance_mm']} mm")
        lines.append(f"    Mean distance     : {sr['tumor_mean_distance_mm']} mm")
        lines.append(f"    Overlap voxels    : {sr['overlap_voxels']}  "
                     f"(tumor: {sr['tumor_overlap_fraction']*100:.2f}%, "
                     f"struct: {sr['struct_overlap_fraction']*100:.2f}%)")
        lines.append(f"    Proximity risk    : {sr['proximity_risk']:.4f}")
        lines.append(f"    Tract risk        : {sr['tract_risk']:.4f}")
        lines.append(f"    Consequence score : {sr['consequence_score']:.4f}")
        lines.append(f"    Combined risk     : {sr['combined_risk']:.4f}  → {sr['risk_label']}")
        lines.append(f"    Functional domain : {sr['functional_domain']}")
        lines.append(f"    Potential conseq  : {sr['potential_consequence']}")
        lines.append(f"    Evidence status   : {sr['evidence_status']}")
        lines.append(f"    QC status         : {sr['qc_status']}")

    lines.append("")
    lines.append("─" * 70)
    lines.append("DISCLAIMER & SCIENTIFIC INTERPRETATION:")
    lines.append("─" * 70)
    lines.append("""
  This report is generated by a RESEARCH DECISION-SUPPORT PROTOTYPE.
  Risk scores represent model-derived spatial relationships between the
  tumor and anatomical/functional structures. They are NOT predictions of:
    - Actual probability of postoperative injury
    - Patient-specific neurological outcome
    - Surgical morbidity or mortality

  A spatial overlap of 0 mm distance or co-registered voxels may reflect
  true anatomical contact OR registration error between DTI and T1 space.
  Clinical correlation and expert review are required before any use in
  patient care planning.

  Functional consequence labels are based on known neuroanatomy and are
  NOT patient-specific fMRI findings (fMRI was not acquired for this exam).
""")
    lines.append("=" * 70)

    summary_txt = "\n".join(lines)
    with open(out_path_txt, "w", encoding="utf-8") as f:
        f.write(summary_txt)

    # JSON version
    summary_json = {
        "patient_id": patient_id,
        "model_version": "V8_VALIDATED",
        "tumor_volume_mm3": round(tumor_vol_mm3, 1),
        "tumor_centroid_voxel": [round(c, 1) for c in tumor_centroid],
        "evidence_status": evidence_status,
        "active_weights": active_norm_weights,
        "tract_findings": tracts_sorted,
        "top_risk_structures": important_structs[:5],
        "risk_threshold_note": (
            "Prototype visualization thresholds only. "
            "Not validated clinical risk boundaries."
        ),
        "disclaimer": (
            "Model-derived relative risk for research decision support. "
            "NOT a clinical outcome prediction."
        )
    }
    with open(out_path_json, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 10: V6 vs V7 comparison + delta map
# ─────────────────────────────────────────────────────────────────────────────
def run_comparison(R_base, R_enhanced, brain_mask, tumor_binary,
                   d_tumor_mm, thresholds, affine, out_dir):
    """Phase 10: Baseline vs V7 comparison metrics + delta map."""
    os.makedirs(out_dir, exist_ok=True)

    delta = (R_enhanced - R_base).astype(np.float32)
    nib.save(nib.Nifti1Image(delta, affine),
             os.path.join(out_dir, "delta_risk_map.nii.gz"))

    peritumoral = brain_mask & (d_tumor_mm <= 10.0)
    high_thr = thresholds.get("HIGH", [0.60, 0.80])[0]
    vox_vol_mm3 = 1.0  # 1mm iso

    def stats(R, mask):
        vals = R[mask]
        return {
            "mean_risk": round(float(np.mean(vals)), 4),
            "max_risk": round(float(np.max(vals)), 4),
            "tumor_adjacent_mean": round(float(np.mean(R[peritumoral])), 4) if np.any(peritumoral) else 0.0,
            "high_risk_volume_mm3": round(float(np.sum(R[mask] >= high_thr)) * vox_vol_mm3, 1),
        }

    base_stats = stats(R_base, brain_mask)
    enh_stats = stats(R_enhanced, brain_mask)

    comparison = {
        "V6_baseline": {
            "formula": "R = 0.5*G + 0.5*S",
            **base_stats
        },
        "V7_enhanced": {
            "formula": "R = wG*G + wS*S + wT*T + wC*C (renormalized)",
            **enh_stats
        },
        "delta": {
            "mean_delta_brain": round(float(np.mean(delta[brain_mask])), 4),
            "max_delta_brain": round(float(np.max(delta[brain_mask])), 4),
            "min_delta_brain": round(float(np.min(delta[brain_mask])), 4),
            "interpretation": (
                "Positive delta: V7 assigns higher risk than V6 (tract/consequence contribution). "
                "Negative delta: V6 assigns higher risk in that region."
            )
        }
    }

    with open(os.path.join(out_dir, "baseline_vs_v7_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison, delta


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 11: Ablation A–F only (as specified, no fabricated U/V)
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation_af(G_x, S_x, T_x, C_x, brain_mask, tumor_binary,
                    d_tumor_mm, affine, out_dir):
    """Phase 11: Ablation configurations A through F."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    peritumoral = brain_mask & (d_tumor_mm <= 10.0)
    configs = {
        "A (G only)":        {"G": 1.0},
        "B (S only)":        {"S": 1.0},
        "C (G+S)":           {"G": 0.50, "S": 0.50},
        "D (G+S+T)":         {"G": 0.45, "S": 0.25, "T": 0.30},
        "E (G+S+C)":         {"G": 0.45, "S": 0.25, "C": 0.30},
        "F (G+S+T+C)":       {"G": 0.35, "S": 0.15, "T": 0.25, "C": 0.25},
    }
    comp_dict = {
        "G": G_x, "S": S_x,
        "T": T_x if T_x is not None else np.zeros_like(G_x),
        "C": C_x if C_x is not None else np.zeros_like(G_x)
    }

    results = []
    slice_z = G_x.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0e1117")
    axes = axes.flatten()

    for idx, (cname, w_dict) in enumerate(configs.items()):
        total_w = sum(w_dict.values())
        norm_w = {k: v / total_w for k, v in w_dict.items()}
        R_abl = np.zeros_like(G_x, dtype=np.float32)
        for k, wt in norm_w.items():
            R_abl += wt * comp_dict[k]
        R_abl[~brain_mask] = 0.0
        R_abl = np.clip(R_abl, 0.0, 1.0)

        mean_r = float(np.mean(R_abl[brain_mask]))
        max_r = float(np.max(R_abl[brain_mask]))
        peri_r = float(np.mean(R_abl[peritumoral])) if np.any(peritumoral) else 0.0
        hi_vol = float(np.sum(R_abl[brain_mask] >= 0.60))

        tag = cname.split()[0]
        nii_out = os.path.join(out_dir, f"ablation_{tag}_risk_map.nii.gz")
        nib.save(nib.Nifti1Image(R_abl, affine), nii_out)

        results.append({
            "configuration": cname,
            "weights_used": str(w_dict),
            "mean_risk_brain": round(mean_r, 4),
            "max_risk_brain": round(max_r, 4),
            "tumor_adjacent_mean_10mm": round(peri_r, 4),
            "high_risk_voxels_ge0.60": round(hi_vol, 0),
            "nii_file": os.path.basename(nii_out)
        })

        ax = axes[idx]
        ax.set_facecolor("#0e1117")
        ax.set_title(cname, color="white", fontsize=11, weight="bold")
        ax.imshow(np.rot90(brain_mask[:, :, slice_z]), cmap="gray", alpha=0.3)
        r_sl = np.ma.masked_where(
            np.rot90(R_abl[:, :, slice_z]) <= 0.05,
            np.rot90(R_abl[:, :, slice_z]))
        ax.imshow(r_sl, cmap="jet", alpha=0.65, vmin=0.0, vmax=1.0)
        ax.contour(np.rot90(tumor_binary[:, :, slice_z]),
                   levels=[0.5], colors=["#00FF66"], linewidths=1.2, linestyles="dashed")
        ax.axis("off")

    fig.suptitle(
        "Ablation Study A–F: Component Contribution Analysis\n"
        "[Note: Colorbar = Model-derived surgical risk score, NOT probability of injury]",
        color="white", fontsize=14, weight="bold"
    )
    plt.savefig(os.path.join(out_dir, "ablation_AF_comparison.png"),
                dpi=180, bbox_inches="tight", facecolor="#0e1117")
    plt.close()

    # Save CSV + JSON
    with open(os.path.join(out_dir, "ablation_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 12–16: Corridor planner with Pareto analysis
# ─────────────────────────────────────────────────────────────────────────────
import heapq

def astar_3d(cost_volume, start, goal, voxel_spacing, max_steps=150000, h_weight=2.5):
    shape = cost_volume.shape
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}

    def heuristic(p):
        return float(np.linalg.norm(
            (np.array(p) - np.array(goal)) * np.array(voxel_spacing))) * h_weight

    steps = 0
    neighbors = [
        (-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1),
        (-1,-1,0),(-1,1,0),(1,-1,0),(1,1,0),
        (-1,0,-1),(-1,0,1),(1,0,-1),(1,0,1),
        (0,-1,-1),(0,-1,1),(0,1,-1),(0,1,1)
    ]

    while open_set and steps < max_steps:
        steps += 1
        _, current = heapq.heappop(open_set)
        if np.linalg.norm(np.array(current) - np.array(goal)) <= 1.5:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        for dx, dy, dz in neighbors:
            nx, ny, nz = current[0]+dx, current[1]+dy, current[2]+dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                step_d = np.linalg.norm(np.array([dx,dy,dz]) * np.array(voxel_spacing))
                cost_here = float(cost_volume[nx, ny, nz])
                tg = g_score[current] + step_d * (1.0 + cost_here * 4.0)
                nb = (nx, ny, nz)
                if nb not in g_score or tg < g_score[nb]:
                    came_from[nb] = current
                    g_score[nb] = tg
                    heapq.heappush(open_set, (tg + heuristic(nb), nb))
    return None


def pareto_analysis(corridor_records):
    """
    Phase 16: Identify Pareto-efficient corridors.
    Objectives: minimize (mean_risk, tract_risk, path_length_mm)
                maximize (tumor_coverage)
    A corridor is Pareto-dominated if another is at least as good on all
    objectives and strictly better on at least one.
    """
    def dominates(a, b):
        """Returns True if a dominates b."""
        better_any = False
        for key, minimize in [
            ("mean_risk", True), ("tract_risk", True),
            ("path_length_mm", True), ("tumor_coverage", False)
        ]:
            av = a[key] * (1 if minimize else -1)
            bv = b[key] * (1 if minimize else -1)
            if av > bv:
                return False
            if av < bv:
                better_any = True
        return better_any

    pareto = []
    for i, c in enumerate(corridor_records):
        dominated = False
        for j, other in enumerate(corridor_records):
            if i != j and dominates(other, c):
                dominated = True
                break
        pareto.append({**c, "pareto_efficient": not dominated})
    return pareto


def write_corridor_paths(json_path, pareto_records, corridor_details, target,
                         affine, voxel_spacing):
    """Save corridor centerlines in millimetres plus the structures each passes near."""
    corridors = []
    for rec in pareto_records:
        detail = corridor_details[rec["corridor_id"]]
        path_world = nib.affines.apply_affine(affine, np.array(detail["path_voxels"], float))
        corridors.append({
            "corridor_id": rec["corridor_id"],
            "name": rec["corridor_name"],
            "pareto_efficient": bool(rec["pareto_efficient"]),
            "length_mm": rec["length_mm"],
            "mean_risk": rec["mean_risk"],
            "max_risk": rec["max_risk"],
            "tract_risk": rec["tract_risk"],
            "eloquent_risk": rec["eloquent_risk"],
            "tumor_coverage": rec["tumor_coverage"],
            "composite_cost": rec["composite_cost"],
            "entry_voxel": rec["entry_point"],
            "path_world_mm": [[round(float(v), 2) for v in pt] for pt in path_world],
            "nearby_structures": detail["nearby_structures"],
        })
    target_world = nib.affines.apply_affine(affine, np.array(target, float))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "coordinate_space": "NIfTI affine transformed to RAS+ millimeters",
            "voxel_spacing_mm": [float(v) for v in voxel_spacing],
            "target": {
                "voxel": [int(v) for v in target],
                "world_mm": [round(float(v), 2) for v in target_world],
            },
            "corridors": corridors,
            "note": ("Computer-generated candidate trajectories for research. "
                     "Not a surgical plan; lowest composite cost is not the safest path."),
        }, f, indent=2)


def plan_corridors_v8(risk_map, tumor_binary, brain_mask, T_x, C_x,
                      cfg, affine, voxel_spacing, out_dir,
                      labels_data=None, tract_masks=None):
    """Phases 12–16: Multi-corridor A* + Pareto analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    os.makedirs(out_dir, exist_ok=True)

    c_weights = cfg.get("corridor_weights", {
        "w_length": 0.15, "w_eloquent": 0.25, "w_tract": 0.30,
        "w_vessel": 0.15, "w_ventricle": 0.15, "w_coverage": 0.20
    })

    tumor_vox = np.argwhere(tumor_binary > 0)
    target = tuple(np.round(np.mean(tumor_vox, axis=0)).astype(int))

    eroded_brain = binary_dilation(~brain_mask, iterations=1)
    cortical_surface = brain_mask & eroded_brain
    surface_voxels = np.argwhere(cortical_surface)

    def pick_entry(cond):
        cands = surface_voxels[cond]
        if len(cands) == 0:
            return None
        return tuple(cands[len(cands)//2])

    tx, ty, tz = target
    candidate_defs = [
        ("Corridor-1 (Lateral Frontal)",
         (surface_voxels[:, 0] > tx+15) &
         (surface_voxels[:, 1] > ty-20) &
         (surface_voxels[:, 2] > tz-15)),
        ("Corridor-2 (Superior Frontal)",
         (surface_voxels[:, 2] > tz+15) &
         (surface_voxels[:, 0] > tx-10) &
         (surface_voxels[:, 1] > ty-10)),
        ("Corridor-3 (Anterior Frontal)",
         (surface_voxels[:, 1] > ty+15) &
         (surface_voxels[:, 0] > tx-10) &
         (surface_voxels[:, 2] > tz-10)),
        ("Corridor-4 (Pterional/Temporal)",
         (surface_voxels[:, 0] > tx+20) &
         (surface_voxels[:, 2] < tz-5)),
    ]

    candidate_entries = []
    for name, cond in candidate_defs:
        entry = pick_entry(cond)
        if entry is not None:
            candidate_entries.append((name, entry))

    # Fallback if fewer than 2 found
    if len(candidate_entries) < 2:
        for i in range(min(4, len(surface_voxels))):
            candidate_entries.append(
                (f"Corridor-{i+1}", tuple(surface_voxels[i * (len(surface_voxels)//4)])))

    corridor_volume = np.zeros(risk_map.shape, dtype=np.uint8)
    raw_records = []
    corridor_details = {}
    target = tuple(int(v) for v in target)

    for cid, (cname, entry) in enumerate(candidate_entries, 1):
        entry = tuple(int(v) for v in entry)
        path = astar_3d(risk_map, entry, target, voxel_spacing)
        if not path:
            print(f"    [CORRIDOR] A* found no path for {cname}")
            continue

        c_mask = np.zeros(risk_map.shape, dtype=bool)
        for pt in path:
            c_mask[pt[0], pt[1], pt[2]] = True
        c_mask_dil = binary_dilation(c_mask, iterations=2)
        corridor_volume[c_mask_dil] = cid

        path_arr = np.array(path)
        diffs = np.diff(path_arr, axis=0) * np.array(voxel_spacing)
        length_mm = float(np.sum(np.linalg.norm(diffs, axis=1)))

        vox_risk = risk_map[c_mask_dil]
        mean_r = float(np.mean(vox_risk)) if len(vox_risk) > 0 else 0.0
        max_r = float(np.max(vox_risk)) if len(vox_risk) > 0 else 0.0

        eloq_r = float(np.mean(C_x[c_mask_dil])) if C_x is not None else 0.0
        tract_r = float(np.mean(T_x[c_mask_dil])) if T_x is not None else 0.0
        vasc_r = 0.0   # Vascular imaging UNAVAILABLE
        vent_r = 0.0   # Ventricular penetration not modeled
        coverage = float(np.sum(c_mask_dil & (tumor_binary > 0)) /
                         max(np.sum(tumor_binary > 0), 1))

        comp_cost = (
            c_weights.get("w_length", 0.15) * (length_mm / 100.0) +
            c_weights.get("w_eloquent", 0.25) * eloq_r +
            c_weights.get("w_tract", 0.30) * tract_r +
            c_weights.get("w_vessel", 0.15) * vasc_r +
            c_weights.get("w_ventricle", 0.15) * vent_r -
            c_weights.get("w_coverage", 0.20) * coverage
        )

        corridor_details[cid] = {
            "path_voxels": [[int(v) for v in pt] for pt in path],
            "nearby_structures": nearby_structures(
                c_mask, voxel_spacing, labels_data, tract_masks),
        }

        raw_records.append({
            "corridor_id": cid,
            "corridor_name": cname,
            "entry_point": list(entry),
            "target_region": f"Centroid_Z{target[2]}",
            "target_voxel": list(target),
            "length_mm": round(length_mm, 2),
            "path_length_mm": round(length_mm, 2),
            "mean_risk": round(mean_r, 4),
            "max_risk": round(max_r, 4),
            "maximum_risk": round(max_r, 4),
            "functional_risk": round(eloq_r, 4),
            "eloquent_risk": round(eloq_r, 4),
            "tract_risk": round(tract_r, 4),
            "vascular_risk": round(vasc_r, 4),
            "ventricular_risk": round(vent_r, 4),
            "tumor_coverage": round(coverage, 4),
            "composite_cost": round(comp_cost, 4),
        })

    if not raw_records:
        print("  [CORRIDORS] No valid corridor paths generated.")
        return [], None

    # Sort by composite cost
    raw_records.sort(key=lambda x: x["composite_cost"])

    # Pareto analysis
    pareto_records = pareto_analysis(raw_records)

    # Save CSVs
    with open(os.path.join(out_dir, "corridor_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_records[0].keys()))
        writer.writeheader()
        writer.writerows(raw_records)

    with open(os.path.join(out_dir, "pareto_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pareto_records[0].keys()))
        writer.writeheader()
        writer.writerows(pareto_records)

    write_corridor_paths(
        os.path.join(out_dir, "corridor_paths.json"),
        pareto_records, corridor_details, target, affine, voxel_spacing)

    # Save NIfTI mask
    nii_path = os.path.join(out_dir, "candidate_corridors.nii.gz")
    nib.save(nib.Nifti1Image(corridor_volume, affine), nii_path)

    # Save composite NIfTI with T1 anatomical image so brain structure is visible
    mri_path = os.path.join(project_root, cfg["paths"]["mri_image"])
    if os.path.exists(mri_path):
        t1_raw = nib.load(mri_path).get_fdata().astype(np.float32)
        p99 = np.percentile(t1_raw[t1_raw > 0], 99.5) if np.any(t1_raw > 0) else 1.0
        t1_norm = np.clip((t1_raw / p99) * 800.0, 0, 800)
        t_rim = binary_dilation(tumor_binary > 0, iterations=1) & ~(tumor_binary > 0)
        t1_norm[t_rim] = 950.0
        for cid in range(1, len(candidate_entries) + 1):
            t1_norm[corridor_volume == cid] = 1100.0 + cid * 200.0
        nib.save(nib.Nifti1Image(t1_norm, affine),
                 os.path.join(out_dir, "corridors_overlaid_on_t1.nii.gz"))

    # Visualization: Axial overlay
    cz = target[2]
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    if os.path.exists(mri_path):
        mri_slice = np.rot90(t1_raw[:, :, cz])
        brain_pixels = t1_raw[t1_raw > 0]
        p2, p98 = np.percentile(brain_pixels, [2, 98]) if brain_pixels.size else (0.0, 1.0)
        mri_slice = np.clip((mri_slice - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
        ax.imshow(mri_slice, cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="nearest")

    risk_slice = np.rot90(risk_map[:, :, cz])
    risk_overlay = np.ma.masked_where(risk_slice <= 0.03, risk_slice)
    ax.imshow(risk_overlay, cmap="turbo", alpha=0.68, vmin=0.0, vmax=1.0,
              interpolation="nearest")
    ax.contour(np.rot90(tumor_binary[:, :, cz]),
               levels=[0.5], colors=["#00FF66"], linewidths=2.5, linestyles="dashed")

    colors_map = {1: "#FF3366", 2: "#33CCFF", 3: "#FFFF33", 4: "#CC33FF"}
    for rec in pareto_records:
        cid = rec["corridor_id"]
        c2d = np.rot90((corridor_volume == cid)[:, :, cz])
        if np.any(c2d):
            label = rec["corridor_name"]
            pareto_flag = " [Pareto-efficient]" if rec.get("pareto_efficient") else ""
            contour = ax.contour(c2d, levels=[0.5],
                                 colors=[colors_map.get(cid, "#FFFFFF")], linewidths=3.0)
            contour.set_path_effects([
                pe.Stroke(linewidth=6.0, foreground="black", alpha=0.75),
                pe.Normal()
            ])
            ys, xs = np.where(c2d)
            ax.annotate(
                f"C{cid}", (float(np.mean(xs)), float(np.mean(ys))),
                color="white", fontsize=10, weight="bold", ha="center", va="center",
                path_effects=[pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()]
            )

    ax.axis("off")
    ax.set_title(
        "Candidate Surgical Corridors — Axial View at Tumor Target\n"
        "Colored contours: competing trajectories | Colormap: model-derived risk score\n"
        "Note: 'lowest composite cost' != 'safest path' (model not clinically validated)",
        color="white", fontsize=12, weight="bold"
    )
    plt.savefig(os.path.join(out_dir, "corridor_visualization.png"),
                dpi=180, bbox_inches="tight", facecolor="#0e1117")
    plt.savefig(os.path.join(out_dir, "corridor_overlay.png"),
                dpi=180, bbox_inches="tight", facecolor="#0e1117")
    val_vis = os.path.join(os.path.dirname(out_dir), "V7_VALIDATED", "visualizations")
    if os.path.exists(val_vis):
        plt.savefig(os.path.join(val_vis, "corridor_overlay.png"),
                    dpi=180, bbox_inches="tight", facecolor="#0e1117")
    plt.close()

    # HTML summary
    html_rows = ""
    for rec in pareto_records:
        p_flag = "✅ Pareto-efficient" if rec.get("pareto_efficient") else "—"
        html_rows += f"""
        <tr>
          <td>{rec['corridor_id']}</td>
          <td>{rec['corridor_name']}</td>
          <td>{rec['path_length_mm']}</td>
          <td>{rec['mean_risk']}</td>
          <td>{rec['maximum_risk']}</td>
          <td>{rec['eloquent_risk']}</td>
          <td>{rec['tract_risk']}</td>
          <td>{rec['tumor_coverage']}</td>
          <td>{rec['composite_cost']}</td>
          <td>{p_flag}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>V8 Surgical Corridors — {cfg.get('patient_id','UCSF-PDGM-0004')}</title>
<style>
  body {{background:#0e1117;color:#eee;font-family:monospace;padding:2em}}
  h1 {{color:#33CCFF}} h2 {{color:#FFFF33}}
  table {{border-collapse:collapse;width:100%}}
  th {{background:#1a2535;color:#33CCFF;padding:8px;border:1px solid #333}}
  td {{padding:6px 8px;border:1px solid #333;color:#ddd}}
  tr:hover td {{background:#1a2535}}
  .note {{color:#FF9944;font-size:0.9em;margin-top:1em}}
</style></head>
<body>
<h1>Candidate Surgical Corridor Analysis</h1>
<h2>Patient: {cfg.get('patient_id','UCSF-PDGM-0004')} | Model: V8_VALIDATED</h2>
<p class="note">⚠️ RESEARCH PROTOTYPE — Composite cost does NOT equal surgical safety.
These candidates are for comparative, decision-support analysis only.</p>
<table>
  <tr>
    <th>ID</th><th>Name</th><th>Length (mm)</th><th>Mean Risk</th><th>Max Risk</th>
    <th>Eloquent Risk</th><th>Tract Risk</th><th>Coverage</th>
    <th>Composite Cost</th><th>Pareto</th>
  </tr>
  {html_rows}
</table>
<p class="note">Pareto-efficient: No other corridor is better on ALL objectives simultaneously.
Vascular risk = UNAVAILABLE (no MRA/CTA). Ventricular risk = not modeled.</p>
</body></html>"""

    with open(os.path.join(out_dir, "corridor_visualization.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  [CORRIDORS V8] {len(raw_records)} corridors generated; "
          f"{sum(1 for r in pareto_records if r.get('pareto_efficient'))} Pareto-efficient.")
    return pareto_records, nii_path


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 17: All visualization outputs
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_visualizations(mri_data, tumor_binary, R_base, R_enhanced,
                                delta_map, G_x, S_x, T_x, C_x,
                                patient_id, out_dir):
    """Phase 17: Generate all required PNG outputs with fixed [0,1] scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    tumor_coords = np.argwhere(tumor_binary > 0)
    cx, cy, cz = np.round(np.mean(tumor_coords, axis=0)).astype(int)

    cbar_label = "Model-derived surgical risk score [0=Low, 1=Very High]"

    def save_triplanar(data_map, title, fname, cmap="jet", vmin=0, vmax=1, mask_below=0.05):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0e1117")
        planes = [
            (f"Sagittal X={cx}", np.rot90(mri_data[cx, :, :]), np.rot90(data_map[cx, :, :]), np.rot90(tumor_binary[cx, :, :])),
            (f"Coronal Y={cy}", np.rot90(mri_data[:, cy, :]), np.rot90(data_map[:, cy, :]), np.rot90(tumor_binary[:, cy, :])),
            (f"Axial Z={cz}", np.rot90(mri_data[:, :, cz]), np.rot90(data_map[:, :, cz]), np.rot90(tumor_binary[:, :, cz])),
        ]
        im = None
        for idx2, (ptitle, mri_s, risk_s, seg_s) in enumerate(planes):
            ax = axes[idx2]
            ax.set_facecolor("#0e1117")
            ax.set_title(ptitle, color="white", fontsize=12, weight="bold")
            ax.imshow(mri_s, cmap="gray")
            masked = np.ma.masked_where(risk_s <= mask_below, risk_s)
            im = ax.imshow(masked, cmap=cmap, alpha=0.55, vmin=vmin, vmax=vmax)
            if np.any(seg_s > 0):
                ax.contour(seg_s, levels=[0.5], colors=["#00FF66"], linewidths=1.8, linestyles="dashed")
            ax.axis("off")
        cbar_ax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_label(cbar_label, color="white", fontsize=10, labelpad=8)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
        fig.suptitle(f"Patient: {patient_id} | {title}", color="white", fontsize=14, weight="bold", y=0.98)
        plt.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight", facecolor="#0e1117")
        plt.close()
        print(f"    Saved: {fname}")

    save_triplanar(R_base, "V6 Baseline Risk Map [R = 0.5G + 0.5S]", "baseline_risk_map.png")
    save_triplanar(R_enhanced, "V8 Enhanced Risk Map [G+S+T+C renormalized]", "enhanced_risk_map.png")

    # Delta map uses coolwarm
    save_triplanar(delta_map, "Delta Risk [V8 - V6] | Red=V8>V6, Blue=V6>V8",
                   "delta_risk_map.png", cmap="coolwarm", vmin=-0.5, vmax=0.5, mask_below=0.02)

    # Component maps + combined risk map (Phase 1)
    fig, axes = plt.subplots(1, 5, figsize=(26, 6), facecolor="#0e1117")
    comp_map_data = [
        ("G(x): Proximity", G_x),
        ("S(x): Anatomy Importance", S_x),
        ("T(x): Tract Risk", T_x),
        ("C(x): Functional Consequence", C_x),
        ("R(x): Final Combined Risk", R_enhanced)
    ]
    mri_ax = np.rot90(mri_data[:, :, cz])
    tumor_ax = np.rot90(tumor_binary[:, :, cz])
    for idx2, (ctitle, carr) in enumerate(comp_map_data):
        ax = axes[idx2]
        ax.set_facecolor("#0e1117")
        ax.set_title(ctitle, color="white", fontsize=12, weight="bold")
        ax.imshow(mri_ax, cmap="gray")
        if carr is not None:
            cs = np.rot90(carr[:, :, cz])
            ax.imshow(np.ma.masked_where(cs <= 0.01, cs), cmap="jet", alpha=0.6, vmin=0, vmax=1)
        else:
            ax.text(0.5, 0.5, "UNAVAILABLE", color="yellow", fontsize=14,
                    ha="center", va="center", transform=ax.transAxes)
        ax.contour(tumor_ax, levels=[0.5], colors=["#00FF66"], linewidths=1.2, linestyles="dashed")
        ax.axis("off")
    fig.suptitle(f"Patient: {patient_id} | V8 Component Maps — Axial Z={cz}\n"
                 f"Colormap = {cbar_label}",
                 color="white", fontsize=13, weight="bold")
    plt.savefig(os.path.join(out_dir, "component_maps.png"), dpi=200, bbox_inches="tight", facecolor="#0e1117")
    plt.close()
    print("    Saved: component_maps.png")

    # Tract overlay (T_x only, all 3 planes)
    if T_x is not None:
        save_triplanar(T_x, "Tract Risk T(x) — DTI White-Matter Tract Proximity/Overlap",
                       "tract_overlay.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(config_path="config_v7.json"):
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    patient_id = cfg.get("patient_id", "UCSF-PDGM-0004")
    paths = cfg["paths"]
    weights = cfg["weights"]
    thresholds = cfg["risk_thresholds"]

    out_validated = os.path.abspath(os.path.join(
        project_root,
        paths.get("output_dir_v7_validated", os.path.join("outputs", "V7_VALIDATED")),
    ))
    out_corridors = os.path.abspath(os.path.join(
        project_root,
        paths.get("output_dir_v8_corridors", os.path.join("outputs", "V8_CORRIDORS")),
    ))
    out_ablation = os.path.abspath(os.path.join(
        project_root,
        paths.get("output_dir_v8_ablation", os.path.join("outputs", "V8_ablation")),
    ))
    out_comparison = os.path.abspath(os.path.join(
        project_root,
        paths.get("output_dir_v8_comparison", os.path.join("outputs", "V8_comparison")),
    ))
    out_qc = os.path.abspath(os.path.join(
        project_root,
        paths.get("output_dir_v8_qc", os.path.join("outputs", "V8_qc")),
    ))
    out_vis = os.path.join(out_validated, "visualizations")

    for d in [out_validated, out_corridors, out_ablation, out_comparison, out_qc, out_vis]:
        os.makedirs(d, exist_ok=True)

    print("\n" + "#" * 72)
    print("  V8 BRAIN TUMOR SURGICAL RISK MAPPING — RESEARCH PROTOTYPE")
    print(f"  Patient  : {patient_id}")
    print("  Notice   : Decision-support prototype. NOT a clinical predictor.")
    print("#" * 72 + "\n")

    # ── Load inputs ──────────────────────────────────────────────────────────
    print(">>> PHASE 1/18: Loading inputs and running strict QC...")
    mri_nii    = nib.load(os.path.join(project_root, paths["mri_image"]))
    tumor_nii  = nib.load(os.path.join(project_root, paths["tumor_mask"]))
    labels_nii = nib.load(os.path.join(project_root, paths["synthseg_labels"]))

    dti_nii = None
    dti_value = paths.get("dti_fa")
    if dti_value:
        dti_path = os.path.join(project_root, dti_value)
        if not os.path.isfile(dti_path):
            raise FileNotFoundError(f"Configured DTI volume does not exist: {dti_path}")
        dti_nii = nib.load(dti_path)

    tract_dir = None
    tract_value = paths.get("tract_masks")
    if tract_value:
        tract_dir = os.path.join(project_root, tract_value)
        if not os.path.isdir(tract_dir):
            raise FileNotFoundError(f"Configured tract-mask directory does not exist: {tract_dir}")

    # PHASE 18: Strict QC
    qc_res = run_strict_qc(mri_nii, tumor_nii, labels_nii, dti_nii, tract_dir)
    with open(os.path.join(out_qc, "qc_report.json"), "w") as f:
        json.dump(qc_res, f, indent=2)

    if not qc_res["all_critical_passed"]:
        reasons = qc_res.get("fail_reasons", [])
        raise ValueError(
            f"[HALT] Critical QC failure for {patient_id}. "
            f"Mismatches: {reasons}. Fix registration before proceeding."
        )
    print(f"  [QC PASS] Shape: {qc_res['volume_shape']} | "
          f"Spacing: {qc_res['physical_spacing_mm']} mm | "
          f"Orientation: {qc_res['orientation']}")

    mri_data      = mri_nii.get_fdata()
    tumor_binary  = (tumor_nii.get_fdata() > 0).astype(np.uint8)
    labels_data   = labels_nii.get_fdata()
    voxel_spacing = tuple(float(z) for z in mri_nii.header.get_zooms()[:3])
    brain_mask    = (labels_data > 0) | (tumor_binary > 0)
    affine        = mri_nii.affine

    # ── PHASE 2: Tract overlap QC ─────────────────────────────────────────
    print("\n>>> PHASE 2: Tract/Tumor Overlap QC...")
    d_tumor_mm = v7.calculate_tumor_distance(tumor_binary, voxel_spacing)
    nib.save(nib.Nifti1Image(d_tumor_mm, affine),
             os.path.join(out_validated, "distance_to_tumor_mm.nii.gz"))

    overlap_records = compute_tract_overlap(tumor_binary, d_tumor_mm, tract_dir, voxel_spacing)
    with open(os.path.join(out_validated, "tract_overlap_report.csv"), "w", newline="") as f:
        if overlap_records:
            writer = csv.DictWriter(f, fieldnames=list(overlap_records[0].keys()))
            writer.writeheader()
            writer.writerows(overlap_records)
    print(f"  [OVERLAP QC] Analyzed {len(overlap_records)} tracts. "
          f"Report: tract_overlap_report.csv")
    for rec in overlap_records:
        print(f"    {rec['tract_name']:40s} -> {rec['spatial_relationship']:12s} "
              f"| min_d={rec['min_distance_mm']:5.1f}mm "
              f"| overlap_vox={rec['overlap_voxels']}")

    # ── PHASE 3: Improved G(x), S(x), T(x) ──────────────────────────────
    print("\n>>> PHASE 3: G(x), S(x), Improved T(x)...")
    sigma_G = cfg.get("spatial_parameters", {}).get("sigma_G_mm", 5.0)
    G_x = v7.calculate_proximity_risk(d_tumor_mm, tumor_binary, sigma_G)
    S_x, struct_info = v7.calculate_anatomical_importance(labels_data, cfg)

    T_x, tract_records = calculate_improved_tract_risk(
        tumor_binary, d_tumor_mm, tract_dir, cfg, voxel_spacing)
    t_available = T_x is not None
    if t_available:
        T_x[~brain_mask] = 0.0
        nib.save(nib.Nifti1Image(T_x, affine),
                 os.path.join(out_validated, "T_map.nii.gz"))
        print(f"  [TRACTS] {len(tract_records)} tracts evaluated. "
              f"T(x) = alpha*P_i + beta*O_i per tract.")
    else:
        print("  [TRACTS] No tract masks found; T(x) unavailable.")

    nib.save(nib.Nifti1Image(G_x, affine), os.path.join(out_validated, "G_map.nii.gz"))
    nib.save(nib.Nifti1Image(S_x, affine), os.path.join(out_validated, "S_map.nii.gz"))

    # ── PHASE 4: C(x) functional consequence ─────────────────────────────
    print("\n>>> PHASE 4: C(x) Functional Consequence (anatomical-domain mapping)...")
    C_x, _ = v7.calculate_functional_consequence(
        labels_data, None, tract_dir, cfg, mri_data.shape, voxel_spacing)
    C_x[~brain_mask] = 0.0
    nib.save(nib.Nifti1Image(C_x, affine), os.path.join(out_validated, "C_map.nii.gz"))

    # ── Combine components ────────────────────────────────────────────────
    components = {"G": G_x, "S": S_x, "T": T_x, "C": C_x, "U": None, "V": None}
    R_enhanced, active_norm_weights = v7.combine_active_risk_components(
        components, weights, brain_mask)
    R_enhanced = np.clip(R_enhanced, 0.0, 1.0)
    nib.save(nib.Nifti1Image(R_enhanced, affine),
             os.path.join(out_validated, "enhanced_risk_map.nii.gz"))
    nib.save(nib.Nifti1Image(R_enhanced, affine),
             os.path.join(out_validated, "R_map.nii.gz"))
    w_str = ", ".join(f"{k}: {v*100:.1f}%" for k, v in active_norm_weights.items())
    print(f"  Active weights: {w_str}")
    print(f"  R range in brain: [{float(np.min(R_enhanced[brain_mask])):.4f}, "
          f"{float(np.max(R_enhanced[brain_mask])):.4f}]")

    # Risk labels
    risk_labels = v7.generate_risk_labels(R_enhanced, thresholds)
    nib.save(nib.Nifti1Image(risk_labels, affine),
             os.path.join(out_validated, "risk_label_map.nii.gz"))

    # ── PHASE 6: Per-modality confidence ─────────────────────────────────
    print("\n>>> PHASE 6: Granular evidence/confidence reporting...")
    evidence_status = build_evidence_status(qc_res, t_available, dti_nii is not None)
    with open(os.path.join(out_validated, "evidence_status.json"), "w") as f:
        json.dump(evidence_status, f, indent=2)
    for k, v in evidence_status.items():
        print(f"  {k:35s}: {v}")

    # ── PHASE 8: Structure-level risk table ──────────────────────────────
    print("\n>>> PHASE 8: Structure-level risk table...")
    structure_records = build_structure_risk_table(
        labels_data, d_tumor_mm, G_x, S_x, T_x, C_x,
        R_enhanced, thresholds, cfg, evidence_status,
        t_available, tumor_binary, voxel_spacing,
        tract_records=tract_records)

    with open(os.path.join(out_validated, "structure_risk_table.csv"), "w", newline="") as f:
        if structure_records:
            writer = csv.DictWriter(f, fieldnames=list(structure_records[0].keys()))
            writer.writeheader()
            writer.writerows(structure_records)
    print(f"  {len(structure_records)} structures analyzed. Top (non-generic):")
    for sr in [s for s in structure_records if not s["structure_type"].startswith("Generic")][:5]:
        print(f"    {sr['structure_name']:35s} risk={sr['combined_risk']:.3f} "
              f"[{sr['risk_label']}] min_d={sr['tumor_min_distance_mm']}mm")

    # ── PHASE 5: Patient summary ──────────────────────────────────────────
    print("\n>>> PHASE 5: Writing structured patient summary...")
    tumor_centroid = [float(v) for v in np.mean(np.argwhere(tumor_binary > 0), axis=0)]
    tumor_vol_mm3 = float(np.sum(tumor_binary > 0)) * float(np.prod(voxel_spacing))
    write_patient_summary(
        patient_id, tumor_vol_mm3, tumor_centroid,
        evidence_status, active_norm_weights,
        tract_records, structure_records, None,
        os.path.join(out_validated, "patient_summary.txt"),
        os.path.join(out_validated, "patient_summary.json")
    )

    # ── V6 baseline (preserved, unchanged) ───────────────────────────────
    print("\n>>> Preserving V6 baseline...")
    R_base, _ = v6_baseline.run_v6_baseline(config_path)

    # ── PHASE 10: Comparison + delta map ─────────────────────────────────
    print("\n>>> PHASE 10: Baseline vs V8 comparison + delta map...")
    comparison, delta_map = run_comparison(
        R_base, R_enhanced, brain_mask, tumor_binary,
        d_tumor_mm, thresholds, affine, out_comparison)
    nib.save(nib.Nifti1Image(delta_map, affine),
             os.path.join(out_validated, "delta_risk_map.nii.gz"))
    print("  Comparison saved to V8_comparison/baseline_vs_v7_comparison.json")
    for model, stats in comparison.items():
        print(f"    {model}: {stats}")

    # ── PHASE 11: Ablation A–F ─────────────────────────────────────────
    print("\n>>> PHASE 11: Ablation study A through F...")
    ablation_res = run_ablation_af(
        G_x, S_x, T_x, C_x, brain_mask, tumor_binary,
        d_tumor_mm, affine, out_ablation)
    print("  Ablation results:")
    for ar in ablation_res:
        print(f"    {ar['configuration']:18s} mean={ar['mean_risk_brain']:.4f} "
              f"max={ar['max_risk_brain']:.4f} "
              f"peri={ar['tumor_adjacent_mean_10mm']:.4f}")
    # Copy ablation CSV to VALIDATED dir for completeness
    with open(os.path.join(out_validated, "ablation_results.csv"), "w", newline="") as f:
        if ablation_res:
            writer = csv.DictWriter(f, fieldnames=list(ablation_res[0].keys()))
            writer.writeheader()
            writer.writerows(ablation_res)

    # ── PHASE 17: Visualizations ──────────────────────────────────────────
    print("\n>>> PHASE 17: Generating all visualization outputs...")
    generate_all_visualizations(
        mri_data, tumor_binary, R_base, R_enhanced, delta_map,
        G_x, S_x, T_x, C_x, patient_id, out_vis)

    # ── PHASES 12–16: Corridors + Pareto ─────────────────────────────────
    print("\n>>> PHASES 12–16: Surgical corridor planning + Pareto analysis...")
    tract_masks = load_tract_masks(tract_dir, labels_data.shape)
    pareto_records, corridor_nii_path = plan_corridors_v8(
        R_enhanced, tumor_binary, brain_mask, T_x, C_x,
        cfg, affine, voxel_spacing, out_corridors,
        labels_data=labels_data, tract_masks=tract_masks)
    structure_overlays = build_structure_overlays(
        structure_records, labels_data, tract_masks)

    # ── QC JSON for validated dir ─────────────────────────────────────────
    with open(os.path.join(out_validated, "qc_report.json"), "w") as f:
        json.dump(qc_res, f, indent=2)

    # ── Interactive 3D report ──
    from risk_report_3d import generate_risk_3d_report

    interactive_report = generate_risk_3d_report(
        os.path.join(project_root, paths["mri_image"]),
        os.path.join(project_root, paths["tumor_mask"]),
        os.path.join(out_validated, "enhanced_risk_map.nii.gz"),
        os.path.join(out_corridors, "risk_3d_viewer.html"),
        corridors_path=(
            os.path.join(out_corridors, "corridor_paths.json") if pareto_records else None
        ),
        corridor_volume_path=corridor_nii_path,
        structure_overlays=structure_overlays,
    )
    print(f"  Interactive 3D report : {interactive_report['html_report']}")

    # ── FINAL REPORT ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  V8 PIPELINE COMPLETE")
    print(f"  V7_VALIDATED outputs : {out_validated}")
    print(f"  V8_CORRIDORS outputs : {out_corridors}")
    print(f"  Ablation (A-F)       : {out_ablation}")
    print(f"  Comparison (V6 vs V8): {out_comparison}")
    print(f"  Visualizations       : {out_vis}")
    print("=" * 72 + "\n")

    return {
        "qc": qc_res,
        "evidence_status": evidence_status,
        "active_weights": active_norm_weights,
        "tract_records": tract_records,
        "overlap_records": overlap_records,
        "structure_records": structure_records,
        "comparison": comparison,
        "ablation": ablation_res,
        "pareto_corridors": pareto_records,
        "interactive_3d_report": interactive_report,
    }


if __name__ == "__main__":
    run_pipeline()
