"""
v7_risk_components.py
=====================
Modular implementation of the V7 Enhanced Surgical Risk Model components:
  - G(x): Geometric tumor proximity
  - S(x): Anatomical structure importance
  - T(x): DTI white-matter tract risk (tumor-to-tract distance & importance)
  - C(x): Functional consequence severity
  - U(x): Boundary & segmentation uncertainty (if legitimate source exists)
  - V(x): Vascular proximity risk (if true vessel imaging exists)
"""
import os
import json
import csv
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from label_mapping import get_label_info

def run_quality_control(mri_nii, tumor_nii, labels_nii, dti_nii=None):
    """Verify shape, voxel spacing, affine, and orientation consistency."""
    qc = {
        "mri_shape": list(mri_nii.shape),
        "tumor_shape": list(tumor_nii.shape),
        "labels_shape": list(labels_nii.shape),
        "mri_spacing": [round(float(z), 3) for z in mri_nii.header.get_zooms()[:3]],
        "tumor_spacing": [round(float(z), 3) for z in tumor_nii.header.get_zooms()[:3]],
        "labels_spacing": [round(float(z), 3) for z in labels_nii.header.get_zooms()[:3]],
        "shapes_match": bool(mri_nii.shape == tumor_nii.shape == labels_nii.shape),
        "spacing_match": bool(np.allclose(mri_nii.header.get_zooms()[:3], tumor_nii.header.get_zooms()[:3], atol=1e-2)),
        "affine_match": bool(np.allclose(mri_nii.affine, tumor_nii.affine, atol=1e-2)),
        "dti_available": dti_nii is not None
    }
    if dti_nii is not None:
        qc["dti_shape"] = list(dti_nii.shape[:3])
        qc["dti_affine_match"] = bool(np.allclose(mri_nii.affine, dti_nii.affine, atol=1e-2))
    qc["all_passed"] = bool(qc["shapes_match"] and qc["spacing_match"] and qc["affine_match"])
    return qc

def calculate_tumor_distance(tumor_binary, voxel_spacing):
    """Compute physical 3D Euclidean distance in mm from tumor boundary."""
    if np.sum(tumor_binary) == 0:
        raise ValueError("Tumor mask contains 0 positive voxels.")
    distance_map_mm = distance_transform_edt(tumor_binary == 0, sampling=voxel_spacing).astype(np.float32)
    return distance_map_mm

def calculate_proximity_risk(distance_mm, tumor_binary, sigma_G=5.0):
    """G(x) = exp(-d(x) / sigma_G), with 1.0 inside tumor."""
    G_x = np.exp(-distance_mm / sigma_G).astype(np.float32)
    G_x[tumor_binary > 0] = 1.0
    return G_x

def calculate_anatomical_importance(labels_data, config):
    """S(x): Normalized structure importance score in [0, 1]."""
    labels_clean = np.round(labels_data).astype(np.int32)
    S_x = np.zeros_like(labels_clean, dtype=np.float32)
    unique_lbls = np.unique(labels_clean)
    struct_info = {}
    for lbl in unique_lbls:
        lbl = int(lbl)
        name, tier, weight = get_label_info(lbl)
        mask = (labels_clean == lbl)
        S_x[mask] = weight
        struct_info[lbl] = {"name": name, "tier": tier, "weight": weight}
    return S_x, struct_info

def calculate_tract_risk(tumor_binary, distance_tumor_mm, tract_dir, config, voxel_spacing):
    """
    Component T: For each tract i:
      d_i = min physical distance to tumor
      P_i = exp(-d_i / sigma_T)
      T_i = P_i * tract_importance_i
      Voxel-wise T(x) evaluates actual tract geometry with buffer.
    """
    sigma_T = config.get("spatial_parameters", {}).get("sigma_T_mm", 10.0)
    tract_buffer = config.get("spatial_parameters", {}).get("tract_buffer_mm", 2.0)
    tract_defs = config.get("tract_definitions", {})

    T_x = np.zeros_like(distance_tumor_mm, dtype=np.float32)
    tract_records = []

    if not tract_dir or not os.path.isdir(tract_dir):
        return None, []

    tract_files = [f for f in sorted(os.listdir(tract_dir)) if f.lower().endswith(('.nii', '.nii.gz'))]
    if not tract_files:
        return None, []

    for tf in tract_files:
        tname = tf.replace(".nii.gz", "").replace(".nii", "")
        tpath = os.path.join(tract_dir, tf)
        t_nii = nib.load(tpath)
        t_data = t_nii.get_fdata()
        t_mask = t_data > 0
        if not np.any(t_mask):
            continue

        vox_vol_mm3 = np.prod(voxel_spacing)
        vol_mm3 = int(np.sum(t_mask)) * vox_vol_mm3

        d_i = float(np.min(distance_tumor_mm[t_mask]))
        d_mean = float(np.mean(distance_tumor_mm[t_mask]))
        P_i = float(np.exp(-d_i / sigma_T))

        t_def = tract_defs.get(tname, {
            "importance": 0.70,
            "functional_domain": "Association",
            "potential_consequence": "Potential pathway disruption"
        })
        imp_i = float(t_def.get("importance", 0.70))
        T_i = float(P_i * imp_i)

        # Spatially represent tract risk on voxel grid
        d_from_tract = distance_transform_edt(~t_mask, sampling=voxel_spacing)
        t_voxel_risk = T_i * np.exp(-d_from_tract / tract_buffer).astype(np.float32)
        T_x = np.maximum(T_x, t_voxel_risk)

        tract_records.append({
            "Tract_Name": tname,
            "Tract_Volume_mm3": round(vol_mm3, 1),
            "Min_Distance_to_Tumor_mm": round(d_i, 2),
            "Mean_Distance_to_Tumor_mm": round(d_mean, 2),
            "Tract_Proximity_Score": round(P_i, 4),
            "Tract_Importance": round(imp_i, 2),
            "Tract_Risk_Score": round(T_i, 4),
            "Functional_Domain": t_def.get("functional_domain", "Association"),
            "Potential_Consequence": t_def.get("potential_consequence", "Potential white-matter pathway impairment")
        })

    return T_x, tract_records

def calculate_functional_consequence(labels_data, tract_records, tract_dir, config, shape, voxel_spacing):
    """
    Component C: Functional consequence severity mapping.
    Maps voxels belonging to known functional structures or tracts to consequence severity.
    Outputs:
      C_x: continuous consequence severity score [0, 1]
      domain_map: categorical map of functional domain IDs
    """
    conseq_defs = config.get("functional_consequences", {})
    C_x = np.zeros(shape, dtype=np.float32)
    domain_map = np.zeros(shape, dtype=np.int32)
    labels_clean = np.round(labels_data).astype(np.int32)

    # Map anatomical structures to functional domains
    for domain_name, d_info in conseq_defs.items():
        did = d_info.get("domain_id", 0)
        sev = float(d_info.get("severity_score", 0.70))
        struct_names = d_info.get("associated_structures", [])
        for lbl in np.unique(labels_clean):
            name, _, _ = get_label_info(int(lbl))
            if name in struct_names:
                m = (labels_clean == lbl)
                C_x[m] = np.maximum(C_x[m], sev)
                domain_map[m] = did

    # Map DTI tracts to functional domains
    if tract_dir and os.path.isdir(tract_dir):
        for domain_name, d_info in conseq_defs.items():
            did = d_info.get("domain_id", 0)
            sev = float(d_info.get("severity_score", 0.70))
            associated_tracts = d_info.get("associated_tracts", [])
            for at in associated_tracts:
                tpath = os.path.join(tract_dir, f"{at}.nii.gz")
                if not os.path.exists(tpath):
                    tpath = os.path.join(tract_dir, f"{at}.nii")
                if os.path.exists(tpath):
                    t_mask = nib.load(tpath).get_fdata() > 0
                    if t_mask.shape == shape:
                        C_x[t_mask] = np.maximum(C_x[t_mask], sev)
                        domain_map[t_mask] = did

    return C_x, domain_map

def combine_active_risk_components(components: dict, weights: dict, brain_mask: np.ndarray):
    """
    Combine active normalized components with dynamic re-normalization:
    R(x) = sum(w_i * comp_i) / sum(w_i) over available components.
    """
    active_weights = {}
    for comp_name, comp_arr in components.items():
        if comp_arr is not None:
            active_weights[comp_name] = weights.get(f"w{comp_name}", 0.0)

    total_weight = sum(active_weights.values())
    if total_weight == 0:
        total_weight = 1.0

    normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

    R_x = np.zeros(brain_mask.shape, dtype=np.float32)
    for comp_name, comp_arr in components.items():
        if comp_arr is not None:
            R_x += normalized_weights[comp_name] * comp_arr

    R_x[~brain_mask] = 0.0
    R_x = np.clip(R_x, 0.0, 1.0)
    return R_x, normalized_weights

def generate_risk_labels(risk_map: np.ndarray, thresholds: dict):
    """
    Discrete risk labels:
      1: LOW (0.00 <= R < 0.20)
      2: MODERATE (0.20 <= R < 0.40)
      3: ELEVATED (0.40 <= R < 0.60)
      4: HIGH (0.60 <= R < 0.80)
      5: VERY HIGH (0.80 <= R <= 1.00)
    """
    label_map = np.zeros_like(risk_map, dtype=np.uint8)
    label_map[(risk_map >= thresholds["LOW"][0]) & (risk_map < thresholds["LOW"][1])] = 1
    label_map[(risk_map >= thresholds["MODERATE"][0]) & (risk_map < thresholds["MODERATE"][1])] = 2
    label_map[(risk_map >= thresholds["ELEVATED"][0]) & (risk_map < thresholds["ELEVATED"][1])] = 3
    label_map[(risk_map >= thresholds["HIGH"][0]) & (risk_map < thresholds["HIGH"][1])] = 4
    label_map[(risk_map >= thresholds["VERY_HIGH"][0]) & (risk_map <= thresholds["VERY_HIGH"][1])] = 5
    return label_map
