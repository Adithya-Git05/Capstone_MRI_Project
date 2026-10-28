"""
run_patient.py
==============
Unified Patient Execution & Multi-Patient Batch Orchestrator.

Enables running the complete pipeline for:
  1. A single specific patient:
     python pipeline/run_patient.py --patient UCSF-PDGM-0004
  2. Any prepared patient from data/ or Risk_input/:
     python pipeline/run_patient.py --patient UCSF-PDGM-0005
  3. Interactive selection:
     python pipeline/run_patient.py --interactive
  4. Full cohort batch run (e.g. 10 patients):
     python pipeline/run_patient.py --all

Guarantees:
  - Patient-Isolated Outputs: All results are written to outputs/<patient_id>/
    ensuring Patient A never overwrites Patient B.
  - Cohort Summary Matrix: Appends metrics (tumor volume, grade, best corridor, 
    risk reduction) into outputs/cohort_summary.csv.
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import nibabel as nib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
pipeline_path = os.path.join(project_root, 'pipeline')
sys.path.append(project_root)
sys.path.append(pipeline_path)

from prepare_patient_from_risk_input import prepare_single_patient, RISK_INPUT, DATA_ROOT

def discover_available_patients():
    """Finds all patients available in data/ and Risk_input/."""
    patients = set()
    if os.path.exists(DATA_ROOT):
        for d in os.listdir(DATA_ROOT):
            p_dir = os.path.join(DATA_ROOT, d)
            if os.path.isdir(p_dir) and (os.path.exists(os.path.join(p_dir, "tumor_mask.nii.gz")) or any(f.endswith("-seg.nii.gz") for f in os.listdir(p_dir))):
                patients.add(d)
                
    if os.path.exists(RISK_INPUT):
        for d in os.listdir(RISK_INPUT):
            if os.path.isdir(os.path.join(RISK_INPUT, d)) and d.endswith("_nifti"):
                patients.add(d.replace("_nifti", ""))
                
    return sorted(list(patients))

def create_patient_config(patient_id):
    """
    Creates a dedicated patient configuration dictionary with isolated output directories.
    """
    patient_data_dir = os.path.join(DATA_ROOT, patient_id)
    
    # Auto-prepare if only in Risk_input
    if not os.path.exists(os.path.join(patient_data_dir, "tumor_mask.nii.gz")):
        risk_input_dir = os.path.join(RISK_INPUT, f"{patient_id}_nifti")
        if os.path.exists(risk_input_dir):
            print(f"  [AUTO-PREPARE] Preparing {patient_id} from Risk_input...")
            prepare_single_patient(patient_id)
            
    # Resolve image paths
    mri_file = os.path.join("data", patient_id, "t1.nii.gz")
    tumor_file = os.path.join("data", patient_id, "tumor_mask.nii.gz")
    synthseg_file = os.path.join("synthseg", "output", f"{patient_id}-synthseg.nii.gz")
    
    # Fallback to SynthSeg file inside risk_map if not in synthseg/output
    if not os.path.exists(os.path.join(project_root, synthseg_file)):
        alt_synth = os.path.join("risk_map", patient_id, "synthseg_labels.nii.gz")
        if os.path.exists(os.path.join(project_root, alt_synth)):
            synthseg_file = alt_synth
            
    # Check if files exist
    if not os.path.exists(os.path.join(project_root, mri_file)):
        # Check alternative filenames (e.g. BraTS)
        candidates = [f for f in os.listdir(patient_data_dir) if "t1" in f.lower() and f.endswith(".nii.gz")]
        if candidates:
            mri_file = os.path.join("data", patient_id, candidates[0])
            
    if not os.path.exists(os.path.join(project_root, tumor_file)):
        candidates = [f for f in os.listdir(patient_data_dir) if "seg" in f.lower() and f.endswith(".nii.gz")]
        if candidates:
            tumor_file = os.path.join("data", patient_id, candidates[0])
            
    # Load base template config
    base_cfg_path = os.path.join(project_root, "configs", "config_v7.json")
    if os.path.exists(base_cfg_path):
        with open(base_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    # Patient-isolated output directory structure
    patient_out_base = os.path.join("outputs", patient_id)
    
    cfg["prototype_version"] = "V8_PATIENT_SPECIFIC"
    cfg["patient_id"] = patient_id
    cfg["paths"] = {
        "mri_image": mri_file,
        "tumor_mask": tumor_file,
        "synthseg_labels": synthseg_file,
        "dti_fa": os.path.join("data", patient_id, "dti_fa.nii.gz"),
        "tract_masks": os.path.join("data", patient_id, "tract_masks"),
        "vessel_mask": os.path.join("data", patient_id, "vessel_mask.nii.gz"),
        "fmri_activation": os.path.join("data", patient_id, "fmri_activation.nii.gz"),
        "output_dir_v6": os.path.join(patient_out_base, "V6_baseline"),
        "output_dir_v7": os.path.join(patient_out_base, "V7_enhanced"),
        "output_dir_comp": os.path.join(patient_out_base, "comparison"),
        "output_dir_qc": os.path.join(patient_out_base, "qc"),
        "output_dir_corridors": os.path.join(patient_out_base, "V8_CORRIDORS"),
        "output_dir_ablation": os.path.join(patient_out_base, "ablation")
    }
    
    patient_cfg_file = os.path.join(project_root, f"config_{patient_id}.json")
    with open(patient_cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        
    return patient_cfg_file, cfg

def update_cohort_summary(patient_id, metrics):
    """Appends patient metrics to outputs/cohort_summary.csv."""
    cohort_csv = os.path.join(project_root, "outputs", "cohort_summary.csv")
    fieldnames = [
        "patient_id", "tumor_volume_cm3", "tumor_centroid", 
        "histological_grade", "best_corridor_id", "best_corridor_name", 
        "trajectory_length_mm", "mean_risk", "peak_hazard", 
        "baseline_mean_risk", "risk_reduction_pct"
    ]
    
    file_exists = os.path.exists(cohort_csv)
    rows = []
    if file_exists:
        with open(cohort_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["patient_id"] != patient_id:
                    rows.append(r)
                    
    rows.append(metrics)
    
    with open(cohort_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [COHORT] Updated cohort summary table: {cohort_csv}")

def run_single_patient(patient_id):
    """Executes the complete pipeline for a single patient."""
    print("\n" + "=" * 76)
    print(f"  PROCESSING PATIENT: {patient_id}")
    print("=" * 76)
    
    cfg_file, cfg = create_patient_config(patient_id)
    
    # Pre-check: Ensure anatomical parcellation exists
    synth_path = os.path.join(project_root, cfg["paths"]["synthseg_labels"])
    if not os.path.exists(synth_path):
        print(f"\n[PREREQUISITE NOTICE] Anatomical parcellation (SynthSeg) not found for '{patient_id}'.")
        print(f"  Looked at: {synth_path}")
        print(f"  -> The V7/V8 pipeline requires whole-brain anatomical parcellation.")
        print(f"  -> Fully pre-processed multimodal patients ready to run right now: 'UCSF-PDGM-0004' and 'UCSF-PDGM-0005'.")
        print(f"  -> For raw BraTS cases, please run SynthSeg 2.0 first to generate '{cfg['paths']['synthseg_labels']}'.")
        return False
        
    print(f"  Configuration generated: {cfg_file}")
    print(f"  Isolated output path: outputs/{patient_id}/")
    
    # 1. Run Master V8 Pipeline
    from run_v8_pipeline import run_pipeline as run_v8
    print("\n>>> [1/3] Running Master Surgical Risk & Corridor Engine...")
    res = run_v8(cfg_file)
    
    # 2. Run Grade-Adaptive Model
    print("\n>>> [2/3] Running Grade-Adaptive Tumor Modeling...")
    from grade_adaptive_risk import classify_tumor_phenotype, get_grade_adaptive_parameters, compute_grade_adaptive_risk_map, plot_adaptive_comparison
    
    patient_data_dir = os.path.join(project_root, "data", patient_id)
    mri_p = os.path.join(project_root, cfg["paths"]["mri_image"])
    tumor_p = os.path.join(project_root, cfg["paths"]["tumor_mask"])
    flair_p = os.path.join(patient_data_dir, "flair.nii.gz")
    
    clf_rep, tumor_data, zooms = classify_tumor_phenotype(mri_p, tumor_p, flair_p)
    adapt_params = get_grade_adaptive_parameters(clf_rep)
    
    # Ensure patient output directory exists
    v7_patient_out = os.path.join(project_root, cfg["paths"]["output_dir_v7"])
    os.makedirs(v7_patient_out, exist_ok=True)
    
    # 3. Multi-Objective Pareto Analysis
    print("\n>>> [3/3] Running Multi-Objective Pareto Frontier Analysis...")
    from pareto_frontier_analysis import load_data, compute_naive_baseline, evaluate_pareto_front, plot_pareto_frontier, generate_interactive_html
    
    # Update Pareto to use patient specific corridors
    corridor_dir = os.path.join(project_root, cfg["paths"]["output_dir_corridors"])
    corridor_csv = os.path.join(corridor_dir, "corridor_results.csv")
    
    # Fallback to general V8_CORRIDORS if not found in patient folder
    if not os.path.exists(corridor_csv):
        alt_corridor_csv = os.path.join(project_root, "outputs", "V8_CORRIDORS", "corridor_results.csv")
        if os.path.exists(alt_corridor_csv):
            corridor_csv = alt_corridor_csv
    
    corridors = []
    if os.path.exists(corridor_csv):
        with open(corridor_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                corridors.append({
                    "corridor_id": int(row["corridor_id"]),
                    "corridor_name": row["corridor_name"],
                    "entry_point": row["entry_point"],
                    "target_voxel": row["target_voxel"],
                    "length_mm": float(row["length_mm"]),
                    "mean_risk": float(row["mean_risk"]),
                    "max_risk": float(row["max_risk"]),
                    "tract_risk": float(row["tract_risk"]),
                    "eloquent_risk": float(row.get("eloquent_risk", 0.0)),
                    "tumor_coverage": float(row.get("tumor_coverage", 0.0)),
                    "composite_cost": float(row["composite_cost"])
                })
                
    risk_p = os.path.join(project_root, "outputs", "V7_VALIDATED", "R_enhanced.nii.gz")
    if not os.path.exists(risk_p):
        risk_p = os.path.join(project_root, cfg["paths"]["output_dir_v7"], "enhanced_risk_map.nii.gz")
        
    baseline = compute_naive_baseline(risk_p, None, tumor_p)
    all_corridors = evaluate_pareto_front(corridors, baseline)
    
    os.makedirs(corridor_dir, exist_ok=True)
    if all_corridors:
        plot_pareto_frontier(all_corridors, baseline, os.path.join(corridor_dir, "pareto_tradeoff_frontier.png"))
        generate_interactive_html(all_corridors, baseline, os.path.join(corridor_dir, "pareto_dashboard.html"))
    
    # Aggregate Metrics for Cohort Summary
    best_c = min(all_corridors, key=lambda x: x["mean_risk"]) if len(all_corridors) > 0 else None
    tumor_vox = np.argwhere(tumor_data > 0)
    centroid = [int(v) for v in np.round(np.mean(tumor_vox, axis=0))]
    
    reduction_pct = 0.0
    if baseline and best_c:
        reduction_pct = round(((baseline["mean_risk"] - best_c["mean_risk"]) / baseline["mean_risk"]) * 100.0, 1)
        
    cohort_metrics = {
        "patient_id": patient_id,
        "tumor_volume_cm3": clf_rep["tumor_volume_cm3"],
        "tumor_centroid": str(centroid),
        "histological_grade": clf_rep["predicted_grade"],
        "best_corridor_id": best_c["corridor_id"] if best_c else "N/A",
        "best_corridor_name": best_c["corridor_name"] if best_c else "N/A",
        "trajectory_length_mm": best_c["length_mm"] if best_c else "N/A",
        "mean_risk": best_c["mean_risk"] if best_c else "N/A",
        "peak_hazard": best_c["max_risk"] if best_c else "N/A",
        "baseline_mean_risk": baseline["mean_risk"] if baseline else "N/A",
        "risk_reduction_pct": f"-{reduction_pct}%" if baseline else "N/A"
    }
    update_cohort_summary(patient_id, cohort_metrics)
    
    print("\n" + "=" * 76)
    print(f"  COMPLETED PATIENT: {patient_id}")
    print(f"  All outputs isolated in: outputs/{patient_id}/")
    print("=" * 76 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Multi-Patient Surgical Risk & Corridor Orchestrator.")
    parser.add_argument("--patient", type=str, default=None, help="Target patient ID (e.g. UCSF-PDGM-0004)")
    parser.add_argument("--all", action="store_true", help="Batch run all discovered patients")
    parser.add_argument("--interactive", action="store_true", help="Prompt user interactively to select a patient")
    args = parser.parse_args()
    
    available = discover_available_patients()
    
    if args.interactive:
        print("\n=== AVAILABLE PATIENTS ===")
        for idx, p in enumerate(available, 1):
            print(f"  [{idx}] {p}")
        choice = input("\nEnter patient number or ID: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(available):
            target = available[int(choice) - 1]
        else:
            target = choice
        run_single_patient(target)
    elif args.all:
        print(f"\n[BATCH MODE] Running cohort pipeline for {len(available)} patients: {available}")
        for p in available:
            try:
                run_single_patient(p)
            except Exception as e:
                print(f"[ERROR] Failed processing patient {p}: {e}")
    elif args.patient:
        run_single_patient(args.patient)
    else:
        print("Defaulting to patient: UCSF-PDGM-0004")
        run_single_patient("UCSF-PDGM-0004")

if __name__ == "__main__":
    main()
