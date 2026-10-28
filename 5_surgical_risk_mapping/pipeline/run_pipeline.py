import os
import sys
import json
import argparse

# Ensure project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
pipeline_path = os.path.join(project_root, 'pipeline')
sys.path.append(project_root)
sys.path.append(pipeline_path)

# Import pipeline modules
import check_data
import run_synthseg
import build_risk_map
import visualize
import validation
import qc_report
import run_dti
import run_fmri
import run_vascular
import run_uncertainty
import compute_full_risk


def run_single_pipeline(config_path: str) -> bool:
    """Run the 9-stage pipeline for a given configuration file."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    patient_id = config.get("patient_id", "unknown")
    print("\n" + "#" * 70)
    print("  BRAIN TUMOR 3D SURGICAL RISK MAP PIPELINE (V1 PROTOTYPE)")
    print(f"  Patient : {patient_id}")
    print(f"  Config  : {config_path}")
    print("#" * 70 + "\n")

    # Step 1: Verify data
    print(">>> STEP 1: Verifying Patient Data and Spatial Alignment...")
    if not check_data.check_patient_data(config_path):
        print(f"[FAIL] Data verification failed for {patient_id}. Aborting.")
        return False

    # Step 2: SynthSeg
    labels_file = config["paths"]["synthseg_labels"]
    if not os.path.exists(labels_file):
        print(f"\n>>> STEP 2: SynthSeg labels not found at '{labels_file}'. Running SynthSeg...")
        run_synthseg.run_synthseg(config_path)
    else:
        print(f"\n>>> STEP 2: SynthSeg labels already present at '{labels_file}'. Skipping.")

    # Step 3: Build risk map
    print("\n>>> STEP 3: Computing Proximity, Assigning Importance & Building Risk Map...")
    build_risk_map.build_surgical_risk_map(config_path)

    # Step 4: Visualization
    print("\n>>> STEP 4: Generating Multi-Planar 3D Risk Map Visualization...")
    visualize.visualize_risk_map(config_path)

    # Validation & QC
    risk_path   = config['paths']['risk_map_nii']
    labels_path = config['paths']['synthseg_labels']
    png_path    = config['paths']['visualization_png']
    validation.validate_zero_overlap(risk_path, labels_path)
    validation.validate_visualization(png_path)
    qc_report.generate_qc_report(config_path)

    # Modality analyses
    print("\n>>> STEP 5: DTI Tract-Risk Overlap...")
    run_dti.run_dti(config_path)

    print("\n>>> STEP 6: fMRI Activation-Risk Overlap...")
    run_fmri.run_fmri(config_path)

    print("\n>>> STEP 7: Vascular-Risk Overlap...")
    run_vascular.run_vascular(config_path)

    print("\n>>> STEP 8: Uncertainty Map...")
    run_uncertainty.run_uncertainty(config_path)

    # Step 9: Combined risk map
    print("\n>>> STEP 9: Computing Combined Risk Map...")
    combined_risk_path = compute_full_risk.compute_combined_risk(config_path)
    print(f"[FULL RISK] Combined risk map saved to {combined_risk_path}")

    print("\n" + "=" * 70)
    print(f"  PIPELINE EXECUTION COMPLETE FOR {patient_id}!")
    print(f"  Risk Map NIfTI:       {config['paths']['risk_map_nii']}")
    print(f"  Structure Table CSV:  {config['paths']['structure_table_csv']}")
    print(f"  Visualization PNG:    {config['paths']['visualization_png']}")
    print("=" * 70 + "\n")
    return True


def ensure_patient_config(patient_id: str) -> str:
    """Generate or update a patient-specific config and return its path."""
    base_config_path = os.path.join(project_root, "configs", "config.json")
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["patient_id"] = patient_id
    out_dir = f"risk_map/{patient_id}"
    os.makedirs(os.path.join(project_root, out_dir), exist_ok=True)

    p = cfg["paths"]
    p["mri_image"]           = f"data/{patient_id}/t1.nii.gz"
    p["tumor_mask"]          = f"data/{patient_id}/tumor_mask.nii.gz"
    p["synthseg_labels"]     = f"synthseg/output/{patient_id}-synthseg.nii.gz"
    p["output_dir"]          = out_dir
    p["risk_map_nii"]        = f"{out_dir}/risk_map.nii.gz"
    p["structure_table_csv"] = f"{out_dir}/structure_table.csv"
    p["visualization_png"]   = f"{out_dir}/risk_map_slices.png"
    p["dti_fa"]              = f"data/{patient_id}/dti_fa.nii.gz"
    p["tract_masks"]         = f"data/{patient_id}/tract_masks"
    p["fmri_activation"]     = f"data/{patient_id}/fmri_activation.nii.gz"
    p["vessel_mask"]         = f"data/{patient_id}/vessel_mask.nii.gz"
    p["fmri_risk"]           = f"{out_dir}/fmri_risk.nii.gz"
    p["vascular_risk"]       = f"{out_dir}/vascular_risk.nii.gz"
    p["uncertainty_map"]     = f"{out_dir}/uncertainty_map.nii.gz"
    p["bvec"]                = f"bvec/{patient_id}_bvec"
    p["bval"]                = f"bval/{patient_id}_bval"
    p["dwi"]                 = f"data/{patient_id}/dwi.nii.gz"
    p["t1c"]                 = f"data/{patient_id}/t1c.nii.gz"
    p["t2"]                  = f"data/{patient_id}/t2.nii.gz"
    p["flair"]               = f"data/{patient_id}/flair.nii.gz"

    patient_cfg_path = os.path.join(project_root, f"config_{patient_id}.json")
    with open(patient_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return patient_cfg_path


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor 3D Surgical Risk Map Pipeline")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a specific JSON config file"
    )
    parser.add_argument(
        "--patient",
        default=None,
        help="Run pipeline for a specific patient ID (e.g. UCSF-PDGM-0004)"
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Run pipeline in batch for all patients listed in a manifest file (e.g. sample_manifest.txt)"
    )
    args = parser.parse_args()

    if args.patient:
        patient_cfg = ensure_patient_config(args.patient)
        success = run_single_pipeline(patient_cfg)
        sys.exit(0 if success else 1)

    if args.manifest:
        manifest_path = args.manifest
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(project_root, manifest_path)
        if not os.path.exists(manifest_path):
            print(f"[ERROR] Manifest file not found: {manifest_path}")
            sys.exit(1)
        with open(manifest_path, "r", encoding="utf-8") as f:
            patients = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"[MANIFEST] Processing {len(patients)} patient(s): {patients}")
        for pid in patients:
            p_cfg = ensure_patient_config(pid)
            run_single_pipeline(p_cfg)
        sys.exit(0)

    # Default fallback: config.json
    default_config = args.config or os.path.join(project_root, "configs", "config.json")
    success = run_single_pipeline(default_config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
