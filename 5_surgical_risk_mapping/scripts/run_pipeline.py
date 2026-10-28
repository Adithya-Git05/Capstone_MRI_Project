import os
import sys
import json

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_data import check_patient_data
from scripts.run_synthseg import run_synthseg
from scripts.build_risk_map import build_surgical_risk_map
from scripts.visualize import visualize_risk_map

def main():
    config_path = "configs/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    print("\n" + "#" * 70)
    print("  BRAIN TUMOR 3D SURGICAL RISK MAP PIPELINE (V1 PROTOTYPE)")
    print("#" * 70 + "\n")

    # Step 1: Stage 0 & 1 Verification
    print(">>> STEP 1: Verifying Patient Data and Spatial Alignment...")
    valid = check_patient_data(config_path)
    if not valid:
        print("[FAIL] Aborting pipeline: Data verification failed.")
        sys.exit(1)

    # Step 2: Stage 2 SynthSeg Parcellation
    labels_file = config["paths"]["synthseg_labels"]
    if not os.path.exists(labels_file):
        print(f"\n>>> STEP 2: SynthSeg labels not found at '{labels_file}'. Running SynthSeg 2.0...")
        run_synthseg(config_path)
    else:
        print(f"\n>>> STEP 2: SynthSeg labels already present at '{labels_file}'. Skipping execution.")

    # Step 3: Stages 3 - 9 Distance Transform & Risk Map Building
    print("\n>>> STEP 3: Computing Proximity, Assigning Importance & Building Risk Map...")
    build_surgical_risk_map(config_path)

    # Step 4: Stage 10 Visualization
    print("\n>>> STEP 4: Generating Multi-Planar 3D Risk Map Visualization...")
    visualize_risk_map(config_path)

    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETE!")
    print(f"  Risk Map NIfTI:       {config['paths']['risk_map_nii']}")
    print(f"  Structure Table CSV:  {config['paths']['structure_table_csv']}")
    print(f"  Visualization PNG:    {config['paths']['visualization_png']}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
