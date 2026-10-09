import os
import sys
import json
import subprocess

def run_synthseg(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    license_file = os.path.join(project_root, "license.txt")
    
    synthseg_in = os.path.join(project_root, "synthseg", "input")
    synthseg_out = os.path.join(project_root, "synthseg", "output")
    os.makedirs(synthseg_in, exist_ok=True)
    os.makedirs(synthseg_out, exist_ok=True)

    patient_id = config["patient_id"]
    mri_rel_path = config["paths"]["mri_image"]
    mri_abs_path = os.path.join(project_root, mri_rel_path)

    mri_fname = os.path.basename(mri_rel_path)
    synthseg_mri_target = os.path.join(synthseg_in, mri_fname)
    if not os.path.exists(synthseg_mri_target):
        import shutil
        shutil.copy2(mri_abs_path, synthseg_mri_target)
        print(f"Copied MRI to: {synthseg_mri_target}")

    out_fname = f"{patient_id}-synthseg.nii.gz"
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{license_file}:/license.txt:ro",
        "-v", f"{synthseg_in}:/input:ro",
        "-v", f"{synthseg_out}:/output",
        "-e", "FS_LICENSE=/license.txt",
        "freesurfer/freesurfer:8.2.0",
        "mri_synthseg",
        "--i", f"/input/{mri_fname}",
        "--o", f"/output/{out_fname}",
        "--autocrop",
        "--fast",
        "--threads", "4"
    ]

    print("=" * 60)
    print("STAGE 2: Running SynthSeg Anatomical Parcellation")
    print("=" * 60)
    print("Executing command:")
    print(" ".join(docker_cmd))

    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"SynthSeg execution failed with return code {result.returncode}")

    print(f"\n[PASS] SynthSeg output successfully generated at: {os.path.join(synthseg_out, out_fname)}")

if __name__ == "__main__":
    run_synthseg()
