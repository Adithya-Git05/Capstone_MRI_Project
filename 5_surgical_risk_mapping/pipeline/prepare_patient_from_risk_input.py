"""
prepare_patient_from_risk_input.py
===================================
Setup script to prepare one or more patients from Risk_input/ for the surgical risk-map pipeline.

What it does for each patient:
  1. Gzip-converts core structural scans into data/<patient>/
  2. Gzip-converts real tumor segmentation into data/<patient>/tumor_mask.nii.gz
  3. Creates zero-filled vessel_mask.nii.gz placeholder
  4. Copies eddy-corrected b-vectors -> bvec/<patient>_bvec
  5. Copies real UCSF-PDGM b-values -> bval/<patient>_bval
  6. Copies & gzips DTI FA map -> data/<patient>/dti_fa.nii.gz
  7. Creates tract_masks/ directory and high-FA white matter tract ROI
  8. Gzip-converts brain_segmentation -> synthseg/output/<patient>-synthseg.nii.gz (SynthSeg substitute)
  9. Updates config.json with patient paths
 10. Updates sample_manifest.txt with all prepared patients
"""
import os
import sys
import shutil
import json
import argparse
import pathlib
import numpy as np
import nibabel as nib

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RISK_INPUT   = PROJECT_ROOT / "Risk_input"
DATA_ROOT    = PROJECT_ROOT / "data"
BVAL_DIR     = PROJECT_ROOT / "bval"
BVEC_DIR     = PROJECT_ROOT / "bvec"
MANIFEST     = PROJECT_ROOT / "sample_manifest.txt"
CONFIG_PATH  = PROJECT_ROOT / "configs" / "config.json"


def gzip_nii(src: pathlib.Path, dst: pathlib.Path, label: str) -> None:
    """Load a .nii (or any NIfTI) and re-save as compressed .nii.gz."""
    if dst.exists():
        print(f"  [SKIP - already exists] {label}")
        return
    img = nib.load(str(src))
    nib.save(img, str(dst))
    print(f"  [gzip] {src.name}  ->  {dst.name}  ({dst.stat().st_size // 1024} KB)")


def step1_core_scans(src_dir: pathlib.Path, dst_dir: pathlib.Path, patient_id: str) -> None:
    print("\n[1] Copying & gzip-compressing core structural scans ...")
    mapping = {
        f"{patient_id}_T1.nii":    "t1.nii.gz",
        f"{patient_id}_T1c.nii":   "t1c.nii.gz",
        f"{patient_id}_T2.nii":    "t2.nii.gz",
        f"{patient_id}_FLAIR.nii": "flair.nii.gz",
        f"{patient_id}_DWI.nii":   "dwi.nii.gz",
    }
    for src_name, dst_name in mapping.items():
        src = src_dir / src_name
        if not src.exists():
            raise FileNotFoundError(f"Missing required scan: {src}")
        gzip_nii(src, dst_dir / dst_name, dst_name)


def step2_tumor_mask(src_dir: pathlib.Path, dst_dir: pathlib.Path, patient_id: str) -> None:
    print("\n[2] Copying & gzip-compressing tumor segmentation ...")
    src = src_dir / f"{patient_id}_tumor_segmentation.nii"
    if not src.exists():
        raise FileNotFoundError(f"Missing tumor segmentation: {src}")
    gzip_nii(src, dst_dir / "tumor_mask.nii.gz", "tumor_mask.nii.gz")


def step3_vessel_mask(dst_dir: pathlib.Path) -> None:
    print("\n[3] Creating zero-filled vessel_mask.nii.gz placeholder ...")
    dst = dst_dir / "vessel_mask.nii.gz"
    if dst.exists():
        print("  [SKIP - already exists] vessel_mask.nii.gz")
        return
    ref = nib.load(str(dst_dir / "t1.nii.gz"))
    zero = np.zeros(ref.shape[:3], dtype=np.uint8)
    nib.save(nib.Nifti1Image(zero, ref.affine), str(dst))
    print(f"  [zero mask] {dst.name}  ({dst.stat().st_size // 1024} KB)")


def step4_bvec(src_dir: pathlib.Path, patient_id: str) -> pathlib.Path:
    print("\n[4] Copying b-vectors ...")
    src_per_patient = src_dir / f"{patient_id}_DTI_eddy.eddy_rotated_bvecs"
    src_shared = BVEC_DIR / "UCSF-PDGM_DTI.bvec"
    dst = BVEC_DIR / f"{patient_id}_bvec"
    if dst.exists():
        print(f"  [SKIP - already exists] {dst.name}")
        return dst
    if src_per_patient.exists():
        shutil.copy(src_per_patient, dst)
        print(f"  [copy] {src_per_patient.name}  ->  {dst.name}")
    elif src_shared.exists():
        shutil.copy(src_shared, dst)
        print(f"  [copy] UCSF-PDGM_DTI.bvec  ->  {dst.name}  (shared)")
    else:
        raise FileNotFoundError(f"No bvec source found for {patient_id}")
    return dst


def step5_bval(bvec_path: pathlib.Path, patient_id: str) -> pathlib.Path:
    print("\n[5] Setting up b-values ...")
    dst = BVAL_DIR / f"{patient_id}_bval"
    real_bval = BVAL_DIR / "UCSF-PDGM_DTI.bval"
    if dst.exists():
        print(f"  [SKIP - already exists] {dst.name}")
        return dst
    if real_bval.exists():
        bvec = np.loadtxt(str(bvec_path))
        n_dirs_bvec = bvec.shape[1] if bvec.ndim == 2 else len(bvec)
        bvals_real = np.array(real_bval.read_text().split(), dtype=float)
        if len(bvals_real) == n_dirs_bvec:
            shutil.copy(real_bval, dst)
            print(f"  [copy] UCSF-PDGM_DTI.bval  ->  {dst.name}  ({len(bvals_real)} values, real data)")
            return dst
        else:
            print(f"  [WARN] Real bval has {len(bvals_real)} entries but bvec has {n_dirs_bvec} columns.")
    # Fallback
    bvec = np.loadtxt(str(bvec_path))
    n_dirs = bvec.shape[1] if bvec.ndim == 2 else len(bvec)
    bvals = np.array([0] + [2000] * (n_dirs - 1))
    np.savetxt(str(dst), bvals.reshape(1, -1), fmt="%d")
    print(f"  [synthetic] {n_dirs} entries (b0=0, b=2000)  ->  {dst.name}")
    return dst


def step6_dti_fa(src_dir: pathlib.Path, dst_dir: pathlib.Path, patient_id: str) -> None:
    print("\n[6] Setting up DTI FA map ...")
    src = src_dir / f"{patient_id}_DTI_eddy_FA.nii"
    dst = dst_dir / "dti_fa.nii.gz"
    if dst.exists():
        print(f"  [SKIP - already exists] {dst.name}")
        return
    if src.exists():
        gzip_nii(src, dst, dst.name)
    else:
        print(f"  [SKIP] {src.name} not found.")


def step7_tract_masks(dst_dir: pathlib.Path) -> None:
    print("\n[7] Setting up tract masks ...")
    tract_dir = dst_dir / "tract_masks"
    tract_dir.mkdir(parents=True, exist_ok=True)
    dst_tract = tract_dir / "white_matter_tracts.nii.gz"
    if dst_tract.exists():
        print(f"  [SKIP - already exists] {dst_tract.name}")
        return
    fa_path = dst_dir / "dti_fa.nii.gz"
    if fa_path.exists():
        fa_nii = nib.load(str(fa_path))
        fa_data = fa_nii.get_fdata()
        # Coherent white matter pathways have high fractional anisotropy (FA >= 0.40)
        wm_mask = (fa_data >= 0.40).astype(np.uint8)
        nib.save(nib.Nifti1Image(wm_mask, fa_nii.affine, header=fa_nii.header), str(dst_tract))
        print(f"  [tract mask] Generated {dst_tract.name} (FA >= 0.40, {int(np.sum(wm_mask))} voxels)")
    else:
        print("  [SKIP] No dti_fa.nii.gz found to extract tract mask.")


def step8_synthseg_substitute(src_dir: pathlib.Path, patient_id: str) -> None:
    print("\n[8] Setting up SynthSeg labels from PDGM brain segmentation ...")
    src = src_dir / f"{patient_id}_brain_segmentation.nii"
    synthseg_out = PROJECT_ROOT / "synthseg" / "output"
    synthseg_out.mkdir(parents=True, exist_ok=True)
    dst = synthseg_out / f"{patient_id}-synthseg.nii.gz"
    if dst.exists():
        print(f"  [SKIP - already exists] {dst.name}")
        return
    if not src.exists():
        print(f"  [WARN] brain_segmentation not found at {src}; SynthSeg step will need Docker.")
        return
    gzip_nii(src, dst, dst.name)
    print(f"  [substituted] PDGM brain_segmentation -> {dst.name}")


def step9_update_config(patient_id: str) -> None:
    print("\n[9] Updating config.json ...")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["patient_id"] = patient_id
    p = cfg["paths"]
    p["mri_image"]          = f"data/{patient_id}/t1.nii.gz"
    p["tumor_mask"]         = f"data/{patient_id}/tumor_mask.nii.gz"
    p["synthseg_labels"]    = f"synthseg/output/{patient_id}-synthseg.nii.gz"
    p["dti_fa"]             = f"data/{patient_id}/dti_fa.nii.gz"
    p["tract_masks"]        = f"data/{patient_id}/tract_masks"
    p["fmri_activation"]    = f"data/{patient_id}/fmri_activation.nii.gz"
    p["vessel_mask"]        = f"data/{patient_id}/vessel_mask.nii.gz"
    p["bvec"]               = f"bvec/{patient_id}_bvec"
    p["bval"]               = f"bval/{patient_id}_bval"
    p["dwi"]                = f"data/{patient_id}/dwi.nii.gz"
    p["t1c"]                = f"data/{patient_id}/t1c.nii.gz"
    p["t2"]                 = f"data/{patient_id}/t2.nii.gz"
    p["flair"]              = f"data/{patient_id}/flair.nii.gz"
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"  [config] {CONFIG_PATH} updated for {patient_id}")


def update_manifest(prepared_patients: list) -> None:
    existing = []
    if MANIFEST.exists():
        existing = [line.strip() for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    merged = sorted(list(set(existing + prepared_patients)))
    MANIFEST.write_text("\n".join(merged) + "\n", encoding="utf-8")
    print(f"\n[manifest] Updated {MANIFEST} with {len(merged)} patient(s): {merged}")


def prepare_single_patient(patient_id: str) -> bool:
    src_dir = RISK_INPUT / f"{patient_id}_nifti"
    if not src_dir.exists():
        src_dir = RISK_INPUT / patient_id
    if not src_dir.exists():
        print(f"[ERROR] Source directory not found in Risk_input for: {patient_id}")
        return False

    dst_dir = DATA_ROOT / patient_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    BVAL_DIR.mkdir(parents=True, exist_ok=True)
    BVEC_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  Preparing Patient: {patient_id}")
    print("=" * 60)

    step1_core_scans(src_dir, dst_dir, patient_id)
    step2_tumor_mask(src_dir, dst_dir, patient_id)
    step3_vessel_mask(dst_dir)
    bvec_path = step4_bvec(src_dir, patient_id)
    step5_bval(bvec_path, patient_id)
    step6_dti_fa(src_dir, dst_dir, patient_id)
    step7_tract_masks(dst_dir)
    step8_synthseg_substitute(src_dir, patient_id)
    step9_update_config(patient_id)

    print(f"\n[SUCCESS] Patient {patient_id} prepared successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare patient datasets from Risk_input.")
    parser.add_argument("--patient", type=str, default=None, help="Specific patient ID (e.g. UCSF-PDGM-0004)")
    parser.add_argument("--all", action="store_true", help="Prepare all patients found in Risk_input")
    args = parser.parse_args()

    # Discover patients in Risk_input
    discovered = []
    if RISK_INPUT.exists():
        for d in sorted(RISK_INPUT.iterdir()):
            if d.is_dir():
                pid = d.name.replace("_nifti", "")
                discovered.append(pid)

    if not discovered:
        print("[ERROR] No patient directories found in Risk_input.")
        sys.exit(1)

    target_patients = []
    if args.patient:
        target_patients = [args.patient]
    elif args.all or len(discovered) > 0:
        target_patients = discovered

    prepared = []
    for pid in target_patients:
        if prepare_single_patient(pid):
            prepared.append(pid)

    if prepared:
        update_manifest(prepared)
        print("\n" + "=" * 60)
        print("  All preparation steps complete!")
        print("  To run the pipeline:")
        print("    python pipeline/run_pipeline.py")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
