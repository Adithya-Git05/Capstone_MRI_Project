"""
setup_tract_masks.py
====================
Extracts distinct anatomical white-matter tract masks from patient DTI FA map
for patient UCSF-PDGM-0004:
  1. Corticospinal Tract (CST) - motor pathway
  2. Superior Longitudinal Fasciculus (SLF) - fronto-parietal association / language
  3. Optic Radiations (OR) - visual pathway
  4. Corpus Callosum (CC) - interhemispheric commissure
"""
import os
import pathlib
import numpy as np
import nibabel as nib

def extract_patient_tracts(patient_dir: str):
    p_path = pathlib.Path(patient_dir)
    fa_file = p_path / "dti_fa.nii.gz"
    seg_file = p_path / "tumor_mask.nii.gz"
    tract_dir = p_path / "tract_masks"
    tract_dir.mkdir(parents=True, exist_ok=True)

    if not fa_file.exists():
        print(f"[WARN] No DTI FA map found at {fa_file}")
        return

    fa_nii = nib.load(str(fa_file))
    fa = fa_nii.get_fdata()
    affine = fa_nii.affine
    header = fa_nii.header

    # Coordinate grids
    X, Y, Z = np.ogrid[:fa.shape[0], :fa.shape[1], :fa.shape[2]]
    brain = fa > 0.05

    tracts = {}

    # 1. Corticospinal Tract (CST): descending motor projection through internal capsule
    cst_roi = (fa >= 0.38) & brain & (X >= 135) & (X <= 175) & (Y >= 115) & (Y <= 155) & (Z >= 45) & (Z <= 130)
    tracts["corticospinal_tract"] = cst_roi.astype(np.uint8)

    # 2. Superior Longitudinal Fasciculus (SLF): association bundle
    slf_roi = (fa >= 0.35) & brain & (X >= 145) & (X <= 190) & (Y >= 110) & (Y <= 175) & (Z >= 90) & (Z <= 135)
    tracts["superior_longitudinal_fasciculus"] = slf_roi.astype(np.uint8)

    # 3. Optic Radiations (OR): posterior projection
    opt_roi = (fa >= 0.35) & brain & (X >= 125) & (X <= 165) & (Y >= 45) & (Y <= 100) & (Z >= 60) & (Z <= 95)
    tracts["optic_radiations"] = opt_roi.astype(np.uint8)

    # 4. Corpus Callosum (CC): midline commissure
    cc_roi = (fa >= 0.45) & brain & (X >= 110) & (X <= 130) & (Y >= 75) & (Y <= 170) & (Z >= 75) & (Z <= 105)
    tracts["corpus_callosum"] = cc_roi.astype(np.uint8)

    for name, mask in tracts.items():
        out_path = tract_dir / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine, header), str(out_path))
        print(f"  [TRACT] Saved {name}.nii.gz ({int(np.sum(mask))} voxels)")

if __name__ == "__main__":
    extract_patient_tracts("data/UCSF-PDGM-0004")
