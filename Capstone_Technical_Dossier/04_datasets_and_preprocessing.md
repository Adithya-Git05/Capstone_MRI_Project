# 04 — Datasets and preprocessing

## BRISC 2025

The local `data/BRISC/brisc2025` tree has 5,000 classification training images and 1,000 official-folder test images. Folder classes and counts are: glioma 1,147/254, meningioma 1,329/306, no_tumor 1,067/140 and pituitary 1,457/300 (train/test). Patient identifiers are unavailable. The classifier converts to grayscale, pads to 160x160 in the checked-in config, uses bilinear resizing, scales to [0,1], then normalizes to [-1,1]. Training augmentation uses horizontal flips (p=0.5), rotations up to 7 degrees, brightness delta 0.08 and contrast delta 0.08. Exact duplicate image hashes are grouped across train-derived partitions. The test-folder duplicate audit finds seven distinct hashes also in official training (nine train-file copies); training code excludes the matching train copies from fitting. This does not establish patient-level independence or rule out near-duplicates.

The segmentation tree contains 3,933 paired train images/masks and 860 test pairs. The local inventory totals 4,793 pairs, while upstream README text says 6,000 samples; this discrepancy is documented and not reconciled. Images are grayscale JPG, masks grayscale PNG, matched by stem. Raw antialiased masks are thresholded at 128; images use bilinear and masks nearest-neighbor resizing to 256x256 for attention U-Net runs. The attention run has sample-level train/validation split (3,343/590); patient-level grouping cannot be established.

## Local BraTS-format 3D subset

The local directory has 271 case folders and 1,355 NIfTI files (four MRI modalities plus segmentation). All checked files are 182x218x182, 1 mm spacing and LAS orientation; the loader reorients to RAS and performs nonzero-voxel z-score normalization per modality. Input order is `[t2f,t1n,t1c,t2w]`. The active composite masks are independent and overlapping: WT=[1,2,3], TC=[1,3], ET=[3], RC=[4]. Raw label semantics and release provenance are provisional/unverified; do not present this as a confirmed BraTS challenge release ontology. The legacy `dataset.json` says 1,351 training cases, unlike 271 locally available cases.

The actual SegResNet run used 215 train and 56 grouped-validation cases, 108 training groups and 10 validation groups, grouped by folder-name prefix before the final hyphen. This is a filename-derived grouping, not a verified patient split; no official test split exists. The default config's `folds: 5` does not establish that five folds were run; the current engine's recorded run is a grouped train/validation split.

## Spatial-risk example and RAG corpus

The saved risk/corridor result concerns one example only. Its QC reports dimensions 240x240x155, spacing 1 mm and LPS orientation. Legacy outputs describe structural MRI, SynthSeg labels, tumor mask and DTI/FA-threshold atlas ROI tract masks. The public source/cohort and label provenance are not established in these artifacts. The integrated adapter instead reorients inputs to RAS and does not accept DTI/tract inputs.

The RAG audit records 40 papers and 1,454 chunks. Its metadata has 25 missing license values, all 25 evaluation questions are pending_annotation, and evidence-type labels require manual review. The included demo patient context is explicitly synthetic.

Detailed counts and provenance appear in `dataset_inventory.csv`; mapping details and code-derived SynthSeg label names are in `label_mapping.csv`.
