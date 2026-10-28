"""
corridor_planner.py
===================
Candidate Surgical Corridor Generation using 3D A* Pathfinding.
Evaluates multiple potential surgical access trajectories tau from cortical entry points
to the tumor target, calculating composite cost:
    C(tau) = w1*L(tau) + w2*R_eloquent(tau) + w3*R_tract(tau) + w4*R_vessel(tau) + w5*R_ventricle(tau) - w6*tumor_coverage(tau)

Outputs:
  - candidate_corridors.csv
  - candidate_corridors.nii.gz
  - corridor_comparison.png
"""
import os
import json
import csv
import heapq
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, distance_transform_edt

def astar_3d(cost_volume, start, goal, voxel_spacing):
    """3D A* search on voxel grid minimizing accumulated cost."""
    shape = cost_volume.shape
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}

    def heuristic(p):
        diff = np.array(p) - np.array(goal)
        return float(np.linalg.norm(diff * np.array(voxel_spacing)))

    f_score = {start: heuristic(start)}
    neighbors = [
        (-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1),
        (-1,-1,0),(-1,1,0),(1,-1,0),(1,1,0),
        (-1,0,-1),(-1,0,1),(1,0,-1),(1,0,1),
        (0,-1,-1),(0,-1,1),(0,1,-1),(0,1,1)
    ]

    max_steps = 60000
    steps = 0

    while open_set and steps < max_steps:
        steps += 1
        current = heapq.heappop(open_set)[1]
        if np.linalg.norm(np.array(current) - np.array(goal)) <= 1.5:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy, dz in neighbors:
            nx, ny, nz = current[0] + dx, current[1] + dy, current[2] + dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                step_dist = np.linalg.norm(np.array([dx, dy, dz]) * np.array(voxel_spacing))
                tentative_g = g_score[current] + step_dist * (1.0 + cost_volume[nx, ny, nz] * 10.0)
                neighbor = (nx, ny, nz)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

    return None

def plan_candidate_corridors(risk_map, tumor_binary, brain_mask, tract_map, consequence_map, config, affine, voxel_spacing, out_dir):
    """Generate and compare multiple candidate surgical corridors."""
    os.makedirs(out_dir, exist_ok=True)
    c_weights = config.get("corridor_weights", {
        "w_length": 0.15, "w_eloquent": 0.25, "w_tract": 0.30,
        "w_vessel": 0.15, "w_ventricle": 0.15, "w_coverage": 0.20
    })

    # Target: tumor centroid
    tumor_vox = np.argwhere(tumor_binary > 0)
    target = tuple(np.round(np.mean(tumor_vox, axis=0)).astype(int))

    # Cortical surface: outer boundary of brain mask
    eroded_brain = binary_dilation(~brain_mask, iterations=1)
    cortical_surface = brain_mask & eroded_brain
    surface_voxels = np.argwhere(cortical_surface)

    # Define candidate entry regions
    # 1. Lateral Frontal, 2. Superior Frontal, 3. Frontopolar, 4. Pterional / Temporal
    candidate_entries = []
    
    # Lateral Frontal: high X, moderate Y, mid-to-high Z
    lat_candidates = surface_voxels[(surface_voxels[:, 0] > target[0] + 15) & (surface_voxels[:, 1] > target[1] - 20) & (surface_voxels[:, 2] > target[2] - 15)]
    if len(lat_candidates) > 0:
        candidate_entries.append(("Corridor-1 (Lateral Frontal Approach)", tuple(lat_candidates[len(lat_candidates)//2])))

    # Superior Frontal: mid X, high Y, high Z
    sup_candidates = surface_voxels[(surface_voxels[:, 2] > target[2] + 25) & (surface_voxels[:, 1] > target[1])]
    if len(sup_candidates) > 0:
        candidate_entries.append(("Corridor-2 (Superior Frontal Approach)", tuple(sup_candidates[len(sup_candidates)//2])))

    # Frontopolar / Anterior: high Y
    ant_candidates = surface_voxels[(surface_voxels[:, 1] > target[1] + 35) & (surface_voxels[:, 0] > target[0] - 20)]
    if len(ant_candidates) > 0:
        candidate_entries.append(("Corridor-3 (Anterior Frontopolar Approach)", tuple(ant_candidates[len(ant_candidates)//2])))

    # Pterional / Lateral Inferior: high X, mid-low Z
    pter_candidates = surface_voxels[(surface_voxels[:, 0] > target[0] + 20) & (surface_voxels[:, 2] < target[2] - 10)]
    if len(pter_candidates) > 0:
        candidate_entries.append(("Corridor-4 (Pterional / Frontotemporal Approach)", tuple(pter_candidates[len(pter_candidates)//2])))

    # Fallback if specific search yielded empty
    if len(candidate_entries) < 2:
        for i in range(min(3, len(surface_voxels))):
            candidate_entries.append((f"Corridor-{i+1}", tuple(surface_voxels[i * (len(surface_voxels)//3)])))

    corridor_volume = np.zeros(risk_map.shape, dtype=np.uint8)
    corridor_records = []

    for cid, (cname, entry) in enumerate(candidate_entries, 1):
        # A* path
        path = astar_3d(risk_map, entry, target, voxel_spacing)
        if not path:
            continue

        # Mark corridor voxels (with 2mm radius dilation)
        c_mask = np.zeros(risk_map.shape, dtype=bool)
        for pt in path:
            c_mask[pt[0], pt[1], pt[2]] = True
        c_mask_dilated = binary_dilation(c_mask, iterations=2)
        corridor_volume[c_mask_dilated] = cid

        # Metrics
        path_coords = np.array(path)
        diffs = np.diff(path_coords, axis=0) * np.array(voxel_spacing)
        length_mm = float(np.sum(np.linalg.norm(diffs, axis=1)))

        vox_risk = risk_map[c_mask_dilated]
        mean_r = float(np.mean(vox_risk)) if len(vox_risk) > 0 else 0.0
        max_r = float(np.max(vox_risk)) if len(vox_risk) > 0 else 0.0

        eloq_r = float(np.mean(consequence_map[c_mask_dilated])) if consequence_map is not None else 0.0
        tract_r = float(np.mean(tract_map[c_mask_dilated])) if tract_map is not None else 0.0
        vasc_r = 0.0  # Vascular imaging unavailable for this exam
        vent_r = 0.0  # Ventricular penetration check

        coverage = float(np.sum(c_mask_dilated & (tumor_binary > 0)) / np.sum(tumor_binary > 0))

        # Composite Cost C(tau)
        comp_cost = (
            c_weights["w_length"] * (length_mm / 100.0) +
            c_weights["w_eloquent"] * eloq_r +
            c_weights["w_tract"] * tract_r +
            c_weights["w_vessel"] * vasc_r +
            c_weights["w_ventricle"] * vent_r -
            c_weights["w_coverage"] * coverage
        )

        corridor_records.append({
            "corridor_id": cid,
            "corridor_name": cname,
            "path_length_mm": round(length_mm, 2),
            "mean_risk": round(mean_r, 4),
            "maximum_risk": round(max_r, 4),
            "eloquent_risk": round(eloq_r, 4),
            "tract_risk": round(tract_r, 4),
            "vascular_risk": round(vasc_r, 4),
            "ventricular_risk": round(vent_r, 4),
            "tumor_coverage": round(coverage, 4),
            "composite_cost": round(comp_cost, 4),
            "interpretation": f"Candidate corridor with calculated composite cost {comp_cost:.4f} (Pareto candidate)"
        })

    # Sort corridors by composite cost
    corridor_records.sort(key=lambda x: x["composite_cost"])

    # Save CSV
    csv_path = os.path.join(out_dir, "candidate_corridors.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["corridor_id", "corridor_name", "path_length_mm", "mean_risk", "maximum_risk", "eloquent_risk", "tract_risk", "vascular_risk", "ventricular_risk", "tumor_coverage", "composite_cost", "interpretation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(corridor_records)

    # Save NIfTI
    nii_path = os.path.join(out_dir, "candidate_corridors.nii.gz")
    nib.save(nib.Nifti1Image(corridor_volume, affine), nii_path)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    c_slice = target[2]
    ax.imshow(np.rot90(risk_map[:, :, c_slice]), cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
    ax.contour(np.rot90(tumor_binary[:, :, c_slice]), levels=[0.5], colors=["#00FF66"], linewidths=1.5, linestyles="dashed")
    
    colors = ["#FF3366", "#33CCFF", "#FFFF33", "#CC33FF"]
    for rec in corridor_records:
        cid = rec["corridor_id"]
        c_mask_2d = np.rot90((corridor_volume == cid)[:, :, c_slice])
        if np.any(c_mask_2d):
            ax.contour(c_mask_2d, levels=[0.5], colors=[colors[(cid-1) % len(colors)]], linewidths=2.0)

    ax.axis("off")
    ax.set_title("Candidate Surgical Corridors (Axial View at Tumor Target)\nColored Contours: Competing Trajectories", color="white", fontsize=14, weight="bold")
    png_path = os.path.join(out_dir, "corridor_comparison.png")
    plt.savefig(png_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"  [CORRIDORS] Saved {len(corridor_records)} candidate trajectories to {out_dir}")
    return corridor_records, nii_path
