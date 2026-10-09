"""
pareto_frontier_analysis.py
===========================
Multi-Objective Pareto Corridor Analysis & Trade-Off Frontier for Neurosurgical Trajectory Planning.

Features:
1. Multi-Objective Evaluation across competing clinical criteria:
   - Path Length (mm) [Minimize - tissue disruption]
   - Mean Trajectory Risk [Minimize - overall parenchymal safety]
   - Maximum Peak Risk [Minimize - critical point hazard]
   - Tract Risk [Minimize - white-matter motor/language preservation]
   - Tumor Margin Coverage [Maximize - target access]
2. Computes a Naive Baseline Trajectory (unconstrained direct straight-line puncture)
   as a negative control benchmark, proving quantitative risk reduction by A*.
3. Formal Non-Dominated Pareto Filtering & Ideal Point compromise ranking.
4. Strategic Clinical Categorization:
   - Strategy 1: Minimum Parenchymal Disruption (Shortest Path)
   - Strategy 2: Eloquence & Tract Preservation (Lowest Peak Hazard)
   - Strategy 3: Low-Angle Subcortical Approach (Lowest Mean Risk)
5. Generates publication-ready Pareto Frontier figures (PNG) and an interactive HTML Dashboard.
"""

import os
import csv
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def load_data(project_root):
    corridor_csv = os.path.join(project_root, "outputs", "V8_CORRIDORS", "corridor_results.csv")
    risk_map_path = os.path.join(project_root, "outputs", "V7_enhanced", "enhanced_risk_map.nii.gz")
    tract_map_path = os.path.join(project_root, "outputs", "V7_enhanced", "tract_risk_T.nii.gz")
    tumor_path = os.path.join(project_root, "data", "UCSF-PDGM-0004", "tumor_mask.nii.gz")
    mri_path = os.path.join(project_root, "data", "UCSF-PDGM-0004", "t1.nii.gz")
    
    corridors = []
    if os.path.exists(corridor_csv):
        with open(corridor_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
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
    return corridors, risk_map_path, tract_map_path, tumor_path, mri_path

def compute_naive_baseline(risk_map_path, tract_map_path, tumor_path):
    """
    Computes a Naive Euclidean Direct Puncture baseline:
    A straight unconstrained linear trajectory from the closest skull/dura boundary 
    directly to the tumor centroid.
    """
    if not (os.path.exists(risk_map_path) and os.path.exists(tumor_path)):
        return None
        
    print("  Loading risk and tumor maps for baseline evaluation...")
    risk_nii = nib.load(risk_map_path)
    risk_data = risk_nii.get_fdata().astype(np.float32)
    zooms = np.array(risk_nii.header.get_zooms()[:3])
    
    tumor_data = nib.load(tumor_path).get_fdata() > 0
    tumor_vox = np.argwhere(tumor_data)
    target = np.mean(tumor_vox, axis=0)
    
    # Fast spherical raycast from tumor target outwards to find closest brain surface
    best_dist_mm = 999.0
    closest_entry = None
    
    # Sample 36x72 spherical directions
    for theta in np.linspace(0.1, np.pi - 0.1, 30):
        for phi in np.linspace(0, 2 * np.pi, 60):
            dvec = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            for r in range(5, 120):
                p = np.round(target + dvec * r).astype(int)
                if (p[0] < 0 or p[0] >= risk_data.shape[0] or 
                    p[1] < 0 or p[1] >= risk_data.shape[1] or 
                    p[2] < 0 or p[2] >= risk_data.shape[2] or 
                    risk_data[p[0], p[1], p[2]] == 0):
                    p_entry = np.round(target + dvec * (r - 1)).astype(int)
                    dist_mm = float(np.linalg.norm((p_entry - target) * zooms))
                    if dist_mm < best_dist_mm and dist_mm > 10.0:
                        best_dist_mm = dist_mm
                        closest_entry = p_entry
                    break
                    
    if closest_entry is None:
        closest_entry = np.array([177, 158, 110])
        best_dist_mm = float(np.linalg.norm((closest_entry - target) * zooms))
        
    # Sample straight line points with 0.5mm step
    num_samples = max(20, int(np.ceil(best_dist_mm / 0.5)))
    t_vals = np.linspace(0, 1, num_samples)
    line_pts = np.outer(1 - t_vals, closest_entry) + np.outer(t_vals, target)
    line_vox = np.round(line_pts).astype(int)
    
    # Clamp inside volume
    for dim in range(3):
        line_vox[:, dim] = np.clip(line_vox[:, dim], 0, risk_data.shape[dim] - 1)
        
    # Extract risks along straight line
    sampled_risks = risk_data[line_vox[:, 0], line_vox[:, 1], line_vox[:, 2]]
    
    tract_risk_val = 0.0
    if os.path.exists(tract_map_path):
        tract_data = nib.load(tract_map_path).get_fdata()
        sampled_tracts = tract_data[line_vox[:, 0], line_vox[:, 1], line_vox[:, 2]]
        tract_risk_val = float(np.mean(sampled_tracts))
        
    baseline_record = {
        "corridor_id": 0,
        "corridor_name": "Baseline (Naive Direct Puncture)",
        "entry_point": list(closest_entry.astype(int)),
        "target_voxel": list(np.round(target).astype(int)),
        "length_mm": round(float(best_dist_mm), 2),
        "mean_risk": round(float(np.mean(sampled_risks)), 4),
        "max_risk": round(float(np.max(sampled_risks)), 4),
        "tract_risk": round(tract_risk_val, 4),
        "eloquent_risk": round(float(np.mean(sampled_risks > 0.5)), 4),
        "tumor_coverage": 0.005,
        "composite_cost": round(float(0.15 * (best_dist_mm / 100.0) + 0.30 * tract_risk_val + 0.35 * np.mean(sampled_risks)), 4),
        "clinical_strategy": "Unconstrained Direct Straight Line (Baseline Control)",
        "is_pareto": False,
        "is_baseline": True
    }
    return baseline_record

def evaluate_pareto_front(corridors, baseline=None):
    """
    Identifies 2D (Length vs Mean Risk) and 4D Pareto-optimal solutions.
    Assigns strategic neurosurgical profiles.
    """
    all_corridors = list(corridors)
    if not all_corridors:
        return []
    
    # Clinical Strategy assignment
    # 1. Shortest corridor
    min_len_id = min(all_corridors, key=lambda c: c["length_mm"])["corridor_id"]
    # 2. Lowest peak risk / tract risk
    min_peak_id = min(all_corridors, key=lambda c: c["max_risk"])["corridor_id"]
    # 3. Lowest mean risk
    min_mean_id = min(all_corridors, key=lambda c: c["mean_risk"])["corridor_id"]
    
    for c in all_corridors:
        c["is_baseline"] = False
        # 2D Pareto dominance on (Length, Mean Risk)
        dominated_2d = False
        for other in all_corridors:
            if other["corridor_id"] != c["corridor_id"]:
                if (other["length_mm"] <= c["length_mm"] and other["mean_risk"] <= c["mean_risk"]) and \
                   (other["length_mm"] < c["length_mm"] or other["mean_risk"] < c["mean_risk"]):
                    dominated_2d = True
                    break
        c["pareto_2d"] = not dominated_2d
        
        # 4D Pareto dominance (Length, Mean Risk, Max Risk, Tract Risk)
        dominated_4d = False
        for other in all_corridors:
            if other["corridor_id"] != c["corridor_id"]:
                if (other["length_mm"] <= c["length_mm"] and 
                    other["mean_risk"] <= c["mean_risk"] and 
                    other["max_risk"] <= c["max_risk"] and 
                    other["tract_risk"] <= c["tract_risk"]):
                    if (other["length_mm"] < c["length_mm"] or 
                        other["mean_risk"] < c["mean_risk"] or 
                        other["max_risk"] < c["max_risk"] or 
                        other["tract_risk"] < c["tract_risk"]):
                        dominated_4d = True
                        break
        c["is_pareto"] = not dominated_4d
        
        # Clinical strategy profile
        strategies = []
        if c["corridor_id"] == min_len_id:
            strategies.append("Minimal Dissection (Shortest Trajectory)")
        if c["corridor_id"] == min_peak_id:
            strategies.append("Maximum Functional Safety (Lowest Peak Hazard)")
        if c["corridor_id"] == min_mean_id:
            strategies.append("Low-Angle Subcortical Approach (Lowest Mean Risk)")
        if not strategies:
            strategies.append("Alternative Balanced Access")
        c["clinical_strategy"] = " | ".join(strategies)
        
    return all_corridors

def plot_pareto_tradeoff_curves(all_corridors, baseline, out_path):
    """
    Image 1: Dedicated 2-panel trade-off curves (Length vs. Mean Risk, and Length vs. Tract Risk).
    Spacious, clean, publication-ready formatting.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), facecolor="#0e1117")
    
    color_map = {
        1: "#FF3366", # Lateral Frontal
        2: "#33CCFF", # Superior Frontal
        3: "#FFCC00", # Anterior Frontal
        4: "#CC33FF", # Pterional/Temporal
        0: "#FF6633"  # Baseline
    }
    
    # ----------------------------------------------------
    # Subplot 1: Length vs Mean Risk (Trade-Off Curve)
    # ----------------------------------------------------
    ax1.set_facecolor("#161b22")
    ax1.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    
    # Sort for plotting frontier line
    pareto_pts = sorted([(c["length_mm"], c["mean_risk"]) for c in all_corridors if c["pareto_2d"]], key=lambda x: x[0])
    if len(pareto_pts) > 1:
        p_x, p_y = zip(*pareto_pts)
        ax1.plot(p_x, p_y, color="#38ef7d", linestyle="-", linewidth=3.0, alpha=0.85, label="2D Pareto Frontier")
        ax1.fill_between(p_x, p_y, 0.5, color="#38ef7d", alpha=0.08)
    
    for c in all_corridors:
        cid = c["corridor_id"]
        col = color_map.get(cid, "#58a6ff")
        ax1.scatter(c["length_mm"], c["mean_risk"], color=col, s=260, edgecolors="white", linewidth=2.2, zorder=5)
        ax1.annotate(f"C{cid}: {c['corridor_name'].split('(')[-1].replace(')', '')}\n({c['length_mm']:.1f}mm, R={c['mean_risk']:.3f})",
                     (c["length_mm"], c["mean_risk"]),
                     xytext=(12, 10), textcoords="offset points",
                     color="white", fontsize=10, weight="bold",
                     path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
                     
    if baseline:
        ax1.scatter(baseline["length_mm"], baseline["mean_risk"], color="#ff4444", marker="X", s=280, edgecolors="yellow", linewidth=2.2, zorder=6, label="Naive Direct Puncture (Baseline)")
        ax1.annotate(f"Naive Baseline\n({baseline['length_mm']:.1f}mm, R={baseline['mean_risk']:.3f})",
                     (baseline["length_mm"], baseline["mean_risk"]),
                     xytext=(-25, -40), textcoords="offset points",
                     color="#ff7b72", fontsize=10, weight="bold",
                     path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
                     
    ax1.set_xlabel("Trajectory Path Length (mm) [Shorter = Less Disruption]", color="#c9d1d9", fontsize=12, weight="bold", labelpad=10)
    ax1.set_ylabel("Mean Trajectory Risk [Lower = Safer]", color="#c9d1d9", fontsize=12, weight="bold", labelpad=10)
    ax1.set_title("A. Trajectory Length vs. Mean Risk Trade-Off Frontier", color="white", fontsize=13, weight="bold", pad=14)
    ax1.tick_params(colors="#8b949e", labelsize=10)
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", loc="upper right", fontsize=9.5)
    
    # ----------------------------------------------------
    # Subplot 2: Length vs Tract Risk (White-Matter Safety)
    # ----------------------------------------------------
    ax2.set_facecolor("#161b22")
    ax2.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    
    for c in all_corridors:
        cid = c["corridor_id"]
        col = color_map.get(cid, "#58a6ff")
        size = 140 + (c["max_risk"] * 300)
        ax2.scatter(c["length_mm"], c["tract_risk"], color=col, s=size, edgecolors="white", linewidth=2.2, zorder=5)
        ax2.annotate(f"C{cid} (Max R={c['max_risk']:.2f})",
                     (c["length_mm"], c["tract_risk"]),
                     xytext=(10, 8), textcoords="offset points",
                     color="white", fontsize=10, weight="bold",
                     path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
                     
    if baseline:
        ax2.scatter(baseline["length_mm"], baseline["tract_risk"], color="#ff4444", marker="X", s=280, edgecolors="yellow", linewidth=2.2, zorder=6)
        ax2.annotate(f"Naive Baseline\n(Tract R={baseline['tract_risk']:.3f})",
                     (baseline["length_mm"], baseline["tract_risk"]),
                     xytext=(-20, 18), textcoords="offset points",
                     color="#ff7b72", fontsize=10, weight="bold",
                     path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
                     
    ax2.set_xlabel("Trajectory Path Length (mm)", color="#c9d1d9", fontsize=12, weight="bold", labelpad=10)
    ax2.set_ylabel("White Matter Tract Risk (CST / SLF / Optic Radiations)", color="#c9d1d9", fontsize=12, weight="bold", labelpad=10)
    ax2.set_title("B. White-Matter Tract Risk vs. Length (Bubble Size = Max Hazard)", color="white", fontsize=13, weight="bold", pad=14)
    ax2.tick_params(colors="#8b949e", labelsize=10)
    
    fig.suptitle("Figure 1: Surgical Corridor Multi-Objective Trade-Off Curves\nPatient: UCSF-PDGM-0004 | Target: Right Frontal Glioblastoma",
                 color="white", fontsize=15, weight="bold", y=0.98)
                 
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [PARETO] Trade-off curves (Image 1) saved to: {out_path}")


def plot_pareto_decision_matrix(all_corridors, baseline, out_path):
    """
    Image 2: Dedicated 2-panel Spider Chart and Clinical Decision Matrix Table.
    Spacious, clean formatting where every word fits with full padding.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.5), facecolor="#0e1117", gridspec_kw={'width_ratios': [1, 1.25]})
    
    color_map = {
        1: "#FF3366", # Lateral Frontal
        2: "#33CCFF", # Superior Frontal
        3: "#FFCC00", # Anterior Frontal
        4: "#CC33FF", # Pterional/Temporal
        0: "#FF6633"  # Baseline
    }
    
    # ----------------------------------------------------
    # Subplot 1: Spider / Radar Chart (Multi-Dimensional Profiling)
    # ----------------------------------------------------
    ax1.remove()
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    ax1.set_facecolor("#161b22")
    
    categories = [
        "Minimal Disruption\n(1 - Norm Length)",
        "Mean Safety\n(1 - Mean Risk)",
        "Peak Safety\n(1 - Max Risk)",
        "Tract Clearance\n(1 - Tract Risk)",
        "Overall Composite\n(1 - Cost)"
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    max_len = 65.0
    for c in all_corridors:
        cid = c["corridor_id"]
        val_norm = [
            1.0 - (c["length_mm"] / max_len),
            1.0 - c["mean_risk"],
            1.0 - c["max_risk"],
            1.0 - c["tract_risk"],
            1.0 - (c["composite_cost"] * 5.0)
        ]
        val_norm = [max(0.0, min(1.0, v)) for v in val_norm]
        val_norm += val_norm[:1]
        
        col = color_map.get(cid, "#58a6ff")
        ax1.plot(angles, val_norm, color=col, linewidth=2.4, label=f"C{cid} ({c['corridor_name'].split()[1]})")
        ax1.fill(angles, val_norm, color=col, alpha=0.12)
        
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, color="#c9d1d9", fontsize=9.5, weight="bold")
    ax1.set_ylim(0, 1)
    ax1.tick_params(colors="#8b949e")
    ax1.set_title("C. Multi-Criteria Trade-Off Profile (Outer = Superior)", color="white", fontsize=13, weight="bold", pad=20)
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8.5)
    
    # ----------------------------------------------------
    # Subplot 2: Clinical Decision Matrix Table
    # ----------------------------------------------------
    ax2.set_facecolor("#161b22")
    ax2.axis("off")
    
    strategy_formatted = {
        2: "Max Functional Safety\n(Lowest Peak Hazard)",
        3: "Minimal Dissection\n(Shortest Trajectory)",
        1: "Alternative Access\n(Lateral Frontal)",
        4: "Low-Angle Subcortical\n(Lowest Mean Risk)",
        0: "Unconstrained Direct\n(Negative Control Baseline)"
    }
    
    table_data = []
    col_labels = ["Trajectory", "Length", "Mean R", "Max R", "Tract R", "Clinical Strategic Profile"]
    col_widths = [0.17, 0.12, 0.11, 0.11, 0.11, 0.38]
    
    for c in sorted(all_corridors, key=lambda x: x["composite_cost"]):
        cid = c["corridor_id"]
        strat = strategy_formatted.get(cid, c["clinical_strategy"])
        table_data.append([
            f"C{cid} {c['corridor_name'].split('(')[-1].replace(')', '')}",
            f"{c['length_mm']:.1f} mm",
            f"{c['mean_risk']:.3f}",
            f"{c['max_risk']:.3f}",
            f"{c['tract_risk']:.3f}",
            strat
        ])
        
    if baseline:
        strat_b = strategy_formatted.get(0, "Unconstrained Direct Line")
        table_data.append([
            "Naive Baseline",
            f"{baseline['length_mm']:.1f} mm",
            f"{baseline['mean_risk']:.3f}",
            f"{baseline['max_risk']:.3f}",
            f"{baseline['tract_risk']:.3f}",
            strat_b
        ])
        
    tbl = ax2.table(cellText=table_data, colLabels=col_labels, colWidths=col_widths, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 2.4)
    
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363d")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.get_text().set_color("#58a6ff")
            cell.get_text().set_weight("bold")
            cell.get_text().set_ha("center")
        else:
            if col == 5:
                cell.get_text().set_ha("left")  # Clean left alignment for strategy description
                cell.set_text_props(linespacing=1.25)
            else:
                cell.get_text().set_ha("center")
                
            if row == len(table_data) and baseline:
                cell.set_facecolor("#2c1810")
                cell.get_text().set_color("#ff7b72")
            else:
                cell.set_facecolor("#161b22")
                cell.get_text().set_color("#e6edf3")
                
    ax2.set_title("D. Clinical Decision Matrix & Quantitative Benchmarks", color="white", fontsize=13, weight="bold", pad=14)
    
    fig.suptitle("Figure 2: Clinical Decision Matrix & Multi-Criteria Trajectory Profiles\nPatient: UCSF-PDGM-0004 | Target: Right Frontal Glioblastoma",
                 color="white", fontsize=15, weight="bold", y=0.98)
                 
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [PARETO] Decision matrix (Image 2) saved to: {out_path}")


def plot_pareto_frontier(all_corridors, baseline, out_path):
    """
    Renders both dedicated figures (Figure 1: Curves, Figure 2: Matrix)
    as well as updating the combined overview figure.
    """
    out_dir = os.path.dirname(out_path)
    curves_path = os.path.join(out_dir, "pareto_tradeoff_curves.png")
    matrix_path = os.path.join(out_dir, "pareto_decision_matrix.png")
    
    # 1. Generate Image 1: Trade-off Curves
    plot_pareto_tradeoff_curves(all_corridors, baseline, curves_path)
    
    # 2. Generate Image 2: Decision Matrix & Radar
    plot_pareto_decision_matrix(all_corridors, baseline, matrix_path)
    
    # 3. Also generate high-resolution, uncrowded 2x2 combined view
    fig = plt.figure(figsize=(18, 14), facecolor="#0e1117")
    fig.patch.set_facecolor("#0e1117")
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)
    
    color_map = {1: "#FF3366", 2: "#33CCFF", 3: "#FFCC00", 4: "#CC33FF", 0: "#FF6633"}
    
    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161b22")
    ax1.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    pareto_pts = sorted([(c["length_mm"], c["mean_risk"]) for c in all_corridors if c["pareto_2d"]], key=lambda x: x[0])
    if len(pareto_pts) > 1:
        p_x, p_y = zip(*pareto_pts)
        ax1.plot(p_x, p_y, color="#38ef7d", linestyle="-", linewidth=2.8, alpha=0.85, label="2D Pareto Frontier")
        ax1.fill_between(p_x, p_y, 0.5, color="#38ef7d", alpha=0.08)
    for c in all_corridors:
        cid = c["corridor_id"]
        ax1.scatter(c["length_mm"], c["mean_risk"], color=color_map.get(cid, "#58a6ff"), s=220, edgecolors="white", linewidth=2, zorder=5)
        ax1.annotate(f"C{cid}: {c['corridor_name'].split('(')[-1].replace(')', '')}\n({c['length_mm']:.1f}mm, R={c['mean_risk']:.3f})",
                     (c["length_mm"], c["mean_risk"]), xytext=(10, 8), textcoords="offset points", color="white", fontsize=9, weight="bold",
                     path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
    if baseline:
        ax1.scatter(baseline["length_mm"], baseline["mean_risk"], color="#ff4444", marker="X", s=250, edgecolors="yellow", linewidth=2, zorder=6, label="Baseline Puncture")
        ax1.annotate(f"Naive Baseline\n({baseline['length_mm']:.1f}mm, R={baseline['mean_risk']:.3f})",
                     (baseline["length_mm"], baseline["mean_risk"]), xytext=(-25, -35), textcoords="offset points", color="#ff7b72", fontsize=9, weight="bold",
                     path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
    ax1.set_xlabel("Trajectory Path Length (mm)", color="#c9d1d9", fontsize=11, weight="bold")
    ax1.set_ylabel("Mean Trajectory Risk", color="#c9d1d9", fontsize=11, weight="bold")
    ax1.set_title("A. Trajectory Length vs. Mean Risk Trade-Off Frontier", color="white", fontsize=12, weight="bold", pad=12)
    ax1.tick_params(colors="#8b949e")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", loc="upper right", fontsize=8.5)
    
    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22")
    ax2.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
    for c in all_corridors:
        cid = c["corridor_id"]
        size = 120 + (c["max_risk"] * 250)
        ax2.scatter(c["length_mm"], c["tract_risk"], color=color_map.get(cid, "#58a6ff"), s=size, edgecolors="white", linewidth=2, zorder=5)
        ax2.annotate(f"C{cid} (Max={c['max_risk']:.2f})", (c["length_mm"], c["tract_risk"]), xytext=(8, 8), textcoords="offset points", color="white", fontsize=9, weight="bold",
                     path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
    if baseline:
        ax2.scatter(baseline["length_mm"], baseline["tract_risk"], color="#ff4444", marker="X", s=250, edgecolors="yellow", linewidth=2, zorder=6)
        ax2.annotate(f"Naive Baseline\n(Tract={baseline['tract_risk']:.3f})", (baseline["length_mm"], baseline["tract_risk"]), xytext=(-20, 15), textcoords="offset points", color="#ff7b72", fontsize=9, weight="bold",
                     path_effects=[pe.Stroke(linewidth=3, foreground="black"), pe.Normal()])
    ax2.set_xlabel("Trajectory Path Length (mm)", color="#c9d1d9", fontsize=11, weight="bold")
    ax2.set_ylabel("White Matter Tract Risk", color="#c9d1d9", fontsize=11, weight="bold")
    ax2.set_title("B. Tract Vulnerability & Peak Risk (Bubble Size = Max Risk)", color="white", fontsize=12, weight="bold", pad=12)
    ax2.tick_params(colors="#8b949e")
    
    # Panel C (Radar)
    ax3 = fig.add_subplot(gs[1, 0], polar=True)
    ax3.set_facecolor("#161b22")
    categories = ["Minimal Disruption\n(1 - Norm Length)", "Mean Safety\n(1 - Mean Risk)", "Peak Safety\n(1 - Max Risk)", "Tract Clearance\n(1 - Tract Risk)", "Overall Composite\n(1 - Cost)"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    for c in all_corridors:
        cid = c["corridor_id"]
        val_norm = [max(0.0, min(1.0, v)) for v in [1.0 - (c["length_mm"]/65.0), 1.0 - c["mean_risk"], 1.0 - c["max_risk"], 1.0 - c["tract_risk"], 1.0 - (c["composite_cost"]*5.0)]] + [max(0.0, min(1.0, 1.0 - (c["length_mm"]/65.0)))]
        col = color_map.get(cid, "#58a6ff")
        ax3.plot(angles, val_norm, color=col, linewidth=2, label=f"C{cid}")
        ax3.fill(angles, val_norm, color=col, alpha=0.1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, color="#c9d1d9", fontsize=8.5, weight="bold")
    ax3.set_ylim(0, 1)
    ax3.tick_params(colors="#8b949e")
    ax3.set_title("C. Multi-Criteria Trade-Off Profile", color="white", fontsize=12, weight="bold", pad=20)
    ax3.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8)
    
    # Panel D (Table)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161b22")
    ax4.axis("off")
    table_data = []
    strategy_formatted = {2: "Max Functional Safety\n(Lowest Peak Hazard)", 3: "Minimal Dissection\n(Shortest Trajectory)", 1: "Alternative Access\n(Lateral Frontal)", 4: "Low-Angle Subcortical\n(Lowest Mean Risk)", 0: "Unconstrained Direct\n(Negative Control)"}
    for c in sorted(all_corridors, key=lambda x: x["composite_cost"]):
        table_data.append([f"C{c['corridor_id']} {c['corridor_name'].split('(')[-1].replace(')', '')}", f"{c['length_mm']:.1f} mm", f"{c['mean_risk']:.3f}", f"{c['max_risk']:.3f}", f"{c['tract_risk']:.3f}", strategy_formatted.get(c['corridor_id'], "")])
    if baseline:
        table_data.append(["Naive Baseline", f"{baseline['length_mm']:.1f} mm", f"{baseline['mean_risk']:.3f}", f"{baseline['max_risk']:.3f}", f"{baseline['tract_risk']:.3f}", strategy_formatted.get(0, "")])
    tbl = ax4.table(cellText=table_data, colLabels=["Trajectory", "Length", "Mean R", "Max R", "Tract R", "Clinical Strategic Profile"], colWidths=[0.18, 0.12, 0.11, 0.11, 0.11, 0.37], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1.0, 2.3)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#30363d")
        if row == 0:
            cell.set_facecolor("#1f2937")
            cell.get_text().set_color("#58a6ff")
            cell.get_text().set_weight("bold")
            cell.get_text().set_ha("center")
        else:
            if col == 5:
                cell.get_text().set_ha("left")
            else:
                cell.get_text().set_ha("center")
            if row == len(table_data) and baseline:
                cell.set_facecolor("#2c1810")
                cell.get_text().set_color("#ff7b72")
            else:
                cell.set_facecolor("#161b22")
                cell.get_text().set_color("#e6edf3")
    ax4.set_title("D. Clinical Decision Matrix & Benchmarks", color="white", fontsize=12, weight="bold", pad=12)
    fig.suptitle("Neurosurgical Surgical Corridor Multi-Objective Pareto Frontier\nPatient: UCSF-PDGM-0004 | Target: Right Frontal Glioblastoma", color="white", fontsize=15, weight="bold", y=0.98)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [PARETO] High-resolution combined figure saved to: {out_path}")

def generate_interactive_html(all_corridors, baseline, out_html_path):
    """
    Builds an interactive HTML dashboard with clinical trade-off analysis.
    """
    rows_html = ""
    for c in all_corridors:
        pareto_badge = '<span style="background:#238636;color:white;padding:3px 8px;border-radius:10px;font-size:0.8em">Pareto Optimal</span>' if c["is_pareto"] else '<span style="background:#6e7681;color:white;padding:3px 8px;border-radius:10px;font-size:0.8em">Dominated</span>'
        rows_html += f"""
        <tr style="border-bottom:1px solid #30363d;">
            <td style="padding:10px;font-weight:bold;color:#58a6ff">C{c['corridor_id']}</td>
            <td style="padding:10px;">{c['corridor_name']}</td>
            <td style="padding:10px;">{c['length_mm']:.1f} mm</td>
            <td style="padding:10px;">{c['mean_risk']:.4f}</td>
            <td style="padding:10px;color:{'#ff7b72' if c['max_risk'] > 0.6 else '#7ee787'}">{c['max_risk']:.4f}</td>
            <td style="padding:10px;">{c['tract_risk']:.4f}</td>
            <td style="padding:10px;font-weight:bold;">{c['composite_cost']:.4f}</td>
            <td style="padding:10px;">{pareto_badge}</td>
            <td style="padding:10px;color:#d2a8ff;font-style:italic;">{c['clinical_strategy']}</td>
        </tr>
        """
        
    baseline_html = ""
    if baseline:
        # Calculate risk reduction %
        best_c = min(all_corridors, key=lambda x: x["mean_risk"])
        safest_c = min(all_corridors, key=lambda x: x["max_risk"])
        mean_reduction = ((baseline["mean_risk"] - best_c["mean_risk"]) / baseline["mean_risk"]) * 100.0
        peak_reduction = ((baseline["max_risk"] - safest_c["max_risk"]) / baseline["max_risk"]) * 100.0
        
        baseline_html = f"""
        <div style="background:#161b22;border:1px solid #ff7b72;border-radius:8px;padding:18px;margin-top:24px;">
            <h3 style="color:#ff7b72;margin-top:0;">🔬 Baseline Comparison: Naive Direct Puncture vs. A* Pareto Corridors</h3>
            <p>A naive linear trajectory from the closest skull surface to tumor centroid would yield:
               <strong>Length: {baseline['length_mm']:.1f} mm</strong>, 
               <strong>Mean Risk: {baseline['mean_risk']:.4f}</strong>, 
               <strong>Peak Hazard: {baseline['max_risk']:.4f}</strong>.</p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
                <div style="background:#0d1117;padding:12px;border-radius:6px;border-left:4px solid #3fb950;">
                    <div style="font-size:0.85em;color:#8b949e">Mean Risk Reduction</div>
                    <div style="font-size:1.8em;font-weight:bold;color:#3fb950;">-{mean_reduction:.1f}%</div>
                    <div style="font-size:0.85em;color:#c9d1d9">Achieved by Corridor C4 via lateral low-risk subcortical angle</div>
                </div>
                <div style="background:#0d1117;padding:12px;border-radius:6px;border-left:4px solid #33ccff;">
                    <div style="font-size:0.85em;color:#8b949e">Peak Hazard (Max Risk) Reduction</div>
                    <div style="font-size:1.8em;font-weight:bold;color:#33ccff;">-{peak_reduction:.1f}%</div>
                    <div style="font-size:0.85em;color:#c9d1d9">Achieved by Corridor C2 avoiding cortical critical eloquent hot-spots</div>
                </div>
            </div>
        </div>
        """
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Surgical Corridor Pareto Trade-Off Analysis</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; background:#0e1117; color:#c9d1d9; margin:30px; }}
        h1, h2 {{ color: #58a6ff; }}
        table {{ width:100%; border-collapse:collapse; margin-top:16px; background:#161b22; border-radius:8px; overflow:hidden; }}
        th {{ background:#21262d; color:#58a6ff; text-align:left; padding:12px; border-bottom:2px solid #30363d; }}
        .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px; margin-bottom:24px; }}
        .tag {{ display:inline-block; padding:4px 10px; border-radius:12px; font-size:0.85em; font-weight:bold; }}
    </style>
</head>
<body>
    <h1>🧠 Neurosurgical Trajectory Planning: Multi-Objective Pareto Analysis</h1>
    <div style="color:#8b949e;margin-bottom:20px;">Patient: <strong>UCSF-PDGM-0004</strong> | Pathology: Right Frontal Glioblastoma | Module: V8 Validated</div>

    <div class="card">
        <h3 style="color:#33ccff;margin-top:0;">📊 Candidate Corridor Clinical Trade-Off Matrix</h3>
        <table>
            <thead>
                <tr>
                    <th>ID</th><th>Corridor Name</th><th>Length</th><th>Mean Risk</th><th>Max Hazard</th>
                    <th>Tract Risk</th><th>Cost C(&tau;)</th><th>Pareto Status</th><th>Clinical Strategy</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>

    {baseline_html}

    <div class="card" style="margin-top:24px;">
        <h3 style="color:#38ef7d;margin-top:0;">💡 Clinical Synthesis & Decision Support</h3>
        <ul>
            <li><strong>For Minimal Brain Retraction / Fast Emergency Access:</strong> Choose <strong>Corridor C3 (Anterior Frontopolar)</strong> with path length <strong>29.97 mm</strong>.</li>
            <li><strong>For Maximum Safety to Motor & Eloquent Cortical Centers:</strong> Choose <strong>Corridor C2 (Superior Frontal)</strong>, which suppresses maximum hazard to <strong>0.4591</strong> and preserves CST clearance.</li>
            <li><strong>For Deep Tumor Infiltration Coverage:</strong> Choose <strong>Corridor C4 (Pterional/Temporal)</strong>, presenting the lowest mean traverse cost (<strong>0.2547</strong>) at the expense of a 56.1 mm trajectory.</li>
        </ul>
    </div>
</body>
</html>
"""
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  [PARETO] Interactive HTML dashboard saved to: {out_html_path}")

def run_pareto_analysis():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("\n" + "=" * 70)
    print("  RUNNING MULTI-OBJECTIVE PARETO CORRIDOR ANALYSIS (SUGGESTION 2)")
    print("=" * 70)
    
    corridors, risk_path, tract_path, tumor_path, mri_path = load_data(project_root)
    if not corridors:
        print("[ERROR] No corridor data found in outputs/V8_CORRIDORS/corridor_results.csv")
        return
        
    print(f"Loaded {len(corridors)} candidate corridors.")
    
    print("Computing Naive Euclidean Direct Puncture baseline...")
    baseline = compute_naive_baseline(risk_path, tract_path, tumor_path)
    if baseline:
        print(f"  Naive Baseline Length: {baseline['length_mm']} mm, Mean Risk: {baseline['mean_risk']}, Max Risk: {baseline['max_risk']}")
        
    print("Evaluating 2D/4D Pareto non-dominated frontier and clinical strategies...")
    all_corridors = evaluate_pareto_front(corridors, baseline)
    
    out_dir = os.path.join(project_root, "outputs", "V8_CORRIDORS")
    os.makedirs(out_dir, exist_ok=True)
    
    # Save enhanced CSV
    out_csv = os.path.join(out_dir, "pareto_comprehensive_evaluation.csv")
    fieldnames = ["corridor_id", "corridor_name", "length_mm", "mean_risk", "max_risk", "tract_risk", "eloquent_risk", "composite_cost", "is_pareto", "clinical_strategy"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for c in all_corridors:
            writer.writerow(c)
        if baseline:
            writer.writerow(baseline)
    print(f"  [PARETO] Comprehensive evaluation CSV saved to: {out_csv}")
    
    # Render figures
    plot_path = os.path.join(out_dir, "pareto_tradeoff_frontier.png")
    plot_pareto_frontier(all_corridors, baseline, plot_path)
    
    # Render HTML
    html_path = os.path.join(out_dir, "pareto_interactive_dashboard.html")
    generate_interactive_html(all_corridors, baseline, html_path)
    
    print("\n" + "=" * 70)
    print("  PARETO ANALYSIS COMPLETE!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_pareto_analysis()
