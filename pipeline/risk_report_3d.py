"""Generate a self-contained 3D MRI viewer with tumor, risk, corridors and structures."""

from __future__ import annotations

import base64
import gzip
import hashlib
import html as html_lib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion

try:
    from .viewer_assets import PAGE_TEMPLATE, SLICE_FIGURES, VIEWER_CSS, VIEWER_JS
except ImportError:
    from viewer_assets import PAGE_TEMPLATE, SLICE_FIGURES, VIEWER_CSS, VIEWER_JS

CORRIDOR_COLORS = ["#00e5ff", "#ffe600", "#7cff4d", "#ff7ad9", "#b794ff", "#ffa04d"]
STRUCTURE_COLORS = ["#4f9dff", "#2ec4b6", "#f4a261", "#c77dff", "#90be6d", "#e76f51"]
TUMOR_COLOR = "#ff3158"
MAX_SLICE_VOXELS = 16_000_000
MAX_STRUCTURE_POINTS = 1_500
NEAR_STRUCTURE_LIMIT_MM = 10.0


def _load_aligned_volume(
    path: Path,
    reference: nib.spatialimages.SpatialImage,
    label: str,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{label} volume not found: {path}")
    image = nib.load(str(path))
    if image.shape[:3] != reference.shape[:3]:
        raise ValueError(
            f"{label} shape {image.shape[:3]} does not match MRI shape {reference.shape[:3]}."
        )
    if not np.allclose(image.affine, reference.affine, atol=1e-4):
        raise ValueError(f"{label} affine does not match the MRI affine.")
    if len(image.shape) != 3:
        raise ValueError(f"{label} must be a 3D NIfTI volume, got shape {image.shape}.")
    values = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN or infinite values.")
    return values


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sampling_stride(shape: Sequence[int], max_points: int) -> int:
    if max_points < 1:
        raise ValueError("max_points must be a positive integer.")
    return max(1, math.ceil((math.prod(shape) / max_points) ** (1 / 3)))


def _world_grid(
    shape: Sequence[int],
    stride: int,
    affine: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [np.arange(0, int(length), stride, dtype=np.float32) for length in shape]
    voxel_grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    world_grid = nib.affines.apply_affine(affine, voxel_grid)
    return world_grid[..., 0], world_grid[..., 1], world_grid[..., 2]


def _canonical(array: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reorient to RAS+ so slice views have a fixed left/right/up orientation."""
    image = nib.Nifti1Image(np.ascontiguousarray(array), affine)
    canonical = nib.as_closest_canonical(image)
    return np.asarray(canonical.dataobj), canonical.affine


def _pack(volume: np.ndarray) -> str:
    raw = np.ascontiguousarray(volume, dtype=np.uint8).tobytes()
    return base64.b64encode(gzip.compress(raw, compresslevel=6, mtime=0)).decode("ascii")


def _point_array(value: Any, label: str, minimum: int = 1) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=float).reshape(-1, 3)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a list of [x, y, z] points.") from exc
    if len(points) < minimum or not np.isfinite(points).all():
        raise ValueError(f"{label} must contain finite [x, y, z] points.")
    return points


def _load_corridors(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Corridor file not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    corridors = document.get("corridors") if isinstance(document, dict) else None
    if not isinstance(corridors, list):
        raise ValueError("Corridor file must contain a 'corridors' list.")
    for corridor in corridors:
        corridor_id = corridor.get("corridor_id")
        if not isinstance(corridor_id, int) or not 1 <= corridor_id <= 255:
            raise ValueError("Each corridor needs an integer corridor_id from 1 to 255.")
        _point_array(corridor.get("path_world_mm"), "Corridor path_world_mm", minimum=2)
        for key in ("length_mm", "mean_risk", "max_risk"):
            if not math.isfinite(float(corridor[key])):
                raise ValueError(f"Corridor {key} must be finite.")
    target = document.get("target", {}).get("world_mm")
    if target is not None:
        _point_array(target, "Corridor target world_mm")
    return document


def _surface_points(mask: np.ndarray, affine: np.ndarray, limit: int) -> np.ndarray:
    surface = np.argwhere(mask & ~binary_erosion(mask, border_value=0))
    if len(surface) == 0:
        surface = np.argwhere(mask)
    if len(surface) > limit:
        surface = surface[np.linspace(0, len(surface) - 1, limit).astype(int)]
    return nib.affines.apply_affine(affine, surface)


def _near_text(structures: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for item in structures:
        distance = float(item["min_distance_mm"])
        where = "path crosses it" if distance == 0 else f"{distance:.0f} mm"
        parts.append(f"{item['name']} ({where})")
    return "; ".join(parts) if parts else f"None within {NEAR_STRUCTURE_LIMIT_MM:.0f} mm"


def _layer_html(layer: Dict[str, Any]) -> str:
    traces = ",".join(str(index) for index in layer["traces"])
    swatch = (
        f'<span class="swatch" style="background:{layer["color"]}"></span>'
        if layer.get("color") else ""
    )
    detail = f"<small>{html_lib.escape(layer['detail'])}</small>" if layer.get("detail") else ""
    checked = " checked" if layer["visible"] else ""
    return (
        f'<label class="layer"><input type="checkbox" data-key="{layer["key"]}" '
        f'data-traces="{traces}"{checked}>'
        f'<span>{swatch}{html_lib.escape(layer["label"])}{detail}</span></label>'
    )


def _layers_panel(layers: Sequence[Dict[str, Any]]) -> str:
    sections: List[str] = ["<h2 style=\"margin-top:0\">Layers</h2>"]
    current_group = None
    for layer in layers:
        if layer["group"] != current_group:
            current_group = layer["group"]
            sections.append(f"<h3>{html_lib.escape(current_group)}</h3>")
        sections.append(_layer_html(layer))
    return "".join(sections)


def _tables_html(
    corridors: Sequence[Dict[str, Any]],
    structures: Sequence[Dict[str, Any]],
) -> str:
    esc = html_lib.escape
    parts: List[str] = []
    if structures:
        rows = ""
        for item in structures:
            gap = (
                "Overlaps the tumor" if item["distance_mm"] == 0
                else "{:.1f} mm".format(item["distance_mm"])
            )
            rows += (
                "<tr>"
                f"<td>{esc(str(item['name']))}</td>"
                f"<td>{esc(str(item['kind']))}</td>"
                f"<td>{esc(str(item['domain']))}</td>"
                f"<td>{gap}</td>"
                f"<td>{esc(str(item['risk_label']).replace('_', ' ').title())}</td>"
                "</tr>"
            )
        parts.append(
            "<h2>Named structures near the tumor</h2>"
            "<p class=\"note\">Distance is the shortest gap between the structure and the supplied tumor "
            "annotation. Map category is an experimental research label.</p>"
            "<table><thead><tr><th>Structure</th><th>Type</th><th>Associated function</th>"
            "<th>Distance from tumor</th><th>Map category</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    if corridors:
        rows = "".join(
            "<tr>"
            f"<td>{esc(str(item['name']))}</td>"
            f"<td>{float(item['length_mm']):.1f} mm</td>"
            f"<td>{float(item['mean_risk']):.2f}</td>"
            f"<td>{float(item['max_risk']):.2f}</td>"
            f"<td>{esc(_near_text(item.get('nearby_structures', [])))}</td>"
            "</tr>"
            for item in corridors
        )
        parts.append(
            "<h2>Candidate approach corridors</h2>"
            "<p class=\"note\">Computer-generated straight-to-target trajectories from the brain surface to the "
            "tumor centre, routed to avoid high map scores. They are shown in the order produced by the pipeline "
            "and are not ranked or recommended. Map scores run from 0 (lowest) to 1 (highest) and are experimental.</p>"
            "<table><thead><tr><th>Approach</th><th>Path length</th><th>Average map score</th>"
            "<th>Peak map score</th><th>Named structures within 10 mm of the path</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    return "".join(parts)


def generate_risk_3d_report(
    mri_path: str,
    tumor_mask_path: str,
    risk_map_path: str,
    output_path: str,
    *,
    corridors_path: Optional[str] = None,
    corridor_volume_path: Optional[str] = None,
    structure_overlays: Optional[Sequence[Dict[str, Any]]] = None,
    max_points: int = 120_000,
) -> Dict[str, Any]:
    """Write an interactive HTML viewer from aligned volumes and optional overlays."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "The interactive 3D report requires Plotly. Install project requirements."
        ) from exc

    mri_file = Path(mri_path).expanduser().resolve()
    tumor_file = Path(tumor_mask_path).expanduser().resolve()
    risk_file = Path(risk_map_path).expanduser().resolve()
    html_file = Path(output_path).expanduser().resolve()
    if not mri_file.is_file():
        raise FileNotFoundError(f"MRI volume not found: {mri_file}")

    mri_image = nib.load(str(mri_file))
    if len(mri_image.shape) != 3:
        raise ValueError(f"MRI must be a 3D NIfTI volume, got shape {mri_image.shape}.")
    mri = np.asarray(mri_image.dataobj, dtype=np.float32)
    if not np.isfinite(mri).all():
        raise ValueError("MRI contains NaN or infinite values.")

    tumor = _load_aligned_volume(tumor_file, mri_image, "Tumor mask") > 0
    if not np.any(tumor):
        raise ValueError("Tumor mask is empty.")
    risk = _load_aligned_volume(risk_file, mri_image, "Surgical risk map")
    if np.any(risk < 0):
        raise ValueError("Surgical risk map contains negative values.")
    if np.any(risk > 1.00001):
        raise ValueError("Surgical risk map values must be in the normalized [0, 1] range.")

    corridor_document: Optional[Dict[str, Any]] = None
    corridors: List[Dict[str, Any]] = []
    corridors_file: Optional[Path] = None
    corridor_volume: Optional[np.ndarray] = None
    corridor_volume_file: Optional[Path] = None
    if corridors_path:
        corridors_file = Path(corridors_path).expanduser().resolve()
        corridor_document = _load_corridors(corridors_file)
        corridors = list(corridor_document["corridors"])
        if corridor_volume_path:
            corridor_volume_file = Path(corridor_volume_path).expanduser().resolve()
            corridor_volume = _load_aligned_volume(
                corridor_volume_file, mri_image, "Corridor volume")
            if np.any(corridor_volume < 0) or np.any(corridor_volume > 255):
                raise ValueError("Corridor volume labels must be integers from 0 to 255.")

    overlays = list(structure_overlays or [])
    if len(overlays) > 255:
        raise ValueError("At most 255 structure overlays can be displayed.")
    for overlay in overlays:
        mask = overlay.get("mask")
        if mask is None or tuple(np.shape(mask)) != tuple(mri.shape):
            raise ValueError(
                f"Structure overlay '{overlay.get('name')}' does not match the MRI shape."
            )

    stride = _sampling_stride(mri.shape, max_points)
    sampled = (slice(None, None, stride),) * 3
    x, y, z = _world_grid(mri.shape, stride, mri_image.affine)
    mri_sample = mri[sampled]
    risk_sample = risk[sampled]
    tumor_surface_world = _surface_points(tumor, mri_image.affine, 25_000)
    valid_mri = mri_sample != 0
    if not np.any(valid_mri):
        raise ValueError("MRI has no nonzero voxels to display.")
    low, high = np.percentile(mri_sample[valid_mri], [2, 98])
    if high <= low:
        raise ValueError("MRI nonzero voxels have no intensity range to display.")
    anatomy = np.clip((mri_sample - low) / (high - low), 0.0, 1.0)
    anatomy[~valid_mri] = 0.0
    risk_max = max(0.151, float(risk_sample.max()))

    figure = go.Figure()
    layers: List[Dict[str, Any]] = []

    def add_layer(key, label, group, traces, *, color=None, detail=None, visible=True):
        layers.append({
            "key": key, "label": label, "group": group, "traces": list(traces),
            "color": color, "detail": detail, "visible": visible,
        })

    first = len(figure.data)
    figure.add_trace(go.Volume(
        x=x.ravel(), y=y.ravel(), z=z.ravel(), value=anatomy.ravel(),
        isomin=0.12, isomax=1.0, opacity=0.055, surface_count=14,
        colorscale="Gray", showscale=False, showlegend=False,
        caps={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
        name="MRI anatomy",
        hovertemplate="MRI intensity: %{value:.2f}<extra></extra>",
    ))
    add_layer("anatomy", "MRI (patient anatomy)", "Patient", [first], detail="3D view only")

    first = len(figure.data)
    figure.add_trace(go.Volume(
        x=x.ravel(), y=y.ravel(), z=z.ravel(), value=risk_sample.ravel(),
        isomin=0.15, isomax=risk_max, opacity=0.12, surface_count=12,
        colorscale="Turbo", showlegend=False,
        colorbar={"title": "Risk map score", "x": 1.0, "y": 0.5, "len": 0.5},
        caps={"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
        name="Surgical risk",
        hovertemplate="Risk score: %{value:.3f}<extra></extra>",
    ))
    add_layer("risk", "Surgical-risk map (research)", "Patient", [first],
              detail="Warmer colors = higher experimental score")

    first = len(figure.data)
    figure.add_trace(go.Scatter3d(
        x=tumor_surface_world[:, 0], y=tumor_surface_world[:, 1], z=tumor_surface_world[:, 2],
        mode="markers", marker={"size": 2.5, "color": TUMOR_COLOR, "opacity": 1.0},
        name="Tumor", hoverinfo="skip", showlegend=False,
    ))
    add_layer("tumor", "Tumor (supplied annotation)", "Patient", [first], color=TUMOR_COLOR)

    corridor_layers: List[Dict[str, Any]] = []
    if corridors:
        for corridor in corridors:
            corridor_id = int(corridor["corridor_id"])
            color = CORRIDOR_COLORS[(corridor_id - 1) % len(CORRIDOR_COLORS)]
            points = _point_array(corridor["path_world_mm"], "Corridor path_world_mm", 2)
            name = html_lib.escape(str(corridor["name"]))
            hover = (
                f"<b>{name}</b><br>Length {float(corridor['length_mm']):.1f} mm"
                f"<br>Average map score {float(corridor['mean_risk']):.2f}, "
                f"peak {float(corridor['max_risk']):.2f}"
                f"<br>Nearby: {html_lib.escape(_near_text(corridor.get('nearby_structures', [])))}"
                "<extra></extra>"
            )
            first = len(figure.data)
            figure.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="lines",
                line={"width": 9, "color": color}, name=str(corridor["name"]),
                hovertemplate=hover, showlegend=False,
            ))
            figure.add_trace(go.Scatter3d(
                x=[points[0, 0]], y=[points[0, 1]], z=[points[0, 2]],
                mode="markers+text", text=[f"C{corridor_id}"], textposition="top center",
                textfont={"color": color, "size": 13},
                marker={"size": 6, "color": color, "symbol": "diamond"},
                name=f"{corridor['name']} entry", hovertemplate=hover, showlegend=False,
            ))
            add_layer(
                f"corridor-{corridor_id}", f"C{corridor_id}: {corridor['name']}", "Candidate corridors",
                [first, first + 1], color=color,
                detail=f"{float(corridor['length_mm']):.0f} mm, average map score {float(corridor['mean_risk']):.2f}",
            )
            corridor_layers.append({"id": corridor_id, "name": str(corridor["name"]), "color": color})
        target_world = (corridor_document or {}).get("target", {}).get("world_mm")
        if target_world is not None:
            target_point = _point_array(target_world, "Corridor target world_mm")[0]
            first = len(figure.data)
            figure.add_trace(go.Scatter3d(
                x=[target_point[0]], y=[target_point[1]], z=[target_point[2]], mode="markers",
                marker={"size": 7, "color": "#ffffff", "symbol": "cross"},
                name="Corridor target", hovertemplate="Corridor target (tumor centre)<extra></extra>",
                showlegend=False,
            ))
            add_layer("target", "Corridor target (tumor centre)", "Candidate corridors", [first],
                      color="#ffffff")

    structure_layers: List[Dict[str, Any]] = []
    structure_rows: List[Dict[str, Any]] = []
    structure_volume: Optional[np.ndarray] = None
    for number, overlay in enumerate(overlays, 1):
        mask = np.asarray(overlay["mask"]) > 0
        if not np.any(mask):
            continue
        color = STRUCTURE_COLORS[(number - 1) % len(STRUCTURE_COLORS)]
        points = _surface_points(mask, mri_image.affine, MAX_STRUCTURE_POINTS)
        distance = float(overlay.get("distance_mm", 0.0))
        gap = "overlaps the tumor" if distance == 0 else f"{distance:.1f} mm from the tumor"
        name = str(overlay["name"])
        first = len(figure.data)
        figure.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            marker={"size": 2, "color": color, "opacity": 0.4}, name=name,
            hovertemplate=f"<b>{html_lib.escape(name)}</b><br>{gap}<extra></extra>",
            showlegend=False, visible=False,
        ))
        add_layer(f"structure-{number}", name, "Named structures near the tumor", [first],
                  color=color, detail=gap, visible=False)
        structure_layers.append({"id": number, "name": name, "color": color})
        structure_rows.append({
            "name": name, "kind": str(overlay.get("kind", "")),
            "domain": str(overlay.get("domain", "")), "distance_mm": distance,
            "risk_label": str(overlay.get("risk_label", "")),
        })
        if structure_volume is None:
            structure_volume = np.zeros(mri.shape, dtype=np.uint8)
        structure_volume[mask & (structure_volume == 0)] = number

    case_title = mri_file.parent.name
    escaped_case_title = html_lib.escape(case_title)
    axis_style = {
        "showticklabels": False, "showgrid": True, "gridcolor": "#26324a",
        "backgroundcolor": "#10141c", "zerolinecolor": "#26324a", "showbackground": True,
    }
    figure.update_layout(
        scene={
            "xaxis": {**axis_style, "title": "Patient left to right"},
            "yaxis": {**axis_style, "title": "Back to front"},
            "zaxis": {**axis_style, "title": "Bottom to top"},
            "aspectmode": "data",
            "camera": {"projection": {"type": "orthographic"}},
        },
        paper_bgcolor="#10141c", plot_bgcolor="#10141c", font={"color": "#e8edf5"},
        showlegend=False, margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=760,
    )
    plot_html = figure.to_html(
        full_html=False, include_plotlyjs=True, div_id="viewer3d",
        config={"responsive": True, "displaylogo": False, "showSendToCloud": False},
    )

    # Slice views use the same grid orientation for every overlay (RAS+).
    mri_u8 = np.zeros(mri.shape, dtype=np.uint8)
    nonzero = mri != 0
    mri_u8[nonzero] = np.round(np.clip((mri[nonzero] - low) / (high - low), 0.0, 1.0) * 255)
    risk_u8 = np.round(np.clip(risk, 0.0, 1.0) * 255).astype(np.uint8)
    slice_stride = _sampling_stride(mri.shape, MAX_SLICE_VOXELS)
    cut = (slice(None, None, slice_stride),) * 3

    def to_slice_grid(volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        canonical, canonical_affine = _canonical(volume, mri_image.affine)
        return canonical[cut], canonical_affine @ np.diag([slice_stride] * 3 + [1.0])

    mri_grid, slice_affine = to_slice_grid(mri_u8)
    risk_grid, _ = to_slice_grid(risk_u8)
    tumor_canonical, _ = _canonical(tumor.astype(np.uint8), mri_image.affine)
    tumor_grid = tumor_canonical[cut]
    centre = np.argwhere(tumor_canonical > 0).mean(axis=0) / slice_stride
    centre = np.clip(np.round(centre), 0, np.array(mri_grid.shape) - 1).astype(int)
    corridor_grid = None
    if corridor_volume is not None:
        corridor_grid, _ = to_slice_grid(np.round(corridor_volume).astype(np.uint8))
    structure_grid = None
    if structure_volume is not None:
        structure_grid, _ = to_slice_grid(structure_volume)
    spacing = np.sqrt((slice_affine[:3, :3] ** 2).sum(axis=0))
    case_data = {
        "dims": [int(v) for v in mri_grid.shape],
        "spacing": [round(float(v), 4) for v in spacing],
        "affine": slice_affine.tolist(),
        "center": [int(v) for v in centre],
        "risk_range": [0.15, round(risk_max, 4)],
        "layers": [{"key": item["key"], "visible": item["visible"]} for item in layers],
        "corridors": corridor_layers,
        "structures": structure_layers,
        "volumes": {
            "mri": _pack(mri_grid),
            "risk": _pack(risk_grid),
            "tumor": _pack(tumor_grid),
            "corridors": _pack(corridor_grid) if corridor_grid is not None else None,
            "structures": _pack(structure_grid) if structure_grid is not None else None,
        },
    }
    case_json = json.dumps(case_data, separators=(",", ":")).replace("<", "\\u003c")

    disclaimer = (
        "Research visualization only. The surgical-risk map is calculated by a spatial risk "
        "pipeline from the supplied tumor annotation and anatomical labels. Its scores and "
        "categories are not clinically validated and do not predict outcomes."
    )
    if corridors:
        disclaimer += (
            " Candidate corridors are computer-generated trajectories, not a surgical plan; "
            "the lowest composite score is not necessarily the safest path."
        )

    summary_items = ["the supplied tumor annotation", "a research surgical-risk map"]
    if corridors:
        summary_items.append("candidate approach corridors")
    if structure_rows:
        summary_items.append("nearby named structures")
    summary = (
        "The patient's MRI with " + ", ".join(summary_items[:-1]) + " and "
        + summary_items[-1] + "."
    )
    page_values = {
        "SUMMARY": html_lib.escape(summary),
        "CSS": VIEWER_CSS,
        "JS": VIEWER_JS,
        "TITLE": escaped_case_title,
        "PLOT": plot_html,
        "LAYERS": _layers_panel(layers),
        "SLICES": SLICE_FIGURES,
        "TABLES": _tables_html(corridors, structure_rows),
        "DISCLAIMER": html_lib.escape(disclaimer),
        "CASE_DATA": case_json,
    }
    page = re.sub(
        r"__([A-Z_]+)__",
        lambda match: page_values.get(match.group(1), match.group(0)),
        PAGE_TEMPLATE,
    )
    html_file.parent.mkdir(parents=True, exist_ok=True)
    html_file.write_text(page, encoding="utf-8")

    provenance = {
        "case_id": case_title,
        "coordinate_space": "NIfTI affine transformed to RAS+ millimeters",
        "shape": list(mri_image.shape[:3]),
        "affine": mri_image.affine.tolist(),
        "sampling_stride_voxels": stride,
        "sampled_grid_shape": list(x.shape),
        "slice_sampling_stride_voxels": slice_stride,
        "inputs": {
            "mri": {"file": mri_file.name, "sha256": _file_sha256(mri_file)},
            "tumor_mask": {"file": tumor_file.name, "sha256": _file_sha256(tumor_file)},
            "risk_map": {"file": risk_file.name, "sha256": _file_sha256(risk_file)},
            "corridors": (
                {"file": corridors_file.name, "sha256": _file_sha256(corridors_file)}
                if corridors_file is not None else None
            ),
            "corridor_volume": (
                {"file": corridor_volume_file.name, "sha256": _file_sha256(corridor_volume_file)}
                if corridor_volume_file is not None else None
            ),
        },
        "corridor_count": len(corridors),
        "named_structures": [row["name"] for row in structure_rows],
        "risk_map_is_neural_explanation": False,
        "disclaimer": disclaimer,
        "html_report": html_file.name,
    }
    metadata_file = html_file.with_suffix(".json")
    metadata_file.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def _resolve_project_path(value: str, project_root: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else project_root / path)


def main() -> None:
    import argparse

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build an interactive 3D report from aligned MRI and risk-map volumes."
    )
    parser.add_argument("--config", default="config_v7.json", help="Risk pipeline JSON config.")
    parser.add_argument(
        "--risk-map",
        default="outputs\\V7_VALIDATED\\enhanced_risk_map.nii.gz",
        help="Generated surgical risk-map NIfTI path.",
    )
    parser.add_argument(
        "--corridors", default="outputs\\V8_CORRIDORS\\corridor_paths.json",
        help="Corridor path JSON from the V8 pipeline (skipped if missing).",
    )
    parser.add_argument(
        "--corridor-volume", default="outputs\\V8_CORRIDORS\\candidate_corridors.nii.gz",
        help="Corridor label NIfTI from the V8 pipeline (skipped if missing).",
    )
    parser.add_argument(
        "--output", default="outputs\\V8_CORRIDORS\\risk_3d_viewer.html",
        help="Self-contained HTML report output path.",
    )
    parser.add_argument("--max-points", type=int, default=120_000)
    args = parser.parse_args()

    config_path = Path(_resolve_project_path(args.config, project_root))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    corridors_file = Path(_resolve_project_path(args.corridors, project_root))
    corridor_volume_file = Path(_resolve_project_path(args.corridor_volume, project_root))
    use_corridors = corridors_file.is_file() and corridor_volume_file.is_file()
    report = generate_risk_3d_report(
        _resolve_project_path(paths["mri_image"], project_root),
        _resolve_project_path(paths["tumor_mask"], project_root),
        _resolve_project_path(args.risk_map, project_root),
        _resolve_project_path(args.output, project_root),
        corridors_path=str(corridors_file) if use_corridors else None,
        corridor_volume_path=str(corridor_volume_file) if use_corridors else None,
        max_points=args.max_points,
    )
    print(f"3D report: {report['html_report']}")


if __name__ == "__main__":
    main()
