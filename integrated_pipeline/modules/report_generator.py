from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
import textwrap
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


DISCLAIMER = (
    "Research decision-support prototype only. Not a diagnosis or treatment recommendation. "
    "All outputs require review by qualified clinicians."
)


def generate_report(result: dict[str, Any]) -> bytes:
    """Create a PDF exclusively from structured results actually provided."""
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        lines = [
            f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            "Project: Brain Tumor Analysis",
            f"Study: {result.get('case_id', 'not provided')}",
            "Ground truth: not supplied; all segmentation masks are model predictions.",
            f"3D model: {result.get('model_3d', 'not run')}",
            f"2D segmentation model: {result.get('model_2d_segmentation', 'not run')}",
            f"2D classification model: {result.get('model_2d_classification', 'not run')}",
            f"Input dimensions: {result.get('input_shape', 'not available')}",
            f"Model-space dimensions: {result.get('output_shape', 'not available')}",
            f"Model-space orientation: {result.get('orientation', 'not available')}",
            f"Model-space voxel spacing (mm): {result.get('output_spacing_mm', 'not available')}",
            f"Input voxel-to-world affine: {result.get('input_affine', 'not available')}",
            f"Model-space voxel-to-world affine: {result.get('output_affine', 'not available')}",
            f"Selected axial slice index: {result.get('slice_index', 'not available')}",
            f"Inference device: {result.get('inference_device', 'not recorded')}",
            f"Software versions: {result.get('software_versions', {})}",
            f"Seed: {result.get('seed', 'not recorded')}",
            f"3D prediction threshold: {result.get('prediction_threshold', 'not recorded')}",
            f"3D region definitions: {result.get('region_definitions', {})}",
            (
                "SynthSeg provenance: user-supplied; label provenance is not independently verified."
                if result.get("stages", {})
                .get("surgical risk mapping", {})
                .get("status")
                in {"completed", "failed"}
                else "SynthSeg labels not supplied; V8 risk mapping was not run."
            ),
            f"Input modalities: {', '.join(result.get('modalities', []))}",
            "Resolved 3D configuration: "
            + json.dumps(result.get("model_configuration", {}), sort_keys=True, default=str),
            "",
            "Stages:",
        ]
        for name, stage in result.get("stages", {}).items():
            lines.append(f"- {name}: {stage.get('status', 'not run')}")
            for reason in stage.get("reasons", []):
                lines.append(f"    {reason}")
            if stage.get("note"):
                lines.append(f"    Note: {stage['note']}")
        classification = result.get("classification")
        if classification:
            lines.extend(
                [
                    "",
                    "Classification model output (not a diagnosis):",
                    f"Predicted class: {classification['predicted_class']}",
                    f"Class probabilities (model output): {classification['probabilities']}",
                ]
            )
        segmentation_metadata = result.get("segmentation2d_metadata")
        if segmentation_metadata:
            lines.extend(
                [
                    "",
                    "2D segmentation configuration:",
                    json.dumps(segmentation_metadata, sort_keys=True, default=str),
                ]
            )
        risk = result.get("risk_mapping")
        if risk:
            lines.extend(
                [
                    "",
                    "V8 risk-map provenance (not clinical injury probabilities):",
                    risk.get("note", "Research-prototype score."),
                    f"Active component weights: {risk.get('active_weights', {})}",
                    f"Evidence availability: {risk.get('evidence_status', {})}",
                    "Spatial QC: "
                    + json.dumps(risk.get("qc", {}), sort_keys=True, default=str),
                    f"Component maps available: {list(risk.get('component_maps', {}))}",
                ]
            )
            for corridor in risk.get("corridors", []):
                lines.append(
                    "Corridor output: "
                    + json.dumps(corridor, sort_keys=True, default=str)
                )
        lines.extend(["", DISCLAIMER])
        _add_text_pages(pdf, "Brain Tumor Analysis - MRI Research Report", lines)

        if result.get("slice_image") is not None:
            image = np.asarray(result["slice_image"])
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            axes[0].imshow(image, cmap="gray")
            region_masks = result.get("region_masks", {})
            if "WT" in region_masks:
                whole_tumor_slice = np.asarray(region_masks["WT"])[:, :, result["slice_index"]]
                axes[0].imshow(
                    np.ma.masked_where(whole_tumor_slice == 0, whole_tumor_slice),
                    cmap="autumn",
                    alpha=0.45,
                )
            axes[0].set_title(f"3D WT mask - axial RAS slice {result.get('slice_index')}")
            axes[0].axis("off")
            axes[1].imshow(image, cmap="gray")
            if result.get("segmentation2d") is not None:
                axes[1].imshow(
                    np.ma.masked_where(result["segmentation2d"] == 0, result["segmentation2d"]),
                    cmap="winter",
                    alpha=0.45,
                )
            axes[1].set_title("2D predicted mask overlay")
            axes[1].axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            if result.get("risk_mapping") is not None:
                risk_map = np.asarray(result["risk_mapping"]["risk_map"])
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.imshow(image, cmap="gray")
                risk_slice = risk_map[:, :, result["slice_index"]]
                ax.imshow(
                    np.ma.masked_where(risk_slice <= 0, risk_slice),
                    cmap="turbo",
                    alpha=0.55,
                    vmin=0,
                    vmax=1,
                )
                corridor_mask = result["risk_mapping"].get("corridor_mask")
                if corridor_mask is not None:
                    corridor_slice = (
                        np.asarray(corridor_mask)[:, :, result["slice_index"]] > 0
                    ).astype(np.uint8)
                    if corridor_slice.any():
                        ax.contour(
                            corridor_slice,
                            levels=[0.5],
                            colors=["cyan"],
                            linewidths=0.6,
                        )
                ax.set_title("V8 normalized composite score and candidate corridors")
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        rag = result.get("rag")
        if rag:
            answer = rag.get("answer", {})
            paragraphs = [
                f"Context ID: {rag.get('queried_case_id', 'not provided')}",
                f"Question: {rag.get('query', 'not recorded')}",
                "Context: structured outputs from this run; raw uploaded MRI was not transmitted.",
                "Summary: " + str(answer.get("summary", "No summary returned.")),
            ]
            paragraphs.extend(
                f"Evidence-supported consideration: {item}"
                for item in answer.get("evidence_supported_considerations", [])
            )
            paragraphs.extend(f"Uncertainty: {item}" for item in answer.get("uncertainties", []))
            if rag.get("evidence_status") == "insufficient_evidence":
                paragraphs.append(str(rag.get("evidence_warning", "No evidence was retrieved.")))
            retrieved = rag.get("retrieved_evidence", [])
            for citation in rag.get("citations", []):
                source = next(
                    (
                        evidence
                        for evidence in retrieved
                        if isinstance(evidence, dict)
                        and (
                            (
                                citation.get("chunk_id")
                                and citation.get("chunk_id") == evidence.get("chunk_id")
                            )
                            or (
                                citation.get("paper_id")
                                and citation.get("paper_id") == evidence.get("paper_id")
                            )
                        )
                    ),
                    {},
                )
                metadata = source.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                paragraphs.append(
                    "Citation: "
                    f"{citation.get('title') or source.get('title', 'Untitled')} "
                    f"({citation.get('paper_id') or source.get('paper_id', 'unknown')}); "
                    f"authors={metadata.get('authors', 'not provided')}; "
                    f"year={metadata.get('year', 'not provided')}; "
                    f"DOI={metadata.get('doi', 'not provided')}; "
                    f"citation_status={citation.get('status', 'not provided')}; "
                    f"URL={metadata.get('source_url', 'not provided')}"
                )
            paragraphs.extend(f"Warning: {warning}" for warning in rag.get("warnings", []))
            paragraphs.append(DISCLAIMER)
            _add_text_pages(pdf, "RAG Response and Retrieved Sources", paragraphs)
    return buffer.getvalue()


def _add_text_pages(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    lines = [
        wrapped
        for paragraph in paragraphs
        for wrapped in (
            textwrap.wrap(
                str(paragraph), width=88, break_long_words=False, break_on_hyphens=False
            )
            or [""]
        )
    ]
    for start in range(0, max(1, len(lines)), 48):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle(title, fontsize=17, y=0.96)
        fig.text(
            0.08,
            0.9,
            "\n".join(lines[start : start + 48]),
            va="top",
            fontsize=9,
            family="monospace",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
