"""Generate a clinician-readable research draft from supplied case data."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _nearby_text(structures: List[Dict[str, Any]]) -> str:
    parts = []
    for item in structures:
        distance = float(item["min_distance_mm"])
        where = "the path passes through it" if distance == 0 else f"{distance:.0f} mm away"
        parts.append(f"{item['name']} ({where})")
    return "; ".join(parts) if parts else "None of the mapped structures within 10 mm"


def _corridor_section(corridors: List[Dict[str, Any]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{_escape(item['name'])}</td>"
        f"<td>{float(item['length_mm']):.0f} mm</td>"
        f"<td>{float(item['mean_risk']):.2f}</td>"
        f"<td>{_escape(_nearby_text(item.get('nearby_structures', [])))}</td>"
        "</tr>"
        for item in corridors
    )
    return (
        "<h2>Candidate approach routes (research)</h2>"
        f"<p>The research software drew {len(corridors)} possible routes from the brain surface to "
        "the centre of the supplied tumor annotation, steering away from areas with higher map scores. "
        "They are listed in the order the software produced them, which is not a recommendation. They are not ranked, "
        "are not a surgical plan, and do not replace review of the original images by the treating team.</p>"
        "<table><thead><tr><th>Approach</th><th>Route length</th><th>Average map score (0 to 1)</th>"
        "<th>Mapped structures within 10 mm of the route</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<p>Only structures present in the supplied labels and tract files are considered. "
        "Blood vessels and functional imaging were not available, so they are not part of these routes.</p>"
    )


def _tumor_details_html(details: Dict[str, Any]) -> str:
    rows = []
    if details.get("side"):
        crossing = " The annotation extends across the estimated midline." if details.get(
            "crosses_midline") else ""
        rows.append((
            "Location (approximate)",
            f"Mainly in the {details['side']} half of the brain.{crossing} "
            "Estimated from image geometry, not from a radiologist's anatomical review.",
        ))
    extent = details.get("extent_mm")
    if extent:
        rows.append((
            "Overall size of the annotation",
            f"About {extent['left_right']:.0f} mm left-right, {extent['front_back']:.0f} mm "
            f"front-back and {extent['up_down']:.0f} mm top-bottom (outer limits of the annotation).",
        ))
    if not rows:
        return ""
    body = "".join(f"<tr><th>{_escape(k)}</th><td>{_escape(v)}</td></tr>" for k, v in rows)
    return f"<table><tbody>{body}</tbody></table>"


def _viewer_section(viewer_html: str) -> str:
    return (
        "<h2>Interactive 3D case visualization</h2>"
        "<p>The patient's MRI with the supplied tumor annotation, the research surgical-risk map, "
        "candidate approach routes and nearby named structures. Drag to rotate, scroll to zoom, use "
        "the <strong>Layers</strong> panel to show or hide items, and click in the slice images to "
        "move through the scan. Requires a current desktop browser.</p>"
        f'<iframe class="viewer" title="3D case visualization" srcdoc="{html.escape(viewer_html, quote=True)}">'
        "</iframe>"
    )


def _validated_classification(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Tumor classification must be a dictionary.")
    cleaned = {
        key: str(value.get(key) or "").strip() for key in ("label", "source", "date")
    }
    if not cleaned["label"]:
        raise ValueError("Tumor classification needs a label.")
    if not cleaned["source"]:
        raise ValueError("Tumor classification needs a source, such as pathology or a validated model.")
    if any(len(item) > 300 for item in cleaned.values()):
        raise ValueError("Tumor classification fields must be 300 characters or fewer.")
    return {key: item for key, item in cleaned.items() if item}


def _classification_section(classification: Optional[Dict[str, Any]]) -> str:
    if not classification:
        return (
            "<h2>Tumor classification</h2>"
            "<p><strong>Not available.</strong> No tumor classification was supplied for this case, "
            "and this software does not determine tumor type or grade. A classification from "
            "pathology or a validated model can be added with its source.</p>"
        )
    rows = [("Classification", classification["label"]), ("Source", classification["source"])]
    if classification.get("date"):
        rows.append(("Date", classification["date"]))
    body = "".join(f"<tr><th>{_escape(k)}</th><td>{_escape(v)}</td></tr>" for k, v in rows)
    return (
        "<h2>Tumor classification</h2>"
        f"<table><tbody>{body}</tbody></table>"
        "<p>This classification was supplied with the case. It was not determined or verified by "
        "this software, and it is shown here so the imaging results can be read alongside it.</p>"
    )


def generate_clinical_case_report(
    metadata: Dict[str, Any],
    risk_summary: Optional[Dict[str, Any]],
    output_path: Path,
    corridors: Optional[List[Dict[str, Any]]] = None,
    *,
    tumor_details: Optional[Dict[str, Any]] = None,
    viewer_html: Optional[str] = None,
    sample_note: Optional[str] = None,
    classification: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write an HTML case summary without adding unverified medical conclusions."""
    classification = _validated_classification(classification)
    classification_section = _classification_section(classification)
    classification_note = (
        "The tumor classification shown was supplied with the case and was not determined or verified by this software."
        if classification else
        "No tumor classification is included because none was supplied."
    )
    supplied = metadata.get("patient_details", {})
    age = supplied.get("age")
    sex = supplied.get("sex")
    history = supplied.get("history", "").strip()
    case_id = metadata["case_id"]
    missing = metadata.get("missing_inputs", [])

    details = [
        ("Pseudonymous case ID", case_id),
        ("Age", f"{age} years" if age is not None else "Not supplied"),
        ("Sex", sex or "Not supplied"),
        ("Clinical history", history or "Not supplied"),
    ]
    detail_rows = "".join(
        f"<tr><th>{_escape(label)}</th><td>{_escape(value)}</td></tr>"
        for label, value in details
    )

    if risk_summary:
        volume_cm3 = float(risk_summary["tumor_volume_mm3"]) / 1000.0
        risk_findings = [
            item for item in risk_summary.get("top_risk_structures", [])
            if str(item.get("structure_name", "")).lower() != "white matter tracts"
        ]

        def relationship(item: Dict[str, Any]) -> str:
            distance = float(item.get("tumor_min_distance_mm", 0))
            return "Overlaps the annotated tumor" if distance == 0 else f"{distance:.0f} mm from the tumor"

        risk_rows = "".join(
            "<tr>"
            f"<td>{_escape(item.get('structure_name', 'Unnamed structure'))}</td>"
            f"<td>{_escape(item.get('functional_domain', 'Not specified'))}</td>"
            f"<td>{_escape(relationship(item))}</td>"
            f"<td>{_escape(str(item.get('risk_label', 'Not classified')).replace('_', ' ').title())}</td>"
            "</tr>"
            for item in risk_findings
        )
        lesion_section = (
            "<p>A lesion/tumor mask was supplied with this case. Its mask-derived "
            f"volume is <strong>{volume_cm3:.2f} cm³</strong>. This is a measurement "
            "of the supplied mask, not a diagnosis or an automated segmentation result.</p>"
            + _tumor_details_html(tumor_details or {})
        )
        mapping_section = (
            "<h2>Structures near the tumor (research map)</h2>"
            "<p>The research map highlights these anatomical structures in relation "
            "to the supplied annotation. Categories are experimental, are not "
            "clinically validated, and are not probabilities of injury or patient outcomes.</p>"
            "<table><thead><tr><th>Structure</th><th>Function it supports</th>"
            "<th>Relationship to tumor</th><th>Map category</th></tr></thead>"
            f"<tbody>{risk_rows or '<tr><td colspan=\"4\">No structures were listed.</td></tr>'}</tbody></table>"
        )
        data_status = (
            "A research-only map was generated from the supplied image annotation. "
            "It is not a prediction of complications or outcome."
        )
        if corridors:
            mapping_section += _corridor_section(corridors)
        if viewer_html:
            mapping_section += _viewer_section(viewer_html)
    else:
        lesion_section = (
            "<p>No lesion-volume measurement or spatial risk map was generated. "
            "The required image annotations were not available or the mapping stage did not run.</p>"
        )
        mapping_section = ""
        data_status = (
            "Mapping was not completed because required image annotations were not supplied."
            if missing
            else "No spatial map was generated for this case."
        )

    html_report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Imaging case summary — {_escape(case_id)}</title>
<style>
body{{font:16px/1.6 Arial,sans-serif;color:#202b38;max-width:1000px;margin:40px auto;padding:0 20px}}
h1{{font-size:28px}}h2{{font-size:20px;margin-top:30px}}
.notice{{padding:14px;background:#fff3d4;border-left:4px solid #c28b17}}
.sample{{padding:14px;margin:14px 0;background:#e6f0fb;border-left:4px solid #2b6cb0}}
iframe.viewer{{width:100%;height:1700px;border:1px solid #dce2e8;border-radius:6px;background:#0f141c}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #dce2e8}}
th{{width:32%;color:#425466}}footer{{border-top:1px solid #dce2e8;margin-top:34px;padding-top:14px;color:#526174;font-size:13px}}
@media print{{body{{margin:0 auto}}}}
</style></head><body>
<h1>Imaging case summary</h1>
<p><strong>Research draft — requires review by a qualified clinician.</strong></p>
{f'<div class="sample"><strong>Sample report.</strong> {_escape(sample_note)}</div>' if sample_note else ''}
<div class="notice">This report is generated from user-supplied imaging and, when available, a research spatial map.
It is not a diagnosis, clinical interpretation, or treatment recommendation.</div>
<h2>Patient details supplied</h2>
<table><tbody>{detail_rows or '<tr><td colspan=\"2\">No optional patient details supplied.</td></tr>'}</tbody></table>
<h2>Imaging and case summary</h2>
<p>Four aligned structural MRI volumes were provided: T1, contrast-enhanced T1, T2, and FLAIR.</p>
{lesion_section}
{classification_section}
{mapping_section}
<h2>Review notes</h2>
<p>{_escape(data_status)} This software did not produce a diagnosis or determine the tumor type or grade.
{_escape(classification_note)}
No prognosis or treatment plan is inferred.
A qualified clinician should review the original images and supplied annotation.</p>
<footer>Research prototype only. Not for clinical use. All measurements and map categories require
independent image review and clinical validation.</footer>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_report, encoding="utf-8")
    record = {
        "case_id": case_id,
        "patient_details": {
            "age": age,
            "sex": sex,
            "history": history or None,
        },
        "risk_mapping_status": "completed" if risk_summary else "blocked",
        "missing_inputs": missing,
        "risk_mapping_block_reason": metadata.get("risk_mapping_block_reason"),
        "risk_summary": risk_summary,
        "tumor_details": tumor_details,
        "tumor_classification": classification,
        "candidate_corridors": [
            {key: item.get(key) for key in (
                "name", "length_mm", "mean_risk", "max_risk", "nearby_structures")}
            for item in (corridors or [])
        ],
        "provenance": {
            "risk_mapping_pipeline": "V8 research prototype" if risk_summary else "not run",
            "lesion_mask_source": "user supplied" if risk_summary else None,
            "automatic_mri_model_inference": "not run",
            "tumor_classification_source": (
                "supplied with the case; not determined by this software"
                if classification else "not available"
            ),
            "clinical_validation": "not established",
        },
        "safety": "Research draft only; not a diagnosis or treatment recommendation.",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )
    return record
