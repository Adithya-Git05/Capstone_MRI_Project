from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from integrated_pipeline.modules.rag_adapter import (
    build_patient_context,
    query_analysis_context,
)
from integrated_pipeline.modules.gradcam import classifier_gradcam
from integrated_pipeline.modules.report_generator import generate_report
from integrated_pipeline.pipeline import PROJECT_ROOT, run_pipeline


def _run_signature(
    current_case_id: str,
    uploaded: Mapping[str, Any],
    checkpoint_3d: str,
    checkpoint_seg: str,
    checkpoint_cls: str,
    risk_config: str,
    current_slice: int,
    automatic_slice: bool,
    use_segmentation: bool,
    use_classification: bool,
    synthseg_upload: Any = None,
    allow_incomplete_uploads: bool = False,
) -> str:
    digest = hashlib.sha256()
    digest.update(current_case_id.encode("utf-8"))
    for modality in ("t2f", "t1n", "t1c", "t2w"):
        item = uploaded.get(modality)
        if item is None and not allow_incomplete_uploads:
            raise ValueError(f"Missing uploaded modality {modality}.")
        digest.update(modality.encode("ascii"))
        if item is None:
            digest.update(b"missing")
        elif isinstance(item, bytes):
            digest.update(hashlib.sha256(item).digest())
        else:
            file_id = getattr(item, "file_id", "")
            if file_id:
                digest.update(repr((file_id, item.name, item.size)).encode("utf-8"))
            else:
                digest.update(hashlib.sha256(item.getbuffer()).digest())
    if synthseg_upload is None:
        digest.update(b"no-synthseg")
    elif isinstance(synthseg_upload, bytes):
        digest.update(hashlib.sha256(synthseg_upload).digest())
    else:
        file_id = getattr(synthseg_upload, "file_id", "")
        digest.update(
            repr((file_id, synthseg_upload.name, synthseg_upload.size)).encode("utf-8")
            if file_id
            else hashlib.sha256(synthseg_upload.getbuffer()).digest()
        )
    config = (
        checkpoint_3d,
        checkpoint_seg if use_segmentation else "",
        checkpoint_cls if use_classification else "",
        risk_config,
        "automatic" if automatic_slice else "manual",
        str(current_slice),
    )
    digest.update(repr(config).encode("utf-8"))
    for checkpoint in (
        checkpoint_3d,
        checkpoint_seg if use_segmentation else "",
        checkpoint_cls if use_classification else "",
        risk_config,
    ):
        if checkpoint:
            path = Path(checkpoint).expanduser()
            try:
                details = path.stat()
                digest.update(f"{details.st_size}:{details.st_mtime_ns}".encode("ascii"))
            except OSError:
                digest.update(b"missing")
    return digest.hexdigest()


st.set_page_config(page_title="MRI Research Analysis", layout="wide")
st.title("MRI Research Analysis Pipeline")
st.warning(
    "Research prototype only. Model predictions are not diagnoses or treatment advice; "
    "all outputs require qualified clinical review."
)
st.caption(
    "3D inference expects four registered BraTS NIfTI volumes ordered t2f, t1n, t1c, t2w. "
    "MRI files are processed locally by this app and are not forwarded to RAG or model services."
)

case_id = st.text_input("Study ID (letters, numbers, underscore, hyphen)", "demo-study")
st.info(
    "To run V8 surgical risk mapping, also provide true SynthSeg anatomical labels "
    "registered to the uploaded t1c volume. The V8 engine will halt if spatial QC fails."
)
synthseg_upload = st.file_uploader(
    "Optional SynthSeg label volume registered to t1c (.nii/.nii.gz)",
    type=["nii", "gz"],
    key="synthseg_upload",
)
upload_cols = st.columns(4)
uploaded_files = {}
for column, modality in zip(upload_cols, ("t2f", "t1n", "t1c", "t2w")):
    with column:
        uploaded_files[modality] = st.file_uploader(
            f"{modality} volume (.nii/.nii.gz)", type=["nii", "gz"], key=f"upload_{modality}"
        )

default_3d = PROJECT_ROOT / "2_3d_segmentation" / "outputs" / "segmentation3d" / "segresnet" / "20261019_151137_seed2026" / "best.pt"
default_seg = PROJECT_ROOT / "3_2d_segmentation" / "outputs" / "segmentation2d" / "02_attention_boundary_unet_import_20261019_1209" / "best.pt"
default_cls = PROJECT_ROOT / "4_2d_classification" / "outputs" / "classification" / "20261019_104128_seed2026" / "best.pt"

with st.expander("Model and optional input configuration", expanded=True):
    checkpoint_3d_text = st.text_input("3D checkpoint", str(default_3d))
    run_2d_seg = st.checkbox("Run available 2D segmentation checkpoint", default=default_seg.is_file())
    checkpoint_seg_text = st.text_input("2D segmentation checkpoint", str(default_seg), disabled=not run_2d_seg)
    run_cls = st.checkbox("Run available 2D classification checkpoint", default=default_cls.is_file())
    checkpoint_cls_text = st.text_input("2D classification checkpoint", str(default_cls), disabled=not run_cls)
    default_risk_config = PROJECT_ROOT / "5_surgical_risk_mapping" / "configs" / "config_v7.json"
    risk_config_text = st.text_input(
        "Optional V8 config template",
        str(default_risk_config) if default_risk_config.is_file() else "",
    )
    automatic_slice = st.checkbox("Select tumor-centre slice automatically", value=True)
    selected_slice = st.number_input(
        "Manual axial RAS slice index", min_value=0, value=0, disabled=automatic_slice
    )

if st.button("Run configured pipeline", type="primary"):
    if any(uploaded_files[name] is None for name in uploaded_files):
        st.error("Upload all four required MRI modalities.")
    elif not Path(checkpoint_3d_text).expanduser().is_file():
        st.error(f"3D checkpoint not found: {checkpoint_3d_text}")
    else:
        status = st.status("Preparing integrated MRI analysis.", expanded=True)
        try:
            current_signature = _run_signature(
                case_id, uploaded_files, checkpoint_3d_text, checkpoint_seg_text,
                checkpoint_cls_text, risk_config_text, int(selected_slice), automatic_slice,
                run_2d_seg, run_cls, synthseg_upload=synthseg_upload
            )
            result = run_pipeline(
                case_id,
                uploaded_files,
                Path(checkpoint_3d_text),
                Path(checkpoint_seg_text) if run_2d_seg else None,
                Path(checkpoint_cls_text) if run_cls else None,
                None if automatic_slice else int(selected_slice),
                Path(risk_config_text) if risk_config_text.strip() else None,
                synthseg_upload=synthseg_upload,
                progress=lambda message: status.update(label=message, state="running"),
            )
            st.session_state["pipeline_result"] = result
            st.session_state["pipeline_signature"] = current_signature
            st.session_state.pop("rag_result", None)
            st.session_state.pop("rag_error", None)
            status.update(label="Analysis finished.", state="complete")
        except (
            EOFError,
            OSError,
            pickle.UnpicklingError,
            ValueError,
            RuntimeError,
            KeyError,
            IndexError,
            TypeError,
            ImportError,
        ) as error:
            status.update(label="Analysis failed.", state="error")
            st.error(f"Pipeline failed: {error}")

result = st.session_state.get("pipeline_result")
if result:
    current_uploads = {
        name: uploaded_files[name]
        for name in uploaded_files
        if uploaded_files[name] is not None
    }
    current_signature = _run_signature(
        case_id,
        current_uploads,
        checkpoint_3d_text,
        checkpoint_seg_text,
        checkpoint_cls_text,
        risk_config_text,
        int(selected_slice),
        automatic_slice,
        run_2d_seg,
        run_cls,
        synthseg_upload=synthseg_upload,
        allow_incomplete_uploads=True,
    )
    if current_signature != st.session_state.get("pipeline_signature"):
        st.session_state.pop("pipeline_result", None)
        st.session_state.pop("pipeline_signature", None)
        st.session_state.pop("rag_result", None)
        st.session_state.pop("rag_error", None)
        result = None

if result:
    st.subheader("Stage status")
    st.json(result["stages"])
    st.write(
        f"Input shape: {result['input_shape']} | RAS output shape: {result['output_shape']} | "
        f"RAS spacing (mm): {result['output_spacing_mm']}"
    )
    mask_regions = list(result["region_masks"])
    left, right = st.columns(2)
    with left:
        region = st.selectbox("3D segmentation region", mask_regions)
        fig, ax = plt.subplots()
        ax.imshow(result["slice_image"], cmap="gray")
        ax.imshow(
            np.ma.masked_where(result["region_masks"][region][:, :, result["slice_index"]] == 0,
                               result["region_masks"][region][:, :, result["slice_index"]]),
            cmap="autumn",
            alpha=0.45,
        )
        ax.set_title(f"{region} mask - axial RAS slice {result['slice_index']}")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots()
        ax.imshow(result["slice_image"], cmap="gray")
        if result.get("segmentation2d") is not None:
            ax.imshow(
                np.ma.masked_where(result["segmentation2d"] == 0, result["segmentation2d"]),
                cmap="winter",
                alpha=0.5,
            )
        ax.set_title("2D predicted mask overlay")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    if result.get("classification"):
        st.subheader("Classification model output — not a diagnosis")
        st.json(result["classification"])
        try:
            gradcam_image, target_layer = classifier_gradcam(
                Path(result["classification"]["checkpoint"]),
                result["slice_image"],
                result["classification"]["predicted_class"],
            )
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            st.info(f"Grad-CAM is unavailable for this checkpoint: {error}")
        else:
            st.image(
                gradcam_image,
                caption=f"Grad-CAM for the predicted class at {target_layer}; visualization only.",
                use_container_width=True,
            )
    risk_result = result.get("risk_mapping")
    if risk_result:
        st.subheader("V8 research risk mapping - not clinical injury probabilities")
        st.caption(risk_result["note"])
        risk_slice = risk_result["risk_map"][:, :, result["slice_index"]]
        fig, ax = plt.subplots()
        ax.imshow(result["slice_image"], cmap="gray")
        ax.imshow(
            np.ma.masked_where(risk_slice <= 0, risk_slice),
            cmap="turbo",
            alpha=0.55,
            vmin=0,
            vmax=1,
        )
        ax.set_title("V8 normalized composite score - axial RAS slice")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        st.subheader("Candidate corridor costs (research prototype)")
        if risk_result["corridors"]:
            st.dataframe(risk_result["corridors"], use_container_width=True)
        else:
            st.info("The V8 engine did not generate any valid corridor candidates.")

    st.subheader("RAG evidence lookup (optional)")
    st.caption(
        "The uploaded MRI is never sent. With your authorization, only this run's structured "
        "model outputs, provenance, and your question are sent to the configured RAG service. "
        "Check that the RAG generation backend is authorized for this data."
    )
    rag_url = st.text_input("Existing RAG API URL", "http://localhost:8000")
    rag_case_id = st.text_input("RAG context ID (blank uses this study ID)")
    question = st.text_area("Evidence question")
    authorize_rag_request = st.checkbox(
        "I authorize sending this study ID, structured analysis results, and question to the RAG API.",
        value=False,
    )
    current_rag_signature = hashlib.sha256(
        repr(
            (
                rag_url.strip(),
                rag_case_id.strip(),
                question.strip(),
                authorize_rag_request,
            )
        ).encode("utf-8")
    ).hexdigest()
    if st.session_state.get("rag_signature") != current_rag_signature:
        st.session_state.pop("rag_result", None)
        st.session_state.pop("rag_signature", None)
        st.session_state.pop("rag_error", None)
    if st.button("Query current-run evidence"):
        if not authorize_rag_request:
            st.error("Confirm authorization before sending the case ID and question.")
        else:
            try:
                rag_case_id_value = rag_case_id.strip() or result["case_id"]
                context = build_patient_context(rag_case_id_value, result)
                rag_result = query_analysis_context(
                    rag_url, rag_case_id_value, question, context
                )
                rag_result["queried_case_id"] = rag_case_id_value
                rag_result["query"] = question.strip()
                st.session_state["rag_result"] = rag_result
                st.session_state["rag_signature"] = current_rag_signature
                st.session_state.pop("rag_error", None)
            except (ValueError, RuntimeError, ImportError) as error:
                st.session_state["rag_error"] = str(error)
                st.error(f"RAG query failed: {error}")
    if st.session_state.get("rag_result"):
        st.json(st.session_state["rag_result"])

    report_result = dict(result)
    report_result["rag"] = st.session_state.get("rag_result")
    if st.session_state.get("rag_error"):
        report_result["stages"] = dict(report_result["stages"])
        report_result["stages"]["RAG"] = {
            "status": "failed",
            "reasons": [st.session_state["rag_error"]],
        }
    if report_result["rag"]:
        report_result["stages"] = dict(report_result["stages"])
        report_result["stages"]["RAG"] = {
            "status": (
                "insufficient evidence"
                if report_result["rag"].get("evidence_status") == "insufficient_evidence"
                else "completed"
            ),
            "reasons": [
                "Sent structured outputs from current run under context ID "
                f"{report_result['rag'].get('queried_case_id', 'unknown')}; "
                "raw MRI was not sent."
            ],
        }
    st.download_button(
        "Download research PDF report",
        data=generate_report(report_result),
        file_name=f"{result['case_id']}_research_report.pdf",
        mime="application/pdf",
    )
