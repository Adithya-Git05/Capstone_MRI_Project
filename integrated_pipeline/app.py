from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from integrated_pipeline.modules.gradcam import classifier_gradcam
from integrated_pipeline.modules.model_adapters import (
    _to_pil,
    classify_slice,
    segment_slice,
)
from integrated_pipeline.modules.model_registry import (
    load_model_registry,
    verify_checkpoint_integrity,
)
from integrated_pipeline.modules.mri_io import (
    MODALITIES,
    UploadedMRI,
    load_mri_upload,
    select_uploaded_slice,
    validate_uploaded_study,
)
from integrated_pipeline.modules.rag_adapter import (
    build_patient_context,
    query_analysis_context,
    rag_readiness,
)
from integrated_pipeline.modules.report_generator import (
    generate_markdown_report,
    generate_report,
)


@st.cache_data(show_spinner=False, max_entries=1)
def _cached_upload(payload: bytes, filename: str) -> UploadedMRI:
    return load_mri_upload(payload, filename)


@st.cache_data(show_spinner=False, max_entries=1)
def _cached_study_validation(
    case_id: str, uploaded: dict[str, bytes]
) -> dict[str, object]:
    return validate_uploaded_study(case_id, uploaded)


def _signature(
    payload: bytes,
    filename: str,
    case_id: str,
    modality: str,
    slice_index: int,
    checkpoints: dict[str, str],
    study_uploads: dict[str, bytes] | None = None,
) -> str:
    digest = sha256(payload)
    digest.update(repr((filename, case_id, modality, slice_index)).encode("utf-8"))
    for modality_key, modality_payload in sorted((study_uploads or {}).items()):
        digest.update(modality_key.encode("utf-8"))
        digest.update(sha256(modality_payload).digest())
    for model_id, checkpoint in sorted(checkpoints.items()):
        digest.update(model_id.encode("utf-8"))
        digest.update(checkpoint.encode("utf-8"))
        path = Path(checkpoint).expanduser()
        try:
            details = path.stat()
            digest.update(f"{details.st_size}:{details.st_mtime_ns}".encode("ascii"))
        except OSError:
            digest.update(b"missing")
    return digest.hexdigest()


def _checkpoint_status(path: str, model: dict[str, Any]) -> str:
    if not path or not Path(path).expanduser().is_file():
        return "not available"
    try:
        integrity = verify_checkpoint_integrity(Path(path), model)
    except (OSError, ValueError) as error:
        return f"integrity check failed: {error}"
    return f"available ({integrity})"


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_volume_compatibility(
    checkpoint_path: str, checkpoint_size: int, checkpoint_mtime_ns: int
) -> dict[str, Any]:
    del checkpoint_size, checkpoint_mtime_ns
    registry = load_model_registry()
    model_spec = registry["segmentation_3d"]
    path = Path(checkpoint_path)
    if not path.is_file():
        return {"compatible": False, "status": "checkpoint missing"}
    try:
        integrity = verify_checkpoint_integrity(path, model_spec)
        import torch

        runtime = {"torch_version": torch.__version__, "monai_version": version("monai")}
    except (ImportError, OSError, ValueError, PackageNotFoundError) as error:
        return {"compatible": False, "status": f"verification failed: {error}"}

    if model_spec.get("architecture") == "SegResNet":
        try:
            from src.segmentation3d.monai_engine import verify_segresnet_checkpoint

            verification = verify_segresnet_checkpoint(path, model_spec)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            return {"compatible": False, "status": f"strict checkpoint check failed: {error}"}
        return {
            "compatible": True,
            "status": (
                f"Strict SegResNet state-dict load passed ({integrity}; "
                f"{verification['state_dict_tensor_count']} tensors, PyTorch "
                f"{verification['torch_version']}, MONAI {verification['monai_version']}). "
                "Inference compatibility is confirmed; clinical performance is unvalidated."
            ),
        }

    tested = model_spec.get("compatibility_test", {}).get("strict_load", {})
    tested_torch_release = str(tested.get("torch_version", "")).split("+", 1)[0]
    current_torch_release = str(runtime["torch_version"]).split("+", 1)[0]
    if (
        tested_torch_release != current_torch_release
        or tested.get("monai_version") != runtime["monai_version"]
    ):
        return {
            "compatible": False,
            "status": (
                "The checkpoint was strictly tested with "
                f"PyTorch {tested.get('torch_version')} / MONAI {tested.get('monai_version')}; "
                f"current runtime is PyTorch {runtime['torch_version']} / MONAI {runtime['monai_version']}."
            ),
        }
    return {
        "compatible": True,
        "status": (
            f"Strict state-dict compatibility verified for this checkpoint "
            f"({integrity}; tested PyTorch {tested.get('torch_version')} / "
            f"MONAI {runtime['monai_version']}, current PyTorch "
            f"{runtime['torch_version']}). The inference run repeats strict loading."
        ),
    }


def _model_sidebar_status(
    task: str, path: str, model: dict[str, Any], result: dict[str, Any] | None
) -> str:
    stage = (
        result.get("segmentation", {})
        if task == "2D segmentation"
        else result.get("classification", {})
    ) if result else {}
    if stage.get("status") == "completed" and stage.get("checkpoint") == str(
        Path(path).expanduser().resolve()
    ):
        return "Loaded for current study"
    availability = _checkpoint_status(path, model)
    if availability == "not available":
        return "Checkpoint missing"
    if availability.startswith("integrity check failed"):
        return availability
    return "Checkpoint available; not loaded"


def _compact_model_status(
    task: str, path: str, model: dict[str, Any], result: dict[str, Any] | None
) -> str:
    status = _model_sidebar_status(task, path, model, result)
    if status == "Loaded for current study":
        return "Loaded"
    if status == "Checkpoint missing":
        return "Missing"
    if status.startswith("Checkpoint available"):
        return "Ready"
    return "Check"


@st.cache_data(show_spinner=False, ttl=20)
def _cached_rag_readiness(api_url: str) -> dict[str, Any]:
    return rag_readiness(api_url)


def _format_cited_answer(response: dict[str, Any]) -> str:
    answer = response.get("answer", {})
    summary = answer.get("summary") if isinstance(answer, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        warnings = response.get("warnings", [])
        detail = next(
            (item for item in warnings if isinstance(item, str) and item.strip()),
            None,
        )
        if detail:
            return f"Response generation failed: {detail}"
        return "The analysis summary was not returned by the service."

    evidence = {
        item["chunk_id"]: item
        for item in response.get("retrieved_evidence", [])
        if isinstance(item, dict) and isinstance(item.get("chunk_id"), str)
    }
    references: list[str] = []
    citation_numbers: dict[str, int] = {}
    for citation in response.get("citations", []):
        if not isinstance(citation, dict) or citation.get("status") != "verified":
            continue
        chunk_id = citation.get("chunk_id")
        if (
            not isinstance(chunk_id, str)
            or chunk_id not in evidence
            or chunk_id in citation_numbers
        ):
            continue
        item = evidence[chunk_id]
        metadata = item.get("metadata") or {}
        citation_number = len(references) + 1
        citation_numbers[chunk_id] = citation_number
        details = [
            str(citation.get("title") or item.get("title") or "").strip(),
            ", ".join(metadata["authors"])
            if isinstance(metadata.get("authors"), list)
            else str(metadata.get("authors") or "").strip(),
            str(metadata.get("year") or "").strip(),
            str(metadata.get("journal") or "").strip(),
            f"DOI: {metadata['doi']}" if metadata.get("doi") else "",
        ]
        reference = ". ".join(part for part in details if part)
        citation_text = f"{reference}." if reference else ""
        references.append(
            f"[{citation_number}] {citation_text} (knowledge-base passage {chunk_id})"
        )

    for chunk_id, citation_number in citation_numbers.items():
        summary = summary.replace(f"[{chunk_id}]", f"[{citation_number}]")
    missing_markers = [
        f"[{number}]"
        for number in citation_numbers.values()
        if f"[{number}]" not in summary
    ]
    if missing_markers:
        summary = f"{summary.rstrip()} {' '.join(missing_markers)}"
    if references:
        summary = f"{summary.rstrip()}\n\nKnowledge-base sources:\n" + "\n".join(references)
    return summary


def _render_chat(
    chat_tab: Any,
    case_id: str,
    current_result: dict[str, Any] | None,
    volume_result: dict[str, Any] | None = None,
    modality_inputs: dict[str, dict[str, Any]] | None = None,
) -> None:
    with chat_tab:
        st.subheader("Chat / Ask MRI Assistant")
        st.caption(
            "Ask general brain-MRI research questions or follow up on retrieved literature. "
            "This is not a diagnostic assistant and does not provide patient-care advice."
        )
        include_results = st.checkbox(
            "Include structured results from this session (no MRI image or voxel data is sent)",
            disabled=current_result is None and volume_result is None,
            key="chat_include_structured_results",
        )
        st.caption(
            "When enabled, the question and structured results are sent to Google Gemini "
            "through the configured API. Do not include identifiable clinical information."
        )
        with st.expander("RAG service and knowledge-base setup", expanded=False):
            api_url = st.text_input(
                "Local RAG API URL",
                "http://127.0.0.1:8000",
                key="chat_rag_api_url",
            )
        readiness = _cached_rag_readiness(api_url)
        if readiness["ready"]:
            st.success(
                "Literature retrieval API and knowledge base are available; "
                f"Gemini model {readiness['gemini_model']} is configured."
            )
        else:
            st.warning("The chat is visible, but evidence-grounded answers are not ready.")
            for item in readiness["missing"]:
                st.write(f"- {item}")
            with st.expander("Setup instructions", expanded=True):
                for item in readiness["setup"]:
                    st.code(item, language="powershell")
                st.caption(
                    "Full setup guide: 6_rag_treatment_planning\\README.md"
                )
            st.caption(
                "The chat will report these blockers when a question is submitted; it will "
                "not generate an answer or cite sources without the configured retrieval pipeline."
            )

        chat_key = "rag_chat_" + re.sub(r"[^A-Za-z0-9_-]", "_", case_id)
        messages = st.session_state.setdefault(chat_key, [])
        for message in messages:
            with st.chat_message(message["role"]):
                response = message.get("response")
                if message["role"] == "assistant" and isinstance(response, dict):
                    if response.get("status") == "completed":
                        st.text(_format_cited_answer(response))
                    else:
                        st.text(message["content"])
                else:
                    st.text(message["content"])
        prompt = st.chat_input(
            "Ask a brain-MRI research question",
            key="rag_chat_input",
        )
        if prompt:
            user_message = {"role": "user", "content": prompt}
            messages.append(user_message)
            if not readiness["ready"]:
                response = {
                    "status": "blocked",
                    "missing": readiness["missing"],
                    "setup": readiness["setup"],
                }
                assistant_content = (
                    "I could not retrieve or generate an answer because the RAG prerequisites "
                    "listed above are unavailable. No sources were retrieved."
                )
            else:
                context_case_id = (
                    case_id
                    if include_results and (current_result or volume_result)
                    else "general-literature-question"
                )
                if include_results and (current_result or volume_result):
                    patient_context = build_patient_context(
                        case_id,
                        current_result or {},
                        volume_result,
                        modality_inputs,
                    )
                else:
                    patient_context = {
                        "case_id": context_case_id,
                        "data_type": "GENERAL_LITERATURE_QUERY",
                        "provenance": {
                            "source": "general literature question",
                            "raw_mri_sent": False,
                            "patient_findings_supplied": False,
                        },
                        "available_modalities": [],
                        "missing_modalities": ["No patient MRI context was supplied."],
                        "quality_indicators": {
                            "clinical_validation": False,
                            "patient_specific_interpretation": False,
                        },
                    }
                try:
                    with st.spinner("Retrieving and checking literature evidence…"):
                        response = query_analysis_context(
                            api_url,
                            context_case_id,
                            prompt,
                            patient_context,
                        )
                    response["status"] = "completed"
                    response["query"] = prompt
                    assistant_content = _format_cited_answer(response)
                except (OSError, ValueError, RuntimeError, ImportError) as error:
                    response = {"status": "failed", "reason": str(error)}
                    assistant_content = f"RAG request failed: {error}"
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "response": response,
                }
            )
            st.rerun()


def _render_2d_analysis_sections(
    slice_image: np.ndarray,
    upload: UploadedMRI,
    selected_slice: int,
    current_result: dict[str, Any] | None,
) -> None:
    analysis_tabs = st.tabs(
        ["MRI & Preprocessing", "2D Segmentation", "2D Classification"]
    )
    with analysis_tabs[0]:
        st.subheader("MRI & Preprocessing")
        first, second = st.columns(2)
        first.image(slice_image, caption="Original selected MRI slice", clamp=True)
        second.image(
            np.asarray(
                _to_pil(
                    slice_image,
                    percentile_window=upload.metadata.get("format") == "NIfTI",
                )
            ),
            caption=(
                "Per-slice percentile-windowed preview (NIfTI intensity adaptation)"
                if upload.metadata.get("format") == "NIfTI"
                else "Grayscale 8-bit preview (training-compatible intensity scale)"
            ),
            clamp=True,
        )
        st.write(
            {
                "format": upload.metadata["format"],
                "dimensions": upload.metadata["shape"],
                "original_dimensions": upload.metadata.get("original_shape"),
                "selected_slice": selected_slice if upload.data.ndim == 3 else None,
                "voxel_spacing_mm": upload.metadata.get("voxel_spacing_mm"),
                "orientation": upload.metadata.get("orientation"),
                "grayscale_conversion": upload.metadata.get("grayscale_conversion", False),
                "preprocessing": (
                    "PNG/JPEG input keeps its grayscale 8-bit intensity scale. NIfTI is "
                    "reoriented to closest canonical RAS without resampling, then "
                    "percentile-windowed per slice as an explicit domain adaptation."
                ),
            }
        )
    with analysis_tabs[1]:
        st.subheader("2D Segmentation")
        if current_result is None:
            st.info("Run 2D analysis from Overview to produce a mask.")
        else:
            segmentation = current_result["segmentation"]
            st.write(f"Stage status: **{segmentation['status']}**")
            if segmentation["status"] == "completed":
                opacity = st.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)
                _render_overlay(slice_image, segmentation["mask"], opacity)
                left, right = st.columns(2)
                left.metric("Foreground area", segmentation["foreground_area"])
                right.metric(
                    "Inference time",
                    f"{segmentation['inference_time_seconds']:.3f} sec",
                )
                st.caption(
                    f"{segmentation['architecture']} · "
                    f"{segmentation['checkpoint_integrity']} · "
                    f"Classes: {segmentation['classes']}"
                )
                st.info(segmentation["limitation"])
                st.download_button(
                    "Download binary mask (PNG)",
                    _png_bytes(segmentation["mask"] * 255),
                    file_name="brain_mri_segmentation_mask.png",
                    mime="image/png",
                )
            else:
                st.warning(segmentation.get("limitation", "No segmentation output available."))
    with analysis_tabs[2]:
        st.subheader("2D Classification")
        if current_result is None:
            st.info("Run 2D analysis from Overview to produce classifier output.")
        else:
            classification = current_result["classification"]
            st.write(f"Stage status: **{classification['status']}**")
            if classification["status"] == "completed":
                st.metric("Predicted BRISC folder class", classification["predicted_class"])
                st.bar_chart(classification["probabilities"])
                st.caption(
                    f"{classification['architecture']} · "
                    f"{classification['checkpoint_integrity']} · "
                    f"Time: {classification['inference_time_seconds']:.3f} sec"
                )
                st.info(classification["limitation"])
                st.caption(
                    "Scores are uncalibrated model outputs, not clinical probabilities."
                )
            else:
                st.warning(
                    classification.get("limitation", "No classification output available.")
                )


def _render_spatial_status(
    *,
    case_id: str,
    uploads: dict[str, bytes],
    input_attested: bool,
    study_validated: bool,
    checkpoint: Path,
    model_spec: dict[str, Any],
    checkpoint_compatible: bool,
    compatibility_status: str,
    synthseg_upload: Any,
    run_signature: str,
) -> None:
    spatial_tabs = st.tabs(["Spatial Risk Assessment", "Surgical Corridors"])
    volume_result = st.session_state.get("volume_analysis")
    if not isinstance(volume_result, dict) or volume_result.get("run_signature") != run_signature:
        volume_result = None
    with spatial_tabs[0]:
        st.subheader("Spatial Risk Assessment")
        st.caption(
            "3D SegResNet produces TC, WT, and ET tumor masks. The separate V8 spatial-risk "
            "mapper additionally requires registered SynthSeg anatomical labels and its "
            "configured risk-map template."
        )
        if not study_validated:
            st.warning(
                "Upload four co-registered NIfTI modalities with 1 mm isotropic spacing "
                "to run this SegResNet."
            )
        if not input_attested:
            st.warning(
                "Confirm that all four modalities are co-registered and from the same study. "
                "Shared patient identity cannot be established from NIfTI geometry alone."
            )
        if not checkpoint_compatible:
            st.error(f"3D inference is unavailable: {compatibility_status}")

        run_disabled = not (
            study_validated and input_attested and checkpoint_compatible
            and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", case_id)
        )
        if st.button(
            "Run 3D segmentation",
            type="primary",
            disabled=run_disabled,
            help=(
                "Uses the registered SegResNet checkpoint and its TC/WT/ET output contract. "
                "The run performs strict checkpoint loading and checks system memory."
            ),
        ):
            from integrated_pipeline.pipeline import run_pipeline

            status = st.status("Preparing 3D inference.", expanded=True)

            def report_progress(message: str) -> None:
                status.update(label=message, state="running")

            try:
                result = run_pipeline(
                    case_id=case_id,
                    uploaded=uploads,
                    checkpoint_3d=checkpoint,
                    synthseg_upload=synthseg_upload,
                    progress=report_progress,
                    model_spec=model_spec,
                )
            except (
                FileNotFoundError,
                ImportError,
                KeyError,
                MemoryError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                st.session_state["volume_analysis"] = {
                    "status": "failed",
                    "reason": str(error),
                    "run_signature": run_signature,
                }
                status.update(label="3D inference did not complete.", state="error")
            else:
                result["status"] = "completed"
                result["run_signature"] = run_signature
                st.session_state["volume_analysis"] = result
                status.update(
                    label="3D inference completed; clinical performance remains unvalidated.",
                    state="complete",
                )

        if volume_result and volume_result.get("status") == "failed":
            st.error(f"3D inference failed: {volume_result['reason']}")
        elif volume_result:
            st.success(
                "Checkpoint loaded and TC/WT/ET inference completed. "
                "Clinical performance is unvalidated; no RC mask is produced."
            )
            st.caption(
                "RC (resection cavity) is not one of this checkpoint's three output channels. "
                "No RC mask is generated or inferred from TC/WT/ET; this model's supported "
                "outputs are only TC, WT, and ET."
            )
            metadata = volume_result["segmentation3d_metadata"]
            st.caption(
                f"Device: {volume_result['inference_device']} · "
                f"PyTorch {volume_result['software_versions']['torch']} · "
                f"MONAI {volume_result['software_versions']['monai']} · "
                f"Native geometry preserved: {volume_result['orientation'] is None} · "
                f"Inference time: {metadata['inference_time_seconds']:.1f} sec"
            )
            memory_usage = metadata.get("memory_usage", {})
            peak_working_set = memory_usage.get("process_after", {}).get(
                "peak_working_set_bytes"
            )
            current_working_set = memory_usage.get("process_after", {}).get(
                "working_set_bytes"
            )
            available_before = metadata.get("memory_preflight", {}).get(
                "available_bytes"
            )
            available_ram = memory_usage.get("available_ram_after_bytes")
            if (
                isinstance(peak_working_set, int)
                and isinstance(current_working_set, int)
                and isinstance(available_before, int)
                and isinstance(available_ram, int)
            ):
                st.caption(
                    f"Process RAM now: {current_working_set / 1024**3:.2f} GiB · "
                    f"Process peak since app start: {peak_working_set / 1024**3:.2f} GiB · "
                    f"System free RAM before/after: {available_before / 1024**3:.2f} / "
                    f"{available_ram / 1024**3:.2f} GiB"
                )
            counts = metadata.get("mask_voxel_count", {})
            metric_columns = st.columns(3)
            for column, region in zip(metric_columns, ("TC", "WT", "ET")):
                column.metric(f"{region} mask voxels", f"{counts.get(region, 'n/a'):,}" if isinstance(counts.get(region), int) else "n/a")
            if volume_result.get("segmentation_visualization"):
                st.image(
                    volume_result["segmentation_visualization"],
                    caption="T1ce with model-generated ET, TC, and WT overlays.",
                    use_container_width=True,
                )
            for region in volume_result["region_order"]:
                st.download_button(
                    f"Download {region} mask (NIfTI)",
                    volume_result["mask_nifti_files"][region],
                    file_name=f"{case_id}_{region}.nii.gz",
                    mime="application/gzip",
                    key=f"download_3d_{region}",
                )

            risk = volume_result["stages"].get("surgical risk mapping", {})
            if risk.get("status") == "completed":
                st.success("V8 spatial risk mapping completed with its spatial QC checks.")
                st.caption(risk.get("note", "Research-prototype result; not clinical advice."))
                risk_mapping = volume_result.get("risk_mapping")
                if isinstance(risk_mapping, dict):
                    risk_map = np.asarray(risk_mapping["risk_map"], dtype=np.float32)
                    risk_slice_index = int(volume_result["slice_index"])
                    if not 0 <= risk_slice_index < risk_map.shape[2]:
                        st.error("Risk-map slice index is outside the returned volume.")
                    else:
                        figure, axis = plt.subplots(figsize=(7, 6))
                        axis.imshow(volume_result["slice_image"], cmap="gray")
                        risk_slice = risk_map[:, :, risk_slice_index]
                        axis.imshow(
                            np.ma.masked_where(risk_slice <= 0, risk_slice),
                            cmap="magma",
                            alpha=0.55,
                            vmin=0,
                            vmax=1,
                        )
                        corridor_mask = risk_mapping.get("corridor_mask")
                        if corridor_mask is not None and np.any(corridor_mask):
                            corridor_slice = np.asarray(corridor_mask)[
                                :, :, risk_slice_index
                            ] > 0
                            axis.contour(
                                corridor_slice.astype(np.uint8),
                                levels=[0.5],
                                colors=["cyan"],
                                linewidths=0.8,
                            )
                        axis.set_title(
                            f"Experimental V8 spatial-risk score · slice {risk_slice_index}"
                        )
                        axis.axis("off")
                        figure.tight_layout()
                        st.pyplot(figure, use_container_width=True)
                        plt.close(figure)
                    st.json(
                        {
                            "active_weights": risk_mapping["active_weights"],
                            "evidence_status": risk_mapping["evidence_status"],
                            "DTI_and_tract_data_available": risk_mapping["qc"].get(
                                "dti_available", False
                            ),
                        }
                    )
                    st.download_button(
                        "Download spatial risk map (NIfTI)",
                        risk_mapping["risk_map_nifti"],
                        file_name=f"{case_id}_spatial_risk.nii.gz",
                        mime="application/gzip",
                        key="download_spatial_risk_map",
                    )
                    if risk_mapping.get("interactive_report_html"):
                        st.download_button(
                            "Download interactive 3D risk and corridor report",
                            risk_mapping["interactive_report_html"],
                            file_name=f"{case_id}_risk_corridors.html",
                            mime="text/html",
                            key="download_interactive_risk_report",
                        )
            elif risk.get("status") == "failed":
                st.error("V8 spatial risk mapping failed; no risk map is presented as valid.")
                st.write(risk.get("reasons", []))
            else:
                st.info(
                    "Spatial risk mapping was not run. It requires a supplied, aligned "
                    "SynthSeg label volume and an available V8 configuration."
                )
                if risk.get("reasons"):
                    st.write(risk["reasons"])
        else:
            st.info("3D inference has not been run for this study.")

        if synthseg_upload is None:
            st.caption("No SynthSeg labels supplied; spatial risk mapping cannot run.")
    with spatial_tabs[1]:
        st.subheader("Surgical Corridor Planning")
        if volume_result and volume_result.get("status") == "completed":
            risk = volume_result.get("risk_mapping")
            corridor_stage = volume_result["stages"].get("Surgical corridors", {})
            if corridor_stage.get("status") == "completed" and isinstance(risk, dict):
                st.warning(
                    "Experimental V8 candidate outputs only; not surgical recommendations "
                    "or a validated path-planning result."
                )
                corridors = risk.get("corridors", [])
                if corridors:
                    st.dataframe(corridors, use_container_width=True, hide_index=True)
                    if risk.get("corridor_mask_nifti"):
                        st.download_button(
                            "Download candidate corridor mask (NIfTI)",
                            risk["corridor_mask_nifti"],
                            file_name=f"{case_id}_candidate_corridors.nii.gz",
                            mime="application/gzip",
                            key="download_candidate_corridors",
                        )
                else:
                    st.info("The completed V8 run returned no candidate corridors.")
            else:
                st.info(
                    "Corridor analysis was not run because valid spatial-risk prerequisites "
                    "were unavailable."
                )
                if corridor_stage.get("reasons"):
                    st.write(corridor_stage["reasons"])
        else:
            st.info(
                "Run successful 3D segmentation and satisfy the V8 spatial-risk prerequisites "
                "before corridor outputs can be evaluated."
            )


def _render_overlay(image: np.ndarray, mask: np.ndarray, opacity: float) -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(image, cmap="gray")
    axis.imshow(
        np.ma.masked_where(mask == 0, mask),
        cmap="autumn",
        alpha=opacity,
        vmin=0,
        vmax=1,
    )
    axis.set_title("2D tumor foreground overlay")
    axis.axis("off")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.asarray(array).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _area_description(mask: np.ndarray, metadata: dict[str, object]) -> str:
    pixel_count = int(np.count_nonzero(mask))
    affine = metadata.get("affine")
    if affine is None:
        return f"{pixel_count} pixels; physical area unavailable"
    matrix = np.asarray(affine, dtype=np.float64)
    pixel_area_mm2 = float(np.linalg.norm(np.cross(matrix[:3, 0], matrix[:3, 1])))
    if not np.isfinite(pixel_area_mm2) or pixel_area_mm2 <= 0:
        return f"{pixel_count} pixels; physical area unavailable"
    return f"{pixel_count} pixels ({pixel_count * pixel_area_mm2:.2f} mm^2 on this slice)"


def _empty_result(
    case_id: str,
    filename: str,
    modality: str,
    upload: UploadedMRI,
    slice_index: int,
) -> dict[str, Any]:
    metadata = upload.metadata
    return {
        "case_id": case_id,
        "input": {
            "filename": filename,
            "modality": modality,
            "shape": metadata.get("shape"),
            "original_shape": metadata.get("original_shape"),
            "orientation": metadata.get("orientation"),
            "voxel_spacing_mm": metadata.get("voxel_spacing_mm"),
            "preprocessing_status": "Input validated; model-specific preprocessing has not run.",
            "slice_index": slice_index if upload.data.ndim == 3 else "not applicable",
        },
        "segmentation": {
            "status": "not run",
            "limitation": "No model output was produced.",
        },
        "classification": {
            "status": "not run",
            "limitation": "No model output was produced.",
        },
        "explainability": {
            "status": "not run",
            "observation": "No heatmap was generated.",
        },
        "spatial_risk": {
            "status": "missing required input",
            "missing_data": "A compatible 3D segmentation mask and registered anatomical labels are required.",
        },
        "surgical_corridors": {
            "status": "missing required input",
            "missing_data": "A validated 3D spatial risk result and required anatomical inputs are unavailable.",
        },
        "stages": {
            "Input validation": {"status": "completed", "reasons": []},
            "2D segmentation": {
                "status": "not run",
                "reasons": ["Run the 2D analysis to produce a prediction."],
            },
            "2D classification": {
                "status": "not run",
                "reasons": ["Run the 2D analysis to produce a prediction."],
            },
            "3D segmentation": {
                "status": "not run",
                "reasons": [
                    "3D inference has not been run for this study."
                ],
            },
            "Spatial risk mapping": {
                "status": "missing required input",
                "reasons": [
                    "A 2D mask cannot satisfy the 3D spatial and anatomical prerequisites."
                ],
            },
            "Surgical corridors": {
                "status": "not run",
                "reasons": ["Run the 3D workflow and satisfy spatial-risk prerequisites."],
            },
            "RAG": {"status": "not run", "reasons": []},
        },
        "rag": None,
    }


def _run_analysis(
    result: dict[str, Any],
    slice_image: np.ndarray,
    metadata: dict[str, object],
    segmentation_checkpoint: str,
    classification_checkpoint: str,
    segmentation_spec: dict[str, Any],
    classification_spec: dict[str, Any],
) -> dict[str, Any]:
    for task, checkpoint, runner, model_spec in (
        ("2D segmentation", segmentation_checkpoint, segment_slice, segmentation_spec),
        (
            "2D classification",
            classification_checkpoint,
            classify_slice,
            classification_spec,
        ),
    ):
        if not checkpoint or not Path(checkpoint).expanduser().is_file():
            reason = (
                f"Selected checkpoint is not present: {checkpoint}"
                if checkpoint
                else "No checkpoint path is configured."
            )
            result["stages"][task] = {"status": "not configured", "reasons": [reason]}
            result["segmentation" if task == "2D segmentation" else "classification"] = {
                "status": "not configured",
                "checkpoint": checkpoint or "not configured",
                "limitation": reason,
            }
            continue
        try:
            integrity = verify_checkpoint_integrity(
                Path(checkpoint),
                model_spec,
            )
            if task == "2D segmentation":
                mask, model_info = runner(
                    Path(checkpoint),
                    slice_image,
                    percentile_window=metadata.get("format") == "NIfTI",
                )
                result["segmentation"] = {
                    "status": "completed",
                    "architecture": model_info["architecture"],
                    "checkpoint": model_info["checkpoint"],
                    "checkpoint_integrity": integrity,
                    "classes": model_info["classes"],
                    "foreground_area": _area_description(mask, metadata),
                    "inference_time_seconds": model_info["inference_time_seconds"],
                    "preprocessing": model_info["preprocessing"],
                    "limitation": (
                        "BRISC 2D binary foreground output; compatibility with other MRI domains "
                        "is unvalidated. Area is from one slice, not tumor volume."
                    ),
                    "mask": mask,
                    "metadata": model_info,
                }
            else:
                classification = runner(
                    Path(checkpoint),
                    slice_image,
                    percentile_window=metadata.get("format") == "NIfTI",
                )
                result["classification"] = {
                    "status": "completed",
                    **classification,
                    "checkpoint_integrity": integrity,
                    "limitation": (
                        "BRISC folder-class prediction, not pathology-confirmed tumor subtype; "
                        "domain compatibility is unvalidated."
                    ),
                }
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
            result["stages"][task] = {
                "status": "failed",
                "reasons": [str(error)],
            }
            result["segmentation" if task == "2D segmentation" else "classification"] = {
                "status": "failed",
                "checkpoint": checkpoint,
                "limitation": str(error),
            }
        else:
            result["stages"][task] = {
                "status": "completed",
                "reasons": [],
            }
    return result


def main() -> None:
    st.set_page_config(page_title="Brain MRI Analysis", page_icon=None, layout="wide")
    dark_mode = st.sidebar.toggle("Dark mode", key="dark_mode")
    theme = (
        {
            "scheme": "dark",
            "background": "#0E1117",
            "sidebar": "#111821",
            "surface": "#161B22",
            "input": "#1F2937",
            "text": "#E6EDF3",
            "button_text": "#D6E4F0",
            "muted": "#B8C6D4",
            "border": "#303A46",
            "accent": "#36A4C5",
            "alert": "#1B2A38",
            "alert_text": "#DCEAF5",
            "code": "#171E27",
        }
        if dark_mode
        else {
            "scheme": "light",
            "background": "#F4F7FA",
            "sidebar": "#EDF3F8",
            "surface": "#FFFFFF",
            "input": "#FFFFFF",
            "text": "#172B4D",
            "button_text": "#173A5E",
            "muted": "#294B69",
            "border": "#D6E0EA",
            "accent": "#0B6D8B",
            "alert": "#E8F1F7",
            "alert_text": "#17324D",
            "code": "#EDF2F7",
        }
    )
    st.markdown(
        f"""
        <style>
        :root {{ color-scheme: {theme["scheme"]}; }}
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background: {theme["background"]} !important;
            color: {theme["text"]} !important;
        }}
        [data-testid="stSidebar"] {{
            background: {theme["sidebar"]} !important;
            color: {theme["text"]} !important;
            border-right: 1px solid {theme["border"]};
        }}
        .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp label, .stApp [data-testid="stCaption"],
        .stApp [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            color: {theme["text"]} !important;
        }}
        .stApp input, .stApp textarea,
        .stApp [data-baseweb="select"] > div,
        .stApp [data-testid="stFileUploader"] section {{
            background-color: {theme["input"]} !important;
            color: {theme["text"]} !important;
            border-color: {theme["border"]} !important;
        }}
        .stApp [data-testid="stFileUploader"] button {{
            background: {theme["accent"]} !important;
            color: #FFFFFF !important;
            border-color: {theme["accent"]} !important;
        }}
        .stApp [role="option"], .stApp [role="listbox"] {{
            background: {theme["surface"]} !important;
            color: {theme["text"]} !important;
        }}
        .stApp button {{
            color: {theme["button_text"]} !important;
            border-color: {theme["border"]} !important;
        }}
        .stApp button[kind="primary"] {{
            background: {theme["accent"]} !important;
            color: #FFFFFF !important;
            border-color: {theme["accent"]} !important;
        }}
        .stApp [data-testid="stAlert"] {{
            background: {theme["alert"]} !important;
            color: {theme["alert_text"]} !important;
            border: 1px solid {theme["border"]};
        }}
        .stApp [data-testid="stAlert"] p,
        .stApp [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {{
            color: {theme["alert_text"]} !important;
        }}
        .stApp [data-testid="stMetric"] {{
            background: {theme["surface"]} !important;
            color: {theme["text"]} !important;
            border: 1px solid {theme["border"]};
            border-radius: 8px;
            padding: 10px 12px;
        }}
        .stApp [data-testid="stMetric"] label,
        .stApp [data-testid="stMetric"] [data-testid="stMetricValue"],
        .stApp [data-testid="stMetric"] [data-testid="stMetricLabel"] {{
            color: {theme["text"]} !important;
        }}
        .stApp [role="tab"] {{ color: {theme["muted"]} !important; }}
        .stApp [role="tab"][aria-selected="true"] {{ color: {theme["accent"]} !important; }}
        .stApp [data-testid="stCode"] {{
            background: {theme["code"]} !important;
            color: {theme["text"]} !important;
        }}
        .stApp [data-testid="stCode"] pre,
        .stApp [data-testid="stCode"] code {{
            background: {theme["code"]} !important;
            color: {theme["text"]} !important;
        }}
        .block-container {{ max-width: 1440px; padding-top: 1.4rem; padding-bottom: 2rem; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {theme["border"]}; }}
        @media (max-width: 900px) {{
            .block-container {{ padding: 1rem 1rem 2rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Brain MRI Analysis")
    st.caption("AI-assisted tumor characterization, segmentation, and spatial risk assessment.")
    st.warning(
        "Research prototype only. This application is not an independently validated clinical "
        "decision-making system; outputs are not a diagnosis or treatment recommendation."
    )

    try:
        registry = load_model_registry()
    except ValueError as error:
        st.error(f"Model registry configuration failed: {error}")
        st.stop()
        return

    segmentation_spec = registry["segmentation_2d"]
    classification_spec = registry["classification_2d"]
    volume_spec = registry["segmentation_3d"]

    st.sidebar.header("MRI workflow")
    case_id = st.sidebar.text_input("Study identifier", "research-case")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", case_id):
        st.sidebar.error(
            "Study identifier must contain 1-64 letters, numbers, underscores, or hyphens."
        )

    with st.sidebar.expander("Patient study input (all four modalities)", expanded=True):
        st.caption(
            "One case intake: upload the four co-registered NIfTI sequences together. "
            "The 2D models use one selected sequence; the 3D model requires all four."
        )
        modality_labels = {
            "t2f": "T2-FLAIR",
            "t1n": "T1 native",
            "t1c": "T1 contrast-enhanced",
            "t2w": "T2-weighted",
        }
        multimodal_uploads = {}
        for modality_key in MODALITIES:
            multimodal_uploads[modality_key] = st.file_uploader(
                f"{modality_labels[modality_key]} NIfTI",
                type=["nii", "gz"],
                key=f"study_modality_{modality_key}",
                help=f"BraTS input channel: {modality_key}. Upload one 3D NIfTI volume.",
            )
        input_attested = st.checkbox(
            "I confirm all four scans are co-registered and from this same study",
            key="3d_input_attested",
            help=(
                "Registration is checked from NIfTI geometry. Shared patient identity "
                "cannot be independently established from these files."
            ),
        )
        synthseg_upload = st.file_uploader(
            "Optional SynthSeg anatomical labels (NIfTI)",
            type=["nii", "gz"],
            key="study_synthseg_labels",
            help=(
                "Registered anatomical labels are required by the experimental spatial "
                "risk mapper. This file alone does not enable risk mapping."
            ),
        )
        selected_2d_source = st.selectbox(
            "2D slice source sequence",
            MODALITIES,
            index=MODALITIES.index("t1c"),
            format_func=lambda key: modality_labels[key],
            key="study_2d_source",
        )
        supplied_modalities = {
            key: upload.getvalue()
            for key, upload in multimodal_uploads.items()
            if upload is not None
        }
        study_validation = None
        if supplied_modalities and len(supplied_modalities) != len(MODALITIES):
            missing_labels = [
                modality_labels[key]
                for key in MODALITIES
                if key not in supplied_modalities
            ]
            st.warning(
                "Four-modality validation requires all sequences. Missing: "
                + ", ".join(missing_labels)
            )
        elif len(supplied_modalities) == len(MODALITIES):
            try:
                study_validation = _cached_study_validation(case_id, supplied_modalities)
            except (OSError, ValueError, RuntimeError, TypeError) as error:
                st.error(f"Four-modality input validation failed: {error}")
            else:
                spacing = tuple(study_validation["voxel_spacing_mm"])
                spacing_valid = np.allclose(
                    spacing, volume_spec["input_spacing_mm"], atol=0.01
                )
                if spacing_valid:
                    st.success(
                        "All four volumes are registered and validated "
                        f"(shape {study_validation['shape']}, spacing {spacing} mm, "
                        f"{study_validation['orientation']} orientation)."
                    )
                else:
                    st.error(
                        "The 3D SegResNet requires 1.0 mm isotropic volumes; "
                        f"received spacing {spacing} mm. No 3D resampling is applied."
                    )
                st.info(
                    "The checkpoint's stated output order is TC/WT/ET. The exact training "
                    "preprocessing is not embedded in the supplied checkpoint."
                )
        if synthseg_upload is not None:
            st.caption(
                "SynthSeg labels supplied. They will be checked against the uploaded MRI "
                "when the 3D/risk workflow runs."
            )
    uploaded_file = None
    modality = modality_labels[selected_2d_source]
    if study_validation is not None:
        uploaded_file = multimodal_uploads[selected_2d_source]

    with st.sidebar.expander("Advanced model configuration", expanded=False):
        segmentation_default = str(segmentation_spec["resolved_checkpoint"])
        classification_default = str(classification_spec["resolved_checkpoint"])
        segmentation_checkpoint = st.text_input(
            "2D segmentation checkpoint", segmentation_default
        )
        classification_checkpoint = st.text_input(
            "2D classification checkpoint", classification_default
        )
        st.text_input(
            "3D SegResNet checkpoint",
            str(volume_spec["resolved_checkpoint"]),
            disabled=True,
        )
        st.caption(
            "The registered 3D checkpoint is enabled only after file-integrity and "
            "strict state-dict compatibility checks."
        )
    sidebar_result = st.session_state.get("analysis_result")
    st.sidebar.caption(
        "2D segmentation · "
        + _model_sidebar_status(
            "2D segmentation",
            segmentation_checkpoint,
            segmentation_spec,
            sidebar_result,
        )
    )
    st.sidebar.caption(
        "2D classification · "
        + _model_sidebar_status(
            "2D classification",
            classification_checkpoint,
            classification_spec,
            sidebar_result,
        )
    )
    volume_checkpoint = Path(volume_spec["resolved_checkpoint"])
    volume_compatibility = _cached_volume_compatibility(
        str(volume_checkpoint),
        volume_checkpoint.stat().st_size if volume_checkpoint.is_file() else 0,
        volume_checkpoint.stat().st_mtime_ns if volume_checkpoint.is_file() else 0,
    )
    if volume_compatibility["compatible"]:
        st.sidebar.success(
            "3D SegResNet checkpoint integrity and strict load verified. "
            "Inference repeats strict loading; clinical performance unvalidated."
        )
    else:
        st.sidebar.error(f"3D checkpoint unavailable: {volume_compatibility['status']}")
    volume_signature_uploads = dict(supplied_modalities)
    if synthseg_upload is not None:
        volume_signature_uploads["synthseg"] = synthseg_upload.getvalue()
    volume_run_signature = sha256(
        (
            _signature(
                b"",
                "",
                case_id,
                "",
                0,
                {"segmentation_3d": str(volume_checkpoint)},
                volume_signature_uploads,
            )
            + f":attested={input_attested}"
        ).encode("utf-8")
    ).hexdigest()
    previous_volume = st.session_state.get("volume_analysis")
    if (
        isinstance(previous_volume, dict)
        and previous_volume.get("run_signature") != volume_run_signature
    ):
        st.session_state.pop("volume_analysis", None)

    upload: UploadedMRI | None = None
    slice_image: np.ndarray | None = None
    selected_slice = 0
    input_error: str | None = None
    if uploaded_file is not None:
        try:
            upload = _cached_upload(uploaded_file.getvalue(), uploaded_file.name)
            if upload.data.ndim == 3:
                selected_slice = st.sidebar.slider(
                    "Axial slice (canonical RAS axis 2)",
                    min_value=0,
                    max_value=upload.data.shape[2] - 1,
                    value=upload.data.shape[2] // 2,
                )
            slice_image = select_uploaded_slice(upload, selected_slice)
        except (OSError, ValueError, RuntimeError, IndexError) as error:
            input_error = str(error)
            st.sidebar.error(f"Input validation failed: {input_error}")
    if upload is not None and len(supplied_modalities) == len(MODALITIES):
        study_slices: dict[str, np.ndarray] = {}
        try:
            for modality_key, modality_upload in multimodal_uploads.items():
                modality_volume = _cached_upload(
                    modality_upload.getvalue(), modality_upload.name
                )
                study_slices[modality_key] = select_uploaded_slice(
                    modality_volume, selected_slice
                )
        except (OSError, ValueError, RuntimeError, IndexError) as error:
            input_error = str(error)
            st.sidebar.error(f"Study slice preparation failed: {input_error}")
            upload = None
            slice_image = None

    tabs = st.tabs(
        [
            "Overview",
            "MRI & 2D Analysis",
            "Explainability",
            "Spatial & Surgical",
            "Chat / Ask MRI Assistant",
            "Final Report",
        ]
    )

    current_result: dict[str, Any] | None = None
    if uploaded_file is None or upload is None or slice_image is None:
        with tabs[0]:
            st.subheader("Analysis workflow")
            status_columns = st.columns(4)
            status_columns[0].metric("MRI input", "Waiting")
            status_columns[1].metric(
                "2D segmentation",
                _compact_model_status(
                    "2D segmentation",
                    segmentation_checkpoint,
                    segmentation_spec,
                    None,
                ),
            )
            status_columns[2].metric(
                "2D classification",
                _compact_model_status(
                    "2D classification",
                    classification_checkpoint,
                    classification_spec,
                    None,
                ),
            )
            status_columns[3].metric(
                "3D checkpoint",
                "Compatible"
                if volume_compatibility["compatible"]
                else "Unavailable"
                if volume_checkpoint.is_file()
                else "Missing",
            )
            if volume_compatibility["compatible"]:
                status_columns[3].caption("Inference not run.")
            st.write(
                "Upload all four registered sequences once in the sidebar. After validation, "
                "the selected sequence feeds the 2D models; all four are retained together "
                "for the 3D workflow."
            )
            st.info(
                "Upload and validate all four co-registered NIfTI volumes. The selected "
                "sequence enables the existing 2D models; all four enable the 3D model."
            )
            if study_validation:
                st.success(
                    "Four-modality study validated: "
                    f"{', '.join(study_validation['modalities'])}; "
                    f"shape {study_validation['shape']}."
                )
                st.info(
                    "3D checkpoint integrity and strict state-dict compatibility are "
                    "verified for the registered runtime. Run 3D inference from the "
                    "Spatial & Surgical tab after the required input attestation."
                )
            if input_error:
                st.error(input_error)
        with tabs[1]:
            st.info("Upload and validate MRI data to enable preprocessing and 2D analysis.")
        with tabs[2]:
            st.info("A successful classification result and compatible checkpoint are required.")
        with tabs[3]:
            st.warning(
                "Spatial-risk mapping and corridor planning are not available without "
                "compatible 3D masks and registered anatomical inputs."
            )
        with tabs[5]:
            st.info("Upload and validate one complete study to enable its structured report.")
    else:
        uploaded_bytes = uploaded_file.getvalue()
        current_signature = _signature(
            uploaded_bytes,
            uploaded_file.name,
            case_id,
            modality,
            selected_slice,
            {
                "segmentation_2d": segmentation_checkpoint,
                "classification_2d": classification_checkpoint,
            },
            supplied_modalities,
        )
        if st.session_state.get("analysis_signature") != current_signature:
            for key in (
                "analysis_result",
                "analysis_signature",
                "gradcam_result",
                "rag_result",
                "rag_signature",
            ):
                st.session_state.pop(key, None)

        with tabs[0]:
            st.subheader("Workflow status")
            columns = st.columns(4)
            columns[0].metric("Input validation", "Completed")
            columns[1].metric(
                "2D segmentation",
                _compact_model_status(
                    "2D segmentation",
                    segmentation_checkpoint,
                    segmentation_spec,
                    st.session_state.get("analysis_result"),
                ),
            )
            columns[2].metric(
                "2D classification",
                _compact_model_status(
                    "2D classification",
                    classification_checkpoint,
                    classification_spec,
                    st.session_state.get("analysis_result"),
                ),
            )
            columns[3].metric(
                "3D segmentation",
                "Compatible; not run"
                if volume_compatibility["compatible"]
                else "Unavailable"
                if volume_checkpoint.is_file()
                else "Missing",
            )
            st.write(
                "Each 2D stage uses its own BRISC-trained checkpoint. Classification labels "
                "reflect BRISC image-folder classes, not independently confirmed pathology."
            )
            st.caption(
                f"One validated study: {', '.join(study_validation['modalities'])}; "
                f"dimensions {upload.metadata['shape']}; selected 2D source {modality}."
            )
            if study_validation:
                st.info(
                    "All four volumes are held as one patient study. The BRISC 2D models use "
                    f"only the selected {modality} slice because their input convention is "
                    "single-sequence 2D. The 3D checkpoint outputs ET/TC/WT in its "
                    "documented channel order; clinical performance remains unvalidated."
                )
            if st.button("Run 2D analysis", type="primary"):
                result = _empty_result(
                    case_id, uploaded_file.name, modality, upload, selected_slice
                )
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", case_id):
                    st.error("Enter a valid study identifier before running analysis.")
                else:
                    result["input"].update(
                        {
                            "filename": uploaded_file.name,
                            "preprocessing_status": (
                                "NIfTI reoriented to closest canonical RAS without resampling; "
                                "2D slice extracted; model-specific transforms run with each "
                                "selected model."
                            ),
                        }
                    )
                    status = st.status("Running configured 2D model stages.", expanded=True)
                    result = _run_analysis(
                        result,
                        slice_image,
                        upload.metadata,
                        segmentation_checkpoint,
                        classification_checkpoint,
                        segmentation_spec,
                        classification_spec,
                    )
                    result["slice_image"] = slice_image
                    result["modality_slices"] = study_slices
                    result["input"]["modalities"] = list(study_validation["modalities"])
                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_signature"] = current_signature
                    st.session_state.pop("gradcam_result", None)
                    st.session_state.pop("rag_result", None)
                    status.update(label="Configured stages finished.", state="complete")
            current_result = st.session_state.get("analysis_result")
            if current_result:
                st.subheader("Current analysis")
                st.json(current_result["stages"])

        current_result = st.session_state.get("analysis_result")
        with tabs[1]:
            _render_2d_analysis_sections(
                slice_image, upload, selected_slice, current_result
            )

        with tabs[2]:
            st.subheader("Explainability")
            st.caption(
                "Grad-CAM highlights image regions relevant to a classifier output; it does not "
                "establish diagnostic correctness."
            )
            if current_result is None or current_result["classification"]["status"] != "completed":
                st.info("A successful classification output and compatible checkpoint are required.")
            else:
                if st.button("Generate Grad-CAM"):
                    classification = current_result["classification"]
                    try:
                        heatmap, overlay, target_layer = classifier_gradcam(
                            Path(classification_checkpoint),
                            slice_image,
                            classification["predicted_class"],
                            percentile_window=upload.metadata.get("format") == "NIfTI",
                        )
                    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
                        st.session_state["gradcam_result"] = {
                            "status": "failed",
                            "reason": str(error),
                        }
                    else:
                        st.session_state["gradcam_result"] = {
                            "status": "completed",
                            "heatmap": heatmap,
                            "overlay": overlay,
                            "target_layer": target_layer,
                        }
                gradcam_result = st.session_state.get("gradcam_result")
                if gradcam_result:
                    if gradcam_result["status"] == "completed":
                        first, second = st.columns(2)
                        first.image(gradcam_result["heatmap"], caption="Grad-CAM heatmap")
                        second.image(
                            gradcam_result["overlay"],
                            caption=f"Overlay; target layer {gradcam_result['target_layer']}",
                        )
                    else:
                        st.error(
                            f"Grad-CAM failed: {gradcam_result.get('reason', 'unknown error')}"
                        )

        with tabs[3]:
            _render_spatial_status(
                case_id=case_id,
                uploads=supplied_modalities,
                input_attested=input_attested,
                study_validated=(
                    study_validation is not None
                    and np.allclose(
                        study_validation["voxel_spacing_mm"],
                        volume_spec["input_spacing_mm"],
                        atol=0.01,
                    )
                ),
                checkpoint=volume_checkpoint,
                model_spec=volume_spec,
                checkpoint_compatible=volume_compatibility["compatible"],
                compatibility_status=volume_compatibility["status"],
                synthseg_upload=synthseg_upload,
                run_signature=volume_run_signature,
            )

        with tabs[5]:
            st.subheader("Final Report")
            volume_result = st.session_state.get("volume_analysis")
            if (
                not isinstance(volume_result, dict)
                or volume_result.get("run_signature") != volume_run_signature
                or volume_result.get("status") != "completed"
            ):
                volume_result = None
            if current_result is None and volume_result is None:
                st.info("Run 2D analysis or 3D segmentation to generate a structured report.")
            else:
                if current_result is None:
                    report_result = _empty_result(
                        case_id, uploaded_file.name, modality, upload, selected_slice
                    )
                else:
                    report_result = dict(current_result)
                report_result["stages"] = dict(report_result["stages"])
                report_result["input_shape"] = (
                    current_result["input"].get("shape")
                    if current_result
                    else volume_result["input_shape"]
                )
                report_result["modalities"] = (
                    current_result["input"].get("modalities", [])
                    if current_result
                    else volume_result["modalities"]
                )
                report_result["slice_index"] = selected_slice
                report_result["slice_image"] = slice_image
                report_result["modality_slices"] = study_slices
                report_result["synthseg_supplied"] = synthseg_upload is not None
                report_result["segmentation2d"] = (
                    current_result.get("segmentation", {}).get("mask")
                    if current_result
                    else None
                )
                report_result["model_3d"] = (
                    volume_result["model_3d"]
                    if volume_result
                    else "not run"
                )
                report_result["model_2d_segmentation"] = (
                    current_result.get("segmentation", {}).get("architecture", "not run")
                    if current_result
                    else "not run"
                )
                report_result["model_2d_classification"] = (
                    current_result.get("classification", {}).get("architecture", "not run")
                    if current_result
                    else "not run"
                )
                if volume_result:
                    report_result.update(
                        {
                            "segmentation3d_metadata": volume_result[
                                "segmentation3d_metadata"
                            ],
                            "region_masks": volume_result["region_masks"],
                            "slice_image": volume_result["slice_image"],
                            "slice_index": volume_result["slice_index"],
                            "input_shape": volume_result["input_shape"],
                            "output_shape": volume_result["output_shape"],
                            "input_affine": volume_result["input_affine"],
                            "output_affine": volume_result["output_affine"],
                            "input_spacing_mm": volume_result["input_spacing_mm"],
                            "output_spacing_mm": volume_result["output_spacing_mm"],
                            "orientation": volume_result["orientation"],
                            "inference_device": volume_result["inference_device"],
                            "software_versions": volume_result["software_versions"],
                            "prediction_threshold": volume_result["prediction_threshold"],
                            "region_definitions": volume_result["region_definitions"],
                            "modalities": volume_result["modalities"],
                            "segmentation_visualization": volume_result[
                                "segmentation_visualization"
                            ],
                            "risk_mapping": volume_result["risk_mapping"],
                            "stages": {
                                **report_result["stages"],
                                **volume_result["stages"],
                            },
                        }
                    )
                gradcam_result = st.session_state.get("gradcam_result")
                if current_result and gradcam_result and gradcam_result.get("status") == "completed":
                    report_result["gradcam"] = gradcam_result
                if current_result and gradcam_result:
                    if gradcam_result["status"] == "completed":
                        target = current_result["classification"]["predicted_class"]
                        report_result["explainability"] = {
                            "status": "completed",
                            "method": (
                                f"Grad-CAM for predicted BRISC class {target}; "
                                f"target layer {gradcam_result['target_layer']}"
                            ),
                            "observation": (
                                "A model-relevance heatmap was generated; it does not "
                                "establish diagnostic correctness."
                            ),
                        }
                        report_result["stages"]["Explainability"] = {
                            "status": "completed",
                            "reasons": [],
                        }
                    else:
                        reason = gradcam_result.get("reason", "Grad-CAM failed.")
                        report_result["explainability"] = {
                            "status": "failed",
                            "method": "Grad-CAM",
                            "observation": reason,
                        }
                        report_result["stages"]["Explainability"] = {
                            "status": "failed",
                            "reasons": [reason],
                        }
                report_result["rag"] = st.session_state.get("rag_result")
                if current_result and report_result["rag"]:
                    rag_status = report_result["rag"].get("status", "completed")
                    report_result["stages"]["RAG"] = {
                        "status": (
                            "failed"
                            if rag_status == "failed"
                            else report_result["rag"].get(
                                "evidence_status", "completed"
                            )
                        ),
                        "reasons": (
                            [report_result["rag"].get("reason", "RAG request failed.")]
                            if rag_status == "failed"
                            else []
                        ),
                    }
                report = generate_markdown_report(report_result)
                st.markdown(report)
                st.download_button(
                    "Download Markdown report",
                    report,
                    file_name="brain_mri_analysis_report.md",
                    mime="text/markdown",
                )
                st.download_button(
                    "Download PDF report",
                    generate_report(report_result),
                    file_name="brain_mri_analysis_report.pdf",
                    mime="application/pdf",
                )

    active_volume_result = st.session_state.get("volume_analysis")
    if (
        not isinstance(active_volume_result, dict)
        or active_volume_result.get("run_signature") != volume_run_signature
        or active_volume_result.get("status") != "completed"
    ):
        active_volume_result = None
    active_modality_inputs = None
    if study_validation is not None:
        active_modality_inputs = {
            key: {
                "label": {
                    "t2f": "T2-FLAIR",
                    "t1n": "T1 native",
                    "t1c": "T1 contrast-enhanced",
                    "t2w": "T2-weighted",
                }[key],
                "uploaded": True,
                "shape": list(study_validation["shape"]),
                "voxel_spacing_mm": list(study_validation["voxel_spacing_mm"]),
                "orientation": study_validation["orientation"],
            }
            for key in MODALITIES
        }
    _render_chat(
        tabs[4],
        case_id,
        current_result,
        active_volume_result,
        active_modality_inputs,
    )


if __name__ == "__main__":
    main()
