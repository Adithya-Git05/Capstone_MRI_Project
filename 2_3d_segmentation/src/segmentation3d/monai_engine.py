from __future__ import annotations

import csv
import codecs
import gc
import inspect
import json
import os
import pickle
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.config import load_config, resolve_path, runtime_device
from src.common.reproducibility import seed_everything, worker_seed
from src.common.run import (
    configure_logging,
    create_run_dir,
    update_run_metadata,
    write_run_metadata,
)
from src.data.brats import (
    DEFAULT_MODALITIES,
    DEFAULT_REGIONS,
    REGION_LABELS,
    REGION_ORDER,
    BraTSCase,
    BraTSVolumeDataset,
    grouped_train_validation_split,
    list_cases,
    load_full_volume,
    normalize_region_labels,
)
from src.segmentation3d.metrics import region_metrics
from src.segmentation3d.models import build_monai_model


ONTOLOGY_ID = "brats_post_treatment_composites_v1"


def _safe_checkpoint_globals() -> list[type[Any] | Any]:
    numpy_core = getattr(np, "_core", np.core)
    safe: list[type[Any] | Any] = [
        numpy_core.multiarray.scalar,
        np.dtype,
        np.float32,
        np.float64,
        np.int64,
        codecs.encode,
    ]
    numpy_dtypes = getattr(np, "dtypes", None)
    if numpy_dtypes is not None:
        for name in ("Float32DType", "Float64DType"):
            dtype = getattr(numpy_dtypes, name, None)
            if dtype is not None:
                safe.append(dtype)
    return safe


def _load_model_card_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    try:
        with torch.serialization.safe_globals(_safe_checkpoint_globals()):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError(f"Could not safely read the SwinUNETR checkpoint: {error}") from error
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("The pretrained checkpoint must contain a state_dict mapping.")
    state_dict = checkpoint["state_dict"]
    if not state_dict or not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("The checkpoint state_dict must contain only tensor weights.")
    if all(key.startswith("net.") for key in state_dict):
        state_dict = {key.removeprefix("net."): value for key, value in state_dict.items()}
    elif any(key.startswith("net.") for key in state_dict):
        raise ValueError("Checkpoint state_dict mixes wrapped and unwrapped parameter names.")
    return checkpoint, state_dict


def _build_model_card_model(model_spec: dict[str, Any]) -> torch.nn.Module:
    modalities = model_spec.get("input_modalities")
    output_order = model_spec.get("output_order")
    inference = model_spec.get("inference")
    if (
        model_spec.get("architecture") != "SwinUNETR"
        or not isinstance(modalities, list)
        or len(modalities) != 4
        or not isinstance(output_order, list)
        or len(output_order) != 3
        or "RC" in output_order
        or not isinstance(inference, dict)
    ):
        raise ValueError("The registry does not describe the documented 4-in/3-out checkpoint.")
    feature_size = model_spec.get("feature_size")
    if not isinstance(feature_size, int) or feature_size <= 0:
        raise ValueError("The SwinUNETR feature_size is missing from the model registry.")
    roi_size = inference.get("roi_size")
    if (
        not isinstance(roi_size, list)
        or len(roi_size) != 3
        or any(not isinstance(value, int) or value < 32 or value % 32 for value in roi_size)
    ):
        raise ValueError("The SwinUNETR inference ROI must contain three dimensions.")
    return build_monai_model(
        "swinunetr",
        patch_size=tuple(int(value) for value in roi_size),
        in_channels=4,
        out_channels=3,
        feature_size=feature_size,
    )


def verify_model_card_checkpoint(
    checkpoint_path: Path, model_spec: dict[str, Any]
) -> dict[str, Any]:
    """Strictly match the inspected checkpoint tensors to the documented model."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"3D segmentation checkpoint not found: {checkpoint_path}")
    checkpoint, state_dict = _load_model_card_checkpoint(checkpoint_path)
    model = _build_model_card_model(model_spec)
    try:
        load_parameters = inspect.signature(model.load_state_dict).parameters
        load_options = {"strict": True}
        if "assign" in load_parameters:
            load_options["assign"] = True
        model.load_state_dict(state_dict, **load_options)
    except RuntimeError as error:
        raise ValueError(
            "The checkpoint tensors are not strictly compatible with the configured "
            f"MONAI SwinUNETR: {error}"
        ) from error
    input_shape = tuple(state_dict["swinViT.patch_embed.proj.weight"].shape)
    output_shape = tuple(state_dict["out.conv.conv.weight"].shape)
    if input_shape[1] != 4 or input_shape[0] != model_spec["feature_size"]:
        raise ValueError(f"Unexpected SwinUNETR input projection shape: {input_shape}.")
    if output_shape[0] != 3:
        raise ValueError(f"Expected three output channels; received {output_shape}.")
    result = {
        "status": "compatible",
        "checkpoint": str(checkpoint_path),
        "checkpoint_key": "state_dict",
        "state_dict_tensor_count": len(state_dict),
        "input_projection_shape": list(input_shape),
        "output_head_shape": list(output_shape),
        "prefix_removed": "net." if all(key.startswith("net.") for key in checkpoint["state_dict"]) else None,
        "torch_version": torch.__version__,
        "monai_version": __import__("monai").__version__,
        "strict_load": "All keys matched successfully",
    }
    del model, state_dict, checkpoint
    gc.collect()
    return result


def _available_memory_bytes() -> int:
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError("Could not determine available system memory.")
        return int(status.ullAvailPhys)
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError) as error:
        raise RuntimeError("Could not determine available system memory.") from error


def _check_inference_memory(
    checkpoint_path: Path,
    image: torch.Tensor,
    output_channels: int,
    roi_size: tuple[int, int, int],
) -> dict[str, int]:
    available = _available_memory_bytes()
    checkpoint_bytes = checkpoint_path.stat().st_size
    image_bytes = image.numel() * image.element_size()
    output_bytes = output_channels * int(np.prod(image.shape[1:])) * 4
    roi_working_bytes = 4 * int(np.prod(roi_size)) * 4 * 64
    reserve_bytes = 256 * 1024**2
    required = (
        2 * checkpoint_bytes
        + image_bytes
        + 2 * output_bytes
        + roi_working_bytes
        + reserve_bytes
    )
    if available < required:
        raise MemoryError(
            "3D CPU inference was not started: available RAM is "
            f"{available / 1024**3:.2f} GiB; the estimated working set for one "
            f"{roi_size[0]}x{roi_size[1]}x{roi_size[2]} window is "
            f"{required / 1024**3:.2f} GiB. Free RAM and retry, or use a machine with "
            "more memory. This estimate is conservative but cannot guarantee that "
            "inference will fit."
        )
    return {
        "available_bytes": available,
        "estimated_minimum_bytes": required,
        "input_bytes": image_bytes,
        "output_bytes": output_bytes,
        "roi_working_bytes": roi_working_bytes,
        "reserve_bytes": reserve_bytes,
    }


def _resolve_inference_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("3D inference supports only CPU or CUDA devices.")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested, but CUDA is unavailable.")
    if selected.type == "cuda" and (
        selected.index is not None and selected.index >= torch.cuda.device_count()
    ):
        raise ValueError(f"CUDA device index {selected.index} is not available.")
    return selected


def _check_cuda_inference_memory(
    checkpoint_path: Path,
    image: torch.Tensor,
    roi_size: tuple[int, int, int],
    device: torch.device,
) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    checkpoint_bytes = checkpoint_path.stat().st_size
    image_bytes = image.numel() * image.element_size()
    output_bytes = 3 * int(np.prod(image.shape[1:])) * 4
    roi_working_bytes = 4 * int(np.prod(roi_size)) * 4 * 64
    reserve_bytes = 256 * 1024**2
    required = checkpoint_bytes + output_bytes + roi_working_bytes + reserve_bytes
    if free_bytes < required:
        raise MemoryError(
            "3D CUDA inference was not started: available GPU memory is "
            f"{free_bytes / 1024**3:.2f} GiB; the estimated minimum is "
            f"{required / 1024**3:.2f} GiB for one {roi_size[0]}x{roi_size[1]}x"
            f"{roi_size[2]} window. Use a GPU with more memory or select CPU."
        )
    return {
        "available_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "estimated_minimum_bytes": int(required),
        "checkpoint_bytes": int(checkpoint_bytes),
        "input_bytes": int(image_bytes),
        "output_bytes": int(output_bytes),
        "roi_working_bytes": int(roi_working_bytes),
        "reserve_bytes": int(reserve_bytes),
    }


def _cuda_memory_usage(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "available_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _process_memory_usage() -> dict[str, int]:
    if os.name != "nt":
        return {}
    import ctypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    ):
        raise RuntimeError("Could not determine current process memory usage.")
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
    }


def _save_pretrained_visualization(
    output_dir: Path,
    input_slice: np.ndarray,
    probability_slices: np.ndarray,
    output_order: list[str],
    threshold: float,
) -> None:
    figure, axes = plt.subplots(1, len(output_order) + 1, figsize=(4 * (len(output_order) + 1), 4))
    axes[0].imshow(input_slice, cmap="gray")
    axes[0].set_title("T1ce input")
    axes[0].axis("off")
    colors = {"ET": "red", "TC": "lime", "WT": "cyan"}
    from matplotlib.colors import to_rgba

    for axis, name, probability_slice in zip(axes[1:], output_order, probability_slices):
        axis.imshow(input_slice, cmap="gray")
        mask = probability_slice >= threshold
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = to_rgba(colors.get(name, "red"), alpha=0.5)
        axis.imshow(overlay)
        axis.set_title(f"{name} mask")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "segmentation_visualization.png", dpi=120)
    plt.close(figure)


def predict_model_card_case(
    checkpoint_path: Path,
    case_directory: Path,
    output_dir: Path,
    model_spec: dict[str, Any],
    device: str | torch.device = "cpu",
) -> Path:
    """Run low-memory SwinUNETR inference and save native-space ET/TC/WT masks."""
    from src.data.brats import load_input_volume

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    case_directory = Path(case_directory).resolve()
    _require_new_or_empty_output_dir(output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"3D segmentation checkpoint not found: {checkpoint_path}")
    modalities = tuple(model_spec["input_modalities"])
    case = BraTSCase(
        case_id=case_directory.name,
        patient_group=case_directory.name.rsplit("-", 1)[0],
        directory=case_directory,
    )
    volume = load_input_volume(case, modalities=modalities, orientation=None)
    image = volume["image"].to(dtype=torch.float32)
    if image.ndim != 4 or image.shape[0] != 4 or not torch.isfinite(image).all():
        raise ValueError("Preprocessed SwinUNETR input must be finite with shape [4, X, Y, Z].")
    inference = model_spec["inference"]
    roi_size = tuple(int(value) for value in inference["roi_size"])
    if int(inference["sw_batch_size"]) != 1:
        raise ValueError("Low-memory SwinUNETR inference requires sw_batch_size=1.")
    if list(model_spec["output_order"]) != ["ET", "TC", "WT"]:
        raise ValueError("SwinUNETR output order must be exactly [ET, TC, WT].")
    selected_device = _resolve_inference_device(device)
    memory = _check_inference_memory(
        checkpoint_path, image, output_channels=3, roi_size=roi_size
    )
    cuda_memory_preflight = (
        _check_cuda_inference_memory(
            checkpoint_path, image, roi_size, selected_device
        )
        if selected_device.type == "cuda"
        else None
    )
    process_memory_before = _process_memory_usage()
    inference_started = time.perf_counter()
    checkpoint, state_dict = _load_model_card_checkpoint(checkpoint_path)
    model = _build_model_card_model(model_spec)
    try:
        load_parameters = inspect.signature(model.load_state_dict).parameters
        load_options = {"strict": True}
        if "assign" in load_parameters:
            load_options["assign"] = True
        model.load_state_dict(state_dict, **load_options)
    except RuntimeError as error:
        raise ValueError(f"SwinUNETR checkpoint compatibility check failed: {error}") from error
    del state_dict, checkpoint
    gc.collect()

    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    from monai.inferers import sliding_window_inference

    previous_thread_count = torch.get_num_threads()
    inference_thread_count = (
        min(previous_thread_count, 4)
        if selected_device.type == "cpu"
        else previous_thread_count
    )
    if inference_thread_count != previous_thread_count:
        torch.set_num_threads(inference_thread_count)
    try:
        model = model.to(selected_device).eval()
        with torch.inference_mode():
            logits = sliding_window_inference(
                inputs=image.unsqueeze(0),
                roi_size=roi_size,
                sw_batch_size=int(inference["sw_batch_size"]),
                predictor=model,
                overlap=float(inference["overlap"]),
                sw_device=selected_device,
                device=torch.device("cpu"),
            )
            if logits.ndim != 5 or logits.shape[1] != 3:
                raise ValueError(
                    f"Model output must have shape [1, 3, X, Y, Z]; received {tuple(logits.shape)}."
                )
            if not torch.isfinite(logits).all():
                raise ValueError("SwinUNETR produced non-finite logits.")
            probability_tensor = torch.sigmoid(logits[0])
            del logits
            probabilities = probability_tensor.numpy()
            del probability_tensor
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise MemoryError(
                f"SwinUNETR {selected_device.type.upper()} inference ran out of memory "
                f"using ROI {roi_size} and sliding-window batch size 1. Reduce other "
                "memory use or select a device with more available memory."
            ) from error
        raise
    finally:
        if inference_thread_count != previous_thread_count:
            torch.set_num_threads(previous_thread_count)
        del model
        gc.collect()
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()

    output_order = list(model_spec["output_order"])
    expected_shape = tuple(int(value) for value in image.shape[1:])
    if probabilities.shape != (3, *expected_shape):
        raise ValueError(
            f"Probability map shape {probabilities.shape} does not match input volume {expected_shape}."
        )
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.any(probabilities > 0)
    ):
        raise ValueError("SwinUNETR probability maps are empty, non-finite, or outside [0, 1].")
    threshold = float(inference["threshold"])
    output_dir.mkdir(parents=True, exist_ok=True)
    header = volume["header"].copy()
    header.set_data_dtype(np.uint8)
    affine = np.asarray(volume["affine"], dtype=np.float64)
    import nibabel as nib

    probability_summary: dict[str, dict[str, float]] = {}
    mask_voxel_count: dict[str, int] = {}
    for index, name in enumerate(output_order):
        probability_map = probabilities[index]
        mask = probability_map >= threshold
        probability_summary[name] = {
            "minimum": float(probability_map.min()),
            "maximum": float(probability_map.max()),
            "mean": float(probability_map.mean()),
        }
        mask_voxel_count[name] = int(mask.sum())
        output = nib.Nifti1Image(mask.astype(np.uint8), affine, header.copy())
        nib.save(output, str(output_dir / f"{case.case_id}_{name}.nii.gz"))
        del mask, output
    t1ce_index = modalities.index("t1c")
    whole_tumor_index = output_order.index("WT")
    occupied_slices = np.flatnonzero(
        np.any(probabilities[whole_tumor_index] >= threshold, axis=(0, 1))
    )
    slice_index = int(occupied_slices.mean()) if occupied_slices.size else expected_shape[2] // 2
    _save_pretrained_visualization(
        output_dir,
        image[t1ce_index, :, :, slice_index].numpy(),
        probabilities[:, :, :, slice_index],
        output_order,
        threshold,
    )
    inference_time_seconds = time.perf_counter() - inference_started
    process_memory_after = _process_memory_usage()
    available_after = _available_memory_bytes()
    metadata = {
        "pipeline": "pretrained_monai_swinunetr",
        "checkpoint": str(checkpoint_path),
        "case_id": case.case_id,
        "checkpoint_key": "state_dict",
        "modalities": list(modalities),
        "region_order": output_order,
        "unavailable_regions": {"RC": "Not documented for this checkpoint; no RC mask was produced."},
        "output_shape": list(expected_shape),
        "input_affine": affine.tolist(),
        "spacing_mm": list(volume["spacing"]),
        "orientation": None,
        "orientation_policy": "native uploaded voxel axes; no reorientation or resampling",
        "preprocessing": model_spec["preprocessing"],
        "inference": inference,
        "probability_summary": probability_summary,
        "mask_voxel_count": mask_voxel_count,
        "visualization": "segmentation_visualization.png",
        "memory_preflight": memory,
        "memory_usage": {
            "process_before": process_memory_before,
            "process_after": process_memory_after,
            "available_ram_after_bytes": available_after,
            "cuda_preflight": cuda_memory_preflight,
            "cuda_after": _cuda_memory_usage(selected_device),
        },
        "inference_time_seconds": inference_time_seconds,
        "device": str(selected_device),
        "torch_version": torch.__version__,
        "monai_version": __import__("monai").__version__,
        "clinical_performance": "unvalidated",
    }
    (output_dir / "prediction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    del probabilities, image, volume
    gc.collect()
    return output_dir


def _ontology_metadata() -> dict[str, Any]:
    return {
        "ontology_id": ONTOLOGY_ID,
        "channel_order": list(REGION_ORDER),
        "raw_label_definitions": {
            name: list(REGION_LABELS[name]) for name in REGION_ORDER
        },
        "target_representation": "overlapping_binary_masks",
        "release_provenance_verified": False,
    }


def _regions(config: dict[str, Any]) -> dict[str, list[int]]:
    label_scheme = config.get("label_scheme")
    if label_scheme != ONTOLOGY_ID:
        raise ValueError(
            f"Unsupported label_scheme {label_scheme!r}; expected {ONTOLOGY_ID!r}. "
            "Legacy L1-L4 configurations must not be reinterpreted as WT/TC/ET/RC."
        )
    configured_order = config.get("region_order")
    if configured_order != list(REGION_ORDER):
        raise ValueError(
            f"region_order must be {list(REGION_ORDER)}; received {configured_order!r}"
        )
    normalized = normalize_region_labels(config.get("region_labels", DEFAULT_REGIONS))
    return {name: list(normalized[name]) for name in REGION_ORDER}


def _validate_checkpoint_ontology(checkpoint: dict[str, Any]) -> None:
    ontology = checkpoint.get("ontology")
    if ontology is None:
        raise ValueError(
            "Checkpoint has no verified 3D segmentation ontology metadata. "
            "Legacy checkpoints cannot be loaded as WT/TC/ET/RC; retrain with "
            "the approved ontology or use the checkpoint only with its historical protocol."
        )
    if ontology != _ontology_metadata():
        raise ValueError(
            "Checkpoint ontology metadata does not match the approved WT/TC/ET/RC "
            "overlapping-mask ontology; refusing to load it."
        )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing its resolved configuration.")
    _regions(config)


def _load_compatible_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid 3D checkpoint format: {checkpoint_path}")
    _validate_checkpoint_ontology(checkpoint)
    return checkpoint


def _require_new_or_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"Refusing to overwrite existing experiment/evaluation outputs in {output_dir}. "
            "Choose a new output directory."
        )


def _modalities(config: dict[str, Any]) -> tuple[str, ...]:
    modalities = tuple(config.get("modalities", DEFAULT_MODALITIES))
    if len(modalities) != 4 or len(set(modalities)) != 4:
        raise ValueError("BraTS requires four unique MRI modalities")
    return modalities


def _model(config: dict[str, Any], model_name: str):
    try:
        import monai
    except ImportError as error:
        raise RuntimeError("MONAI training requires `pip install -e .[brats]`") from error
    del monai
    model_config = config.get("model", {})
    feature_key = {
        "swinunetr": "swinunetr_feature_size",
        "resnet3d": "resnet3d_feature_size",
        "efficientnet3d": "efficientnet3d_feature_size",
    }.get(model_name, "segresnet_feature_size")
    return build_monai_model(
        model_name,
        patch_size=config.get("patch_size", [96, 96, 96]),
        in_channels=4,
        out_channels=len(REGION_ORDER),
        feature_size=int(model_config.get(feature_key, 48 if model_name == "swinunetr" else 16)),
    )


def _loader(
    dataset: BraTSVolumeDataset,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=worker_seed,
        pin_memory=torch.cuda.is_available(),
    )


def _soft_dice_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch.sigmoid(logits)
    spatial_dims = tuple(range(2, logits.ndim))
    intersection = (probabilities * target).sum(dim=spatial_dims)
    denominator = probabilities.sum(dim=spatial_dims) + target.sum(dim=spatial_dims)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 0.5 * bce + 0.5 * (1.0 - dice.mean())


def _predict_logits(
    model, image: torch.Tensor, patch_size: tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    from monai.inferers import sliding_window_inference

    model.eval()
    with torch.inference_mode():
        prediction = sliding_window_inference(
            inputs=image.unsqueeze(0).to(device),
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )[0]
    return prediction


def _evaluate_cases(
    model,
    cases: list[BraTSCase],
    config: dict[str, Any],
    device: torch.device,
    *,
    output_dir: Path | None = None,
    compute_hd95: bool = True,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    regions = _regions(config)
    region_names = list(REGION_ORDER)
    patch_size = tuple(int(value) for value in config.get("patch_size", [96, 96, 96]))
    modalities = _modalities(config)
    rows: list[dict[str, Any]] = []
    sample_visualizations: list[dict[str, Any]] = []
    losses = []
    model.eval()
    for case in cases:
        volume = load_full_volume(
            case,
            modalities=modalities,
            orientation=str(config.get("orientation", "RAS")),
            region_labels=regions,
        )
        image = volume["image"]
        target = volume["label"].numpy() >= 0.5
        logits = _predict_logits(model, image, patch_size, device)
        losses.append(
            float(
                _soft_dice_bce(
                    logits.unsqueeze(0), volume["label"].unsqueeze(0).to(logits.device)
                ).item()
            )
        )
        prediction = torch.sigmoid(logits).cpu().numpy() >= 0.5
        metrics = region_metrics(
            prediction, target, region_names, volume["spacing"], compute_hd95=compute_hd95
        )
        for region_name, values in metrics.items():
            rows.append({"case_id": case.case_id, "patient_group": case.patient_group,
                         "region": region_name, **values})
        if output_dir is not None:
            _save_prediction_nifti(output_dir, case.case_id, prediction, region_names, volume["affine"])
        if len(sample_visualizations) < 4:
            image_channel = modalities.index("t1c") if "t1c" in modalities else 0
            sample_visualizations.append(
                {
                    "case_id": case.case_id,
                    "image": image[image_channel].numpy(),
                    "target": target,
                    "prediction": prediction,
                }
            )
    if not rows:
        raise ValueError("Cannot evaluate an empty BraTS split")
    return float(np.mean(losses)), rows, sample_visualizations


def _summarize(
    rows: list[dict[str, Any]],
    split: str,
    loss: float | None,
    region_definitions: dict[str, list[int]],
) -> dict[str, Any]:
    region_names = list(REGION_ORDER)
    observed_names = {row["region"] for row in rows}
    if observed_names != set(region_names):
        raise ValueError(
            f"Metric rows must contain exactly {region_names}; received {sorted(observed_names)}"
        )
    per_region = {}
    for region_name in region_names:
        subset = [row for row in rows if row["region"] == region_name]
        valid_hd95 = [row["hd95_mm"] for row in subset if row["hd95_mm"] is not None]
        per_region[region_name] = {
            "case_count": len(subset),
            "mean_dice": float(np.mean([row["dice"] for row in subset])),
            "mean_iou": float(np.mean([row["iou"] for row in subset])),
            "mean_hd95_mm_when_valid": float(np.mean(valid_hd95)) if valid_hd95 else None,
            "hd95_valid_case_count": len(valid_hd95),
        }
    return {
        "split": split,
        "case_count": len({row["case_id"] for row in rows}),
        "region_count": len(region_names),
        "bce_plus_soft_dice_loss": loss,
        "per_region": per_region,
        "region_order": list(REGION_ORDER),
        "region_definitions": {
            name: region_definitions[name] for name in REGION_ORDER
        },
        "prediction_threshold": 0.5,
        "ontology": _ontology_metadata(),
    }


def _save_prediction_nifti(
    output_dir: Path,
    case_id: str,
    prediction: np.ndarray,
    region_names: list[str],
    affine: np.ndarray,
) -> None:
    import nibabel as nib

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, region_name in enumerate(region_names):
        output_path = output_dir / f"{case_id}_{region_name}.nii.gz"
        image = nib.Nifti1Image(prediction[index].astype(np.uint8), affine)
        nib.save(image, str(output_path))


def _save_figures(
    output_dir: Path,
    rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    history: list[dict[str, float]] | None = None,
) -> None:
    if history:
        epochs = [item["epoch"] for item in history]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        axes[0].plot(epochs, [item["train_loss"] for item in history], label="train")
        axes[0].plot(epochs, [item["validation_loss"] for item in history], label="validation")
        axes[0].set(xlabel="Epoch", ylabel="BCE + soft Dice", title="Loss")
        axes[0].legend()
        axes[1].plot(epochs, [item["validation_dice"] for item in history], label="validation")
        axes[1].set(
            xlabel="Epoch",
            ylabel="Mean WT/TC/ET/RC Dice",
            ylim=(0, 1),
            title="Validation",
        )
        axes[1].legend()
        figure.savefig(output_dir / "training_curves.png", dpi=300)
        figure.savefig(output_dir / "training_curves.pdf")
        plt.close(figure)
        with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    if samples:
        figure, axes = plt.subplots(len(samples), 3, figsize=(10, 3.5 * len(samples)), squeeze=False)
        for index, sample in enumerate(samples):
            tumor = sample["target"].any(axis=0)
            if tumor.any():
                z_index = int(np.argmax(tumor.sum(axis=(0, 1))))
            else:
                z_index = sample["image"].shape[-1] // 2
            image = sample["image"][:, :, z_index]
            low, high = np.percentile(image, (1, 99))
            display = np.clip((image - low) / max(float(high - low), 1e-8), 0, 1)
            target = sample["target"][:, :, :, z_index].any(axis=0)
            prediction = sample["prediction"][:, :, :, z_index].any(axis=0)
            axes[index, 0].imshow(display, cmap="gray")
            axes[index, 0].set_title(f"{sample['case_id']} MRI")
            axes[index, 1].imshow(display, cmap="gray")
            axes[index, 1].imshow(target, cmap="Reds", alpha=0.45)
            axes[index, 1].set_title("Ground-truth union overlay (WT/TC/ET/RC)")
            axes[index, 2].imshow(display, cmap="gray")
            axes[index, 2].imshow(prediction, cmap="Blues", alpha=0.45)
            axes[index, 2].set_title("Prediction union overlay (WT/TC/ET/RC)")
            for axis in axes[index]:
                axis.axis("off")
        figure.savefig(output_dir / "representative_3d_slices.png", dpi=300)
        figure.savefig(output_dir / "representative_3d_slices.pdf")
        plt.close(figure)


def _save_evaluation(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    samples: list[dict[str, Any]],
    history: list[dict[str, float]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "per_case_region_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _save_figures(output_dir, rows, samples, history)


def train(config: dict[str, Any], model_name: str) -> Path:
    seed = int(config.get("seed", 2026))
    seed_everything(seed)
    data_root = resolve_path(config["data_root"], env_var="BRATS_ROOT")
    output_root = resolve_path(config["output_root"], env_var="OUTPUT_ROOT") / model_name
    cases = list_cases(data_root)
    training_cases, validation_cases = grouped_train_validation_split(
        cases, float(config.get("validation_fraction", 0.2)), seed
    )
    device = torch.device(runtime_device(str(config.get("device", "auto"))))
    model = _model(config, model_name).to(device)
    patch_size = tuple(int(value) for value in config.get("patch_size", [96, 96, 96]))
    modalities = _modalities(config)
    train_dataset = BraTSVolumeDataset(
        training_cases,
        patch_size=patch_size,
        modalities=modalities,
        orientation=str(config.get("orientation", "RAS")),
        augment=True,
        region_labels=_regions(config),
    )
    batch_size = int(config.get("batch_size", 1))
    train_loader = _loader(
        train_dataset, batch_size, seed, True, int(config.get("num_workers", 0))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.0001)),
        weight_decay=float(config.get("weight_decay", 0.00001)),
    )
    run_dir = create_run_dir(output_root, f"seed{seed}")
    logger = configure_logging(run_dir)
    validation_groups = sorted({case.patient_group for case in validation_cases})
    training_groups = sorted({case.patient_group for case in training_cases})
    if set(training_groups) & set(validation_groups):
        raise RuntimeError("Patient-group leakage detected in BraTS train/validation split")
    write_run_metadata(
        run_dir,
        config,
        pipeline=f"brats_3d_{model_name}",
        split_details={
            "total_cases": len(cases),
            "training_cases": len(training_cases),
            "validation_cases": len(validation_cases),
            "training_group_count": len(training_groups),
            "validation_group_count": len(validation_groups),
            "training_case_ids": sorted(case.case_id for case in training_cases),
            "validation_case_ids": sorted(case.case_id for case in validation_cases),
            "training_group_ids": training_groups,
            "validation_group_ids": validation_groups,
            "grouping_rule": "case folder prefix before final hyphen",
            "group_identity_verified": False,
            "official_test_split_available": False,
            "modalities_in_input_order": list(modalities),
            "region_order": list(REGION_ORDER),
            "region_definitions": _regions(config),
            "ontology": _ontology_metadata(),
            "checkpoint_selection_criterion": (
                "maximum mean per-case Dice across WT, TC, ET, and RC on grouped validation"
            ),
        },
    )
    update_run_metadata(run_dir, device=str(device), data_root=str(data_root.resolve()))
    history: list[dict[str, float]] = []
    best_dice = -1.0
    best_epoch = 0
    stale_epochs = 0
    patience = int(config.get("early_stopping_patience", 30))
    for epoch in range(1, int(config.get("epochs", 300)) + 1):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _soft_dice_bce(model(image), target)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * image.size(0)
        train_loss = train_loss_sum / len(train_loader.dataset)
        validation_loss, validation_rows, _ = _evaluate_cases(
            model, validation_cases, config, device, compute_hd95=False
        )
        validation_dice = float(np.mean([row["dice"] for row in validation_rows]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_dice": validation_dice,
            }
        )
        logger.info(
            "epoch=%d train_loss=%.5f validation_loss=%.5f validation_mean_region_dice=%.4f",
            epoch, train_loss, validation_loss, validation_dice,
        )
        if validation_dice > best_dice:
            best_dice = validation_dice
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "config": {key: value for key, value in config.items() if not key.startswith("_")},
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "checkpoint_selection_criterion": (
                        "maximum mean per-case Dice across WT, TC, ET, and RC on grouped validation"
                    ),
                    "ontology": _ontology_metadata(),
                },
                run_dir / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("early stopping at epoch %d; best epoch=%d", epoch, best_epoch)
                break
    checkpoint = _load_compatible_checkpoint(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    validation_loss, rows, samples = _evaluate_cases(
        model, validation_cases, config, device, output_dir=run_dir / "region_masks"
    )
    summary = _summarize(rows, "grouped_validation", validation_loss, _regions(config))
    _save_evaluation(run_dir, rows, summary, samples, history)
    update_run_metadata(
        run_dir,
        checkpoint=str((run_dir / "best.pt").resolve()),
        metrics=str((run_dir / "metrics.json").resolve()),
        best_epoch=best_epoch,
        completed_epochs=len(history),
        monai=__import__("monai").__version__,
    )
    logger.info("saved best checkpoint and validation artifacts to %s", run_dir)
    return run_dir


def evaluate_validation(checkpoint_path: Path, output_dir: Path) -> Path:
    _require_new_or_empty_output_dir(output_dir)
    checkpoint = _load_compatible_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    seed = int(checkpoint["seed"])
    seed_everything(seed)
    cases = list_cases(resolve_path(config["data_root"], env_var="BRATS_ROOT"))
    training_cases, validation_cases = grouped_train_validation_split(
        cases, float(config.get("validation_fraction", 0.2)), seed
    )
    device = torch.device(runtime_device(str(config.get("device", "auto"))))
    model = _model(config, checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loss, rows, samples = _evaluate_cases(
        model, validation_cases, config, device, output_dir=output_dir / "region_masks"
    )
    _save_evaluation(
        output_dir,
        rows,
        _summarize(rows, "grouped_validation", loss, _regions(config)),
        samples,
    )
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "split": "regenerated grouped validation partition",
                "seed": seed,
                "case_count": len(validation_cases),
                "training_case_count": len(training_cases),
                "training_group_count": len(
                    {case.patient_group for case in training_cases}
                ),
                "validation_group_count": len(
                    {case.patient_group for case in validation_cases}
                ),
                "training_case_ids": sorted(case.case_id for case in training_cases),
                "validation_case_ids": sorted(case.case_id for case in validation_cases),
                "training_group_ids": sorted(
                    {case.patient_group for case in training_cases}
                ),
                "validation_group_ids": sorted(
                    {case.patient_group for case in validation_cases}
                ),
                "grouping_rule": "case folder prefix before final hyphen",
                "patient_group_identity_verified": False,
                "data_root": str(resolve_path(config["data_root"], env_var="BRATS_ROOT")),
                "official_test_split": False,
                "model_name": checkpoint["model_name"],
                "resolved_config": config,
                "device": str(device),
                "torch_version": torch.__version__,
                "monai_version": __import__("monai").__version__,
                "prediction_threshold": 0.5,
                "ontology": _ontology_metadata(),
                "region_order": list(REGION_ORDER),
                "region_definitions": _regions(config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def predict_case(
    checkpoint_path: Path,
    case_directory: Path,
    output_dir: Path,
) -> Path:
    from src.data.brats import load_input_volume

    _require_new_or_empty_output_dir(output_dir)
    checkpoint = _load_compatible_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    case = BraTSCase(
        case_id=case_directory.name,
        patient_group=case_directory.name.rsplit("-", 1)[0],
        directory=case_directory,
    )
    volume = load_input_volume(
        case,
        modalities=_modalities(config),
        orientation=str(config.get("orientation", "RAS")),
    )
    device = torch.device(runtime_device(str(config.get("device", "auto"))))
    model = _model(config, checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    logits = _predict_logits(
        model,
        volume["image"],
        tuple(int(value) for value in config.get("patch_size", [96, 96, 96])),
        device,
    )
    prediction = torch.sigmoid(logits).cpu().numpy() >= 0.5
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_prediction_nifti(
        output_dir, case.case_id, prediction, list(_regions(config)), volume["affine"]
    )
    (output_dir / "prediction_metadata.json").write_text(
        json.dumps(
            {
                "pipeline": f"brats_3d_{checkpoint['model_name']}",
                "checkpoint": str(checkpoint_path.resolve()),
                "seed": int(checkpoint["seed"]),
                "case_id": case.case_id,
                "case_directory": str(case_directory.resolve()),
                "modalities": list(_modalities(config)),
                "region_definitions": _regions(config),
                "region_order": list(REGION_ORDER),
                "prediction_threshold": 0.5,
                "ontology": _ontology_metadata(),
                "orientation": config.get("orientation", "RAS"),
                "resolved_config": config,
                "device": str(device),
                "torch_version": torch.__version__,
                "monai_version": __import__("monai").__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train/evaluate MONAI BraTS 3D models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", default="2_3d_segmentation/configs/segmentation3d.yaml")
    train_parser.add_argument("--model", choices=(
        "segresnet", "swinunetr", "resnet3d", "efficientnet3d"
    ), required=True)
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--checkpoint", required=True, type=Path)
    eval_parser.add_argument("--output-dir", type=Path, required=True)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", required=True, type=Path)
    predict_parser.add_argument("--case-dir", required=True, type=Path)
    predict_parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "train":
        print(train(load_config(args.config), args.model))
    elif args.command == "evaluate":
        print(evaluate_validation(args.checkpoint, args.output_dir))
    else:
        print(predict_case(args.checkpoint, args.case_dir, args.output_dir))
    return 0
