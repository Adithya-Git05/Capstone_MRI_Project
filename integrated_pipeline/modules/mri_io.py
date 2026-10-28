from __future__ import annotations

from dataclasses import dataclass
import gzip
from io import BytesIO
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError


MODALITIES = ("t2f", "t1n", "t1c", "t2w")
MAX_SPATIAL_ELEMENTS = 50_000_000


@dataclass(frozen=True)
class MRIStudy:
    case_id: str
    modality_paths: Mapping[str, Path]
    shape: tuple[int, int, int]
    affine: np.ndarray
    spacing: tuple[float, float, float]


@dataclass(frozen=True)
class UploadedMRI:
    data: np.ndarray
    metadata: dict[str, object]


def load_mri_upload(payload: bytes, filename: str) -> UploadedMRI:
    """Load one supported 2D image or one NIfTI volume for slice-level models."""
    if not payload:
        raise ValueError("The uploaded MRI file is empty.")
    if len(payload) > 200 * 1024 * 1024:
        raise ValueError("MRI uploads must be 200 MB or smaller.")

    lowered_name = filename.lower()
    if lowered_name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ImportError as error:
            raise RuntimeError("NIfTI uploads require nibabel.") from error

        suffix = ".nii.gz" if lowered_name.endswith(".nii.gz") else ".nii"
        try:
            with tempfile.TemporaryDirectory(prefix="mri_upload_") as temporary:
                temporary_path = Path(temporary) / f"upload{suffix}"
                temporary_path.write_bytes(payload)
                image = nib.load(str(temporary_path), mmap=False)
                if len(image.shape) != 3:
                    raise ValueError(
                        f"NIfTI input must be a 3D volume; received shape {image.shape}."
                    )
                if int(np.prod(image.shape)) > MAX_SPATIAL_ELEMENTS:
                    raise ValueError(
                        f"NIfTI volume exceeds the {MAX_SPATIAL_ELEMENTS:,}-voxel limit."
                    )
                original_shape = tuple(int(size) for size in image.shape)
                image = nib.as_closest_canonical(image)
                data = image.get_fdata(dtype=np.float32, caching="unchanged")
                affine = np.asarray(image.affine, dtype=np.float64).copy()
                voxel_spacing = tuple(
                    float(value) for value in nib.affines.voxel_sizes(affine)
                )
                orientation = "".join(nib.aff2axcodes(affine))
        except (EOFError, OSError, nib.filebasedimages.ImageFileError) as error:
            raise ValueError("The uploaded file is not a readable NIfTI volume.") from error

        if not np.isfinite(affine).all() or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-10:
            raise ValueError("NIfTI volume has invalid spatial metadata.")
        if not np.isfinite(data).all():
            raise ValueError("NIfTI volume contains non-finite voxel values.")
        if not np.any(data):
            raise ValueError("NIfTI volume is entirely zero-valued.")
        return UploadedMRI(
            data=data,
            metadata={
                "format": "NIfTI",
                "shape": tuple(int(size) for size in data.shape),
                "original_shape": original_shape,
                "voxel_spacing_mm": voxel_spacing,
                "orientation": orientation,
                "affine": affine.tolist(),
            },
        )

    if Path(lowered_name).suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Supported inputs are PNG, JPEG, NIfTI (.nii), and NIfTI (.nii.gz).")
    try:
        with Image.open(BytesIO(payload)) as opened:
            if opened.width * opened.height > MAX_SPATIAL_ELEMENTS:
                raise ValueError(
                    f"Image dimensions exceed the supported {MAX_SPATIAL_ELEMENTS:,}-pixel limit."
                )
            opened.verify()
        with Image.open(BytesIO(payload)) as opened:
            mode = opened.mode
            image = opened.convert("L")
            data = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a readable PNG or JPEG image.") from error

    if data.ndim != 2 or not np.isfinite(data).all() or not np.any(data):
        raise ValueError("Image must contain finite, non-empty pixel data.")
    return UploadedMRI(
        data=data,
        metadata={
            "format": Path(lowered_name).suffix[1:].upper(),
            "shape": tuple(int(size) for size in data.shape),
            "original_mode": mode,
            "grayscale_conversion": mode != "L",
        },
    )


def select_uploaded_slice(upload: UploadedMRI, index: int) -> np.ndarray:
    if upload.data.ndim == 2:
        if index != 0:
            raise IndexError("A 2D image has only slice index 0.")
        return np.asarray(upload.data, dtype=np.float32)
    if upload.data.ndim != 3:
        raise ValueError(f"Expected a 2D image or 3D volume, got {upload.data.ndim}D.")
    if not 0 <= index < upload.data.shape[2]:
        raise IndexError(f"Axial slice {index} is outside [0, {upload.data.shape[2] - 1}].")
    return np.asarray(upload.data[:, :, index], dtype=np.float32)


def validate_study(case_id: str, modality_paths: Mapping[str, Path]) -> MRIStudy:
    """Validate the four registered BraTS NIfTIs without discarding spatial metadata."""
    try:
        import nibabel as nib
    except ImportError as error:
        raise RuntimeError("MRI input requires nibabel; install integrated_pipeline requirements.") from error

    if not case_id.strip():
        raise ValueError("A non-empty study identifier is required.")
    if any(character in case_id for character in ("/", "\\", "..")):
        raise ValueError("Study identifier cannot contain path separators or '..'.")
    if set(modality_paths) != set(MODALITIES):
        missing = sorted(set(MODALITIES) - set(modality_paths))
        extra = sorted(set(modality_paths) - set(MODALITIES))
        raise ValueError(f"Expected modalities {MODALITIES}; missing={missing}, extra={extra}.")

    reference_shape = None
    reference_affine = None
    reference_spacing = None
    resolved: dict[str, Path] = {}
    for modality in MODALITIES:
        path = Path(modality_paths[modality])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {modality} NIfTI file: {path}")
        try:
            image = nib.load(str(path))
        except (EOFError, OSError, nib.filebasedimages.ImageFileError) as error:
            raise ValueError(f"{modality} is not a readable NIfTI volume: {path}") from error
        if len(image.shape) != 3:
            raise ValueError(f"{modality} must be a 3D volume; received shape {image.shape}.")
        if any(int(size) <= 0 for size in image.shape):
            raise ValueError(f"{modality} has an empty spatial dimension: {image.shape}.")
        if reference_shape is None:
            reference_shape = tuple(int(size) for size in image.shape)
            reference_affine = np.asarray(image.affine, dtype=np.float64).copy()
            if not np.all(np.isfinite(reference_affine)):
                raise ValueError(f"{modality} has a non-finite voxel-to-world affine.")
            if abs(float(np.linalg.det(reference_affine[:3, :3]))) < 1e-10:
                raise ValueError(f"{modality} has a singular voxel-to-world affine.")
            reference_spacing = tuple(
                float(value) for value in nib.affines.voxel_sizes(reference_affine)
            )
        elif image.shape != reference_shape or not np.allclose(
            image.affine, reference_affine, atol=1e-4
        ):
            raise ValueError(
                f"Modality {modality} is not registered to the reference volume "
                "(shape and affine must match)."
            )
        try:
            volume = image.get_fdata(dtype=np.float32, caching="unchanged")
        except (EOFError, OSError, ValueError) as error:
            raise ValueError(f"Could not read voxel data for {modality}: {path}") from error
        if not np.isfinite(volume).all():
            raise ValueError(f"{modality} contains non-finite voxel values.")
        if not np.any(volume):
            raise ValueError(f"{modality} is entirely zero-valued.")
        resolved[modality] = path.resolve()

    return MRIStudy(
        case_id=case_id.strip(),
        modality_paths=resolved,
        shape=reference_shape,
        affine=reference_affine,
        spacing=reference_spacing,
    )


def write_uploaded_study(
    case_id: str, uploaded: Mapping[str, object], directory: Path
) -> MRIStudy:
    """Write uploaded NIfTI bytes using the repository's expected case filenames."""
    if not case_id or not all(
        character.isascii() and (character.isalnum() or character in "_-")
        for character in case_id
    ):
        raise ValueError(
            "Study identifier must contain only letters, numbers, underscores, or hyphens "
            "and cannot contain path separators."
        )
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for modality in MODALITIES:
        if modality not in uploaded:
            raise ValueError(f"Required MRI modality '{modality}' was not supplied.")
        content = uploaded[modality]
        if isinstance(content, (bytes, bytearray, memoryview)):
            payload = content
        elif callable(get_buffer := getattr(content, "getbuffer", None)):
            payload = get_buffer()
        else:
            raise TypeError(
                f"Uploaded {modality} data must be bytes or provide a getbuffer() method."
            )
        if not payload:
            raise ValueError(f"Required MRI modality '{modality}' was empty.")
        path = directory / f"{case_id}-{modality}.nii.gz"
        with path.open("wb") as stream:
            if payload[:2] == b"\x1f\x8b":
                stream.write(payload)
            else:
                with gzip.GzipFile(fileobj=stream, mode="wb") as compressed:
                    compressed.write(payload)
        paths[modality] = path
    return validate_study(case_id, paths)


def validate_uploaded_study(
    case_id: str, uploaded: Mapping[str, object]
) -> dict[str, object]:
    """Validate a four-modality uploaded BraTS study without retaining its files."""
    with tempfile.TemporaryDirectory(prefix="mri_study_") as temporary:
        study = write_uploaded_study(case_id, uploaded, Path(temporary))
        import nibabel as nib

        orientation = "".join(nib.aff2axcodes(study.affine))
        return {
            "case_id": study.case_id,
            "modalities": list(MODALITIES),
            "shape": study.shape,
            "voxel_spacing_mm": study.spacing,
            "orientation": orientation,
        }
