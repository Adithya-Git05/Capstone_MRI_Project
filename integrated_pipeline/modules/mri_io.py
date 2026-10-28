from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Mapping

import numpy as np


MODALITIES = ("t2f", "t1n", "t1c", "t2w")


@dataclass(frozen=True)
class MRIStudy:
    case_id: str
    modality_paths: Mapping[str, Path]
    shape: tuple[int, int, int]
    affine: np.ndarray
    spacing: tuple[float, float, float]


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
