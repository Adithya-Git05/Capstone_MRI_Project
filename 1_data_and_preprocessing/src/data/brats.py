from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.validation import BRATS_MODALITIES

DEFAULT_MODALITIES = ("t2f", "t1n", "t1c", "t2w")
REGION_ORDER = ("WT", "TC", "ET", "RC")
REGION_LABELS = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
    "RC": (4,),
}
DEFAULT_REGIONS = REGION_LABELS


def normalize_region_labels(
    region_labels: Mapping[str, Any] | None = None,
) -> dict[str, tuple[int, ...]]:
    selected = REGION_LABELS if region_labels is None else region_labels
    if set(selected) != set(REGION_ORDER):
        raise ValueError(
            f"BraTS composite regions must be exactly {REGION_ORDER}; "
            f"received {tuple(selected)}"
        )
    normalized = {
        name: tuple(int(label) for label in selected[name])
        for name in REGION_ORDER
    }
    if normalized != REGION_LABELS:
        raise ValueError(
            "BraTS composite region mappings must be WT=[1,2,3], TC=[1,3], "
            "ET=[3], RC=[4]"
        )
    return normalized


def build_region_masks(
    label: np.ndarray,
    region_labels: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Build independent, potentially overlapping masks in canonical channel order."""
    normalized = normalize_region_labels(region_labels)
    return np.stack(
        [np.isin(label, normalized[name]) for name in REGION_ORDER], axis=0
    ).astype(np.float32)


@dataclass(frozen=True)
class BraTSCase:
    case_id: str
    patient_group: str
    directory: Path


def list_cases(data_root: Path) -> list[BraTSCase]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"BraTS data root does not exist: {data_root}")
    cases = []
    for directory in sorted(path for path in data_root.iterdir() if path.is_dir()):
        cases.append(
            BraTSCase(
                case_id=directory.name,
                patient_group=directory.name.rsplit("-", 1)[0],
                directory=directory,
            )
        )
    if not cases:
        raise ValueError(f"No BraTS case directories found at {data_root}")
    return cases


def grouped_train_validation_split(
    cases: list[BraTSCase], validation_fraction: float, seed: int
) -> tuple[list[BraTSCase], list[BraTSCase]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    groups: dict[str, list[BraTSCase]] = {}
    for case in cases:
        groups.setdefault(case.patient_group, []).append(case)
    if len(groups) < 2:
        raise ValueError("At least two patient groups are needed for a grouped split")

    rng = np.random.default_rng(seed)
    group_ids = list(groups)
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda group_id: len(groups[group_id]), reverse=True)
    target_cases = len(cases) * validation_fraction
    validation_ids: set[str] = set()
    validation_case_count = 0
    for group_id in group_ids:
        if len(validation_ids) == len(group_ids) - 1:
            break
        current_distance = abs(target_cases - validation_case_count)
        after_distance = abs(
            target_cases - validation_case_count - len(groups[group_id])
        )
        if after_distance < current_distance or not validation_ids:
            validation_ids.add(group_id)
            validation_case_count += len(groups[group_id])
    if not validation_ids or len(validation_ids) == len(group_ids):
        raise RuntimeError("Could not form a nonempty patient-grouped validation split")
    train = [case for case in cases if case.patient_group not in validation_ids]
    validation = [case for case in cases if case.patient_group in validation_ids]
    return train, validation


def grouped_kfold(cases: list[BraTSCase], fold_count: int, seed: int) -> list[dict[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for case in cases:
        groups.setdefault(case.patient_group, []).append(case.case_id)
    if fold_count < 2 or fold_count > len(groups):
        raise ValueError(f"fold_count must be from 2 to {len(groups)}")
    rng = np.random.default_rng(seed)
    group_ids = list(groups)
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda group_id: len(groups[group_id]), reverse=True)
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    fold_group_counts = [0] * fold_count
    for group_id in group_ids:
        fold_index = min(range(fold_count), key=lambda index: (fold_group_counts[index], len(folds[index])))
        folds[fold_index].extend(groups[group_id])
        fold_group_counts[fold_index] += 1
    all_cases = {case.case_id for case in cases}
    splits = []
    for validation_ids in folds:
        validation_groups = {
            case.patient_group for case in cases if case.case_id in set(validation_ids)
        }
        training_ids = sorted(
            case.case_id for case in cases if case.patient_group not in validation_groups
        )
        validation_ids = sorted(validation_ids)
        if set(training_ids) & set(validation_ids) or set(training_ids) | set(validation_ids) != all_cases:
            raise RuntimeError("Invalid patient-grouped cross-validation split")
        splits.append({"train": training_ids, "val": validation_ids})
    return splits


def _load_case_arrays(
    case: BraTSCase,
    modalities: tuple[str, ...] = DEFAULT_MODALITIES,
    orientation: str | None = "RAS",
    normalize: bool = True,
    region_labels: dict[str, list[int] | tuple[int, ...]] | None = None,
    require_label: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    try:
        import nibabel as nib
        from nibabel.orientations import axcodes2ornt, ornt_transform
    except ImportError as error:
        raise RuntimeError("BraTS volume loading requires nibabel; install the [brats] extra") from error

    unknown = set(modalities) - set(BRATS_MODALITIES[:-1])
    if unknown or len(modalities) != 4 or len(set(modalities)) != 4:
        raise ValueError(f"Expected four unique BraTS modalities; invalid entries: {sorted(unknown)}")
    modality_arrays: dict[str, np.ndarray] = {}
    reference_shape = None
    reference_affine = None
    reference_header = None
    files_to_load = (*modalities, "seg") if require_label else modalities
    for modality in files_to_load:
        path = case.directory / f"{case.case_id}-{modality}.nii.gz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing BraTS modality {modality}: {path}")
        image = nib.load(str(path))
        if reference_shape is None:
            reference_shape = image.shape
            reference_affine = image.affine
            reference_header = image.header.copy()
        elif image.shape != reference_shape or not np.allclose(
            image.affine, reference_affine, atol=1e-4
        ):
            raise ValueError(f"BraTS modalities do not share shape and affine for {case.case_id}")
        data = image.get_fdata(dtype=np.float32, caching="unchanged")
        if orientation is not None:
            source_orientation = nib.orientations.io_orientation(image.affine)
            target_orientation = axcodes2ornt(tuple(orientation))
            transform = ornt_transform(source_orientation, target_orientation)
            data = nib.orientations.apply_orientation(data, transform)
        array = np.asarray(
            data, dtype=np.float32 if modality != "seg" else np.int16
        )
        modality_arrays[modality] = array

    label = modality_arrays.pop("seg") if require_label else np.zeros(reference_shape, dtype=np.int16)
    observed_labels = set(int(value) for value in np.unique(label))
    if not observed_labels.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"Unexpected BraTS labels in {case.case_id}: {sorted(observed_labels)}")
    channels = []
    for modality in modalities:
        image = modality_arrays[modality]
        if normalize:
            nonzero = image != 0
            if nonzero.any():
                mean = float(image[nonzero].mean())
                std = float(image[nonzero].std())
                if std > 1e-8:
                    image = np.where(nonzero, (image - mean) / std, 0.0).astype(np.float32)
                else:
                    image = np.zeros_like(image, dtype=np.float32)
        channels.append(image)

    images = np.stack(channels, axis=0).astype(np.float32, copy=False)
    regions = build_region_masks(label, region_labels)
    if orientation is None:
        output_affine = reference_affine.copy()
    else:
        affine_oriented = nib.orientations.inv_ornt_aff(
            ornt_transform(
                nib.orientations.io_orientation(reference_affine),
                axcodes2ornt(tuple(orientation)),
            ),
            reference_shape,
        )
        output_affine = reference_affine @ affine_oriented
    spacing = tuple(float(v) for v in reference_header.get_zooms()[:3])
    return images, regions, output_affine, spacing


class BraTSVolumeDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        cases: list[BraTSCase],
        patch_size: tuple[int, int, int] | None = None,
        modalities: tuple[str, ...] = DEFAULT_MODALITIES,
        orientation: str = "RAS",
        augment: bool = False,
        region_labels: dict[str, list[int] | tuple[int, ...]] | None = None,
    ) -> None:
        self.cases = cases
        self.patch_size = patch_size
        self.modalities = modalities
        self.orientation = orientation
        self.augment = augment
        self.region_labels = region_labels

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case = self.cases[index]
        images, regions, affine, spacing = _load_case_arrays(
            case, self.modalities, self.orientation, region_labels=self.region_labels
        )
        if self.patch_size:
            images, regions = _random_crop(images, regions, self.patch_size)
        if self.augment:
            for axis in (-1, -2, -3):
                if torch.rand(()) < 0.5:
                    images = np.flip(images, axis=axis).copy()
                    regions = np.flip(regions, axis=axis).copy()
        return {
            "image": torch.from_numpy(images),
            "label": torch.from_numpy(regions),
            "case_id": case.case_id,
            "patient_group": case.patient_group,
            "affine": affine,
            "spacing": spacing,
        }


def load_full_volume(
    case: BraTSCase,
    modalities: tuple[str, ...] = DEFAULT_MODALITIES,
    orientation: str = "RAS",
    region_labels: dict[str, list[int] | tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    images, regions, affine, spacing = _load_case_arrays(
        case, modalities, orientation, region_labels=region_labels
    )
    return {
        "image": torch.from_numpy(images),
        "label": torch.from_numpy(regions),
        "case_id": case.case_id,
        "patient_group": case.patient_group,
        "affine": affine,
        "spacing": spacing,
    }


def load_input_volume(
    case: BraTSCase,
    modalities: tuple[str, ...] = DEFAULT_MODALITIES,
    orientation: str | None = "RAS",
) -> dict[str, Any]:
    if orientation is None:
        try:
            import nibabel as nib
        except ImportError as error:
            raise RuntimeError("BraTS volume loading requires nibabel") from error
        if len(modalities) != 4 or len(set(modalities)) != 4:
            raise ValueError("Expected four unique MRI modalities.")

        reference_shape = None
        reference_affine = None
        reference_header = None
        images = None
        spacing = None
        for channel, modality in enumerate(modalities):
            path = case.directory / f"{case.case_id}-{modality}.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing BraTS modality {modality}: {path}")
            nifti = nib.load(str(path), mmap=True)
            if len(nifti.shape) != 3:
                raise ValueError(f"{modality} must be a 3D volume; received {nifti.shape}.")
            affine = np.asarray(nifti.affine, dtype=np.float64)
            if reference_shape is None:
                reference_shape = tuple(int(size) for size in nifti.shape)
                reference_affine = affine.copy()
                reference_header = nifti.header.copy()
                spacing = tuple(float(value) for value in nifti.header.get_zooms()[:3])
                images = np.empty((len(modalities), *reference_shape), dtype=np.float32)
            elif nifti.shape != reference_shape or not np.allclose(
                affine, reference_affine, atol=1e-4
            ):
                raise ValueError(
                    f"BraTS modalities do not share shape and affine for {case.case_id}"
                )
            data = nifti.get_fdata(dtype=np.float32, caching="unchanged")
            if not np.isfinite(data).all():
                raise ValueError(f"{modality} contains non-finite voxel values.")
            nonzero = data != 0
            if not nonzero.any():
                raise ValueError(f"{modality} is entirely zero-valued.")
            mean = float(np.mean(data, where=nonzero, dtype=np.float64))
            std = float(np.std(data, where=nonzero, dtype=np.float64))
            if std > 1e-8:
                np.subtract(data, mean, out=data)
                np.divide(data, std, out=data)
                data[~nonzero] = 0.0
            else:
                data.fill(0.0)
            images[channel] = data
            del data, nonzero, nifti

        assert images is not None
        return {
            "image": torch.from_numpy(images),
            "case_id": case.case_id,
            "patient_group": case.patient_group,
            "affine": reference_affine,
            "spacing": spacing,
            "header": reference_header,
            "orientation": None,
        }

    images, _, affine, spacing = _load_case_arrays(
        case, modalities, orientation, require_label=False
    )
    try:
        import nibabel as nib
    except ImportError as error:
        raise RuntimeError("BraTS volume loading requires nibabel") from error
    reference_header = nib.load(
        str(case.directory / f"{case.case_id}-{modalities[0]}.nii.gz")
    ).header.copy()
    return {
        "image": torch.from_numpy(images),
        "case_id": case.case_id,
        "patient_group": case.patient_group,
        "affine": affine,
        "spacing": spacing,
        "header": reference_header,
        "orientation": orientation,
    }


def _random_crop(
    images: np.ndarray,
    labels: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    padding = []
    for current, target in zip(images.shape[1:], patch_size):
        missing = max(0, target - current)
        padding.extend((missing // 2, missing - missing // 2))
    pad_width = [(0, 0)]
    for index in range(0, len(padding), 2):
        pad_width.append((padding[index], padding[index + 1]))
    if any(left or right for left, right in pad_width[1:]):
        images = np.pad(images, pad_width, mode="constant")
        labels = np.pad(labels, pad_width, mode="constant")
    starts = [
        int(np.random.randint(0, size - target + 1))
        for size, target in zip(images.shape[1:], patch_size)
    ]
    slices = tuple(slice(start, start + target) for start, target in zip(starts, patch_size))
    return images[(slice(None), *slices)], labels[(slice(None), *slices)]
