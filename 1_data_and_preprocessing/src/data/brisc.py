from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

CLASS_NAMES = ("glioma", "meningioma", "no_tumor", "pituitary")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
Split = Literal["train", "test"]


@dataclass(frozen=True)
class ClassificationRecord:
    path: Path
    label: int


@dataclass(frozen=True)
class SegmentationRecord:
    image_path: Path
    mask_path: Path


def classification_records(data_root: Path, split: Split) -> list[ClassificationRecord]:
    split_root = data_root / "classification_task" / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"BRISC classification split not found: {split_root}")
    records = []
    for class_name in CLASS_NAMES:
        class_dir = split_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"BRISC class directory not found: {class_dir}")
        records.extend(
            ClassificationRecord(path, CLASS_TO_INDEX[class_name])
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    if not records:
        raise ValueError(f"No BRISC classification images found in {split_root}")
    return records


def segmentation_records(data_root: Path, split: Split) -> list[SegmentationRecord]:
    split_root = data_root / "segmentation_task" / split
    image_dir = split_root / "images"
    mask_dir = split_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"BRISC segmentation split is incomplete: {split_root}")
    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }
    if images.keys() != masks.keys():
        missing = sorted(images.keys() - masks.keys())[:5]
        orphaned = sorted(masks.keys() - images.keys())[:5]
        raise ValueError(
            f"BRISC image/mask stems do not match in {split_root}; "
            f"missing masks={missing}, orphan masks={orphaned}"
        )
    records = [
        SegmentationRecord(images[stem], masks[stem]) for stem in sorted(images)
    ]
    if not records:
        raise ValueError(f"No BRISC segmentation pairs found in {split_root}")
    return records


def stratified_split(
    records: list[ClassificationRecord],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    training, validation, _ = duplicate_aware_split(
        records, validation_fraction, 0.0, seed
    )
    return training, validation


def classification_image_hashes(
    records: list[ClassificationRecord],
) -> list[str]:
    hashes = []
    for record in records:
        digest = hashlib.sha256()
        with record.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes.append(digest.hexdigest())
    return hashes


def duplicate_aware_split(
    records: list[ClassificationRecord],
    validation_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if not 0 <= calibration_fraction < 1:
        raise ValueError("calibration_fraction must be in [0, 1)")
    if validation_fraction + calibration_fraction >= 1:
        raise ValueError("validation and calibration fractions must sum to less than one")

    hashes = classification_image_hashes(records)
    groups_by_class: dict[int, dict[str, list[int]]] = {}
    hash_labels: dict[str, int] = {}
    for index, (record, digest) in enumerate(zip(records, hashes, strict=True)):
        previous_label = hash_labels.setdefault(digest, record.label)
        if previous_label != record.label:
            raise ValueError(f"Identical image hash has conflicting labels: {digest}")
        groups_by_class.setdefault(record.label, {}).setdefault(digest, []).append(index)

    rng = np.random.default_rng(seed)
    training_indices: list[int] = []
    validation_indices: list[int] = []
    calibration_indices: list[int] = []
    for groups in groups_by_class.values():
        shuffled = list(groups.values())
        rng.shuffle(shuffled)
        minimum_training_groups = 1
        minimum_calibration_groups = int(calibration_fraction > 0)
        if len(shuffled) <= minimum_training_groups + minimum_calibration_groups:
            raise ValueError(
                "Each class needs distinct image groups for training, validation, "
                "and calibration"
            )
        n_validation = min(
            len(shuffled) - minimum_training_groups - minimum_calibration_groups,
            max(1, round(len(shuffled) * validation_fraction)),
        )
        remaining_groups = len(shuffled) - n_validation
        n_calibration = (
            min(
                remaining_groups - minimum_training_groups,
                max(1, round(len(shuffled) * calibration_fraction)),
            )
            if calibration_fraction > 0
            else 0
        )
        for group in shuffled[:n_validation]:
            validation_indices.extend(group)
        for group in shuffled[n_validation : n_validation + n_calibration]:
            calibration_indices.extend(group)
        for group in shuffled[n_validation + n_calibration :]:
            training_indices.extend(group)

    rng.shuffle(training_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(calibration_indices)
    return training_indices, validation_indices, calibration_indices


class BRISCClassificationDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        records: list[ClassificationRecord],
        image_size: int,
        *,
        resize_mode: Literal["pad", "stretch"] = "stretch",
        augment: bool = False,
        augmentation: Mapping[str, float | bool] | None = None,
    ) -> None:
        if resize_mode not in {"pad", "stretch"}:
            raise ValueError("resize_mode must be 'pad' or 'stretch'")
        self.records = records
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.augment = augment
        self.augmentation = dict(augmentation or {})

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, image: Image.Image) -> Image.Image:
        if random.random() < float(self.augmentation.get("horizontal_flip_probability", 0.5)):
            image = ImageOps.mirror(image)
        rotation = float(self.augmentation.get("rotation_degrees", 7.0))
        if rotation > 0:
            image = image.rotate(
                random.uniform(-rotation, rotation),
                resample=Image.Resampling.BILINEAR,
                fillcolor=0,
            )
        brightness_delta = float(self.augmentation.get("brightness_delta", 0.08))
        contrast_delta = float(self.augmentation.get("contrast_delta", 0.08))
        if brightness_delta > 0:
            factor = random.uniform(1 - brightness_delta, 1 + brightness_delta)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if contrast_delta > 0:
            factor = random.uniform(1 - contrast_delta, 1 + contrast_delta)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("L")
        if self.augment:
            image = self._augment(image)
        size = (self.image_size, self.image_size)
        if self.resize_mode == "pad":
            image = ImageOps.pad(
                image,
                size,
                method=Image.Resampling.BILINEAR,
                color=0,
                centering=(0.5, 0.5),
            )
        else:
            image = image.resize(size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return (tensor - 0.5) / 0.5, record.label


class BRISCSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: list[SegmentationRecord],
        image_size: int,
        mask_threshold: int = 128,
        augment: bool = False,
    ) -> None:
        if not 0 <= mask_threshold <= 255:
            raise ValueError("mask_threshold must be in [0, 255]")
        self.records = records
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        size = (self.image_size, self.image_size)
        with Image.open(record.image_path) as source:
            image = source.convert("L").resize(size, Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
        with Image.open(record.mask_path) as source:
            binary = source.convert("L").point(
                lambda value: 255 if value >= self.mask_threshold else 0
            )
            mask = binary.resize(size, Image.Resampling.NEAREST)
            mask_array = np.asarray(mask, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        if self.augment:
            if torch.rand(()) < 0.5:
                image_tensor = image_tensor.flip(-1)
                mask_tensor = mask_tensor.flip(-1)
            if torch.rand(()) < 0.5:
                image_tensor = image_tensor.flip(-2)
                mask_tensor = mask_tensor.flip(-2)
        return (image_tensor - 0.5) / 0.5, mask_tensor
