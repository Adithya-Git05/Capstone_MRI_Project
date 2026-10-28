from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

from .model_adapters import _checkpoint, _tensor


def classifier_gradcam(
    checkpoint_path: Path,
    slice_image: np.ndarray,
    target_class: str,
    percentile_window: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    from src.classification.model import BRISCClassifier
    from src.data.brisc import CLASS_NAMES

    checkpoint = _checkpoint(checkpoint_path)
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Classifier checkpoint class order does not match BRISC.")
    if target_class not in CLASS_NAMES:
        raise ValueError(f"Unknown BRISC target class {target_class!r}.")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Classifier checkpoint is missing a valid model_config.")
    model = BRISCClassifier(
        width=int(config.get("width", 32)),
        dropout=float(config.get("dropout", 0.25)),
        num_classes=len(CLASS_NAMES),
    )
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as error:
        raise ValueError("Grad-CAM supports BRISCClassifier checkpoints only.") from error
    model.eval()

    activation: list[torch.Tensor] = []
    gradient: list[torch.Tensor] = []
    layer = model.features[14]

    def capture_activation(_module, _inputs, output):
        activation.append(output)
        output.register_hook(lambda grad: gradient.append(grad))

    handle = layer.register_forward_hook(capture_activation)
    try:
        size = int(checkpoint["image_size"])
        resize_mode = checkpoint.get("preprocessing", {}).get("resize_mode", "stretch")
        image = _tensor(
            slice_image,
            size,
            pad=resize_mode == "pad",
            normalize_to_minus_one=True,
            percentile_window=percentile_window,
        )
        logits = model(image)
        model.zero_grad(set_to_none=True)
        logits[0, CLASS_NAMES.index(target_class)].backward()
    finally:
        handle.remove()
    if not activation or not gradient:
        raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
    weights = gradient[0].mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activation[0]).sum(dim=1, keepdim=True))
    heatmap = functional.interpolate(
        heatmap, size=slice_image.shape, mode="bilinear", align_corners=False
    )[0, 0].detach().numpy()
    maximum = float(heatmap.max())
    if maximum > 0:
        heatmap /= maximum

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(slice_image, cmap="gray")
    axis.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axis.axis("off")
    figure.tight_layout(pad=0)
    figure.canvas.draw()
    overlay = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(figure)
    return heatmap, overlay, "BRISCClassifier.features[14]"
