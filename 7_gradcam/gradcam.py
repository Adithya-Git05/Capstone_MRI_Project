from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def classifier_gradcam(
    checkpoint_path: Path, slice_image: np.ndarray, target_class: str
) -> tuple[np.ndarray, str]:
    """Create Grad-CAM for the trained BRISC CNN's final convolutional feature map."""
    from src.classification.model import BRISCClassifier
    from src.data.brisc import CLASS_NAMES
    from .model_adapters import _checkpoint, _tensor

    checkpoint = _checkpoint(checkpoint_path)
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Classifier checkpoint class order does not match BRISC.")
    if target_class not in CLASS_NAMES:
        raise ValueError(f"Unknown BRISC target class {target_class!r}.")
    config = checkpoint["model_config"]
    model = BRISCClassifier(
        width=int(config.get("width", 32)),
        dropout=float(config.get("dropout", 0.25)),
        num_classes=len(CLASS_NAMES),
    )
    try:
        model.load_state_dict(checkpoint["model_state"])
    except (RuntimeError, KeyError) as error:
        raise ValueError("Grad-CAM supports the BRISC custom CNN checkpoint only.") from error
    model.eval()
    activation: list[torch.Tensor] = []
    gradient: list[torch.Tensor] = []
    layer = model.features[14]
    def capture_activation(_module, _inputs, output):
        activation.append(output)
        output.register_hook(lambda grad: gradient.append(grad))

    handle_f = layer.register_forward_hook(capture_activation)
    try:
        size = int(checkpoint["image_size"])
        resize_mode = checkpoint.get("preprocessing", {}).get("resize_mode", "stretch")
        image = _tensor(
            slice_image,
            size,
            pad=resize_mode == "pad",
            normalize_to_minus_one=True,
        )
        logits = model(image)
        class_index = CLASS_NAMES.index(target_class)
        model.zero_grad(set_to_none=True)
        logits[0, class_index].backward()
    finally:
        handle_f.remove()
    if not activation or not gradient:
        raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
    weights = gradient[0].mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activation[0]).sum(dim=1, keepdim=True))
    heatmap = torch.nn.functional.interpolate(
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
    rgba = np.asarray(figure.canvas.buffer_rgba())
    output = rgba[:, :, :3].copy()
    plt.close(figure)
    return output, "BRISCClassifier.features[14]"
