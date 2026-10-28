# 7. Grad-CAM visualizations

The Grad-CAM implementation is `7_gradcam/gradcam.py`; the Streamlit
integration displays it for a compatible custom BRISC classifier checkpoint.
Its supported target and model handoff notes are in `7_gradcam/handoff/`. It is
an activation visualization,
not a segmentation metric, model validation, or clinical explanation.

See [`../integrated_pipeline/README.md`](../integrated_pipeline/README.md)
for launch requirements and the checkpoint/domain-shift limitations.
