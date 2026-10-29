from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parents[1]
output = Path(__file__).with_name("Capstone_Source_Snapshot.zip")
include = ["README.md", "COLAB_IMPLEMENTATION_AUDIT.md", "pyproject.toml", "requirements.txt",
           "src", "pipeline", "integrated_pipeline", "1_data_and_preprocessing/src",
           "1_data_and_preprocessing/scripts", "1_data_and_preprocessing/README.md",
           "2_3d_segmentation/src", "2_3d_segmentation/scripts", "2_3d_segmentation/configs",
           "2_3d_segmentation/README.md", "3_2d_segmentation/src", "3_2d_segmentation/scripts",
           "3_2d_segmentation/configs", "3_2d_segmentation/README.md", "4_2d_classification/src",
           "4_2d_classification/scripts", "4_2d_classification/configs", "4_2d_classification/README.md",
           "5_surgical_risk_mapping/pipeline", "5_surgical_risk_mapping/configs",
           "5_surgical_risk_mapping/scripts", "5_surgical_risk_mapping/README.md",
           "6_rag_treatment_planning/src", "6_rag_treatment_planning/api",
           "6_rag_treatment_planning/scripts", "6_rag_treatment_planning/requirements.txt",
           "6_rag_treatment_planning/README.md", "6_rag_treatment_planning/knowledge_base/chunks.json",
           "7_gradcam/gradcam.py", "7_gradcam/README.md", "notebooks/Capstone_Full_Demo.ipynb",
           "notebooks/colab_demo_helpers.py", "tests/test_colab_demo_helpers.py"]
excluded_parts = {"__pycache__", ".pytest_cache", "outputs", "data", "vector_db", ".git"}
with ZipFile(output, "w", ZIP_DEFLATED) as archive:
    for item in include:
        path = root / item
        if not path.exists():
            continue
        paths = [path] if path.is_file() else path.rglob("*")
        for file in paths:
            if not file.is_file() or excluded_parts.intersection(file.parts):
                continue
            if file.suffix.lower() in {".pt", ".pth", ".ckpt", ".nii", ".gz", ".pdf", ".jpg", ".jpeg", ".png"}:
                continue
            archive.write(file, file.relative_to(root))
with ZipFile(output) as archive:
    assert archive.testzip() is None
print(output, output.stat().st_size)
