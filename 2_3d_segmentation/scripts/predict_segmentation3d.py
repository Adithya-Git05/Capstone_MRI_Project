import sys

from src.segmentation3d.monai_engine import main

if __name__ == "__main__":
    sys.argv.insert(1, "predict")
    raise SystemExit(main())
