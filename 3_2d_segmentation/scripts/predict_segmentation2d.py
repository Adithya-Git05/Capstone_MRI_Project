import sys

from src.segmentation2d.engine import main

if __name__ == "__main__":
    sys.argv.insert(1, "predict")
    raise SystemExit(main())
