import sys

from src.segmentation2d.engine_attention import main

if __name__ == "__main__":
    sys.argv.insert(1, "train")
    raise SystemExit(main())
