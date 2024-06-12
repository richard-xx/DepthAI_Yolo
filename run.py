# coding=utf-8
try:
    import depthai_yolo  # noqa: F401
except ImportError:
    import sys
    from pathlib import Path

    FILE = Path(__file__).resolve()
    ROOT = FILE.parent.joinpath("src")

    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
finally:
    from depthai_yolo.cli import app

if __name__ == "__main__":
    app()
