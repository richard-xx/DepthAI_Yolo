# coding=utf-8
from __future__ import annotations

import argparse
from pathlib import Path

import blobconverter
from depthai_sdk import ArgsParser, OakCamera

ROOT = Path(__file__).parent
model_dir = ROOT.joinpath("models")
blobconverter.set_defaults(output_dir=model_dir, version="2022.1")

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-conf", "--config", help="Trained YOLO JSON config path", default="json/yolov4_tiny_coco_416x416.json", type=str
)
parser.add_argument(
    "-s",
    "--spatial",
    help="Display spatial information",
    action="store_true",
    default=False,
)
args = ArgsParser.parseArgs(parser)

with OakCamera(args=args) as oak:
    color = oak.create_camera("color")
    nn = oak.create_nn(args["config"], color, nn_type="yolo", spatial=True if args["spatial"] else None)
    oak.visualize(nn, fps=True, scale=2 / 3)
    if args["spatial"]:
        oak.visualize(nn.out.spatials, fps=True)
    oak.start(blocking=True)
