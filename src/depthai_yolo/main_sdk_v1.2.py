#!/usr/bin/env python3
# coding=utf-8
from __future__ import annotations

import blobconverter
from depthai_sdk import Previews, getDeviceInfo

try:
    from depthai_sdk import FPSHandler
except ImportError:
    from depthai_sdk.fps import FPSHandler

import argparse
from pathlib import Path

import cv2
import depthai as dai
from depthai_sdk.managers import (
    BlobManager,
    NNetManager,
    PipelineManager,
    PreviewManager,
)

ROOT = Path(__file__).parent
model_dir = ROOT.joinpath("models")
blobconverter.set_defaults(output_dir=model_dir, version="2022.1")

# https://github.com/luxonis/depthai-model-zoo/tree/main/models
depthai_model_zoo = [
    "yolov3_coco_416x416",
    "yolov4_coco_608x608",
    "yolov4_tiny_coco_416x416",
    "yolov5n_coco_416x416",
    "yolov5n_coco_640x352",
    "yolov6n_coco_416x416",
    "yolov6n_coco_640x640",
    "yolov6nr1_coco_512x288",
    "yolov6nr1_coco_640x352",
    "yolov6nr3_coco_416x416",
    "yolov6nr3_coco_640x352",
    "yolov6t_coco_416x416",
    "yolov7_coco_416x416",
    "yolov7tiny_coco_416x416",
    "yolov7tiny_coco_640x352",
    "yolov8n_coco_416x416",
    "yolov8n_coco_640x352",
]
open_model_zoo = [
    "yolo-v3-tf",
    "yolo-v3-tiny-tf",
    "yolo-v4-tf",
    "yolo-v4-tiny-tf",
]
# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-m",
    "--model",
    help="Provide model path for inference",
    default="yolov8n_coco_640x352",
    type=str,
)
parser.add_argument(
    "-c",
    "--config",
    help="Provide config path for inference",
    default="json/yolov8n_coco_640x352.json",
    type=str,
)
parser.add_argument(
    "-s",
    "--spatial",
    help="Display spatial information",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-F",
    "--fullFov",
    help="If to :code:`False`, "
    "it will first center crop the frame to meet the NN aspect ratio and then scale down the image",
    default=True,
    type=bool,
)
parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
args = parser.parse_args()
config_path = args.config

zoo_type = "depthai"
# create blob, NN, and preview managers
if Path(args.model).exists():
    # initialize blob manager with path to the blob
    bm = BlobManager(blobPath=args.model)
else:
    # initialize blob manager with the name of the model otherwise
    nn_name = Path(args.model).stem
    if nn_name in depthai_model_zoo + open_model_zoo:
        if nn_name in open_model_zoo:
            zoo_type = "intel"
        list_file = list(model_dir.glob(f"{nn_name}*.blob"))
        if list_file:
            nn_path = list_file[0]
            bm = BlobManager(blobPath=nn_path)
        else:
            bm = BlobManager(zooName=args.model)

nm = NNetManager(nnFamily="YOLO", inputSize=(416, 416))
nm.readConfig(config_path)  # this will also parse the correct input size

pm = PipelineManager()
pm.createColorCam(
    previewSize=nm.inputSize,
    res=dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    fps=60,
    fullFov=args.fullFov,
    orientation=None,
    colorOrder=dai.ColorCameraProperties.ColorOrder.BGR,
    xout=True,
    xoutVideo=False,
    xoutStill=False,
    control=False,
)
if args.spatial:
    pm.createLeftCam(
        res=dai.MonoCameraProperties.SensorResolution.THE_400_P,
        fps=60,
        orientation=None,
        xout=False,
        control=False,
    )
    pm.createRightCam(
        res=dai.MonoCameraProperties.SensorResolution.THE_400_P,
        fps=60,
        orientation=None,
        xout=False,
        control=False,
    )
    pm.createDepth(
        dct=245,
        median=None,
        sigma=0,
        lr=True,
        lrcThreshold=5,
        extended=False,
        subpixel=False,
        useDisparity=False,
        useDepth=False,
        useRectifiedLeft=False,
        useRectifiedRight=False,
        runtimeSwitch=False,
        alignment=dai.CameraBoardSocket.RGB,
        control=False,
    )
# create preview manager
fps_handler = FPSHandler()
pv = PreviewManager(display=[Previews.color.name], fpsHandler=fps_handler)

# create NN with managers
nn = nm.createNN(
    pipeline=pm.pipeline,
    nodes=pm.nodes,
    blobPath=bm.getBlob(shaves=6, openvinoVersion=pm.pipeline.getOpenVINOVersion(), zooType=zoo_type),
    source=Previews.color.name,
    useDepth=args.spatial,
    minDepth=100,
    maxDepth=10000,
    sbbScaleFactor=0.3,
    fullFov=args.fullFov,
    useImageManip=False,
)
pm.addNn(nn)

# initialize pipeline
with dai.Device(pm.pipeline, getDeviceInfo()) as device:
    if args.spatial and device.getIrDrivers():
        device.setIrLaserDotProjectorBrightness(200)  # in mA, 0..1200
        device.setIrFloodLightBrightness(0)  # in mA, 0..1500
    # create outputs
    pv.createQueues(device)
    nm.createQueues(device)

    nn_data = []

    while True:
        # parse outputs
        pv.prepareFrames()
        in_nn = nm.outputQueue.tryGet()

        if in_nn is not None:
            nn_data = nm.decode(in_nn)
            if args.classes is not None:
                nn_data = [detection for detection in nn_data if detection.label in args.classes]
            # count FPS
            fps_handler.tick("nn")

        nm.draw(pv, nn_data)
        pv.showFrames()

        if cv2.waitKey(1) == ord("q"):
            break
