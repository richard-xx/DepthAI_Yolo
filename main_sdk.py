#!/usr/bin/env python3
# coding=utf-8
from depthai_sdk import Previews, FPSHandler, getDeviceInfo
from depthai_sdk.managers import (
    PipelineManager,
    PreviewManager,
    BlobManager,
    NNetManager,
)
import depthai as dai
import cv2
import argparse
from pathlib import Path

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-m",
    "--model",
    help="Provide model path for inference",
    default="yolov4_tiny_coco_416x416",
    type=str,
)
parser.add_argument(
    "-c",
    "--config",
    help="Provide config path for inference",
    default="json/yolov4-tiny.json",
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
args = parser.parse_args()
CONFIG_PATH = args.config

# create blob, NN, and preview managers
if Path(args.model).exists():
    # initialize blob manager with path to the blob
    bm = BlobManager(blobPath=args.model)
else:
    # initialize blob manager with the name of the model otherwise
    bm = BlobManager(zooName=args.model)

nm = NNetManager(nnFamily="YOLO", inputSize=4)
nm.readConfig(CONFIG_PATH)  # this will also parse the correct input size

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
fpsHandler = FPSHandler()
pv = PreviewManager(display=[Previews.color.name], fpsHandler=fpsHandler)

# create NN with managers
nn = nm.createNN(
    pipeline=pm.pipeline,
    nodes=pm.nodes,
    blobPath=bm.getBlob(
        shaves=6, openvinoVersion=pm.pipeline.getOpenVINOVersion(), zooType="depthai"
    ),
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
    # create outputs
    pv.createQueues(device)
    nm.createQueues(device)

    nnData = []

    while True:

        # parse outputs
        pv.prepareFrames()
        inNn = nm.outputQueue.tryGet()

        if inNn is not None:
            nnData = nm.decode(inNn)
            # count FPS
            fpsHandler.tick("nn")

        nm.draw(pv, nnData)
        pv.showFrames()

        if cv2.waitKey(1) == ord("q"):
            break
