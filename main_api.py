#!/usr/bin/env python3
"""
The code is edited from docs (https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/)
We add parsing from JSON files that contain configuration
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
import blobconverter

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-m",
    "--model",
    help="Provide model name or model path for inference",
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
parser.add_argument(
    "--syncNN",
    help="Show synced frame",
    action="store_true",
    default=False,
)
args = parser.parse_args()
print("args: {}".format(args))

if args.spatial:
    lr = True  # Better handling for occlusions
    extended = False  # Closer-in minimum depth, disparity range is doubled
    subpixel = (
        False  # Better accuracy for longer distance, fractional disparity 32-levels
    )

# parse config
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split("x")))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print("config: {}".format(metadata))

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(
        blobconverter.from_zoo(
            args.model,
            shaves=6,
            zoo_type="depthai",
            use_cache=True,
            output_dir="models",
        )
    )


def drawText(
    frame,
    text,
    org,
    color=(255, 255, 255),
    bg_color=(128, 128, 128),
    fontScale=0.5,
    thickness=1,
):
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale,
        bg_color,
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def drawRect(
    frame, p1, p2, color=(255, 255, 255), bg_color=(128, 128, 128), thickness=1
):
    cv2.rectangle(frame, pt1=p1, pt2=p2, color=bg_color, thickness=thickness + 3)
    cv2.rectangle(frame, pt1=p1, pt2=p2, color=color, thickness=thickness)


def getDeviceInfo(deviceId=None, debug=False) -> dai.DeviceInfo:
    """
    Find a correct :obj:`depthai.DeviceInfo` object, either matching provided :code:`deviceId` or selected by the user (if multiple devices available)
    Useful for almost every app where there is a possibility of multiple devices being connected simultaneously

    Args:
        deviceId (str, optional): Specifies device MX ID, for which the device info will be collected

    Returns:
        depthai.DeviceInfo: Object representing selected device info

    Raises:
        RuntimeError: if no DepthAI device was found or, if :code:`deviceId` was specified, no device with matching MX ID was found
        ValueError: if value supplied by the user when choosing the DepthAI device was incorrect
    """
    deviceInfos = []
    if debug:
        deviceInfos = dai.XLinkConnection.getAllConnectedDevices()
    else:
        deviceInfos = dai.Device.getAllAvailableDevices()

    if len(deviceInfos) == 0:
        raise RuntimeError("No DepthAI device found!")
    else:
        print("Available devices:")
        for i, deviceInfo in enumerate(deviceInfos):
            print(
                f"[{i}] {deviceInfo.name} {deviceInfo.getMxId()} [{deviceInfo.state.name}]"
            )

        if deviceId == "list":
            raise SystemExit(0)
        elif deviceId is not None:
            matchingDevice = next(
                filter(lambda info: info.getMxId() == deviceId, deviceInfos), None
            )
            if matchingDevice is None:
                raise RuntimeError(
                    f"No DepthAI device found with id matching {deviceId} !"
                )
            return matchingDevice
        elif len(deviceInfos) == 1:
            return deviceInfos[0]
        else:
            val = input("Which DepthAI Device you want to use: ")
            try:
                return deviceInfos[int(val)]
            except:
                raise ValueError("Incorrect value supplied: {}".format(val))


def create_pipeline():
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    if args.spatial:
        monoLeft = pipeline.createMonoCamera()
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setFps(60)

        monoRight = pipeline.createMonoCamera()
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setFps(60)

        stereo = pipeline.createStereoDepth()
        stereo.initialConfig.setConfidenceThreshold(245)
        stereo.setLeftRightCheck(lr)
        stereo.setExtendedDisparity(extended)
        stereo.setSubpixel(subpixel)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        if args.syncNN:
            stereo.setOutputSize(*camRgb.getPreviewSize())
        else:
            stereo.setOutputSize(*camRgb.getVideoSize())
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        detectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

        stereo.depth.link(detectionNetwork.inputDepth)
        detectionNetwork.setDepthLowerThreshold(100)
        detectionNetwork.setDepthUpperThreshold(10000)
        detectionNetwork.setBoundingBoxScaleFactor(0.3)
    else:
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    nnOut = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    nnOut.setStreamName("nn")

    # Properties
    camRgb.setPreviewSize(W, H)

    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(60)
    camRgb.setPreviewKeepAspectRatio(not args.fullFov)

    # Network specific settings
    detectionNetwork.setConfidenceThreshold(confidenceThreshold)
    detectionNetwork.setNumClasses(classes)
    detectionNetwork.setCoordinateSize(coordinates)
    detectionNetwork.setAnchors(anchors)
    detectionNetwork.setAnchorMasks(anchorMasks)
    detectionNetwork.setIouThreshold(iouThreshold)
    detectionNetwork.setBlobPath(nnPath)
    # detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)
    detectionNetwork.input.setQueueSize(1)

    # Linking
    camRgb.preview.link(detectionNetwork.input)
    if args.syncNN:
        detectionNetwork.passthrough.link(xoutRgb.input)
    else:
        camRgb.video.link(xoutRgb.input)
    detectionNetwork.out.link(nnOut.input)

    return pipeline


def main():
    # Connect to device and start pipeline
    with dai.Device(create_pipeline(), getDeviceInfo()) as device:
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        frame = None
        detections = []
        startTime = time.monotonic()
        counter = 0
        bboxColors = (
            np.random.random(size=(256, 3)) * 256
        )  # Random Colors for bounding boxes

        # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
        def frameNorm(frame, bbox, NN_SIZE=None):
            if NN_SIZE is not None:
                # Check difference in aspect ratio and apply correction to BBs
                ar_diff = NN_SIZE[0] / NN_SIZE[1] - frame.shape[0] / frame.shape[1]
                sel = 0 if 0 < ar_diff else 1
                bbox[sel::2] *= 1 - abs(ar_diff)
                bbox[sel::2] += abs(ar_diff) / 2
            # Normalize bounding boxes
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(bbox, 0, 1) * normVals).astype(int)

        def displayFrame(name, frame, detections):
            for detection in detections:
                bbox = frameNorm(
                    frame,
                    (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
                    None if args.fullFov else [W, H]
                )
                drawText(
                    frame,
                    labels[detection.label],
                    (bbox[0] + 10, bbox[1] + 20),
                    bboxColors[detection.label],
                )
                drawText(
                    frame,
                    f"{int(detection.confidence * 100)}%",
                    (bbox[0] + 10, bbox[1] + 40),
                    bboxColors[detection.label],
                )
                drawRect(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    bboxColors[detection.label],
                )
                if hasattr(
                    detection, "spatialCoordinates"
                ):  # Display spatial coordinates as well
                    xMeters = detection.spatialCoordinates.x / 1000
                    yMeters = detection.spatialCoordinates.y / 1000
                    zMeters = detection.spatialCoordinates.z / 1000
                    drawText(
                        frame,
                        "X: {:.2f} m".format(xMeters),
                        (bbox[0] + 10, bbox[1] + 60),
                    )
                    drawText(
                        frame,
                        "Y: {:.2f} m".format(yMeters),
                        (bbox[0] + 10, bbox[1] + 75),
                    )
                    drawText(
                        frame,
                        "Z: {:.2f} m".format(zMeters),
                        (bbox[0] + 10, bbox[1] + 90),
                    )
            # Show the frame
            cv2.imshow(
                name,
                frame if args.syncNN else cv2.resize(frame, (0, 0), fx=0.5, fy=0.5),
            )

        while True:
            inRgb = qRgb.get()
            inDet = qDet.get()

            if inRgb is not None:
                frame = inRgb.getCvFrame()
                drawText(
                    frame,
                    "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                    (2, frame.shape[0] - 4),
                    fontScale=1,
                )

            if inDet is not None:
                detections = inDet.detections
                counter += 1

            if frame is not None:
                displayFrame("rgb", frame, detections)

            if cv2.waitKey(1) == ord("q"):
                break


if __name__ == "__main__":
    main()
