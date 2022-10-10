#!/usr/bin/env python3
# coding=utf-8
"""
The code is edited from docs (https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/)
We add parsing from JSON files that contain configuration
"""

import collections
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
    default="True",
)
parser.add_argument(
    "--syncNN",
    help="Show synced frame",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--high_res",
    help="Show synced frame",
    action="store_true",
    default=False,
)
args = parser.parse_args()
if args.fullFov.upper() in ["TRUE", "ON"]:
    args.fullFov = True
elif args.fullFov.upper() in ["FALSE", "OFF"]:
    args.fullFov = False
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


class FPSHandler:
    """
    Class that handles all FPS-related operations. Mostly used to calculate different streams FPS, but can also be
    used to feed the video file based on it's FPS property, not app performance (this prevents the video from being sent
    to quickly if we finish processing a frame earlier than the next video frame should be consumed)
    """

    _fpsBgColor = (0, 0, 0)
    _fpsColor = (255, 255, 255)
    _fpsType = cv2.FONT_HERSHEY_SIMPLEX
    _fpsLineType = cv2.LINE_AA

    def __init__(self, cap=None, maxTicks=100):
        """
        Args:
            cap (cv2.VideoCapture, Optional): handler to the video file object
            maxTicks (int, Optional): maximum ticks amount for FPS calculation
        """
        self._timestamp = None
        self._start = None
        self._framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self._useCamera = cap is None

        self._iterCnt = 0
        self._ticks = {}

        if maxTicks < 2:
            raise ValueError(
                f"Proviced maxTicks value must be 2 or higher (supplied: {maxTicks})"
            )

        self._maxTicks = maxTicks

    def nextIter(self):
        """
        Marks the next iteration of the processing loop. Will use :obj:`time.sleep` method if initialized with video file
        object
        """
        if self._start is None:
            self._start = time.monotonic()

        if not self._useCamera and self._timestamp is not None:
            frameDelay = 1.0 / self._framerate
            delay = (self._timestamp + frameDelay) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._timestamp = time.monotonic()
        self._iterCnt += 1

    def tick(self, name):
        """
        Marks a point in time for specified name
        Args:
            name (str): Specifies timestamp name
        """
        if name not in self._ticks:
            self._ticks[name] = collections.deque(maxlen=self._maxTicks)
        self._ticks[name].append(time.monotonic())

    def tickFps(self, name):
        """
        Calculates the FPS based on specified name
        Args:
            name (str): Specifies timestamps' name
        Returns:
            float: Calculated FPS or :code:`0.0` (default in case of failure)
        """
        if name in self._ticks and len(self._ticks[name]) > 1:
            timeDiff = self._ticks[name][-1] - self._ticks[name][0]
            return (len(self._ticks[name]) - 1) / timeDiff if timeDiff != 0 else 0.0
        else:
            return 0.0

    def fps(self):
        """
        Calculates FPS value based on :func:`nextIter` calls, being the FPS of processing loop
        Returns:
            float: Calculated FPS or :code:`0.0` (default in case of failure)
        """
        if self._start is None or self._timestamp is None:
            return 0.0
        timeDiff = self._timestamp - self._start
        return self._iterCnt / timeDiff if timeDiff != 0 else 0.0

    def printStatus(self):
        """
        Prints total FPS for all names stored in :func:`tick` calls
        """
        print("=== TOTAL FPS ===")
        for name in self._ticks:
            print(f"[{name}]: {self.tickFps(name):.1f}")

    def drawFps(self, frame, name):
        """
        Draws FPS values on requested frame, calculated based on specified name
        Args:
            frame (numpy.ndarray): Frame object to draw values on
            name (str): Specifies timestamps' name
        """
        frameFps = f"{name.upper()} FPS: {round(self.tickFps(name), 1)}"
        # cv2.rectangle(frame, (0, 0), (120, 35), (255, 255, 255), cv2.FILLED)
        cv2.putText(
            frame,
            frameFps,
            (5, 15),
            self._fpsType,
            0.5,
            self._fpsBgColor,
            4,
            self._fpsLineType,
        )
        cv2.putText(
            frame,
            frameFps,
            (5, 15),
            self._fpsType,
            0.5,
            self._fpsColor,
            1,
            self._fpsLineType,
        )

        if "nn" in self._ticks:
            cv2.putText(
                frame,
                f"NN FPS:  {round(self.tickFps('nn'), 1)}",
                (5, 30),
                self._fpsType,
                0.5,
                self._fpsBgColor,
                4,
                self._fpsLineType,
            )
            cv2.putText(
                frame,
                f"NN FPS:  {round(self.tickFps('nn'), 1)}",
                (5, 30),
                self._fpsType,
                0.5,
                self._fpsColor,
                1,
                self._fpsLineType,
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
    elif args.high_res:
        camRgb.video.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)
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
        bboxColors = (
            np.random.random(size=(256, 3)) * 256
        )  # Random Colors for bounding boxes
        fpsHandler = FPSHandler()

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
                    None if args.fullFov else [W, H],
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
                frame
                if not args.high_res
                else cv2.resize(frame, (0, 0), fx=0.5, fy=0.5),
            )

        while True:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

            if inRgb is not None:
                frame = inRgb.getCvFrame()
                fpsHandler.tick("color")

            if inDet is not None:
                detections = inDet.detections
                fpsHandler.tick("nn")

            if frame is not None:
                fpsHandler.drawFps(frame, "color")
                displayFrame("rgb", frame, detections)

            if cv2.waitKey(1) == ord("q"):
                break


if __name__ == "__main__":
    main()
