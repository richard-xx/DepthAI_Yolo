#!/usr/bin/env python3
# coding=utf-8
"""
The code is edited from docs (https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/)
We add parsing from JSON files that contain configuration
"""

import argparse
import collections
import time
from pathlib import Path

import blobconverter
import cv2
import depthai as dai
import numpy as np

import json

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-m",
    "--model",
    required=True,
    help="Provide model name or model path for inference",
    # default="yolov4_tiny_coco_416x416",
    type=Path,
)
parser.add_argument(
    "-c",
    "--config",
    help="Provide config path for inference",
    # default="json/yolov4-tiny.json",
    type=Path,
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
parser.add_argument(
    "--classes",
    nargs='+', 
    type=int, 
    help='filter by class: --classes 0, or --classes 0 2 3'
)
args = parser.parse_args()
if args.fullFov.upper() in ["TRUE", "ON"]:
    args.fullFov = True
elif args.fullFov.upper() in ["FALSE", "OFF"]:
    args.fullFov = False

numClasses = 80

# get model path
nnPath = args.model.resolve().absolute()
if not nnPath.is_file():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = blobconverter.from_zoo(
            nnPath.stem,
            shaves=6,
            zoo_type="depthai",
            use_cache=True,
            output_dir="models",
        )
args.model = nnPath

model = dai.OpenVINO.Blob(args.model)
dim = next(iter(model.networkInputs.values())).dims
nnWidth, nnHeight = dim[:2]
print(f"{nnWidth, nnHeight = }")

output_name, output_tenser = next(iter(model.networkOutputs.items()))
if "yolov6" in output_name:
    numClasses = output_tenser.dims[2] - 5
else:
    numClasses = output_tenser.dims[2] // 3 - 5

# parse config
if args.config is None or not args.config.exists():
    configPath = nnPath.resolve().absolute().with_suffix(".json")
else:
    configPath = args.config.resolve().absolute()

assert configPath.exists(), ValueError("Path {} does not exist!".format(configPath))

args.config = configPath
print("args: {}".format(args))


if args.spatial:
    # Better handling for occlusions
    lr = True
    # Closer-in minimum depth, disparity range is doubled
    extended = False
    # Better accuracy for longer distance, fractional disparity 32-levels
    subpixel = False

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
coordinates = metadata.get("coordinates", 4)
anchors = metadata.get("anchors", [])
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", 0.5)
confidenceThreshold = metadata.get("confidence_threshold", 0.5)

print("config: {}".format(metadata))

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", [])


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
        cv2.putText(frame, frameFps, (5, 15), self._fpsType, 0.5, self._fpsBgColor, 4, self._fpsLineType, )
        cv2.putText(frame, frameFps, (5, 15), self._fpsType, 0.5, self._fpsColor, 1, self._fpsLineType, )

        if "nn" in self._ticks:
            cv2.putText(
                frame, f"NN FPS:  {round(self.tickFps('nn'), 1)}", (5, 30), self._fpsType, 0.5,
                self._fpsBgColor, 4, self._fpsLineType, )
            cv2.putText(
                frame, f"NN FPS:  {round(self.tickFps('nn'), 1)}", (5, 30), self._fpsType, 0.5,
                self._fpsColor, 1, self._fpsLineType, )


def drawText(
    frame, text, org, color=(255, 255, 255), bg_color=(128, 128, 128), fontScale=0.5, thickness=1,
):
    cv2.putText(
        frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, bg_color, thickness + 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA
    )


def drawRect(
    frame, p1, p2, color=(255, 255, 255), bg_color=(128, 128, 128), thickness=1
):
    cv2.rectangle(frame, pt1=p1, pt2=p2, color=bg_color, thickness=thickness + 3)
    cv2.rectangle(frame, pt1=p1, pt2=p2, color=color, thickness=thickness)


def getDeviceInfo(deviceId=None, debug=False, poe=True) -> dai.DeviceInfo:
    """
    Find the DeviceInfo object of a DepthAI device. This function can be used to retrieve the DeviceInfo of a specific
    device, or if multiple devices are connected, the user will be prompted to select one.

    Args:
        deviceId: str, optional
            The unique ID of the DepthAI device to retrieve. If not specified, the user will be prompted to select one.
        debug: bool, optional
            If True, the function will attempt to connect to all available DepthAI devices.
        poe: bool, optional
            If True, only devices connected via Power over Ethernet will be considered.

    Returns:
        dai.DeviceInfo
            The DeviceInfo object of the selected DepthAI device.

    Raises:
        RuntimeError: if no DepthAI device is found.
        RuntimeError: if no DepthAI device is found with a matching ID.
        ValueError: if an incorrect value is supplied when prompted to select a device.
    """
    deviceInfos = []
    if debug:
        # Get all connected devices in debug mode
        deviceInfos = dai.XLinkConnection.getAllConnectedDevices()
    else:
        # Get all available devices in normal mode
        deviceInfos = dai.Device.getAllAvailableDevices()

    # Filter devices based on Power over Ethernet (PoE) connection
    if not poe:
        deviceInfos = [deviceInfo for deviceInfo in deviceInfos if
                       deviceInfo.protocol != dai.XLinkProtocol.X_LINK_TCP_IP]

    # If no devices are found, raise an error
    if len(deviceInfos) == 0:
        raise RuntimeError("No DepthAI device found!")
    else:
        # Print the available devices
        print("Available devices:")
        for i, deviceInfo in enumerate(deviceInfos):
            print(f"[{i}] {deviceInfo.name} {deviceInfo.getMxId()} [{deviceInfo.state.name}]")

        # If the user specifies to list the devices, exit the program
        if deviceId == "list":
            raise SystemExit(0)
        # If the user specifies a device ID, return the matching DeviceInfo object
        elif deviceId is not None:
            matchingDevice = next(filter(lambda info: info.getMxId() == deviceId, deviceInfos), None)
            if matchingDevice is None:
                raise RuntimeError(f"No DepthAI device found with id matching {deviceId} !")
            return matchingDevice
        # If only one device is available, return its DeviceInfo object
        elif len(deviceInfos) == 1:
            return deviceInfos[0]
        # If multiple devices are available, prompt the user to select one and return its DeviceInfo object
        else:
            val = input("Which DepthAI Device you want to use: ")
            try:
                return deviceInfos[int(val)]
            except:
                raise ValueError("Incorrect value supplied: {}".format(val))

def create_pipeline():
    """
    Create a DepthAI pipeline for object detection using YOLO.

    Returns:
        dai.Pipeline
            The DepthAI pipeline object.
    """
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
        detectionNetwork.setDepthLowerThreshold(100)  # mm
        detectionNetwork.setDepthUpperThreshold(10_000) # mm
        detectionNetwork.setBoundingBoxScaleFactor(0.3)
    else:
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    nnOut = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    nnOut.setStreamName("nn")

    # Properties
    camRgb.setPreviewSize(nnWidth, nnHeight)

    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(60)
    camRgb.setPreviewKeepAspectRatio(not args.fullFov)

    # Network specific settings
    detectionNetwork.setConfidenceThreshold(confidenceThreshold)
    detectionNetwork.setNumClasses(numClasses)
    detectionNetwork.setCoordinateSize(coordinates)
    detectionNetwork.setAnchors(anchors)
    detectionNetwork.setAnchorMasks(anchorMasks)
    detectionNetwork.setIouThreshold(iouThreshold)
    detectionNetwork.setBlob(model)
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
        if args.spatial and device.getIrDrivers():
            device.setIrLaserDotProjectorBrightness(200) # in mA, 0..1200
            device.setIrFloodLightBrightness(0) # in mA, 0..1500
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        imageQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectQueue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        frame = None
        detections = []
        # Random Colors for bounding boxes
        bboxColors:list[list[int]] = np.random.randint(256, size=(numClasses, 3), dtype=int).tolist()
        fpsHandler = FPSHandler()

        # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
        def frameNorm(frame, bbox):
            # Normalize bounding boxes
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(bbox, 0, 1) * normVals).astype(int)

        def displayFrame(name, frame, detections):
            for detection in detections:
                bbox = frameNorm(
                    frame,
                    (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
                )
                drawText(
                    frame,
                    labels[detection.label],
                    (bbox[0] + 10, bbox[1] + 20),
                    bboxColors[detection.label],
                )
                drawText(
                    frame,
                    f"{detection.confidence:.2%}",
                    (bbox[0] + 10, bbox[1] + 40),
                    bboxColors[detection.label],
                )
                drawRect(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    bboxColors[detection.label],
                )
                # Display spatial coordinates as well
                if hasattr(detection, "spatialCoordinates"):
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
            cv2.imshow(name, frame)

        while True:
            imageQueueData = imageQueue.tryGet()
            detectQueueData = detectQueue.tryGet()

            if imageQueueData is not None:
                frame = imageQueueData.getCvFrame()
                fpsHandler.tick("color")

            if detectQueueData is not None:
                detections = detectQueueData.detections
                if args.classes is not None:
                    detections = [detection for detection in detections if detection.label in args.classes]
                    
                fpsHandler.tick("nn")

            if frame is not None:
                fpsHandler.drawFps(frame, "color")
                displayFrame("rgb", frame, detections)
                # detections = []

            key = cv2.waitKey(1)

            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"{time.strftime('%Y%m%d_%H%M%S',time.localtime())}.jpg"
                # for chinese dir
                cv2.imencode(".jpg", frame)[1].tofile(filename)
                # cv2.imwrite(filename,frame)
                print(f"save to: {filename}")
            


if __name__ == "__main__":
    main()
