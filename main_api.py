#!/usr/bin/env python3
# coding=utf-8
from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path

import blobconverter
import cv2
import depthai as dai
import numpy as np

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
    required=True,
    help="Provide model name or model path for inference",
    # default="yolov8n_coco_640x352",
    type=Path,
)
parser.add_argument(
    "-c",
    "--config",
    help="Provide config path for inference",
    # default="json/yolov8n_coco_640x352.json",
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
parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
parser.add_argument(
    "-usbs", "--usbSpeed", type=str, default="usb3", choices=["usb2", "usb3"], help="Force USB communication speed."
)

args = parser.parse_args()
if args.fullFov.upper() in ["TRUE", "ON"]:
    args.fullFov = True
elif args.fullFov.upper() in ["FALSE", "OFF"]:
    args.fullFov = False

num_classes = 80

# get model path
nn_path = args.model.resolve().absolute()
if not nn_path.is_file():
    nn_name = nn_path.stem

    if nn_name in depthai_model_zoo + open_model_zoo:
        list_file = list(model_dir.glob(f"{nn_name}*.blob"))
        if list_file:
            nn_path = list_file[0]
        else:
            print(f"Model ({nn_name}) not found in local. Looking into DepthAI model zoo.")
            if nn_name in depthai_model_zoo:
                nn_path = blobconverter.from_zoo(
                    nn_name,
                    shaves=6,
                    zoo_type="depthai",
                    use_cache=True,
                )
            elif nn_name in open_model_zoo:
                nn_path = blobconverter.from_zoo(
                    nn_name,
                    shaves=6,
                    zoo_type="intel",
                    use_cache=True,
                )
    else:
        print(f"Model ({nn_name}) not found in model zoo. check model is in: ")
        print(depthai_model_zoo + open_model_zoo)
        raise FileNotFoundError
    args.config = ROOT.joinpath("json", f"{nn_name}.json")

args.model = nn_path

model = dai.OpenVINO.Blob(args.model)
dim = next(iter(model.networkInputs.values())).dims
nn_width, nn_height = dim[:2]
print(f"nnWidth = {nn_width}, nnHeight = {nn_height}")

output_name, output_tenser = next(iter(model.networkOutputs.items()))
num_classes = output_tenser.dims[2] - 5 if "yolov6" in output_name else output_tenser.dims[2] // 3 - 5

# parse config
if args.config is None or not args.config.exists():
    config_path = nn_path.resolve().absolute().with_suffix(".json")
    if not config_path.exists() and re.search("_openvino_(.*?)_(.*?)shave", nn_path.stem):
        config_path = re.sub("_openvino_(.*?)_(.*?)shave", "", nn_path.stem)
else:
    config_path = args.config.resolve().absolute()

assert config_path.exists(), ValueError(f"Path {config_path} does not exist!")

args.config = config_path
print(f"args: {args}")

if args.spatial:
    # Better handling for occlusions
    lr = True
    # Closer-in minimum depth, disparity range is doubled
    extended = False
    # Better accuracy for longer distance, fractional disparity 32-levels
    subpixel = False

with config_path.open() as f:
    config = json.load(f)
nn_config = config.get("nn_config", {})

# extract metadata
metadata = nn_config.get("NN_specific_metadata", {})
coordinates = metadata.get("coordinates", 4)
anchors = metadata.get("anchors", [])
anchor_masks = metadata.get("anchor_masks", {})
iou_threshold = metadata.get("iou_threshold", 0.5)
confidence_threshold = metadata.get("confidence_threshold", 0.5)

print(f"config: {metadata}")

# parse labels
nn_mappings = config.get("mappings", {})
labels = nn_mappings.get("labels", [])


def main():
    # Connect to a device and start pipeline
    with dai.Device(
        create_pipeline(),
        get_device_info(),
        maxUsbSpeed=dai.UsbSpeed.HIGH if args.usbSpeed == "usb2" else dai.UsbSpeed.SUPER_PLUS,
    ) as device:
        if args.spatial and device.getIrDrivers():
            device.setIrLaserDotProjectorBrightness(200)  # in mA, 0..1200
            device.setIrFloodLightBrightness(0)  # in mA, 0..1500
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        image_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detect_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        frame = None
        detections = []
        # Random Colors for bounding boxes
        bbox_colors = np.random.default_rng().integers(256, size=(num_classes, 3), dtype=int)
        fps_handler = FPSHandler()

        # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
        def frame_norm(frame, bbox):
            # Normalize bounding boxes
            norm_vals = np.full(len(bbox), frame.shape[0])
            norm_vals[::2] = frame.shape[1]
            return (np.clip(bbox, 0, 1) * norm_vals).astype(int)

        def display_frame(name, frame, detections):
            for detection in detections:
                bbox = frame_norm(
                    frame,
                    (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
                )
                draw_text(
                    frame,
                    labels[detection.label],
                    (bbox[0] + 10, bbox[1] + 20),
                    bbox_colors[detection.label],
                )
                draw_text(
                    frame,
                    f"{detection.confidence:.2%}",
                    (bbox[0] + 10, bbox[1] + 40),
                    bbox_colors[detection.label],
                )
                draw_rect(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    bbox_colors[detection.label],
                )
                # Display spatial coordinates as well
                if hasattr(detection, "spatialCoordinates"):
                    x_meters = detection.spatialCoordinates.x / 1000
                    y_meters = detection.spatialCoordinates.y / 1000
                    z_meters = detection.spatialCoordinates.z / 1000
                    draw_text(
                        frame,
                        f"X: {x_meters:.2f} m",
                        (bbox[0] + 10, bbox[1] + 60),
                    )
                    draw_text(
                        frame,
                        f"Y: {y_meters:.2f} m",
                        (bbox[0] + 10, bbox[1] + 75),
                    )
                    draw_text(
                        frame,
                        f"Z: {z_meters:.2f} m",
                        (bbox[0] + 10, bbox[1] + 90),
                    )
            # Show the frame
            cv2.imshow(name, frame)

        while True:
            image_queue_data = image_queue.tryGet()
            detect_queue_data = detect_queue.tryGet()

            if image_queue_data is not None:
                frame = image_queue_data.getCvFrame()
                fps_handler.tick("color")

            if detect_queue_data is not None:
                detections = detect_queue_data.detections
                if args.classes is not None:
                    detections = [detection for detection in detections if detection.label in args.classes]

                fps_handler.tick("nn")

            if frame is not None:
                fps_handler.draw_fps(frame, "color")
                display_frame("rgb", frame, detections)
                # detections = []

            key = cv2.waitKey(1)

            if key == ord("q"):
                break
            if key == ord("s"):
                filename = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.jpg"
                # for chinese dir
                cv2.imencode(".jpg", frame)[1].tofile(filename)
                # cv2.imwrite(filename,frame)
                print(f"save to: {filename}")
        cv2.destroyAllWindows()


class FPSHandler:
    """
    Class that handles all FPS-related operations.

    Mostly used to calculate different streams FPS, but can also be
    used to feed the video file based on its FPS property, not app performance (this prevents the video from being sent
    to quickly if we finish processing a frame earlier than the next video frame should be consumed)
    """

    _fps_bg_color = (0, 0, 0)
    _fps_color = (255, 255, 255)
    _fps_type = cv2.FONT_HERSHEY_SIMPLEX
    _fps_line_type = cv2.LINE_AA

    def __init__(self, cap=None, max_ticks=100):
        """
        Constructor that initializes the class with a video file object and a maximum ticks amount for FPS calculation

        Args:
            cap (cv2.VideoCapture, Optional): handler to the video file object
            max_ticks (int, Optional): maximum ticks amount for FPS calculation
        """
        self._timestamp = None
        self._start = None
        self._framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self._useCamera = cap is None

        self._iterCnt = 0
        self._ticks = {}

        if max_ticks < 2:  # noqa: PLR2004
            msg = f"Proviced max_ticks value must be 2 or higher (supplied: {max_ticks})"
            raise ValueError(msg)

        self._maxTicks = max_ticks

    def next_iter(self):
        """Marks the next iteration of the processing loop. Will use `time.sleep` method if initialized with video file object"""
        if self._start is None:
            self._start = time.monotonic()

        if not self._useCamera and self._timestamp is not None:
            frame_delay = 1.0 / self._framerate
            delay = (self._timestamp + frame_delay) - time.monotonic()
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

    def tick_fps(self, name):
        """
        Calculates the FPS based on specified name

        Args:
            name (str): Specifies timestamps' name

        Returns:
            float: Calculated FPS or `0.0` (default in case of failure)
        """
        if name in self._ticks and len(self._ticks[name]) > 1:
            time_diff = self._ticks[name][-1] - self._ticks[name][0]
            return (len(self._ticks[name]) - 1) / time_diff if time_diff != 0 else 0.0
        return 0.0

    def fps(self):
        """
        Calculates FPS value based on `nextIter` calls, being the FPS of processing loop

        Returns:
            float: Calculated FPS or `0.0` (default in case of failure)
        """
        if self._start is None or self._timestamp is None:
            return 0.0
        time_diff = self._timestamp - self._start
        return self._iterCnt / time_diff if time_diff != 0 else 0.0

    def print_status(self):
        """Prints total FPS for all names stored in :func:`tick` calls"""
        print("=== TOTAL FPS ===")
        for name in self._ticks:
            print(f"[{name}]: {self.tick_fps(name):.1f}")

    def draw_fps(self, frame, name):
        """
        Draws FPS values on requested frame, calculated based on specified name

        Args:
            frame (numpy.ndarray): Frame object to draw values on
            name (str): Specifies timestamps' name
        """
        frame_fps = f"{name.upper()} FPS: {round(self.tick_fps(name), 1)}"
        # cv2.rectangle(frame, (0, 0), (120, 35), (255, 255, 255), cv2.FILLED)
        cv2.putText(
            frame,
            frame_fps,
            (5, 15),
            self._fps_type,
            0.5,
            self._fps_bg_color,
            4,
            self._fps_line_type,
        )
        cv2.putText(
            frame,
            frame_fps,
            (5, 15),
            self._fps_type,
            0.5,
            self._fps_color,
            1,
            self._fps_line_type,
        )

        if "nn" in self._ticks:
            cv2.putText(
                frame,
                f"NN FPS:  {round(self.tick_fps('nn'), 1)}",
                (5, 30),
                self._fps_type,
                0.5,
                self._fps_bg_color,
                4,
                self._fps_line_type,
            )
            cv2.putText(
                frame,
                f"NN FPS:  {round(self.tick_fps('nn'), 1)}",
                (5, 30),
                self._fps_type,
                0.5,
                self._fps_color,
                1,
                self._fps_line_type,
            )


def draw_text(
    frame,
    text,
    org,
    color=(255, 255, 255),
    bg_color=(128, 128, 128),
    font_scale=0.5,
    thickness=1,
):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, bg_color, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_rect(frame, p1, p2, color=(255, 255, 255), bg_color=(128, 128, 128), thickness=1):
    cv2.rectangle(frame, pt1=p1, pt2=p2, color=bg_color, thickness=thickness + 3)
    cv2.rectangle(frame, pt1=p1, pt2=p2, color=color, thickness=thickness)


def get_device_info(device_id=None, debug=False, poe=True) -> dai.DeviceInfo:
    device_infos = dai.XLinkConnection.getAllConnectedDevices() if debug else dai.Device.getAllAvailableDevices()

    # Filter devices based on Power over Ethernet (PoE) connection
    if not poe:
        device_infos = [
            device_info for device_info in device_infos if device_info.protocol != dai.XLinkProtocol.X_LINK_TCP_IP
        ]

    # If no devices are found, raise an error
    if len(device_infos) == 0:
        msg = "No DepthAI device found!"
        raise RuntimeError(msg)

    # Print the available devices
    print("Available devices:")
    for i, device_info in enumerate(device_infos):
        print(f"[{i}] {device_info.name} {device_info.getMxId()} [{device_info.state.name}]")

    # If the user specifies to list the devices, exit the program
    if device_id == "list":
        raise SystemExit(0)
    # If the user specifies a device ID, return the matching DeviceInfo object
    if device_id is not None:
        matching_device = next(filter(lambda info: info.getMxId() == device_id, device_infos), None)
        if matching_device is None:
            msg = f"No DepthAI device found with id matching {device_id} !"
            raise RuntimeError(msg)
        return matching_device
    # If only one device is available, return its DeviceInfo object
    if len(device_infos) == 1:
        return device_infos[0]
    # If multiple devices are available, prompt the user to select one and return its DeviceInfo object
    val = input("Which DepthAI Device you want to use: ")
    if val not in [str(i) for i in range(len(device_infos))]:
        msg = f"No DepthAI device found with id matching {val} !"
        raise RuntimeError(msg)
    return device_infos[int(val)]


def create_pipeline():
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    detection_network = create_stereo(pipeline) if args.spatial else pipeline.create(dai.node.YoloDetectionNetwork)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    nn_out = pipeline.create(dai.node.XLinkOut)

    xout_rgb.setStreamName("rgb")
    nn_out.setStreamName("nn")

    # Properties
    cam_rgb.setPreviewSize(nn_width, nn_height)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(60)
    cam_rgb.setPreviewKeepAspectRatio(not args.fullFov)

    # Network specific settings
    detection_network.setConfidenceThreshold(confidence_threshold)
    detection_network.setNumClasses(num_classes)
    detection_network.setCoordinateSize(coordinates)
    detection_network.setAnchors(anchors)
    detection_network.setAnchorMasks(anchor_masks)
    detection_network.setIouThreshold(iou_threshold)
    detection_network.setBlob(model)
    # detection_network.setNumInferenceThreads(2)
    detection_network.input.setBlocking(False)
    detection_network.input.setQueueSize(1)

    # Linking
    cam_rgb.preview.link(detection_network.input)
    if args.syncNN:
        detection_network.passthrough.link(xout_rgb.input)
    elif args.high_res:
        cam_rgb.video.link(xout_rgb.input)
    else:
        cam_rgb.preview.link(xout_rgb.input)
    detection_network.out.link(nn_out.input)

    return pipeline


def create_stereo(pipeline):
    mono_left = pipeline.createMonoCamera()
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setFps(60)
    mono_right = pipeline.createMonoCamera()
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setFps(60)
    stereo = pipeline.createStereoDepth()
    stereo.initialConfig.setConfidenceThreshold(245)
    stereo.setLeftRightCheck(lr)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    detection_network: dai.node.NeuralNetwork | dai.node.YoloSpatialDetectionNetwork = pipeline.create(
        dai.node.YoloSpatialDetectionNetwork
    )
    stereo.depth.link(detection_network.inputDepth)
    detection_network.setDepthLowerThreshold(100)  # mm
    detection_network.setDepthUpperThreshold(10_000)  # mm
    detection_network.setBoundingBoxScaleFactor(0.3)
    return detection_network


if __name__ == "__main__":
    main()
