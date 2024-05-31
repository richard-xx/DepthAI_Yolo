# coding=utf-8
import collections
import re
import time
from pathlib import Path

import blobconverter
import cv2
import depthai as dai
import numpy as np
from depthai_yolo.download_models import download
from depthai_yolo.nn_config import Config
from depthai_yolo.yolo_define_models import depthai_model_zoo, open_model_zoo

ROOT = Path(__file__).parent
# model_dir = ROOT.joinpath("models")
# blobconverter.set_defaults(output_dir=model_dir, version="2021.4")
blobconverter.set_defaults(version="2021.4")
model_dir = blobconverter.__defaults.get("output_dir")


def parse_yolo_model(config_file, model):
    # get model path
    if not model.is_file():
        nn_name = model.stem

        if nn_name in depthai_model_zoo + open_model_zoo:
            list_file = list(model_dir.glob(f"{nn_name}*.blob"))
            if list_file:
                model = list_file[0]
            else:
                print(f"Model ({nn_name}) not found in local. Looking into DepthAI model zoo.")
                model = download(nn_name)
        else:
            print(f"Model ({nn_name}) not found in model zoo. check model is in: ")
            print(depthai_model_zoo + open_model_zoo)
            raise FileNotFoundError
        config_file = ROOT.joinpath("jsons", f"{nn_name}.json")
    nn_path = model.resolve().absolute()

    model_blob = dai.OpenVINO.Blob(nn_path)
    dim = next(iter(model_blob.networkInputs.values())).dims
    nn_width, nn_height = dim[:2]
    print(f"nnWidth = {nn_width}, nnHeight = {nn_height}")

    output_name, output_tenser = next(iter(model_blob.networkOutputs.items()))
    num_classes = output_tenser.dims[2] - 5 if "yolov6" in output_name else output_tenser.dims[2] // 3 - 5

    # parse config
    if config_file is None or not config_file.exists():
        config_path = nn_path.with_suffix(".json")
        if not config_path.exists() and re.search("_openvino_(.*?)_(.*?)shave", config_path.stem):
            config_path = Path(re.sub("_openvino_(.*?)_(.*?)shave", "", str(config_path)))
    else:
        config_path = config_file.resolve().absolute()
    if not config_path.exists():
        msg = f"Config file {config_path} not found!"
        raise FileNotFoundError(msg)

    config = Config.model_validate_json(config_file.read_bytes())

    config.nn_config.input_size = f"{nn_width}x{nn_height}"
    # config.nn_config.nn_width = nn_width
    # config.nn_config.nn_height = nn_height
    config.nn_config.NN_specific_metadata.classes = num_classes
    config.mappings.labels.extend([f"class_{x}" for x in range(num_classes - config.mappings.nc, num_classes)])
    return config, model_blob


# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def frame_norm(frame, bbox):
    # Normalize bounding boxes
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(bbox, 0, 1) * norm_vals).astype(int)


def display_frame(name, frame, detections, bbox_colors, labels):
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


def get_device_info(device_id=None, *, debug=False, poe=True) -> dai.DeviceInfo:
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
