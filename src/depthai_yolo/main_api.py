#!/usr/bin/env python3
# coding=utf-8
import time
from dataclasses import dataclass, field
from datetime import timedelta

import cv2
import depthai as dai
import numpy as np

from depthai_yolo.utils import FPSHandler, display_frame, get_device_info


def clamp(num, v0, v1):
    return max(v0, min(num, v1))


@dataclass
class CameraControl:
    EXP_STEP = 500  # ms
    ISO_STEP = 50
    LENS_STEP = 3
    WB_STEP = 200

    lens_pos: int = field(default=150)
    exp_time: int = field(default=20_000)
    sens_iso: int = field(default=800)
    wb_manual: int = field(default=6500)

    def __post_init__(self):
        self.lens_pos = clamp(self.lens_pos, 0, 255)
        self.exp_time = clamp(self.exp_time, 1, 33_000)
        self.sens_iso = clamp(self.sens_iso, 100, 1600)
        self.wb_manual = clamp(self.wb_manual, 1000, 12000)

    def increase_lens_pos(self):
        self.lens_pos = clamp(self.lens_pos + self.LENS_STEP, 0, 255)
        print(f"Lens position adjusted to: {self.lens_pos}")

    def decrease_lens_pos(self):
        self.lens_pos = clamp(self.lens_pos - self.LENS_STEP, 0, 255)
        print(f"Lens position adjusted to: {self.lens_pos}")

    def increase_wb_manual(self):
        self.wb_manual = clamp(self.wb_manual + self.WB_STEP, 1000, 12000)
        print(f"White balance adjusted to: {self.wb_manual} K")

    def decrease_wb_manual(self):
        self.wb_manual = clamp(self.wb_manual - self.WB_STEP, 1000, 12000)
        print(f"White balance adjusted to: {self.wb_manual} K")

    def increase_exp_time(self):
        self.exp_time = clamp(self.exp_time + self.EXP_STEP, 1, 33000)
        print(f"Exposure time adjusted to: {self.exp_time / 1000:.3f} ms")

    def decrease_exp_time(self):
        self.exp_time = clamp(self.exp_time - self.EXP_STEP, 1, 33000)
        print(f"Exposure time adjusted to: {self.exp_time / 1000:.3f} ms")

    def increase_sens_iso(self):
        self.sens_iso = clamp(self.sens_iso + self.ISO_STEP, 100, 1600)
        print(f"ISO adjusted to: {self.sens_iso}")

    def decrease_sens_iso(self):
        self.sens_iso = clamp(self.sens_iso - self.ISO_STEP, 100, 1600)
        print(f"ISO adjusted to: {self.sens_iso}")


def main(pipeline_func, **kwargs):
    classes = kwargs.get("classes", [])

    config_data = kwargs["config_data"]
    num_classes = config_data.nn_config.NN_specific_metadata.classes

    camera_control = CameraControl()

    # Connect to a device and start pipeline
    with dai.Device(
        pipeline_func(**kwargs),
        get_device_info(),
        maxUsbSpeed=kwargs.get("usbSpeed"),
    ) as device:
        if kwargs.get("spatial") and device.getIrDrivers():
            device.setIrLaserDotProjectorBrightness(200)  # in mA, 0..1200
            device.setIrFloodLightBrightness(0)  # in mA, 0..1500
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        image_queue = device.getOutputQueue(name="image", maxSize=4, blocking=False)
        detect_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        control_queue = device.getInputQueue(name="control", maxSize=1, blocking=False)

        frame = None
        detections = []
        # Random Colors for bounding boxes
        bbox_colors = np.random.default_rng().integers(256, size=(num_classes, 3), dtype=int).tolist()
        fps_handler = FPSHandler()

        while True:
            image_queue_data = image_queue.tryGet()  # type: dai.ImgFrame | dai.ADatatype | None
            detect_queue_data = detect_queue.tryGet()  # type: dai.ImgDetections | dai.ADatatype | None

            if image_queue_data is not None:
                frame = image_queue_data.getCvFrame()
                camera_control.wb_manual = image_queue_data.getColorTemperature()
                camera_control.sens_iso = image_queue_data.getSensitivity()
                camera_control.exp_time = image_queue_data.getExposureTime().total_seconds() * 10**6
                camera_control.lens_pos = image_queue_data.getLensPosition()
                fps_handler.tick("color")

            if detect_queue_data is not None:
                detections = detect_queue_data.detections
                if len(classes):
                    detections = [detection for detection in detections if detection.label in classes]

                fps_handler.tick("nn")

            if frame is not None:
                fps_handler.draw_fps(frame, "color")
                display_frame("rgb", frame, detections, bbox_colors, config_data.mappings.labels)

            key = cv2.waitKey(1)

            if key == ord("q"):
                break
            parse_key(control_queue, frame, key, camera_control)

        cv2.destroyAllWindows()


def parse_key(control_queue, frame, key, camera_control):
    ctrl = dai.CameraControl()
    send_ctrl = False
    if key == ord("s"):
        filename = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.jpg"
        cv2.imencode(".jpg", frame)[1].tofile(filename)  # Save frame
        print(f"save to: {filename}")

    elif key == ord("t"):
        print("Autofocus trigger (and disable continuous)")
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
        ctrl.setAutoFocusTrigger()
        send_ctrl = True
    elif key == ord("f"):
        print("Autofocus enable, continuous")
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        send_ctrl = True

    elif key == ord("e"):
        print("Autoexposure enable")
        ctrl.setAutoExposureEnable()
        send_ctrl = True

    elif key == ord("b"):
        print("Auto white-balance enable")
        ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        send_ctrl = True

    elif key in {ord(","), ord(".")}:  # Manual focus control
        camera_control.increase_lens_pos() if key == ord(".") else camera_control.decrease_lens_pos()
        ctrl.setManualFocus(camera_control.lens_pos)
        send_ctrl = True

    elif key in {ord("i"), ord("o"), ord("k"), ord("l")}:
        # Manual exposure and ISO control
        if key == ord("i"):
            camera_control.decrease_exp_time()
        elif key == ord("o"):
            camera_control.increase_exp_time()
        elif key == ord("k"):
            camera_control.decrease_sens_iso()
        elif key == ord("l"):
            camera_control.increase_sens_iso()

        ctrl.setManualExposure(timedelta(microseconds=camera_control.exp_time), camera_control.sens_iso)
        send_ctrl = True

    elif key in {ord("n"), ord("m")}:
        # Manual white balance control
        camera_control.increase_wb_manual() if key == ord("m") else camera_control.decrease_wb_manual()
        ctrl.setManualWhiteBalance(camera_control.wb_manual)
        send_ctrl = True
    if send_ctrl:
        control_queue.send(ctrl)
