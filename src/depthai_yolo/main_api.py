#!/usr/bin/env python3
# coding=utf-8
import time

import cv2
import depthai as dai
import numpy as np
from utils import FPSHandler, display_frame, get_device_info


def main(pipeline, **kwargs):
    classes = kwargs.get("classes", [])

    config_data = kwargs["config_data"]
    num_classes = config_data.nn_config.NN_specific_metadata.classes

    # Connect to a device and start pipeline
    with dai.Device(
        pipeline,
        get_device_info(),
        maxUsbSpeed=kwargs.get("usbSpeed"),
    ) as device:
        if kwargs.get("spatial") and device.getIrDrivers():
            device.setIrLaserDotProjectorBrightness(200)  # in mA, 0..1200
            device.setIrFloodLightBrightness(0)  # in mA, 0..1500
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        image_queue = device.getOutputQueue(name="image", maxSize=4, blocking=False)
        detect_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        frame = None
        detections = []
        # Random Colors for bounding boxes
        bbox_colors = np.random.default_rng().integers(256, size=(num_classes, 3), dtype=int).tolist()
        fps_handler = FPSHandler()

        while True:
            image_queue_data = image_queue.tryGet()
            detect_queue_data = detect_queue.tryGet()

            if image_queue_data is not None:
                frame = image_queue_data.getCvFrame()
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
            if key == ord("s"):
                filename = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.jpg"
                # for chinese dir
                cv2.imencode(".jpg", frame)[1].tofile(filename)
                print(f"save to: {filename}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
