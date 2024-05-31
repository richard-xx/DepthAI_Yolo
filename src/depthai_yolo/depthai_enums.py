# coding=utf-8

import depthai as dai

__all__ = [
    "color_res_opts",
    "mono_res_opts",
    "usb_speed_opts",
]

usb_speed_opts = dai.UsbSpeed.__members__

mono_res_opts = dai.MonoCameraProperties.SensorResolution.__members__

color_res_opts = dai.ColorCameraProperties.SensorResolution.__members__
