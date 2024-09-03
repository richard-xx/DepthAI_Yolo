# coding=utf-8
from enum import Enum
from typing import Union

import depthai as dai


class StereoPair(str, Enum):
    """
    Stereo pair for stereo pair inference

    Attributes:
        LR: CAM_LEFT (CAM_B) as the left camera in the camera pair, and CAM_RIGHT (CAM_C) as the right camera in the camera pair
        LC: CAM_LEFT (CAM_B) as the left camera in the camera pair, and CAM_CENTER (CAM_A) as the right camera in the camera pair
        CR: CAM_CENTER(CAM_A) is the left camera in the camera pair, and CAM_RIGHT(CAM_C) is the right camera in the camera pair
    """

    LR = "LR"
    LC = "LC"
    CR = "CR"


cam_left_socket = {
    StereoPair.LR: dai.CameraBoardSocket.CAM_B,
    StereoPair.LC: dai.CameraBoardSocket.CAM_B,
    StereoPair.CR: dai.CameraBoardSocket.CAM_A,
}

cam_right_socket = {
    StereoPair.LR: dai.CameraBoardSocket.CAM_C,
    StereoPair.LC: dai.CameraBoardSocket.CAM_A,
    StereoPair.CR: dai.CameraBoardSocket.CAM_C,
}


def create_pipeline(**kwargs):
    model_data = kwargs.get("model_data")
    config_data = kwargs.get("config_data")
    nn_config = config_data.nn_config
    color_res = kwargs.get("color_res", dai.ColorCameraProperties.SensorResolution.THE_1200_P)
    isp_scale = kwargs.get("isp_scale", (2, 3))
    stereo_pair = kwargs.get("stereo_pair", StereoPair.LR)
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    detection_network = (
        create_stereo(pipeline, cam_rgb=cam_rgb, **kwargs)
        if kwargs.get("spatial", False)
        else pipeline.create(dai.node.YoloDetectionNetwork)
    )

    xout_image = pipeline.create(dai.node.XLinkOut)
    nn_out = pipeline.create(dai.node.XLinkOut)

    xout_image.setStreamName("image")
    nn_out.setStreamName("nn")

    cam_rgb.setPreviewSize(nn_config.nn_width, nn_config.nn_height)
    cam_rgb.setBoardSocket(cam_left_socket[stereo_pair])
    cam_rgb.setResolution(color_res)
    cam_rgb.setIspScale(isp_scale)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(kwargs.get("fps", 30))
    cam_rgb.setPreviewKeepAspectRatio(not kwargs.get("fullFov", False))

    detection_network.setConfidenceThreshold(nn_config.NN_specific_metadata.confidence_threshold)
    detection_network.setNumClasses(nn_config.NN_specific_metadata.classes)
    detection_network.setCoordinateSize(nn_config.NN_specific_metadata.coordinates)
    detection_network.setAnchors(nn_config.NN_specific_metadata.anchors)
    detection_network.setAnchorMasks(nn_config.NN_specific_metadata.anchor_masks)
    detection_network.setIouThreshold(nn_config.NN_specific_metadata.iou_threshold)
    detection_network.setBlob(model_data)
    detection_network.input.setBlocking(False)
    detection_network.input.setQueueSize(1)

    cam_rgb.preview.link(detection_network.input)
    if kwargs.get("syncNN", False):
        detection_network.passthrough.link(xout_image.input)
    elif kwargs.get("high_res", False):
        cam_rgb.video.link(xout_image.input)
    else:
        cam_rgb.preview.link(xout_image.input)
    detection_network.out.link(nn_out.input)

    xin_control = pipeline.create(dai.node.XLinkIn)
    xin_control.setStreamName("control")
    xin_control.out.link(cam_rgb.inputControl)

    return pipeline


def create_stereo(pipeline, **kwargs):
    color_res = kwargs.get("color_res", dai.ColorCameraProperties.SensorResolution.THE_1200_P)
    isp_scale = kwargs.get("isp_scale", (2, 3))
    stereo_pair = kwargs.get("stereo_pair", StereoPair.LR)
    cam_left = kwargs["cam_left"]

    cam_right = pipeline.create(dai.node.ColorCamera)
    cam_right.setBoardSocket(cam_right_socket[stereo_pair])
    cam_right.setResolution(color_res)
    cam_right.setIspScale(isp_scale)
    cam_right.setFps(kwargs.get("fps", 30))

    stereo = pipeline.createStereoDepth()
    stereo.initialConfig.setConfidenceThreshold(245)
    stereo.setLeftRightCheck(kwargs.get("lr_check", False))
    stereo.setExtendedDisparity(kwargs.get("extended_disparity", False))
    stereo.setSubpixel(kwargs.get("subpixel", False))
    stereo.setDepthAlign(cam_left.getBoardSocket())

    detection_network: Union[dai.node.NeuralNetwork, dai.node.YoloSpatialDetectionNetwork] = pipeline.create(
        dai.node.YoloSpatialDetectionNetwork
    )
    detection_network.setDepthLowerThreshold(100)  # mm
    detection_network.setDepthUpperThreshold(10_000)  # mm
    detection_network.setBoundingBoxScaleFactor(0.3)

    cam_left.isp.link(stereo.left)
    cam_right.isp.link(stereo.right)
    stereo.depth.link(detection_network.inputDepth)

    return detection_network
