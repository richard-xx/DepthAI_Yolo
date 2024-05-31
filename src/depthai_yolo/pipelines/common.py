# coding=utf-8
from typing import Union

import depthai as dai


def create_pipeline(**kwargs):
    model_data = kwargs.get("model_data")
    config_data = kwargs.get("config_data")
    nn_config = config_data.nn_config
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    detection_network = (
        create_stereo(pipeline, **kwargs)
        if kwargs.get("spatial", False)
        else pipeline.create(dai.node.YoloDetectionNetwork)
    )

    xout_image = pipeline.create(dai.node.XLinkOut)
    nn_out = pipeline.create(dai.node.XLinkOut)

    xout_image.setStreamName("image")
    nn_out.setStreamName("nn")

    cam_rgb.setPreviewSize(nn_config.nn_width, nn_config.nn_height)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(kwargs.get("color_res", dai.ColorCameraProperties.SensorResolution.THE_1080_P))
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

    return pipeline


def create_stereo(pipeline, **kwargs):
    mono_res = kwargs.get("mono_res", dai.MonoCameraProperties.SensorResolution.THE_720_P)

    mono_left = pipeline.createMonoCamera()
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setResolution(mono_res)
    mono_left.setFps(kwargs.get("fps", 30))
    mono_right = pipeline.createMonoCamera()
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setResolution(mono_res)
    mono_right.setFps(kwargs.get("fps", 30))
    stereo = pipeline.createStereoDepth()
    stereo.initialConfig.setConfidenceThreshold(245)
    stereo.setLeftRightCheck(kwargs.get("lr_check", False))
    stereo.setExtendedDisparity(kwargs.get("extended_disparity", False))
    stereo.setSubpixel(kwargs.get("subpixel", False))
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    detection_network: Union[dai.node.NeuralNetwork, dai.node.YoloSpatialDetectionNetwork] = pipeline.create(
        dai.node.YoloSpatialDetectionNetwork
    )
    detection_network.setDepthLowerThreshold(100)  # mm
    detection_network.setDepthUpperThreshold(10_000)  # mm
    detection_network.setBoundingBoxScaleFactor(0.3)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    stereo.depth.link(detection_network.inputDepth)

    return detection_network
