# coding=utf-8
from typing import Union

import depthai as dai


def create_pipeline(**kwargs):
    model_data = kwargs.get("model_data")
    config_data = kwargs.get("config_data")
    nn_config = config_data.nn_config
    color = kwargs.get("color", False)
    res = (
        kwargs.get("color_res", dai.ColorCameraProperties.SensorResolution.THE_720_P)
        if color
        else kwargs.get("mono_res", dai.MonoCameraProperties.SensorResolution.THE_720_P)
    )

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    mono_right = pipeline.create(dai.node.ColorCamera) if color else pipeline.create(dai.node.MonoCamera)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setResolution(res)
    mono_right.setFps(kwargs.get("fps", 30))

    detection_network = (
        create_stereo(pipeline, res=res, mono_right=mono_right, **kwargs)
        if kwargs.get("spatial", False)
        else pipeline.create(dai.node.YoloDetectionNetwork)
    )

    xout_image = pipeline.create(dai.node.XLinkOut)
    nn_out = pipeline.create(dai.node.XLinkOut)

    xout_image.setStreamName("image")
    nn_out.setStreamName("nn")
    if not color:
        image_manip = pipeline.createImageManip()
        image_manip.initialConfig.setResize(nn_config.nn_width, nn_config.nn_height)
        image_manip.initialConfig.setKeepAspectRatio(not kwargs.get("fullFov", False))
        image_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    else:
        image_manip = None

    # Network specific settings
    detection_network.setConfidenceThreshold(nn_config.NN_specific_metadata.confidence_threshold)
    detection_network.setNumClasses(nn_config.NN_specific_metadata.classes)
    detection_network.setCoordinateSize(nn_config.NN_specific_metadata.coordinates)
    detection_network.setAnchors(nn_config.NN_specific_metadata.anchors)
    detection_network.setAnchorMasks(nn_config.NN_specific_metadata.anchor_masks)
    detection_network.setIouThreshold(nn_config.NN_specific_metadata.iou_threshold)
    detection_network.setBlob(model_data)
    # detection_network.setNumInferenceThreads(2)
    detection_network.input.setBlocking(False)
    detection_network.input.setQueueSize(1)

    # Linking
    if color:
        mono_right.preview.link(detection_network.input)

    else:
        mono_right.out.link(image_manip.inputImage)
        image_manip.out.link(detection_network.input)

    if kwargs.get("syncNN", False):
        detection_network.passthrough.link(xout_image.input)
    elif color and kwargs.get("high_res", False):
        mono_right.video.link(xout_image.input)
    elif color:
        mono_right.preview.link(xout_image.input)
    else:
        mono_right.out.link(xout_image.input)

    detection_network.out.link(nn_out.input)

    xin_control = pipeline.create(dai.node.XLinkIn)
    xin_control.setStreamName("control")
    xin_control.out.link(mono_right.inputControl)

    return pipeline


def create_stereo(pipeline, **kwargs):
    res = kwargs.get("res")
    mono_right = kwargs.get("mono_right")

    mono_left = (
        pipeline.create(dai.node.ColorCamera) if kwargs.get("color", False) else pipeline.create(dai.node.MonoCamera)
    )
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setResolution(res)
    mono_left.setFps(kwargs.get("fps", 30))

    stereo = pipeline.createStereoDepth()
    stereo.initialConfig.setConfidenceThreshold(245)
    stereo.setLeftRightCheck(kwargs.get("lr_check", False))
    stereo.setExtendedDisparity(kwargs.get("extended_disparity", False))
    stereo.setSubpixel(kwargs.get("subpixel", False))
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_C)

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
