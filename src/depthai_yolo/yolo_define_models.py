# coding=utf-8

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

all_model_zoo = depthai_model_zoo + open_model_zoo
