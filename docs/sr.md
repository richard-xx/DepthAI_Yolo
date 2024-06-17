## 介绍
这个脚本用于创建一个基于 depthai 库的 Pipeline，用于处理摄像头输入和目标检测神经网络推理。它可以根据传入的参数配置摄像头和神经网络模型，并构建一个数据流处理的 Pipeline。

## 源码
??? note "create_pipeline_for_sr.py"

    ```python
    --8<-- "src/depthai_yolo/pipelines/sr.py"
    ```

## 用法
这个脚本可以通过调用`create_pipeline`函数来创建一个 Pipeline 对象。函数接受以下参数：

- `model_data`：神经网络模型的数据文件路径。
- `config_data`：神经网络模型的配置文件路径。
- `color`：一个布尔值，指示是否使用彩色摄像头。默认为`False`。
- `color_res`：彩色摄像头的分辨率。默认为`dai.ColorCameraProperties.SensorResolution.THE_720_P`。
- `mono_res`：单色摄像头的分辨率。默认为`dai.MonoCameraProperties.SensorResolution.THE_720_P`。
- `fps`：摄像头的帧率。默认为 30。
- `spatial`：一个布尔值，指示是否使用空间目标检测网络。默认为`False`。
- `fullFov`：一个布尔值，指示是否保持全视场比例。默认为`False`。
- `syncNN`：一个布尔值，指示是否同步神经网络推理。默认为`False`。
- `high_res`：一个布尔值，指示是否使用高分辨率预览。默认为`False`。

## 示例
下面是一个使用示例：

```python
import depthai as dai
from create_pipeline_for_sr import create_pipeline

# 指定模型文件和配置文件路径
model_data = "path/to/model.blob"
config_data = "path/to/config.json"

# 创建 Pipeline
pipeline = create_pipeline(model_data=model_data, config_data=config_data, color=True)

# 创建 Device
with dai.Device(pipeline) as device:
    # 启动 Pipeline
    device.startPipeline()

    # 获取输出流
    image_queue = device.getOutputQueue(name="image", maxSize=4, blocking=False)
    nn_queue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    while True:
        # 从输出队列中获取数据
        image_data = image_queue.get()
        nn_data = nn_queue.get()

        # 处理数据...
```

## 常见问题解答

!!! info "Q: 该脚本支持哪些深度摄像头设备？"

    A: 该脚本使用 depthai 库，支持 OAK_D，OAK-D-SR 以及使用 `OV9*82` 相机模组作为左右相机的 OAK-FFC。

!!! info "Q: 如何指定自定义的模型和配置文件？"

    A: 在调用`create_pipeline`函数时，通过`model_data`和`config_data`参数指定模型和配置文件的路径。

!!! info "Q: 是否支持其他类型的神经网络模型？"

    A: 目前这个脚本支持 YoloDetectionNetwork 和 YoloSpatialDetectionNetwork 两种类型的神经网络模型。
