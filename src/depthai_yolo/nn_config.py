# coding=utf-8
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from typing_extensions import Annotated


class YoloConfig(BaseModel):
    """This schema is useful for parsing the YOLO config file from .json."""

    classes: Annotated[int, Field(..., description="Number of classes.")]
    coordinates: Annotated[int, Field(default=4, description="Number of coordinates.")]
    anchors: Annotated[List[float], Field(default=[], description="List of anchors.")]
    anchor_masks: Annotated[Dict[str, List[int]], Field(default={}, description="Dictionary of anchor masks.")]
    iou_threshold: Annotated[float, Field(default=0.3, description="IOU threshold.")]
    confidence_threshold: Annotated[float, Field(default=0.5, description="Confidence threshold.")]


class Mappings(BaseModel):
    """This schema defines the fields and their types for the model configuration."""

    labels: Annotated[List[str], Field(..., description="List of labels.")]

    @computed_field(description="Number of classes.", return_type=int)
    def nc(self) -> int:
        return len(self.labels)


class Model(BaseModel):
    blob: Annotated[str, Field(default="", description="Path to the blob file.")]
    name: Annotated[str, Field(default="", description="Name of the model.", alias="model_name")]
    zoo: Annotated[str, Field(default="", description="Name of the model zoo.")]
    xml: Annotated[str, Field(default="", description="Path to the xml file.")]
    bin: Annotated[str, Field(default="", description="Path to the bin file.")]


class NNConfig(BaseModel):
    """This schema defines the fields and their types for the neural network configuration."""

    model_config = ConfigDict(validate_assignment=True)
    NN_family: Annotated[str, Field(default="", description="Name of the neural network family.")]
    confidence_threshold: Annotated[
        float, Field(default=None, description="Confidence threshold for the neural network.")
    ]
    NN_specific_metadata: Annotated[
        YoloConfig, Field(default=None, description="Metadata specific to the neural network.")
    ]
    output_format: Annotated[str, Field(..., description="Format of the output from the neural network.")]
    input_size: Annotated[str, Field(..., description="Size of the input to the neural network.")]
    nn_width: Annotated[Optional[int], Field(default=None, description="Width of the neural network input.")]
    nn_height: Annotated[Optional[int], Field(default=None, description="Height of the neural network input.")]

    @model_validator(mode="before")
    @classmethod
    def set_nn_width_and_height(cls, values):
        if values["input_size"] is not None:
            nn_width, nn_height = values["input_size"].split("x")
            values["nn_width"], values["nn_height"] = int(nn_width), int(nn_height)
        return values


class Config(BaseModel):
    """This schema defines the fields and their types for the overall configuration."""

    model: Annotated[Model, Field(..., description="Model configuration.")]
    handler: Annotated[Optional[str], Field(default=None, description="Name of the handler.")]
    nn_config: Annotated[NNConfig, Field(...)]
    openvino_version: Annotated[Optional[str], Field(default=None, description="Version of OpenVINO.")]
    mappings: Annotated[Optional[Mappings], Field(default=None, description="Mappings configuration.")]
    version: Annotated[Optional[int], Field(default=None, description="Version of the configuration.")]
