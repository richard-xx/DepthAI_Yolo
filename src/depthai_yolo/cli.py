# coding=utf-8
from enum import Enum
from pathlib import Path
from typing import List, Optional

import click
import depthai as dai
import typer
from typing_extensions import Annotated

from depthai_yolo.depthai_enums import color_res_opts, mono_res_opts, usb_speed_opts
from depthai_yolo.download_models import download_models
from depthai_yolo.main_api import main
from depthai_yolo.pipelines.common import create_pipeline as create_pipeline_main
from depthai_yolo.pipelines.lr import StereoPair
from depthai_yolo.pipelines.lr import create_pipeline as create_pipeline_lr
from depthai_yolo.pipelines.sr import create_pipeline as create_pipeline_sr
from depthai_yolo.utils import parse_yolo_model
from depthai_yolo.yolo_define_models import all_model_zoo

app = typer.Typer()


class APP(str, Enum):
    OAK_D = "oak"
    OAK_LR = "lr"
    OAK_SR = "sr"


def print_defined_models(ctx, param, value) -> None:
    if not value or ctx.resilient_parsing:
        return
    print("Defined models:")
    for model in all_model_zoo:
        typer.echo(f"\t- {model}")

    ctx.exit()


def download_defined_models(ctx, param, value) -> None:
    if not value or ctx.resilient_parsing:
        return
    download_models("all")
    ctx.exit()


@app.command(name="depthai_yolo")
def run(  # noqa: PLR0913
    app: Annotated[
        APP,
        typer.Argument(
            ...,
            help="Provide app name for inference",
        ),
    ] = APP.OAK_D,
    model_file: Annotated[
        Path,
        typer.Option(
            ...,
            "-m",
            "-w",
            "--model",
            "--weight",
            help="Provide model name or model path for inference",
        ),
    ] = ...,
    config_file: Annotated[
        Optional[Path],
        typer.Option(
            ...,
            "-c",
            "-j",
            "--config",
            "--json",
            help="Provide config path for inference",
        ),
    ] = None,
    classes_id: Annotated[
        Optional[List[int]],
        Optional[List[str]],
        typer.Option(..., "-classes_id", "--classes_id", help="filter by class: --classes 0 or --classes 0 2 3 "),
    ] = None,
    classes_str: Annotated[
        Optional[List[str]],
        typer.Option(
            ..., "-classes_str", "--classes_str", help="filter by class: --classes person or --classes person cup"
        ),
    ] = None,
    usbSpeed: Annotated[
        str,
        typer.Option(
            ...,
            "-usbs",
            "--usbSpeed",
            click_type=click.Choice(usb_speed_opts, case_sensitive=False),
            case_sensitive=False,
            help="Force USB communication speed.",
        ),
    ] = "SUPER_PLUS",
    color_res: Annotated[
        str,
        typer.Option(
            ...,
            click_type=click.Choice(color_res_opts, case_sensitive=False),
            help="Color camera resolution, if using OAK-LR, must be selected from `THE_720_P/THE_400_P` (zoom from THE_1200_P).",
        ),
    ] = "THE_1080_P",
    mono_res: Annotated[
        str,
        typer.Option(..., click_type=click.Choice(mono_res_opts, case_sensitive=False), help="Mono camera resolution"),
    ] = "THE_400_P",
    fps: Annotated[
        float,
        typer.Option(
            ...,
            "-fps",
            "--fps",
            help="Set capture FPS for all cameras.",
        ),
    ] = 30,
    stereo_pair: Annotated[
        StereoPair,
        typer.Option(
            ...,
            "--stereo_pair",
            help="Stereo pair, current only for OAK-LR.",
        ),
    ] = StereoPair.LR,
    *,
    spatial: Annotated[
        bool,
        typer.Option(
            ...,
            "-s",
            "--spatial",
            is_flag=True,
            help="Display spatial information",
        ),
    ] = False,
    lr_check: Annotated[
        bool,
        typer.Option(
            ...,
            "-lr",
            "--lr_check",
            is_flag=True,
            help="If to True, it will perform left-right check on stereo pair, only for `spatial is True`",
        ),
    ] = True,
    extended_disparity: Annotated[
        bool,
        typer.Option(
            ...,
            "-e",
            "--extended_disparity",
            is_flag=True,
            help="If to True, it will enable disparity, only for `spatial is True`",
        ),
    ] = False,
    subpixel: Annotated[
        bool,
        typer.Option(
            ...,
            "-sub",
            "--subpixel",
            is_flag=True,
            help="If to True, it will enable subpixel disparity, only for `spatial is True`",
        ),
    ] = False,
    fullFov: Annotated[
        bool,
        typer.Option(
            ...,
            "-F",
            "--full_fov/--no_full_fov",
            is_flag=True,
            help="If to False, it will first center crop the frame to meet the NN aspect ratio and then scale down the image",
        ),
    ] = True,
    syncNN: Annotated[
        bool,
        typer.Option(..., "-syncNN", "--syncNN/--no_syncNN", is_flag=True, help="Show synced frame"),
    ] = False,
    high_res: Annotated[
        bool,
        typer.Option(..., "-high", "--high_res/--no_high_res", is_flag=True, help="Show synced frame"),
    ] = False,
    color: Annotated[
        bool,
        typer.Option(
            ...,
            "-color",
            "--color/--no_color",
            is_flag=True,
            help="Show lens as color, only for `sr` ",
        ),
    ] = False,
    list_models: Annotated[
        bool,
        typer.Option(
            ...,
            "-list",
            "-ls",
            "--list_models",
            is_eager=True,
            is_flag=True,
            callback=print_defined_models,
            help="List all pre-defined models",
        ),
    ] = False,
    download: Annotated[
        bool,
        typer.Option(
            ...,
            "-d",
            "--download",
            is_flag=True,
            is_eager=True,
            callback=download_defined_models,
            help="Download all pre-defined models",
        ),
    ] = False,
):
    isp_scale = (2, 3)  # only for lr app
    color_res = color_res_opts.get(color_res)

    if app == "lr":
        if color_res == "THE_720_P":
            isp_scale = (2, 3)
        if color_res == "THE_400_P":
            isp_scale = (1, 3)
        color_res = dai.ColorCameraProperties.SensorResolution.THE_1200_P

    mono_res = mono_res_opts.get(mono_res)
    usbSpeed = usb_speed_opts.get(usbSpeed)

    config_data, model_data = parse_yolo_model(config_file, model_file)
    if classes_id is None:
        classes_id = []
    if classes_str is None:
        classes_str = []

    classes_id = [x for x in classes_id if x <= config_data.mappings.nc]
    classes_str = [x for x in classes_str if x in config_data.mappings.labels]
    classes_id.extend([config_data.mappings.labels.index(x) for x in classes_str])
    classes = set(classes_id)

    create_pipeline = create_pipeline_main
    if app == "api":
        create_pipeline = create_pipeline_main

    elif app == "sr":
        create_pipeline = create_pipeline_sr

    elif app == "lr":
        create_pipeline = create_pipeline_lr

    main(
        create_pipeline,
        classes=classes,
        usbSpeed=usbSpeed,
        color_res=color_res,
        mono_res=mono_res,
        fps=fps,
        spatial=spatial,
        lr_check=lr_check,
        extended_disparity=extended_disparity,
        subpixel=subpixel,
        fullFov=fullFov,
        syncNN=syncNN,
        high_res=high_res,
        config_data=config_data,
        model_data=model_data,
        color=color,
        isp_scale=isp_scale,
        stereo_pair=stereo_pair,
    )


if __name__ == "__main__":
    app()
