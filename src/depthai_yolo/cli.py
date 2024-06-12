# coding=utf-8
from pathlib import Path
from typing import List, Optional

import click
import typer
from typing_extensions import Annotated

from depthai_yolo.depthai_enums import color_res_opts, mono_res_opts, usb_speed_opts
from depthai_yolo.main_api import main
from depthai_yolo.pipelines.common import create_pipeline as create_pipeline_main
from depthai_yolo.pipelines.sr import create_pipeline as create_pipeline_sr
from depthai_yolo.utils import parse_yolo_model

app = typer.Typer()


@app.command()
def run(  # noqa: PLR0913
    app: Annotated[
        str,
        typer.Argument(
            ...,
            help="Provide app name for inference",
        ),
    ] = "api",
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
            help="Color camera resolution",
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
    *,
    spatial: Annotated[
        bool,
        typer.Option(
            ...,
            "-s",
            "--spatial/--no_spatial",
            help="Display spatial information",
        ),
    ] = False,
    lr_check: Annotated[
        bool,
        typer.Option(
            ...,
            "-lr",
            "--lr_check/--no_lr_check",
            help="If to True, it will perform left-right check on stereo pair, only for `spatial is True`",
        ),
    ] = True,
    extended_disparity: Annotated[
        bool,
        typer.Option(
            ...,
            "-extended",
            "--extended_disparity/--no_extended",
            help="If to True, it will enable disparity, only for `spatial is True`",
        ),
    ] = False,
    subpixel: Annotated[
        bool,
        typer.Option(
            ...,
            "-sub",
            "--subpixel/--no_subpixel",
            help="If to True, it will enable subpixel disparity, only for `spatial is True`",
        ),
    ] = False,
    fullFov: Annotated[
        bool,
        typer.Option(
            ...,
            "-F",
            "--fullFov/--no_fullFov",
            help="If to False, it will first center crop the frame to meet the NN aspect ratio and then scale down the image",
        ),
    ] = True,
    syncNN: Annotated[
        bool,
        typer.Option(..., "-syncNN", "--syncNN/--no_syncNN", help="Show synced frame"),
    ] = False,
    high_res: Annotated[
        bool,
        typer.Option(..., "-high", "--high_res/--no_high_res", help="Show synced frame"),
    ] = False,
    color: Annotated[
        bool,
        typer.Option(
            ...,
            "-color",
            "--color/--no_color",
            help="Show lens as color, only for `sr` ",
        ),
    ] = False,
):
    color_res = color_res_opts.get(color_res)
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
    )


if __name__ == "__main__":
    app()
