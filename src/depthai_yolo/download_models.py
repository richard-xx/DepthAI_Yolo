# coding=utf-8
import blobconverter
import click
import typer
from depthai_yolo.yolo_define_models import depthai_model_zoo, open_model_zoo
from typing_extensions import Annotated

app = typer.Typer()


@app.command()
def download_models(
    model_name: Annotated[
        str,
        typer.Argument(..., click_type=click.Choice(depthai_model_zoo + open_model_zoo + ["all"]), help="Model name"),
    ] = "all",
):
    """Download all models from the model zoo"""
    if model_name == "all":
        for nn_name in depthai_model_zoo + open_model_zoo:
            download(nn_name)
    else:
        download(model_name)


def download(nn_name):
    nn_path = None
    if nn_name in depthai_model_zoo:
        nn_path = blobconverter.from_zoo(
            nn_name,
            shaves=6,
            zoo_type="depthai",
            use_cache=True,
        )
    elif nn_name in open_model_zoo:
        nn_path = blobconverter.from_zoo(
            nn_name,
            shaves=6,
            zoo_type="intel",
            use_cache=True,
        )
    return nn_path


if __name__ == "__main__":
    app()
