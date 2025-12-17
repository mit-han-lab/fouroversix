from __future__ import annotations

from pathlib import Path
from typing import Annotated

import modal
import typer

from .resources import FOUROVERSIX_INSTALL_PATH, app, get_image

img = get_image(include_tests=True)

with img.imports():
    import pytest


@app.function(image=img, cpu=4, memory=8 * 1024, gpu="B200", timeout=30 * 60)
def run_tests(path: str | None = None, filter: str | None = None) -> None:  # noqa: A002
    """Run tests on a B200 on Modal."""

    args = [(Path(FOUROVERSIX_INSTALL_PATH) / (path or "tests")).as_posix()]

    if filter:
        args.extend(["-k", filter])

    pytest.main(args)


def run_tests_on_modal(
    path: Annotated[str | None, typer.Argument()] = None,
    filter: str | None = None,  # noqa: A002
) -> None:
    with modal.enable_output(), app.run():
        run_tests.remote(path, filter)


if __name__ == "__main__":
    cli_app = typer.Typer()
    cli_app.command()(run_tests_on_modal)
    cli_app()
