from pathlib import Path

import modal

from .resources import (
    FOUROVERSIX_CACHE_PATH,
    FOUROVERSIX_INSTALL_PATH,
    app,
    cache_volume,
    get_image,
)


def add_files_for_build(img: modal.Image) -> modal.Image:
    img = img.run_commands(
        "git clone https://github.com/NVIDIA/cutlass.git "
        f"{FOUROVERSIX_INSTALL_PATH}/third_party/cutlass",
    )

    for file in [
        "LICENSE.md",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "src",
    ]:
        if (Path(__file__).parent.parent / file).is_file():
            img = img.add_local_file(file, f"{FOUROVERSIX_INSTALL_PATH}/{file}")
        else:
            img = img.add_local_dir(file, f"{FOUROVERSIX_INSTALL_PATH}/{file}")

    return img


build_img = get_image(
    dependencies=[],
    extra_env={"MAX_JOBS": "16"},
    extra_pip_dependencies=["build"],
    run_before_copy=add_files_for_build,
)

with build_img.imports():
    from build.__main__ import main as build_main


@app.function(
    image=build_img,
    cpu=16,
    memory=32 * 1024,
    timeout=30 * 60,
    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
)
def build() -> None:
    build_main(
        [
            "--outdir",
            (FOUROVERSIX_CACHE_PATH / "dist").as_posix(),
            FOUROVERSIX_INSTALL_PATH.as_posix(),
        ],
    )
