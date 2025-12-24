from __future__ import annotations

import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import modal
import tomllib

if TYPE_CHECKING:
    from collections.abc import Callable

FOUROVERSIX_CACHE_PATH = Path("/fouroversix")
FOUROVERSIX_INSTALL_PATH = Path("/root/fouroversix")
KERNEL_DEV_MODE = os.getenv("KERNEL_DEV_MODE", "0") == "1"

app = modal.App("fouroversix")
cache_volume = modal.Volume.from_name("fouroversix", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


class Dependency(str, Enum):
    """Dependencies to add to the base image."""

    awq = "awq"
    fast_hadamard_transform = "fast_hadamard_transform"
    flash_attention = "flash_attention"
    fouroversix = "fouroversix"
    fp_quant = "fp_quant"
    qutlass = "qutlass"
    spinquant = "spinquant"


cuda_version_to_image_tag = {
    "12.8": "nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04",
    "12.9": "nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04",
    "13.0": "nvcr.io/nvidia/cuda-dl-base:25.09-cuda13.0-devel-ubuntu24.04",
}


def build_fouroversix_ext() -> None:
    shutil.copytree(
        FOUROVERSIX_CACHE_PATH / "build",
        FOUROVERSIX_INSTALL_PATH / "build",
    )
    subprocess.run(
        ["python", "setup.py", "build_ext", "--inplace"],  # noqa: S607
        check=False,
    )
    shutil.copytree(
        FOUROVERSIX_INSTALL_PATH / "build",
        FOUROVERSIX_CACHE_PATH / "build",
        dirs_exist_ok=True,
    )


def install_flash_attn() -> None:
    subprocess.run(
        ["pip", "install", "flash-attn", "--no-build-isolation"],  # noqa: S607
        check=False,
    )


def install_fouroversix() -> None:
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            FOUROVERSIX_INSTALL_PATH.as_posix(),
        ],
        check=False,
    )


def install_fouroversix_non_editable() -> None:
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            FOUROVERSIX_INSTALL_PATH.as_posix(),
        ],
        check=False,
    )


def install_qutlass() -> None:
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "pip",
            "install",
            "--no-build-isolation",
            f"{FOUROVERSIX_INSTALL_PATH}/third_party/qutlass",
        ],
        check=False,
    )


def get_image(  # noqa: C901, PLR0912
    dependencies: list[Dependency] | None = None,
    *,
    cuda_version: str = "12.9",
    deploy: bool = False,
    extra_env: dict[str, str] | None = None,
    extra_pip_dependencies: list[str] | None = None,
    include_tests: bool = False,
    python_version: str = "3.12",
    pytorch_version: str = "2.9.1",
    run_before_copy: Callable[[modal.Image], modal.Image] | None = None,
) -> modal.Image:
    if dependencies is None:
        dependencies = [Dependency.fouroversix]

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    if not pyproject_path.exists():
        pyproject_path = Path(__file__).parent.parent / "fouroversix" / "pyproject.toml"

    with pyproject_path.open("rb") as f:
        pyproject_data = tomllib.load(f)

    img = (
        modal.Image.from_registry(
            cuda_version_to_image_tag[cuda_version],
            add_python=python_version,
        )
        .entrypoint([])
        .apt_install("clang", "git")
        .uv_pip_install(
            *filter(
                lambda x: not x.startswith("torch"),
                pyproject_data["build-system"]["requires"],
            ),
            *pyproject_data["project"]["optional-dependencies"]["tests"],
            "numpy",
        )
        .uv_pip_install(
            f"torch=={pytorch_version}",
            extra_index_url=(
                f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
            ),
        )
    )

    for dependency in dependencies:
        if dependency == Dependency.awq:
            img = img.add_local_dir(
                "third_party/llm-awq",
                f"{FOUROVERSIX_INSTALL_PATH}/third_party/llm-awq",
                copy=True,
            ).run_commands(
                f"pip install --no-deps {FOUROVERSIX_INSTALL_PATH}/third_party/llm-awq",
            )

        if dependency == Dependency.fast_hadamard_transform:
            img = img.add_local_dir(
                "third_party/fast-hadamard-transform",
                f"{FOUROVERSIX_INSTALL_PATH}/third_party/fast-hadamard-transform",
                copy=True,
            ).run_commands(
                f"pip install {FOUROVERSIX_INSTALL_PATH}/third_party"
                "/fast-hadamard-transform",
            )

        if dependency == Dependency.flash_attention:
            img = img.run_function(
                install_flash_attn,
                cpu=64,
                memory=128 * 1024,
                gpu="B200",
            )

        if dependency == Dependency.fouroversix:
            img = (
                img.env({"CUDA_ARCHS": "100", "FORCE_BUILD": "1", "MAX_JOBS": "32"})
                .add_local_dir(
                    "third_party/cutlass",
                    f"{FOUROVERSIX_INSTALL_PATH}/third_party/cutlass",
                    copy=True,
                )
                .add_local_file(
                    "pyproject.toml",
                    f"{FOUROVERSIX_INSTALL_PATH}/pyproject.toml",
                    copy=True,
                )
                .add_local_file(
                    "setup.py",
                    f"{FOUROVERSIX_INSTALL_PATH}/setup.py",
                    copy=True,
                )
                .add_local_file(
                    "src/fouroversix/__init__.py",
                    f"{FOUROVERSIX_INSTALL_PATH}/src/fouroversix/__init__.py",
                    copy=True,
                )
            )

            if KERNEL_DEV_MODE:
                img = (
                    img.add_local_file(
                        "README.md",
                        f"{FOUROVERSIX_INSTALL_PATH}/README.md",
                        copy=True,
                    )
                    .add_local_file(
                        "LICENSE.md",
                        f"{FOUROVERSIX_INSTALL_PATH}/LICENSE.md",
                        copy=True,
                    )
                    .workdir(FOUROVERSIX_INSTALL_PATH)
                )

            img = img.add_local_dir(
                "src/fouroversix/csrc",
                f"{FOUROVERSIX_INSTALL_PATH}/src/fouroversix/csrc",
                copy=True,
            )

            if KERNEL_DEV_MODE:
                img = img.run_function(
                    build_fouroversix_ext,
                    cpu=32,
                    memory=64 * 1024,
                    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
                ).workdir("/root")
            else:
                img = img.run_function(install_fouroversix, cpu=32, memory=64 * 1024)

        if dependency == Dependency.fp_quant:
            img = img.add_local_dir(
                "third_party/fp-quant",
                f"{FOUROVERSIX_INSTALL_PATH}/fpquant/fpquant_cli",
                copy=True,
            ).run_commands(
                f"pip install {FOUROVERSIX_INSTALL_PATH}/fpquant/fpquant_cli/"
                "inference_lib",
            )

        if dependency == Dependency.qutlass:
            img = (
                img.apt_install("cmake")
                .add_local_dir(
                    "third_party/qutlass",
                    f"{FOUROVERSIX_INSTALL_PATH}/third_party/qutlass",
                    copy=True,
                )
                .run_commands(
                    # Prevent qutlass from trying to clone cutlass during build process
                    f"rm -rf {FOUROVERSIX_INSTALL_PATH}/third_party/qutlass/.git",
                )
                .env({"MAX_JOBS": "32"})
                .run_function(install_qutlass, gpu="B200", cpu=32, memory=64 * 1024)
            )

        if dependency == Dependency.spinquant:
            img = img.add_local_dir(
                "third_party/spinquant",
                f"{FOUROVERSIX_INSTALL_PATH}/spinquant",
                copy=True,
            )

    if extra_pip_dependencies is not None:
        img = img.uv_pip_install(*extra_pip_dependencies)

    img = img.env({"HF_HOME": FOUROVERSIX_CACHE_PATH.as_posix(), **(extra_env or {})})

    if run_before_copy is not None:
        img = run_before_copy(img)

    # Add source files after all dependencies are added so we can avoid rebuilding when
    # they change
    for dependency in dependencies:
        if dependency == Dependency.fouroversix:
            img = img.add_local_dir(
                "src",
                f"{FOUROVERSIX_INSTALL_PATH}/src",
                copy=deploy or KERNEL_DEV_MODE,
            )

            if KERNEL_DEV_MODE:
                img = img.run_function(
                    install_fouroversix_non_editable,
                    cpu=8,
                    memory=32 * 1024,
                    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
                )

    if include_tests:
        img = img.add_local_dir("tests", f"{FOUROVERSIX_INSTALL_PATH}/tests")

    return img
