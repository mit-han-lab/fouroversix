import functools
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any

from packaging.version import Version, parse
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

SKIP_CUDA_BUILD = os.getenv("SKIP_CUDA_BUILD", "0") == "1"


@functools.cache
def get_cuda_archs() -> list[str]:
    return os.getenv("CUDA_ARCHS", "100;110;120").split(";")


def get_cuda_bare_metal_version() -> Version:
    raw_output = subprocess.check_output(  # noqa: S603
        [CUDA_HOME + "/bin/nvcc", "-V"],
        universal_newlines=True,
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    return parse(output[release_idx].split(",")[0])


def get_cuda_gencodes() -> list[str]:
    """
    Add -gencode flags based on nvcc capabilities.

    Uses the following rules:
      - sm_100/120 on CUDA >= 12.8
      - Use 100f on CUDA >= 12.9 (Blackwell family-specific)
      - Map requested 110 -> 101 if CUDA < 13.0 (Thor rename)
      - Embed PTX for newest arch for forward compatibility
    """

    archs = set(get_cuda_archs())
    cuda_version = get_cuda_bare_metal_version()
    cc_flags = []

    # Blackwell requires >= 12.8
    if cuda_version >= Version("12.8"):
        if "100" in archs:
            # CUDA 12.9 introduced "family-specific" for Blackwell (100f)
            if cuda_version >= Version("12.9"):
                cc_flags += ["-gencode", "arch=compute_100f,code=sm_100"]
            else:
                cc_flags += ["-gencode", "arch=compute_100,code=sm_100"]

        # Thor rename: 12.9 uses sm_101; 13.0+ uses sm_110
        if "110" in archs:
            if cuda_version >= Version("13.0"):
                cc_flags += ["-gencode", "arch=compute_110f,code=sm_110"]
            elif cuda_version >= Version("12.9"):
                # Provide Thor support for CUDA 12.9 via sm_101
                cc_flags += ["-gencode", "arch=compute_101f,code=sm_101"]
            # else: no Thor support in older toolkits

        if "120" in archs:
            # sm_120 is supported in CUDA 12.8/12.9+ toolkits
            if cuda_version >= Version("12.9"):
                cc_flags += ["-gencode", "arch=compute_120f,code=sm_120"]
            else:
                cc_flags += ["-gencode", "arch=compute_120,code=sm_120"]

    return cc_flags


class NinjaBuildExtension(BuildExtension):
    """
    Custom build extension that tells Ninja how many jobs to run.

    Credit: https://github.com/Dao-AILab/flash-attention/blob/main/setup.py
    """

    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            try:
                import psutil

                # calculate the maximum allowed NUM_JOBS based on cores
                max_num_jobs_cores = max(1, os.cpu_count() // 2)

                # calculate the maximum allowed NUM_JOBS based on free memory
                free_memory_gb = psutil.virtual_memory().available / (
                    1024**3
                )  # free memory in GB
                max_num_jobs_memory = int(
                    free_memory_gb / 9,
                )  # each JOB peak memory cost is ~8-9GB when threads = 4

                # pick lower value of jobs based on cores vs memory metric to minimize
                # oom and swap usage during compilation
                max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
                os.environ["MAX_JOBS"] = str(max_jobs)
            except ImportError:
                warnings.warn(
                    "psutil not found, install psutil and ninja to get better build "
                    "performance",
                    stacklevel=1,
                )

        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    ext_modules = None

    if SKIP_CUDA_BUILD:
        warnings.warn(
            "SKIP_CUDA_BUILD is set to 1, installing fouroversix without quantization "
            "and matmul kernels",
            stacklevel=1,
        )
    else:
        setup_dir = Path(__file__).parent
        kernels_dir = setup_dir / "src" / "fouroversix" / "csrc"
        sources = [
            path.relative_to(Path(__file__).parent).as_posix()
            for ext in ["**/*.cu", "**/*.cpp"]
            for path in kernels_dir.glob(ext)
        ]
        ext_modules = [
            CUDAExtension(
                "fouroversix._C",
                sources,
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "--expt-relaxed-constexpr",
                        "--use_fast_math",
                        "-DNDEBUG",
                        "-Xcompiler",
                        "-funroll-loops",
                        "-Xcompiler",
                        "-ffast-math",
                        "-Xcompiler",
                        "-finline-functions",
                        *get_cuda_gencodes(),
                    ],
                },
                include_dirs=[
                    setup_dir / "third_party/cutlass/examples/common",
                    setup_dir / "third_party/cutlass/include",
                    setup_dir / "third_party/cutlass/tools/util/include",
                    kernels_dir / "include",
                ],
            ),
        ]

    setup(
        name="fouroversix",
        ext_modules=ext_modules,
        cmdclass={"build_ext": NinjaBuildExtension},
        include_package_data=True,
    )
