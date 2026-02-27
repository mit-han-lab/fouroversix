from __future__ import annotations

import torch
from fouroversix.utils import (
    DataType,
    RoundStyle,
    ScaleRule,
    ScaleType,
    device_supports_cvt_rn_e2m1x2,
    device_supports_cvt_rs_e2m1x4,
)
from fouroversix.quantize import QuantizedTensor, from_blocked
from triton.tools.tensor_descriptor import TensorDescriptor

from .rht import rht_kernel
from .quantize import SCALE_MEGABLOCK_SIZE, quantization_kernel
from .dequantize import dequantize_kernel


def quantize_to_fp4(
    x: torch.Tensor,
    x_amax: torch.Tensor | None = None,
    had: torch.Tensor | None = None,
    *,
    dtype: DataType = DataType.nvfp4,
    round_style: RoundStyle = RoundStyle.nearest,
    scale_rule: ScaleRule = ScaleRule.mse,
    block_scale_2d: bool = False,
    transpose: bool = False,
    rbits: int = -1,
    use_blackwell_cvt_rn_instructions: bool | None = None,
    use_blackwell_cvt_rs_instructions: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if transpose:
        N, M = x.shape
    else:
        M, N = x.shape

    block_size_m = 128
    block_size_n = 4 * dtype.block_size

    tile_size_m = 16
    tile_size_n = block_size_n

    if tile_size_n % dtype.block_size != 0:
        msg = (
            f"Tile size N ({tile_size_n}) must be divisible by the block size "
            f"({dtype.block_size})"
        )
        raise ValueError(msg)

    if block_scale_2d and tile_size_m % dtype.block_size != 0:
        msg = (
            f"Tile size M ({tile_size_m}) must be divisible by the block size "
            f"({dtype.block_size}) when performing 2D block scaling"
        )
        raise ValueError(msg)

    if x_amax is None:
        x_amax = (
            x.abs().max().float()
            if dtype.scale_type != ScaleType.mx
            else torch.ones(1, device=x.device, dtype=torch.float32)
        )

    padded_m = M + (block_size_m - M % block_size_m) % block_size_m
    padded_n = N + (block_size_n - N % block_size_n) % block_size_n

    x_e2m1 = torch.empty(
        (padded_m, padded_n // dtype.quantized_value_type.packing_factor),
        device=x.device,
        dtype=torch.uint8,
    )
    x_sf = torch.empty(
        padded_m * padded_n // dtype.block_size,
        device=x.device,
        dtype=(
            torch.uint8
            if dtype.scale_type != ScaleType.nv
            else dtype.scale_type.torch_dtype
        ),
    )

    grid = lambda _: (  # noqa: E731
        padded_m // block_size_m,
        padded_n // block_size_n,
    )

    x_desc = TensorDescriptor.from_tensor(
        x,
        block_shape=[
            tile_size_m if not transpose else tile_size_n,
            tile_size_n if not transpose else tile_size_m,
        ],
    )
    x_e2m1_desc = TensorDescriptor.from_tensor(
        x_e2m1,
        block_shape=[
            tile_size_m,
            tile_size_n // dtype.quantized_value_type.packing_factor,
        ],
    )
    x_sf_desc = TensorDescriptor.from_tensor(
        x_sf,
        block_shape=[SCALE_MEGABLOCK_SIZE.value],
    )

    if had is not None:
        had_block_size = had.shape[0]

        if M % had_block_size != 0:
            msg = (
                f"The first dimension of A ({M}) must be divisible by the width of H "
                f"({had_block_size})"
            )
            raise ValueError(msg)
        if N % had_block_size != 0:
            msg = (
                f"The second dimension of A ({N}) must be divisible by the width of H "
                f"({had_block_size})"
            )
            raise ValueError(msg)
        if had.shape[0] != had.shape[1]:
            msg = "H must be a square matrix"
            raise ValueError(msg)
        if (had.shape[0] & (had.shape[0] - 1)) != 0:
            msg = "H must have dimensions that are a power of two"
            raise ValueError(msg)

        x_rht = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

        h_desc = TensorDescriptor.from_tensor(
            had,
            block_shape=[had_block_size, had_block_size],
        )
        x_rht_desc = TensorDescriptor.from_tensor(
            x_rht,
            block_shape=[tile_size_m, tile_size_n],
        )

        rht_grid = lambda _: (  # noqa: E731
            padded_m // tile_size_m,
            padded_n // tile_size_n,
        )

        rht_kernel[rht_grid](
            x_desc,
            h_desc,
            x_rht_desc,
            BLOCK_SIZE_M=tile_size_m,
            BLOCK_SIZE_N=tile_size_n,
            TRANSPOSE=transpose,
        )

        transpose = False
        x_amax = x_rht.abs().max().float()

    quantization_kernel[grid](
        x_rht_desc if had is not None else x_desc,
        x_amax,
        x_e2m1_desc,
        x_sf_desc,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        TILE_SIZE_M=tile_size_m,
        TILE_SIZE_N=tile_size_n,
        MAX_QUANTIZED_VALUE=dtype.quantized_value_type.get_maximum_value(scale_rule),
        MAX_SCALE_FACTOR=dtype.scale_type.get_maximum_value(scale_rule),
        TRANSPOSE=transpose,
        QUANTIZED_VALUE_TYPE=dtype.quantized_value_type.value,
        QUANTIZED_VALUE_PACKING_FACTOR=dtype.quantized_value_type.packing_factor,
        ROUND_STYLE=round_style.value,
        SCALE_TYPE=dtype.scale_type.value,
        SCALE_GROUP_SIZE=dtype.block_size,
        SCALE_RULE=scale_rule.value,
        BLOCK_SCALE_2D=block_scale_2d,
        RBITS=rbits,
        USE_BLACKWELL_CVT_RN_INSTRUCTIONS=(
            device_supports_cvt_rn_e2m1x2()
            if use_blackwell_cvt_rn_instructions is None
            else use_blackwell_cvt_rn_instructions
        ),
        USE_BLACKWELL_CVT_RS_INSTRUCTIONS=(
            device_supports_cvt_rs_e2m1x4()
            if use_blackwell_cvt_rs_instructions is None
            else use_blackwell_cvt_rs_instructions
        ),
    )

    if x_sf.dtype != dtype.scale_type.torch_dtype:
        x_sf = x_sf.view(dtype.scale_type.torch_dtype)

    if dtype == DataType.if4:
        x_sf = from_blocked(x_sf, (padded_m, padded_n // dtype.block_size))

    return x_e2m1, x_sf, x_amax


def dequantize_values(
    tensor: QuantizedTensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    use_blackwell_cvt_rn_instructions: bool | None = None,
) -> torch.Tensor:
    block_size_m = 128
    block_size_n = 64

    values_desc = TensorDescriptor.from_tensor(
        tensor.values,
        block_shape=[block_size_m, block_size_n],
    )

    output = torch.empty(
        tensor.padded_shape[0],
        tensor.padded_shape[1],
        device=tensor.values.device,
        dtype=torch.float16,
    )

    output_desc = TensorDescriptor.from_tensor(
        output,
        block_shape=[block_size_m, block_size_n],
    )

    grid = lambda meta: (  # noqa: E731
        tensor.padded_shape[0] // meta["BLOCK_SIZE_M"],
        tensor.padded_shape[1] // meta["BLOCK_SIZE_N"],
    )

    dequantize_kernel[grid](
        values_desc,
        output_desc,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        USE_BLACKWELL_CVT_RN_INSTRUCTIONS=(
            device_supports_cvt_rn_e2m1x2()
            if use_blackwell_cvt_rn_instructions is None
            else use_blackwell_cvt_rn_instructions
        ),
    )

    return output.to(dtype)
