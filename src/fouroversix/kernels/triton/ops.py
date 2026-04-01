from __future__ import annotations

import torch
import triton
import triton.language as tl
from fouroversix.quantize import QuantizedTensor, from_blocked
from fouroversix.utils import DataType, RoundStyle, ScaleRule, ScaleType
from triton.tools.tensor_descriptor import TensorDescriptor

from .constants import SCALE_MEGABLOCK_SIZE
from .dequantize import dequantize_with_tensor_descriptors
from .matmul import matmul_kernel
from .quantize import pseudo_quantization_kernel, quantization_kernel
from .rht import rht_kernel


def quantize(  # noqa: C901, PLR0915
    x: torch.Tensor,
    x_amax: torch.Tensor | None = None,
    had: torch.Tensor | None = None,
    *,
    dtype: DataType = DataType.nvfp4,
    round_style: RoundStyle = RoundStyle.nearest,
    scale_rule: ScaleRule = ScaleRule.mse,
    block_scale_2d: bool = False,
    transpose: bool = False,
    major_compute_capability: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    major_compute_capability = (
        torch.cuda.get_device_capability()[0]
        if major_compute_capability is None
        else major_compute_capability
    )

    if transpose:
        N, M = x.shape
    else:
        M, N = x.shape

    block_size_m = 128
    block_size_n = 4 * dtype.block_size

    tile_size_m = dtype.block_size
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
        dtype=torch.uint8,
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

        with torch.cuda.device(x.device):
            rht_kernel[rht_grid](
                x_desc,
                h_desc,
                x_rht_desc,
                BLOCK_SIZE_M=tile_size_m,
                BLOCK_SIZE_N=tile_size_n,
                TRANSPOSE=transpose,
            )

        transpose = False

        x_amax = (
            x_rht.abs().max().float()
            if dtype.scale_type != ScaleType.mx
            else torch.ones(1, device=x.device, dtype=torch.float32)
        )

    rbits_ptr = (
        torch.randint(0, torch.iinfo(torch.int32).max, (1,), device=x.device)
        if round_style.is_stochastic
        else None
    )

    with torch.cuda.device(x.device):
        quantization_kernel[grid](
            x_rht_desc if had is not None else x_desc,
            x_amax,
            x_e2m1_desc,
            x_sf_desc,
            rbits_ptr,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            TILE_SIZE_M=tile_size_m,
            TILE_SIZE_N=tile_size_n,
            MAX_QUANTIZED_VALUE=dtype.quantized_value_type.get_maximum_value(
                scale_rule,
            ),
            MAX_SCALE_FACTOR=dtype.scale_type.get_maximum_value(scale_rule),
            TRANSPOSE=transpose,
            QUANTIZED_VALUE_TYPE=dtype.quantized_value_type.value,
            QUANTIZED_VALUE_PACKING_FACTOR=dtype.quantized_value_type.packing_factor,
            ROUND_STYLE=round_style.value,
            SCALE_TYPE=dtype.scale_type.value,
            SCALE_GROUP_SIZE=dtype.block_size,
            SCALE_RULE=scale_rule.value,
            BLOCK_SCALE_2D=block_scale_2d,
            MAJOR_COMPUTE_CAPABILITY=major_compute_capability,
        )

    if x_sf.dtype != dtype.scale_type.torch_dtype:
        x_sf = x_sf.view(dtype.scale_type.torch_dtype)

    if dtype in {
        DataType.if3,
        DataType.if3_bs8,
        DataType.if4,
        DataType.if4_bs8,
        DataType.nvint3,
        DataType.nvint3_bs8,
        DataType.nvint4,
        DataType.nvint4_bs8,
        DataType.nvint6,
    }:
        x_sf = from_blocked(x_sf, (padded_m, padded_n // dtype.block_size))

    return x_e2m1, x_sf, x_amax


def pseudo_quantize(
    x: torch.Tensor,
    x_amax: torch.Tensor | None = None,
    had: torch.Tensor | None = None,
    *,
    dtype: DataType = DataType.nvfp4,
    round_style: RoundStyle = RoundStyle.nearest,
    scale_rule: ScaleRule = ScaleRule.mse,
    block_scale_2d: bool = False,
    transpose: bool = False,
    major_compute_capability: int | None = None,
) -> torch.Tensor:
    major_compute_capability = (
        torch.cuda.get_device_capability()[0]
        if major_compute_capability is None
        else major_compute_capability
    )

    if transpose:
        N, M = x.shape
    else:
        M, N = x.shape

    block_size_m = 128
    block_size_n = 4 * dtype.block_size

    tile_size_m = dtype.block_size
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

    x_out = torch.empty(
        (padded_m, padded_n),
        device=x.device,
        dtype=torch.bfloat16,
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
    x_out_desc = TensorDescriptor.from_tensor(
        x_out,
        block_shape=[tile_size_m, tile_size_n],
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

        with torch.cuda.device(x.device):
            rht_kernel[rht_grid](
                x_desc,
                h_desc,
                x_rht_desc,
                BLOCK_SIZE_M=tile_size_m,
                BLOCK_SIZE_N=tile_size_n,
                TRANSPOSE=transpose,
            )

        transpose = False

        x_amax = (
            x_rht.abs().max().float()
            if dtype.scale_type != ScaleType.mx
            else torch.ones(1, device=x.device, dtype=torch.float32)
        )

    rbits_ptr = (
        torch.randint(0, torch.iinfo(torch.int32).max, (1,), device=x.device)
        if round_style.is_stochastic
        else None
    )

    with torch.cuda.device(x.device):
        pseudo_quantization_kernel[grid](
            x_rht_desc if had is not None else x_desc,
            x_amax,
            x_out_desc,
            rbits_ptr,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            TILE_SIZE_M=tile_size_m,
            TILE_SIZE_N=tile_size_n,
            MAX_QUANTIZED_VALUE=dtype.quantized_value_type.get_maximum_value(
                scale_rule,
            ),
            MAX_SCALE_FACTOR=dtype.scale_type.get_maximum_value(scale_rule),
            TRANSPOSE=transpose,
            QUANTIZED_VALUE_TYPE=dtype.quantized_value_type.value,
            ROUND_STYLE=round_style.value,
            SCALE_TYPE=dtype.scale_type.value,
            SCALE_GROUP_SIZE=dtype.block_size,
            SCALE_RULE=scale_rule.value,
            BLOCK_SCALE_2D=block_scale_2d,
            MAJOR_COMPUTE_CAPABILITY=major_compute_capability,
        )

    return x_out[:M, :N]


def dequantize_values(
    tensor: QuantizedTensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    major_compute_capability: int | None = None,
) -> torch.Tensor:
    major_compute_capability = (
        torch.cuda.get_device_capability()[0]
        if major_compute_capability is None
        else major_compute_capability
    )

    block_size_m = 128
    block_size_n = 64

    values_desc = TensorDescriptor.from_tensor(
        tensor.values,
        block_shape=[
            block_size_m,
            block_size_n // tensor.dtype.quantized_value_type.packing_factor,
        ],
    )

    scale_factors_desc = (
        TensorDescriptor.from_tensor(
            tensor.scale_factors,
            block_shape=[SCALE_MEGABLOCK_SIZE.value],
        )
        if tensor.dtype
        in {
            DataType.if3,
            DataType.if3_bs8,
            DataType.if4,
            DataType.if4_bs8,
            DataType.nvint3,
            DataType.nvint3_bs8,
            DataType.nvint4,
            DataType.nvint4_bs8,
            DataType.nvint6,
            DataType.if6_e2m3,
            DataType.if6_e3m2,
        }
        else None
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
        triton.cdiv(tensor.padded_shape[0], meta["BLOCK_SIZE_M"]),
        triton.cdiv(tensor.padded_shape[1], meta["BLOCK_SIZE_N"]),
    )

    dequantize_with_tensor_descriptors[grid](
        values_desc,
        scale_factors_desc,
        output_desc,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        QUANTIZED_VALUE_TYPE=tensor.dtype.quantized_value_type.value,
        QUANTIZED_VALUE_PACKING_FACTOR=tensor.dtype.quantized_value_type.packing_factor,
        SCALE_FACTOR_TYPE=tensor.dtype.scale_type.value,
        OUT_DTYPE=tl.float16,
        MAJOR_COMPUTE_CAPABILITY=major_compute_capability,
    )

    return output.to(dtype)


def matmul(
    input: QuantizedTensor,
    other: QuantizedTensor,
    *,
    out_dtype: DataType,
    major_compute_capability: int | None = None,
) -> torch.Tensor:
    major_compute_capability = (
        torch.cuda.get_device_capability()[0]
        if major_compute_capability is None
        else major_compute_capability
    )

    m = input.original_shape[0]
    n = other.original_shape[0]
    k = input.original_shape[1]

    input_sf_shape = (
        input.padded_shape[0],
        input.padded_shape[1] // input.dtype.block_size,
    )
    other_sf_shape = (
        other.padded_shape[0],
        other.padded_shape[1] // other.dtype.block_size,
    )

    if input.scale_factors_are_in_blackwell_layout:
        input_scale_factors = from_blocked(
            input.scale_factors,
            input_sf_shape,
        )
    else:
        input_scale_factors = input.scale_factors.reshape(input_sf_shape)

    if other.scale_factors_are_in_blackwell_layout:
        other_scale_factors = from_blocked(
            other.scale_factors,
            other_sf_shape,
        )
    else:
        other_scale_factors = other.scale_factors.reshape(other_sf_shape)

    output = torch.empty(
        (m, n),
        device=input.values.device,
        dtype=out_dtype.torch_dtype,
    )

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
    )

    matmul_kernel[grid](
        input.values,
        input_scale_factors.view(torch.uint8),
        input.amax,
        other.values,
        other_scale_factors.view(torch.uint8),
        other.amax,
        output,
        m,
        n,
        k,
        input.values.stride(0),
        input_scale_factors.stride(0),
        input.values.stride(1),
        input_scale_factors.stride(1),
        other.values.stride(0),
        other_scale_factors.stride(0),
        other.values.stride(1),
        other_scale_factors.stride(1),
        output.stride(0),
        output.stride(1),
        INPUT_QUANTIZED_VALUE_TYPE=input.dtype.quantized_value_type.value,
        INPUT_QUANTIZED_VALUE_PACKING_FACTOR=input.dtype.quantized_value_type.packing_factor,
        INPUT_QUANTIZED_VALUE_MAX=input.dtype.quantized_value_type.get_maximum_value(
            input.scale_rule,
        ),
        INPUT_SCALE_TYPE=input.dtype.scale_type.value,
        INPUT_SCALE_FACTOR_MAX=input.dtype.scale_type.get_maximum_value(
            input.scale_rule,
        ),
        INPUT_SCALE_GROUP_SIZE=input.dtype.block_size,
        INPUT_ROUND_STYLE=input.round_style.value,
        OTHER_QUANTIZED_VALUE_TYPE=other.dtype.quantized_value_type.value,
        OTHER_QUANTIZED_VALUE_PACKING_FACTOR=other.dtype.quantized_value_type.packing_factor,
        OTHER_QUANTIZED_VALUE_MAX=other.dtype.quantized_value_type.get_maximum_value(
            other.scale_rule,
        ),
        OTHER_SCALE_TYPE=other.dtype.scale_type.value,
        OTHER_SCALE_FACTOR_MAX=other.dtype.scale_type.get_maximum_value(
            other.scale_rule,
        ),
        OTHER_SCALE_GROUP_SIZE=other.dtype.block_size,
        OTHER_ROUND_STYLE=other.round_style.value,
        INTERMEDIATE_DTYPE=tl.float16,
        OUT_DTYPE=tl.bfloat16,
        MAJOR_COMPUTE_CAPABILITY=major_compute_capability,
        M_N_K_BUCKET=f"{m // 128}x{n // 128}x{k // 128}",
    )

    return output
