from random import randint

import torch
import triton
import triton.language as tl
from fouroversix.utils import DataType

from .constants import (
    IF4_INT_EXPANSION_FACTOR,
    IF4_INT_EXPANSION_FACTOR_RCP,
    QUANTIZED_VALUE_TYPE_IF4,
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def eden_1x16s_fp4_kernel(  # noqa: PLR0915
    x_ptr,
    hadamard_matrix_ptr,
    current_amax_ptr,
    output_ptr,
    next_amax_ptr,
    n_elements: tl.constexpr,
    hadamard_dim: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    seed: int,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # hadamard transform
    offsets_hadamard = tl.arange(0, hadamard_dim * hadamard_dim)
    hadamard_matrix = tl.load(hadamard_matrix_ptr + offsets_hadamard).reshape(
        hadamard_dim,
        hadamard_dim,
    )
    x = tl.reshape(x_flat, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_had = tl.dot(x, hadamard_matrix)  # not TN!, A @ B!

    # write amax for next iter
    tl.atomic_max(next_amax_ptr, tl.max(tl.abs(x_had)).to(tl.float32), sem="relaxed")

    # group
    x_grouped = tl.reshape(x_had, (BLOCK_SIZE // group_size, group_size))

    # amax
    # Not 448 because eden needs space to rescale up a bitsometimes after the correction
    scales_max = 255.99
    val_max = 6.0 / scale_override
    amax = tl.load(current_amax_ptr)
    s_dec = tl.where(
        amax == 0.0,
        1.0,
        amax / scales_max / val_max,
    )

    # scale
    s_dec_b = tl.max(tl.abs(x_grouped), axis=-1, keep_dims=True) / val_max
    s_dec_b_e4m3 = (s_dec_b / s_dec).to(tl.float8e4nv).to(tl.float32)
    s_dec_b_e4m3 = tl.where(
        s_dec_b_e4m3 == 0,
        1.0,
        s_dec_b_e4m3,
    )
    x_scaled = x_grouped / (s_dec_b_e4m3 * s_dec)

    # quantize
    x_scaled_abs = tl.abs(x_scaled)
    x_scaled_sign = tl.where(
        x_scaled > 0,
        1,
        -1,
    )
    x_fp4 = (
        tl.where(
            x_scaled_abs >= 5,  # noqa: PLR2004
            6,
            tl.where(
                x_scaled_abs >= 3.5,  # noqa: PLR2004
                4,
                tl.where(
                    x_scaled_abs >= 2.5,  # noqa: PLR2004
                    3,
                    tl.where(
                        x_scaled_abs >= 1.75,  # noqa: PLR2004
                        2,
                        tl.where(
                            x_scaled_abs >= 1.25,  # noqa: PLR2004
                            1.5,
                            tl.where(
                                x_scaled_abs >= 0.75,  # noqa: PLR2004
                                1,
                                tl.where(
                                    x_scaled_abs >= 0.25,  # noqa: PLR2004
                                    0.5,
                                    0,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        * x_scaled_sign
    )

    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF4:
        # INT4 quantization
        x_int4 = (
            tl.extra.cuda.libdevice.rint(
                tl.clamp(x_scaled * IF4_INT_EXPANSION_FACTOR_RCP, -7.0, 7.0),
            )
            * IF4_INT_EXPANSION_FACTOR
        )

        # Per-group MSE comparison
        diff_fp4 = x_scaled - x_fp4
        diff_int4 = x_scaled - x_int4
        err_fp4 = tl.sum(diff_fp4 * diff_fp4, axis=-1, keep_dims=True)
        err_int4 = tl.sum(diff_int4 * diff_int4, axis=-1, keep_dims=True)

        # Select winner per block
        x_fp4 = tl.where(err_int4 < err_fp4, x_int4, x_fp4)

    # Calculate EDEN scale
    x_scaled = tl.reshape(x_scaled, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_fp4 = tl.reshape(x_fp4, (BLOCK_SIZE // hadamard_dim, hadamard_dim))

    num = tl.sum(x_scaled * x_scaled, axis=-1, keep_dims=True)
    denom = tl.sum(x_scaled * x_fp4, axis=-1, keep_dims=True)

    correction = tl.where(
        denom == 0.0,
        1.0,
        num / denom,
    )

    # Apply EDEN scale
    scales = tl.reshape(
        s_dec_b_e4m3,
        (BLOCK_SIZE // hadamard_dim, hadamard_dim // group_size),
    )
    corrected_scales = tl.reshape(scales * correction, (BLOCK_SIZE // group_size, 1))

    bitscales = tl.cast(corrected_scales.to(tl.float8e4nv), tl.uint8, bitcast=True)
    prevscale = tl.cast((bitscales - 1), tl.float8e4nv, bitcast=True).to(tl.float32)
    currscale = tl.cast((bitscales), tl.float8e4nv, bitcast=True).to(tl.float32)
    nextscale = tl.cast((bitscales + 1), tl.float8e4nv, bitcast=True).to(tl.float32)

    up = tl.where(
        currscale > corrected_scales,
        currscale,
        nextscale,
    )
    down = tl.where(
        currscale > corrected_scales,
        prevscale,
        currscale,
    )

    prob_up = (corrected_scales - down) / (up - down)

    scale_start_idx = pid * (BLOCK_SIZE // group_size)
    scale_offsets = scale_start_idx + tl.arange(0, BLOCK_SIZE // group_size)
    sampled_prob = tl.rand(seed, scale_offsets).reshape(BLOCK_SIZE // group_size, 1)

    scales = tl.where(
        sampled_prob < prob_up,
        up,
        down,
    )
    scales = tl.reshape(scales, (BLOCK_SIZE // group_size, 1))
    x_fp4 = tl.reshape(x_fp4, (BLOCK_SIZE // group_size, group_size))

    # Reshape back to flat form for storage
    x_dequantized = x_fp4 * scales * s_dec
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))

    # store
    tl.store(
        output_ptr + offsets,
        x_dequantized_flat.to(x_ptr.dtype.element_ty),
        mask=mask,
    )


@torch.compiler.disable()
def eden_1x16s_fp4_kernel_wrapper(
    x: torch.Tensor,
    hadamard_matrix: torch.Tensor,
    scale_override: float,
    group_size: int,
    current_amax: torch.Tensor,
    dtype: DataType,
) -> [torch.Tensor, torch.Tensor]:
    hadamard_dim = hadamard_matrix.size(0)
    assert hadamard_matrix.size(1) == hadamard_dim  # noqa: S101
    assert x.numel() % hadamard_dim == 0  # noqa: S101
    assert hadamard_dim % group_size == 0  # noqa: S101

    x = x.contiguous()
    hadamard_matrix = hadamard_matrix.T.contiguous()  # .T.contiguous() + tl.dot -> TN
    output = torch.empty_like(x)
    seed = randint(0, 1000000)  # noqa: S311

    next_amax = torch.zeros_like(current_amax)

    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731

    eden_1x16s_fp4_kernel[grid](
        x_ptr=x,
        hadamard_matrix_ptr=hadamard_matrix,
        current_amax_ptr=current_amax,
        output_ptr=output,
        next_amax_ptr=next_amax,
        n_elements=n_elements,
        hadamard_dim=hadamard_dim,
        scale_override=scale_override,
        group_size=group_size,
        seed=seed,
        QUANTIZED_VALUE_TYPE=dtype.quantized_value_type.value,
    )
    return output, next_amax
