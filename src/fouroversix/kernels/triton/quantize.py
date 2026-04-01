from __future__ import annotations

import triton
import triton.language as tl

from .constants import (
    E2M1_MAX_VALUE,
    E4M3_MAX_FOUROVERSIX,
    IF3_INT_EXPANSION_FACTOR,
    IF3_INT_EXPANSION_FACTOR_RCP,
    IF4_INT_EXPANSION_FACTOR,
    IF4_INT_EXPANSION_FACTOR_RCP,
    IF6_E2M3_INT_EXPANSION_FACTOR,
    IF6_E2M3_INT_EXPANSION_FACTOR_RCP,
    IF6_E3M2_INT_EXPANSION_FACTOR,
    IF6_E3M2_INT_EXPANSION_FACTOR_RCP,
    QUANTIZED_VALUE_TYPE_FP3,
    QUANTIZED_VALUE_TYPE_FP4,
    QUANTIZED_VALUE_TYPE_FP6_E2M3,
    QUANTIZED_VALUE_TYPE_FP6_E3M2,
    QUANTIZED_VALUE_TYPE_IF3,
    QUANTIZED_VALUE_TYPE_IF4,
    QUANTIZED_VALUE_TYPE_IF6_E2M3,
    QUANTIZED_VALUE_TYPE_IF6_E3M2,
    QUANTIZED_VALUE_TYPE_INT3,
    QUANTIZED_VALUE_TYPE_INT4,
    QUANTIZED_VALUE_TYPE_INT6,
    ROUND_STYLE_STOCHASTIC_UNBIASED,
    SCALE_MEGABLOCK_SIZE,
    SCALE_RULE_ABS_MAX,
    SCALE_RULE_MAE,
    SCALE_RULE_MSE,
    SCALE_RULE_STATIC_4,
    SCALE_RULE_STATIC_6,
    SCALE_TYPE_MX,
    SCALE_TYPE_NV,
    SCALE_TYPE_NV_IF,
    UNBIASED_SR_ADJUSTMENT_FACTOR,
)
from .fp3 import convert_to_e2m0, convert_to_e2m0_and_dequantized_fp16
from .fp4 import convert_to_e2m1x2, convert_to_e2m1x2_and_quantized_fp16
from .fp6 import (
    convert_to_e2m3x2,
    convert_to_e2m3x2_and_dequantized_fp16,
    convert_to_e3m2x2,
    convert_to_e3m2x2_and_dequantized_fp16,
)
from .fp8 import convert_e4m3_to_high_precision, convert_to_e4m3_with_rtn
from .int3 import convert_to_int3_and_quantized_fp16
from .int4 import (
    convert_to_int4x2,
    convert_to_int4x2_and_quantized_fp16,
)
from .int6 import convert_to_int6_and_quantized_fp16


@triton.jit
def compute_scale_factors_kernel(
    x_block,
    x_amax_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    MAX_QUANTIZED_VALUE: tl.constexpr,
    MAX_SCALE_FACTOR: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SCALE_EXPANSION_FACTOR: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor]:
    SR_SCALE: tl.constexpr = (
        UNBIASED_SR_ADJUSTMENT_FACTOR
        if ROUND_STYLE == ROUND_STYLE_STOCHASTIC_UNBIASED
        else 1
    )

    x_amax = tl.load(x_amax_ptr)

    x_scale_blocks = x_block.reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // SCALE_GROUP_SIZE,
        SCALE_GROUP_SIZE,
    )

    if x_amax == 0:
        x_scales_hp = tl.full(
            (BLOCK_SIZE_M, BLOCK_SIZE_N // SCALE_GROUP_SIZE),
            0,
            dtype=tl.float32,
        )

        if SCALE_TYPE == SCALE_TYPE_MX:
            x_scales = tl.full(
                (BLOCK_SIZE_M, BLOCK_SIZE_N // SCALE_GROUP_SIZE),
                0,
                dtype=tl.uint8,
            )
    elif SCALE_TYPE == SCALE_TYPE_MX:
        x_scales_hp = tl.div_rn(
            tl.max(x_scale_blocks.abs(), axis=-1),
            MAX_QUANTIZED_VALUE,
        )
        x_scales_e8m0_u32 = x_scales_hp.cast(tl.uint32, bitcast=True)

        # Use the 8-bit exponent as the scale factor
        x_scales_e8m0 = ((x_scales_e8m0_u32 >> 23) & 0xFF).to(tl.uint8)

        # Add one in order to round up
        x_scales = tl.where(
            (x_scales_e8m0_u32 & 0x7FFFFF) == 0,
            x_scales_e8m0,
            x_scales_e8m0 + 1,
        )

        # Convert the rounded-up scale factor back to a 32-bit float
        x_scales_hp = (x_scales.cast(tl.uint32) << 23).cast(tl.float32, bitcast=True)
    elif SCALE_TYPE == SCALE_TYPE_NV or SCALE_TYPE == SCALE_TYPE_NV_IF:
        encode_scale = tl.div_rn(
            MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR * SR_SCALE,
            x_amax,
        )

        x_scales_hp = (
            tl.div_rn(
                tl.max(x_scale_blocks.abs(), axis=-1),
                MAX_QUANTIZED_VALUE * SR_SCALE,
            )
            * encode_scale
        )

        # When x_amax is very small, encode_scale overflows to inf, causing
        # 0 * inf = NaN for zero blocks. Replace NaN with 0.
        x_scales_hp = tl.where(x_scales_hp != x_scales_hp, 0.0, x_scales_hp)

    if BLOCK_SCALE_2D:
        x_scales_hp = (
            tl.max(
                x_scales_hp.reshape(
                    BLOCK_SIZE_M // SCALE_GROUP_SIZE,
                    SCALE_GROUP_SIZE,
                    BLOCK_SIZE_N // SCALE_GROUP_SIZE,
                ).permute(0, 2, 1),
                axis=-1,
            )
            .expand_dims(0)
            .broadcast_to(
                BLOCK_SIZE_M // SCALE_GROUP_SIZE,
                SCALE_GROUP_SIZE,
                BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            )
            .permute(1, 0, 2)
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // SCALE_GROUP_SIZE)
        )

    if SCALE_EXPANSION_FACTOR is not None:
        x_scales_hp = x_scales_hp * SCALE_EXPANSION_FACTOR

    if SCALE_TYPE == SCALE_TYPE_MX:
        x_scales = ((x_scales_hp.to(tl.uint32, bitcast=True) >> 23) & 0xFF).to(tl.uint8)
        x_block_scaled = tl.where(
            x_scales.expand_dims(2) != 0,
            tl.div_rn(
                x_scale_blocks,
                x_scales_hp.to(x_scale_blocks.dtype).expand_dims(2),
            ),
            0,
        )
    elif SCALE_TYPE == SCALE_TYPE_NV or SCALE_TYPE == SCALE_TYPE_NV_IF:
        x_scales = convert_to_e4m3_with_rtn(x_scales_hp, MAJOR_COMPUTE_CAPABILITY)

        decode_scale = tl.div_rn(
            1,
            tl.div_rn(MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR * SR_SCALE, x_amax),
        )

        x_block_scaled = tl.where(
            x_scales.expand_dims(2).cast(tl.uint8, bitcast=True) != 0,
            x_scale_blocks
            * tl.div_rn(
                1,
                decode_scale
                * convert_e4m3_to_high_precision(
                    x_scales,
                    x_amax.dtype,
                    MAJOR_COMPUTE_CAPABILITY,
                ).expand_dims(2),
            ),
            0,
        )

    return x_block_scaled, x_scales, x_amax


@triton.jit
def compute_error_and_select_kernel(
    original_values,
    dequantized_1,
    dequantized_2,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
) -> None:
    diff_1 = dequantized_1 - original_values
    diff_2 = dequantized_2 - original_values

    if SCALE_RULE == SCALE_RULE_ABS_MAX:
        error_1 = tl.max(tl.abs(diff_1), axis=-1)
        error_2 = tl.max(tl.abs(diff_2), axis=-1)
    elif SCALE_RULE == SCALE_RULE_MAE:
        error_1 = tl.sum(tl.abs(diff_1), axis=-1)
        error_2 = tl.sum(tl.abs(diff_2), axis=-1)
    elif SCALE_RULE == SCALE_RULE_MSE:
        error_1 = tl.sum(diff_1 * diff_1, axis=-1)
        error_2 = tl.sum(diff_2 * diff_2, axis=-1)

    if BLOCK_SCALE_2D:
        error_1 = error_1.reshape(
            BLOCK_SIZE_M // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
        ).permute(0, 2, 1)
        error_2 = error_2.reshape(
            BLOCK_SIZE_M // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
        ).permute(0, 2, 1)

        if SCALE_RULE == SCALE_RULE_ABS_MAX:
            error_1 = tl.max(error_1, axis=-1)
            error_2 = tl.max(error_2, axis=-1)
        elif SCALE_RULE == SCALE_RULE_MAE or SCALE_RULE == SCALE_RULE_MSE:
            error_1 = tl.sum(error_1, axis=-1)
            error_2 = tl.sum(error_2, axis=-1)

        error_1 = (
            error_1.expand_dims(0)
            .broadcast_to(
                SCALE_GROUP_SIZE,
                BLOCK_SIZE_M // SCALE_GROUP_SIZE,
                BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            )
            .permute(1, 0, 2)
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // SCALE_GROUP_SIZE)
        )

        error_2 = (
            error_2.expand_dims(0)
            .broadcast_to(
                SCALE_GROUP_SIZE,
                BLOCK_SIZE_M // SCALE_GROUP_SIZE,
                BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            )
            .permute(1, 0, 2)
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // SCALE_GROUP_SIZE)
        )

    return error_1 < error_2


@triton.jit
def generic_block_scaled_quantization_kernel(
    x_block,
    x_amax_ptr,
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MAX_QUANTIZED_VALUE: tl.constexpr,
    MAX_SCALE_FACTOR: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    x_block_scaled, x_scales, _ = compute_scale_factors_kernel(
        x_block,
        x_amax_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SCALE_2D,
        MAX_QUANTIZED_VALUE,
        MAX_SCALE_FACTOR,
        ROUND_STYLE,
        SCALE_TYPE,
        SCALE_GROUP_SIZE,
        None,
        MAJOR_COMPUTE_CAPABILITY,
    )

    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP4:
        x_e2m1 = convert_to_e2m1x2(
            x_block_scaled,
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            SCALE_GROUP_SIZE,
            ROUND_STYLE,
            MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP6_E2M3:
        x_e2m1 = convert_to_e2m3x2(
            x_block_scaled,
            MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP6_E3M2:
        x_e2m1 = convert_to_e3m2x2(
            x_block_scaled,
            MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_INT4:
        x_e2m1 = convert_to_int4x2(
            x_block_scaled,
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            SCALE_GROUP_SIZE,
            ROUND_STYLE,
            1,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_INT6:
        x_e2m1, _ = convert_to_int6_and_quantized_fp16(
            x_block_scaled,
            1,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP3:
        x_e2m1 = convert_to_e2m0(
            x_block_scaled,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_INT3:
        x_e2m1, _ = convert_to_int3_and_quantized_fp16(
            x_block_scaled,
            1,
        )

    return x_e2m1, x_scales


@triton.jit
def intfloat_block_scaled_quantization_kernel(
    x_block,
    x_amax_ptr,
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MAX_QUANTIZED_VALUE: tl.constexpr,
    MAX_SCALE_FACTOR: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    SR_SCALE: tl.constexpr = (
        UNBIASED_SR_ADJUSTMENT_FACTOR
        if ROUND_STYLE == ROUND_STYLE_STOCHASTIC_UNBIASED
        else 1
    )

    x_block_scaled, x_scales, x_amax = compute_scale_factors_kernel(
        x_block,
        x_amax_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SCALE_2D,
        MAX_QUANTIZED_VALUE,
        MAX_SCALE_FACTOR,
        ROUND_STYLE,
        SCALE_TYPE_NV_IF,
        SCALE_GROUP_SIZE,
        None,
        MAJOR_COMPUTE_CAPABILITY,
    )

    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF3:
        INT_EXPANSION_FACTOR: tl.constexpr = IF3_INT_EXPANSION_FACTOR

        x_fp, x_fp_fp16 = convert_to_e2m0_and_dequantized_fp16(
            x_block_scaled,
        )

        x_int, x_int_fp16 = convert_to_int3_and_quantized_fp16(
            x_block_scaled,
            IF3_INT_EXPANSION_FACTOR_RCP,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF4:
        INT_EXPANSION_FACTOR: tl.constexpr = IF4_INT_EXPANSION_FACTOR

        x_fp, x_fp_fp16 = convert_to_e2m1x2_and_quantized_fp16(
            x_block_scaled,
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            ROUND_STYLE,
            SCALE_GROUP_SIZE,
            MAJOR_COMPUTE_CAPABILITY,
        )

        x_int, x_int_fp16 = convert_to_int4x2_and_quantized_fp16(
            x_block_scaled,
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            SCALE_GROUP_SIZE,
            ROUND_STYLE,
            IF4_INT_EXPANSION_FACTOR_RCP,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF6_E2M3:
        INT_EXPANSION_FACTOR: tl.constexpr = IF6_E2M3_INT_EXPANSION_FACTOR

        x_fp, x_fp_fp16 = convert_to_e2m3x2_and_dequantized_fp16(
            x_block_scaled,
            MAJOR_COMPUTE_CAPABILITY,
        )

        x_int, x_int_fp16 = convert_to_int6_and_quantized_fp16(
            x_block_scaled,
            IF6_E2M3_INT_EXPANSION_FACTOR_RCP,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF6_E3M2:
        INT_EXPANSION_FACTOR: tl.constexpr = IF6_E3M2_INT_EXPANSION_FACTOR

        x_fp, x_fp_fp16 = convert_to_e3m2x2_and_dequantized_fp16(
            x_block_scaled,
            MAJOR_COMPUTE_CAPABILITY,
        )

        x_int, x_int_fp16 = convert_to_int6_and_quantized_fp16(
            x_block_scaled,
            IF6_E3M2_INT_EXPANSION_FACTOR_RCP,
        )

    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF4:
        x_fp_dequantized = tl.div_rn(
            (
                x_fp_fp16.to(x_amax.dtype)
                * convert_e4m3_to_high_precision(
                    x_scales,
                    x_amax.dtype,
                    MAJOR_COMPUTE_CAPABILITY,
                ).expand_dims(2)
                * x_amax
            ),
            MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR * SR_SCALE,
        )

        x_int_dequantized = tl.div_rn(
            (
                (
                    x_int_fp16.to(x_amax.dtype)
                    * convert_e4m3_to_high_precision(
                        x_scales,
                        x_amax.dtype,
                        MAJOR_COMPUTE_CAPABILITY,
                    ).expand_dims(2)
                    * x_amax
                )
                * INT_EXPANSION_FACTOR
            ),
            MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR * SR_SCALE,
        )
    else:
        x_fp_dequantized = (
            x_fp_fp16.to(x_amax.dtype)
            * convert_e4m3_to_high_precision(
                x_scales,
                x_amax.dtype,
                MAJOR_COMPUTE_CAPABILITY,
            ).expand_dims(2)
            * x_amax
        ) / (MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR * SR_SCALE)

        x_int_dequantized = (
            (
                x_int_fp16.to(x_amax.dtype)
                * convert_e4m3_to_high_precision(
                    x_scales,
                    x_amax.dtype,
                    MAJOR_COMPUTE_CAPABILITY,
                ).expand_dims(2)
                * x_amax
            )
            * INT_EXPANSION_FACTOR
        ) / (MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR * SR_SCALE)

    int_indicator = compute_error_and_select_kernel(
        x_block.reshape(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
        ),
        x_int_dequantized,
        x_fp_dequantized,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SCALE_RULE,
        SCALE_GROUP_SIZE,
        BLOCK_SCALE_2D,
    )

    x_values = tl.where(int_indicator.expand_dims(2), x_int, x_fp)
    x_scales = tl.where(int_indicator, x_scales + 128, x_scales)

    return x_values, x_scales


@triton.jit
def nvfp4_fouroversix_block_scaled_quantization_kernel(
    x_block,
    x_amax_ptr,
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    SCALE_GROUP_SIZE: tl.constexpr = 16
    SR_SCALE: tl.constexpr = (
        UNBIASED_SR_ADJUSTMENT_FACTOR
        if ROUND_STYLE == ROUND_STYLE_STOCHASTIC_UNBIASED
        else 1
    )

    x_block_scaled_6, x_scales_6, x_amax = compute_scale_factors_kernel(
        x_block,
        x_amax_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SCALE_2D,
        E2M1_MAX_VALUE,
        E4M3_MAX_FOUROVERSIX,
        ROUND_STYLE,
        SCALE_TYPE_NV_IF,
        SCALE_GROUP_SIZE,
        None,
        MAJOR_COMPUTE_CAPABILITY,
    )

    x_6, x_6_fp16 = convert_to_e2m1x2_and_quantized_fp16(
        x_block_scaled_6,
        rbits_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        ROUND_STYLE,
        SCALE_GROUP_SIZE,
        MAJOR_COMPUTE_CAPABILITY,
    )

    x_6_dequantized = tl.div_rn(
        x_6_fp16.to(x_amax.dtype)
        * convert_e4m3_to_high_precision(
            x_scales_6,
            x_amax.dtype,
            MAJOR_COMPUTE_CAPABILITY,
        ).expand_dims(2)
        * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX * SR_SCALE,
    )

    x_block_scaled_4, x_scales_4, _ = compute_scale_factors_kernel(
        x_block,
        x_amax_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SCALE_2D,
        E2M1_MAX_VALUE,
        E4M3_MAX_FOUROVERSIX,
        ROUND_STYLE,
        SCALE_TYPE_NV_IF,
        SCALE_GROUP_SIZE,
        1.5,
        MAJOR_COMPUTE_CAPABILITY,
    )

    x_4, x_4_fp16 = convert_to_e2m1x2_and_quantized_fp16(
        x_block_scaled_4,
        rbits_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        ROUND_STYLE,
        SCALE_GROUP_SIZE,
        MAJOR_COMPUTE_CAPABILITY,
    )

    x_4_dequantized = tl.div_rn(
        x_4_fp16.to(x_amax.dtype)
        * convert_e4m3_to_high_precision(
            x_scales_4,
            x_amax.dtype,
            MAJOR_COMPUTE_CAPABILITY,
        ).expand_dims(2)
        * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX * SR_SCALE,
    )

    four_indicator = compute_error_and_select_kernel(
        x_block.reshape(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
        ),
        x_4_dequantized,
        x_6_dequantized,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SCALE_RULE,
        SCALE_GROUP_SIZE,
        BLOCK_SCALE_2D,
    )

    x_e2m1 = tl.where(four_indicator.expand_dims(2), x_4, x_6)
    x_scales = tl.where(four_indicator, x_scales_4, x_scales_6)

    return x_e2m1, x_scales


@triton.jit
def quantization_kernel(
    x_desc,
    x_amax_ptr,
    x_e2m1_desc,
    x_sf_desc,
    rbits_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_N: tl.constexpr,
    MAX_QUANTIZED_VALUE: tl.constexpr,
    MAX_SCALE_FACTOR: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    QUANTIZED_VALUE_PACKING_FACTOR: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_block_offset = pid_m * BLOCK_SIZE_M
    n_block_offset = pid_n * BLOCK_SIZE_N

    NUM_TILES_M: tl.constexpr = BLOCK_SIZE_M // TILE_SIZE_M
    NUM_TILES_N: tl.constexpr = BLOCK_SIZE_N // TILE_SIZE_N
    SF_PER_TILE_M: tl.constexpr = TILE_SIZE_M
    SF_PER_TILE_N: tl.constexpr = TILE_SIZE_N // SCALE_GROUP_SIZE

    output_scales = tl.zeros(
        (NUM_TILES_M, NUM_TILES_N, SF_PER_TILE_M, SF_PER_TILE_N),
        dtype=tl.uint8,
    )

    output_scales_idx = (
        tl.arange(0, NUM_TILES_M)[:, None, None, None] * NUM_TILES_N
        + tl.arange(0, NUM_TILES_N)[None, :, None, None]
    ).broadcast_to(NUM_TILES_M, NUM_TILES_N, SF_PER_TILE_M, SF_PER_TILE_N)

    for tile_m in range(NUM_TILES_M):
        for tile_n in range(NUM_TILES_N):
            # Load [B, B] block from A or A^T
            if not TRANSPOSE:
                x_block = x_desc.load(
                    [
                        m_block_offset + tile_m * TILE_SIZE_M,
                        n_block_offset + tile_n * TILE_SIZE_N,
                    ],
                )
            else:
                x_block = x_desc.load(
                    [
                        n_block_offset + tile_n * TILE_SIZE_N,
                        m_block_offset + tile_m * TILE_SIZE_M,
                    ],
                ).T

            x_block = x_block.to(tl.float32)

            if SCALE_RULE == SCALE_RULE_STATIC_6 or SCALE_RULE == SCALE_RULE_STATIC_4:
                x_e2m1, x_scales = generic_block_scaled_quantization_kernel(
                    x_block,
                    x_amax_ptr,
                    rbits_ptr,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    MAX_QUANTIZED_VALUE,
                    MAX_SCALE_FACTOR,
                    QUANTIZED_VALUE_TYPE,
                    ROUND_STYLE,
                    SCALE_TYPE,
                    SCALE_GROUP_SIZE,
                    BLOCK_SCALE_2D,
                    MAJOR_COMPUTE_CAPABILITY,
                )
            elif SCALE_TYPE == SCALE_TYPE_NV_IF:
                x_e2m1, x_scales = intfloat_block_scaled_quantization_kernel(
                    x_block,
                    x_amax_ptr,
                    rbits_ptr,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    MAX_QUANTIZED_VALUE,
                    MAX_SCALE_FACTOR,
                    QUANTIZED_VALUE_TYPE,
                    ROUND_STYLE,
                    SCALE_GROUP_SIZE,
                    SCALE_RULE,
                    BLOCK_SCALE_2D,
                    MAJOR_COMPUTE_CAPABILITY,
                )
            else:
                x_e2m1, x_scales = nvfp4_fouroversix_block_scaled_quantization_kernel(
                    x_block,
                    x_amax_ptr,
                    rbits_ptr,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    ROUND_STYLE,
                    SCALE_RULE,
                    BLOCK_SCALE_2D,
                    MAJOR_COMPUTE_CAPABILITY,
                )

            x_e2m1_desc.store(
                [
                    pid_m * BLOCK_SIZE_M + tile_m * TILE_SIZE_M,
                    (pid_n * BLOCK_SIZE_N + tile_n * TILE_SIZE_N)
                    // QUANTIZED_VALUE_PACKING_FACTOR,
                ],
                x_e2m1.reshape(
                    TILE_SIZE_M,
                    TILE_SIZE_N // QUANTIZED_VALUE_PACKING_FACTOR,
                ),
            )

            output_scales = tl.where(
                output_scales_idx == tile_m * NUM_TILES_N + tile_n,
                x_scales[None, None, :, :],
                output_scales,
            )

    output_scales = (
        output_scales.permute(0, 2, 1, 3).reshape(4, 32, 4).permute(1, 0, 2).ravel()
    )

    scale_block_offset = (pid_m * tl.num_programs(1) + pid_n) * SCALE_MEGABLOCK_SIZE
    x_sf_desc.store([scale_block_offset], output_scales)


@triton.jit
def pseudo_generic_block_scaled_quantization_kernel(
    x_block,
    x_amax_ptr,
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MAX_QUANTIZED_VALUE: tl.constexpr,
    MAX_SCALE_FACTOR: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    SR_SCALE: tl.constexpr = (
        UNBIASED_SR_ADJUSTMENT_FACTOR
        if ROUND_STYLE == ROUND_STYLE_STOCHASTIC_UNBIASED
        else 1
    )

    x_block_scaled, x_scales, x_amax = compute_scale_factors_kernel(
        x_block,
        x_amax_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SCALE_2D,
        MAX_QUANTIZED_VALUE,
        MAX_SCALE_FACTOR,
        ROUND_STYLE,
        SCALE_TYPE,
        SCALE_GROUP_SIZE,
        None,
        MAJOR_COMPUTE_CAPABILITY,
    )

    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP4:
        _, x_fp16 = convert_to_e2m1x2_and_quantized_fp16(
            x_block_scaled,
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            ROUND_STYLE,
            SCALE_GROUP_SIZE,
            MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP6_E2M3:
        _, x_fp16 = convert_to_e2m3x2_and_dequantized_fp16(
            x_block_scaled,
            MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP6_E3M2:
        _, x_fp16 = convert_to_e3m2x2_and_dequantized_fp16(
            x_block_scaled,
            MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_INT4:
        _, x_fp16 = convert_to_int4x2_and_quantized_fp16(
            x_block_scaled,
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            SCALE_GROUP_SIZE,
            ROUND_STYLE,
            1,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_INT6:
        _, x_fp16 = convert_to_int6_and_quantized_fp16(
            x_block_scaled,
            1,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP3:
        _, x_fp16 = convert_to_e2m0_and_dequantized_fp16(x_block_scaled)
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_INT3:
        _, x_fp16 = convert_to_int3_and_quantized_fp16(x_block_scaled, 1)

    if SCALE_TYPE == SCALE_TYPE_MX:
        x_scales_hp = (x_scales.cast(tl.uint32) << 23).cast(tl.float32, bitcast=True)
        x_dequantized = x_fp16.to(tl.float32) * x_scales_hp.expand_dims(2)
    elif SCALE_TYPE == SCALE_TYPE_NV or SCALE_TYPE == SCALE_TYPE_NV_IF:
        x_dequantized = tl.div_rn(
            x_fp16.to(x_amax.dtype)
            * convert_e4m3_to_high_precision(
                x_scales,
                x_amax.dtype,
                MAJOR_COMPUTE_CAPABILITY,
            ).expand_dims(2)
            * x_amax,
            MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR * SR_SCALE,
        )

    return x_dequantized.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)


@triton.jit
def pseudo_intfloat_block_scaled_quantization_kernel(
    x_block,
    x_amax_ptr,
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MAX_QUANTIZED_VALUE: tl.constexpr,
    MAX_SCALE_FACTOR: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    SR_SCALE: tl.constexpr = (
        UNBIASED_SR_ADJUSTMENT_FACTOR
        if ROUND_STYLE == ROUND_STYLE_STOCHASTIC_UNBIASED
        else 1
    )

    x_block_scaled, x_scales, x_amax = compute_scale_factors_kernel(
        x_block,
        x_amax_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SCALE_2D,
        MAX_QUANTIZED_VALUE,
        MAX_SCALE_FACTOR,
        ROUND_STYLE,
        SCALE_TYPE_NV_IF,
        SCALE_GROUP_SIZE,
        None,
        MAJOR_COMPUTE_CAPABILITY,
    )

    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF3:
        INT_EXPANSION_FACTOR: tl.constexpr = IF3_INT_EXPANSION_FACTOR

        _, x_fp_fp16 = convert_to_e2m0_and_dequantized_fp16(x_block_scaled)

        _, x_int_fp16 = convert_to_int3_and_quantized_fp16(
            x_block_scaled,
            IF3_INT_EXPANSION_FACTOR_RCP,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF4:
        INT_EXPANSION_FACTOR: tl.constexpr = IF4_INT_EXPANSION_FACTOR

        _, x_fp_fp16 = convert_to_e2m1x2_and_quantized_fp16(
            x_block_scaled,
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            ROUND_STYLE,
            SCALE_GROUP_SIZE,
            MAJOR_COMPUTE_CAPABILITY,
        )

        _, x_int_fp16 = convert_to_int4x2_and_quantized_fp16(
            x_block_scaled,
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            SCALE_GROUP_SIZE,
            ROUND_STYLE,
            IF4_INT_EXPANSION_FACTOR_RCP,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF6_E2M3:
        INT_EXPANSION_FACTOR: tl.constexpr = IF6_E2M3_INT_EXPANSION_FACTOR

        _, x_fp_fp16 = convert_to_e2m3x2_and_dequantized_fp16(
            x_block_scaled,
            MAJOR_COMPUTE_CAPABILITY,
        )

        _, x_int_fp16 = convert_to_int6_and_quantized_fp16(
            x_block_scaled,
            IF6_E2M3_INT_EXPANSION_FACTOR_RCP,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF6_E3M2:
        INT_EXPANSION_FACTOR: tl.constexpr = IF6_E3M2_INT_EXPANSION_FACTOR

        _, x_fp_fp16 = convert_to_e3m2x2_and_dequantized_fp16(
            x_block_scaled,
            MAJOR_COMPUTE_CAPABILITY,
        )

        _, x_int_fp16 = convert_to_int6_and_quantized_fp16(
            x_block_scaled,
            IF6_E3M2_INT_EXPANSION_FACTOR_RCP,
        )

    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF4:
        x_fp_dequantized = tl.div_rn(
            (
                x_fp_fp16.to(x_amax.dtype)
                * convert_e4m3_to_high_precision(
                    x_scales,
                    x_amax.dtype,
                    MAJOR_COMPUTE_CAPABILITY,
                ).expand_dims(2)
                * x_amax
            ),
            MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR * SR_SCALE,
        )

        x_int_dequantized = tl.div_rn(
            (
                (
                    x_int_fp16.to(x_amax.dtype)
                    * convert_e4m3_to_high_precision(
                        x_scales,
                        x_amax.dtype,
                        MAJOR_COMPUTE_CAPABILITY,
                    ).expand_dims(2)
                    * x_amax
                )
                * INT_EXPANSION_FACTOR
            ),
            MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR * SR_SCALE,
        )
    else:
        x_fp_dequantized = (
            x_fp_fp16.to(x_amax.dtype)
            * convert_e4m3_to_high_precision(
                x_scales,
                x_amax.dtype,
                MAJOR_COMPUTE_CAPABILITY,
            ).expand_dims(2)
            * x_amax
        ) / (MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR * SR_SCALE)

        x_int_dequantized = (
            (
                x_int_fp16.to(x_amax.dtype)
                * convert_e4m3_to_high_precision(
                    x_scales,
                    x_amax.dtype,
                    MAJOR_COMPUTE_CAPABILITY,
                ).expand_dims(2)
                * x_amax
            )
            * INT_EXPANSION_FACTOR
        ) / (MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR * SR_SCALE)

    int_indicator = compute_error_and_select_kernel(
        x_block.reshape(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
        ),
        x_int_dequantized,
        x_fp_dequantized,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SCALE_RULE,
        SCALE_GROUP_SIZE,
        BLOCK_SCALE_2D,
    )

    x_dequantized = tl.where(
        int_indicator.expand_dims(2),
        x_int_dequantized,
        x_fp_dequantized,
    )

    return x_dequantized.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)


@triton.jit
def pseudo_nvfp4_fouroversix_block_scaled_quantization_kernel(
    x_block,
    x_amax_ptr,
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    SCALE_GROUP_SIZE: tl.constexpr = 16
    SR_SCALE: tl.constexpr = (
        UNBIASED_SR_ADJUSTMENT_FACTOR
        if ROUND_STYLE == ROUND_STYLE_STOCHASTIC_UNBIASED
        else 1
    )

    x_block_scaled_6, x_scales_6, x_amax = compute_scale_factors_kernel(
        x_block,
        x_amax_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SCALE_2D,
        E2M1_MAX_VALUE,
        E4M3_MAX_FOUROVERSIX,
        ROUND_STYLE,
        SCALE_TYPE_NV_IF,
        SCALE_GROUP_SIZE,
        None,
        MAJOR_COMPUTE_CAPABILITY,
    )

    _, x_6_fp16 = convert_to_e2m1x2_and_quantized_fp16(
        x_block_scaled_6,
        rbits_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        ROUND_STYLE,
        SCALE_GROUP_SIZE,
        MAJOR_COMPUTE_CAPABILITY,
    )

    x_6_dequantized = tl.div_rn(
        x_6_fp16.to(x_amax.dtype)
        * convert_e4m3_to_high_precision(
            x_scales_6,
            x_amax.dtype,
            MAJOR_COMPUTE_CAPABILITY,
        ).expand_dims(2)
        * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX * SR_SCALE,
    )

    x_block_scaled_4, x_scales_4, _ = compute_scale_factors_kernel(
        x_block,
        x_amax_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SCALE_2D,
        E2M1_MAX_VALUE,
        E4M3_MAX_FOUROVERSIX,
        ROUND_STYLE,
        SCALE_TYPE_NV_IF,
        SCALE_GROUP_SIZE,
        1.5,
        MAJOR_COMPUTE_CAPABILITY,
    )

    _, x_4_fp16 = convert_to_e2m1x2_and_quantized_fp16(
        x_block_scaled_4,
        rbits_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        ROUND_STYLE,
        SCALE_GROUP_SIZE,
        MAJOR_COMPUTE_CAPABILITY,
    )

    x_4_dequantized = tl.div_rn(
        x_4_fp16.to(x_amax.dtype)
        * convert_e4m3_to_high_precision(
            x_scales_4,
            x_amax.dtype,
            MAJOR_COMPUTE_CAPABILITY,
        ).expand_dims(2)
        * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX * SR_SCALE,
    )

    four_indicator = compute_error_and_select_kernel(
        x_block.reshape(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
        ),
        x_4_dequantized,
        x_6_dequantized,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SCALE_RULE,
        SCALE_GROUP_SIZE,
        BLOCK_SCALE_2D,
    )

    x_dequantized = tl.where(
        four_indicator.expand_dims(2),
        x_4_dequantized,
        x_6_dequantized,
    )

    return x_dequantized.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)


@triton.jit
def pseudo_quantization_kernel(
    x_desc,
    x_amax_ptr,
    x_out_desc,
    rbits_ptr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_N: tl.constexpr,
    MAX_QUANTIZED_VALUE: tl.constexpr,
    MAX_SCALE_FACTOR: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_block_offset = pid_m * BLOCK_SIZE_M
    n_block_offset = pid_n * BLOCK_SIZE_N

    NUM_TILES_M: tl.constexpr = BLOCK_SIZE_M // TILE_SIZE_M
    NUM_TILES_N: tl.constexpr = BLOCK_SIZE_N // TILE_SIZE_N

    for tile_m in range(NUM_TILES_M):
        for tile_n in range(NUM_TILES_N):
            if not TRANSPOSE:
                x_block = x_desc.load(
                    [
                        m_block_offset + tile_m * TILE_SIZE_M,
                        n_block_offset + tile_n * TILE_SIZE_N,
                    ],
                )
            else:
                x_block = x_desc.load(
                    [
                        n_block_offset + tile_n * TILE_SIZE_N,
                        m_block_offset + tile_m * TILE_SIZE_M,
                    ],
                ).T

            x_block = x_block.to(tl.float32)

            if SCALE_RULE == SCALE_RULE_STATIC_6 or SCALE_RULE == SCALE_RULE_STATIC_4:
                x_dequantized = pseudo_generic_block_scaled_quantization_kernel(
                    x_block,
                    x_amax_ptr,
                    rbits_ptr,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    MAX_QUANTIZED_VALUE,
                    MAX_SCALE_FACTOR,
                    QUANTIZED_VALUE_TYPE,
                    ROUND_STYLE,
                    SCALE_TYPE,
                    SCALE_GROUP_SIZE,
                    BLOCK_SCALE_2D,
                    MAJOR_COMPUTE_CAPABILITY,
                )
            elif SCALE_TYPE == SCALE_TYPE_NV_IF:
                x_dequantized = pseudo_intfloat_block_scaled_quantization_kernel(
                    x_block,
                    x_amax_ptr,
                    rbits_ptr,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    MAX_QUANTIZED_VALUE,
                    MAX_SCALE_FACTOR,
                    QUANTIZED_VALUE_TYPE,
                    ROUND_STYLE,
                    SCALE_GROUP_SIZE,
                    SCALE_RULE,
                    BLOCK_SCALE_2D,
                    MAJOR_COMPUTE_CAPABILITY,
                )
            else:
                x_dequantized = (
                    pseudo_nvfp4_fouroversix_block_scaled_quantization_kernel(
                        x_block,
                        x_amax_ptr,
                        rbits_ptr,
                        TILE_SIZE_M,
                        TILE_SIZE_N,
                        ROUND_STYLE,
                        SCALE_RULE,
                        BLOCK_SCALE_2D,
                        MAJOR_COMPUTE_CAPABILITY,
                    )
                )

            x_out_desc.store(
                [
                    pid_m * BLOCK_SIZE_M + tile_m * TILE_SIZE_M,
                    pid_n * BLOCK_SIZE_N + tile_n * TILE_SIZE_N,
                ],
                x_dequantized.to(tl.bfloat16),
            )
