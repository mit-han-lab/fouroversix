from __future__ import annotations

import triton
import triton.language as tl

from .constants import (
    E2M1_MAX_VALUE,
    E4M3_MAX_FOUROVERSIX,
    E4M3_MAX_VALUE,
    QUANTIZED_VALUE_TYPE_FP4,
    QUANTIZED_VALUE_TYPE_FP6_E2M3,
    QUANTIZED_VALUE_TYPE_FP6_E3M2,
    QUANTIZED_VALUE_TYPE_INT4,
    ROUND_STYLE_STOCHASTIC,
    SCALE_MEGABLOCK_SIZE,
    SCALE_RULE_ABS_MAX,
    SCALE_RULE_MAE,
    SCALE_RULE_MSE,
    SCALE_RULE_STATIC_4,
    SCALE_RULE_STATIC_6,
    SCALE_TYPE_MX,
    SCALE_TYPE_NV,
    SCALE_TYPE_NV_IF,
)
from .fp4 import convert_to_e2m1x2, convert_to_e2m1x2_and_quantized_fp16
from .fp6 import convert_to_e2m3x2, convert_to_e3m2x2
from .fp8 import convert_e4m3_to_high_precision, convert_to_e4m3_with_rtn
from .int4 import convert_to_int4


@triton.jit  # noqa: RET503
def prepare_inputs_for_block_scaling(
    x_block,
    x_amax_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    MAX_QUANTIZED_VALUE: tl.constexpr,
    MAX_SCALE_FACTOR: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SCALE_EXPANSION_FACTOR: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor, tl.tensor]:
    x_amax = tl.load(x_amax_ptr)

    if SCALE_TYPE == SCALE_TYPE_MX:
        x_scale_blocks = x_block.reshape(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
        )
        x_scales_hp = tl.max(x_scale_blocks.abs(), axis=-1) / MAX_QUANTIZED_VALUE
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
        x_scales_hp = (x_scales.cast(tl.uint32) << 23).cast(x_block.dtype, bitcast=True)

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

        x_block_scaled = x_scale_blocks / x_scales.to(x_scale_blocks.dtype).expand_dims(
            2,
        )

        return x_block_scaled, x_scales, x_amax

    if SCALE_TYPE == SCALE_TYPE_NV or SCALE_TYPE == SCALE_TYPE_NV_IF:
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
        else:
            encode_scale = tl.div_rn(MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR, x_amax)
            x_scales_hp = (
                tl.div_rn(tl.max(x_scale_blocks.abs(), axis=-1), MAX_QUANTIZED_VALUE)
                * encode_scale
            )

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
                    SCALE_GROUP_SIZE,
                    BLOCK_SIZE_M // SCALE_GROUP_SIZE,
                    BLOCK_SIZE_N // SCALE_GROUP_SIZE,
                )
                .permute(1, 0, 2)
                .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // SCALE_GROUP_SIZE)
            )

        if SCALE_EXPANSION_FACTOR is not None:
            x_scales_hp = x_scales_hp * SCALE_EXPANSION_FACTOR

        x_scales = convert_to_e4m3_with_rtn(x_scales_hp, MAJOR_COMPUTE_CAPABILITY)

        decode_scale = tl.div_rn(
            1,
            tl.div_rn(MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR, x_amax),
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
def block_scaled_quantization_kernel(
    x_block,
    x_amax_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MAX_QUANTIZED_VALUE: tl.constexpr,
    MAX_SCALE_FACTOR: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    RBITS: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    x_block_scaled, x_scales, _ = prepare_inputs_for_block_scaling(
        x_block,
        x_amax_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SCALE_2D,
        MAX_QUANTIZED_VALUE,
        MAX_SCALE_FACTOR,
        SCALE_TYPE,
        SCALE_GROUP_SIZE,
        None,
        MAJOR_COMPUTE_CAPABILITY,
    )

    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP4:
        x_block_scaled_b1, x_block_scaled_b2 = x_block_scaled.reshape(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // 2,
            2,
        ).split()

        x_e2m1 = convert_to_e2m1x2(
            x_block_scaled_b1,
            x_block_scaled_b2,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            ROUND_STYLE,
            RBITS,
            MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP6_E2M3:
        x_e2m1 = convert_to_e2m3x2(
            x_block_scaled.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N),
            MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP6_E3M2:
        x_e2m1 = convert_to_e3m2x2(
            x_block_scaled.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N),
            MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_INT4:
        x_e2m1 = convert_to_int4(
            x_block_scaled.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N),
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            ROUND_STYLE,
        )

    return x_e2m1, x_scales


@triton.jit
def nvfp4_fouroversix_quantization_kernel(
    x_block,
    x_amax_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    RBITS: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    x_scale_blocks = x_block.reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // SCALE_GROUP_SIZE,
        SCALE_GROUP_SIZE,
    )

    x_block_scaled_6, x_scales_6, x_amax = prepare_inputs_for_block_scaling(
        x_scale_blocks,
        x_amax_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SCALE_2D,
        E2M1_MAX_VALUE,
        E4M3_MAX_FOUROVERSIX,
        SCALE_TYPE,
        SCALE_GROUP_SIZE,
        None,
        MAJOR_COMPUTE_CAPABILITY,
    )

    x_block_scaled_6_b1, x_block_scaled_6_b2 = x_block_scaled_6.reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // 2,
        2,
    ).split()

    x_e2m1_6, x_fp16_6 = convert_to_e2m1x2_and_quantized_fp16(
        x_block_scaled_6_b1,
        x_block_scaled_6_b2,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        ROUND_STYLE,
        SCALE_GROUP_SIZE,
        RBITS,
        MAJOR_COMPUTE_CAPABILITY,
    )

    x_dequantized_6 = tl.div_rn(
        x_fp16_6.to(x_amax.dtype)
        * convert_e4m3_to_high_precision(
            x_scales_6,
            x_amax.dtype,
            MAJOR_COMPUTE_CAPABILITY,
        ).expand_dims(2)
        * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX,
    )

    x_block_scaled_4, x_scales_4, _ = prepare_inputs_for_block_scaling(
        x_scale_blocks,
        x_amax_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SCALE_2D,
        E2M1_MAX_VALUE,
        E4M3_MAX_FOUROVERSIX,
        SCALE_TYPE,
        SCALE_GROUP_SIZE,
        1.5,
        MAJOR_COMPUTE_CAPABILITY,
    )

    x_block_scaled_4_b1, x_block_scaled_4_b2 = x_block_scaled_4.reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // 2,
        2,
    ).split()

    x_e2m1_4, x_fp16_4 = convert_to_e2m1x2_and_quantized_fp16(
        x_block_scaled_4_b1,
        x_block_scaled_4_b2,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        ROUND_STYLE,
        SCALE_GROUP_SIZE,
        RBITS,
        MAJOR_COMPUTE_CAPABILITY,
    )

    x_dequantized_4 = tl.div_rn(
        x_fp16_4.to(x_amax.dtype)
        * convert_e4m3_to_high_precision(
            x_scales_4,
            x_amax.dtype,
            MAJOR_COMPUTE_CAPABILITY,
        ).expand_dims(2)
        * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX,
    )

    diff_6 = x_dequantized_6 - x_scale_blocks
    diff_4 = x_dequantized_4 - x_scale_blocks

    if SCALE_RULE == SCALE_RULE_ABS_MAX:
        six_error = tl.max(tl.abs(diff_6), axis=-1)
        four_error = tl.max(tl.abs(diff_4), axis=-1)
    elif SCALE_RULE == SCALE_RULE_MAE:
        six_error = tl.sum(tl.abs(diff_6), axis=-1)
        four_error = tl.sum(tl.abs(diff_4), axis=-1)
    elif SCALE_RULE == SCALE_RULE_MSE:
        six_error = tl.sum(diff_6 * diff_6, axis=-1)
        four_error = tl.sum(diff_4 * diff_4, axis=-1)

    if BLOCK_SCALE_2D:
        six_error = six_error.reshape(
            BLOCK_SIZE_M // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
        ).permute(0, 2, 1)

        four_error = four_error.reshape(
            BLOCK_SIZE_M // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
        ).permute(0, 2, 1)

        if SCALE_RULE == SCALE_RULE_ABS_MAX:
            six_error = tl.max(six_error, axis=-1)
            four_error = tl.max(four_error, axis=-1)
        elif SCALE_RULE == SCALE_RULE_MAE or SCALE_RULE == SCALE_RULE_MSE:
            six_error = tl.sum(six_error, axis=-1)
            four_error = tl.sum(four_error, axis=-1)

        six_error = (
            six_error.expand_dims(0)
            .broadcast_to(
                SCALE_GROUP_SIZE,
                BLOCK_SIZE_M // SCALE_GROUP_SIZE,
                BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            )
            .permute(1, 0, 2)
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // SCALE_GROUP_SIZE)
        )
        four_error = (
            four_error.expand_dims(0)
            .broadcast_to(
                SCALE_GROUP_SIZE,
                BLOCK_SIZE_M // SCALE_GROUP_SIZE,
                BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            )
            .permute(1, 0, 2)
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // SCALE_GROUP_SIZE)
        )

    x_e2m1 = tl.where(
        (four_error < six_error).expand_dims(2),
        x_e2m1_4.reshape(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE // 2,
        ),
        x_e2m1_6.reshape(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE // 2,
        ),
    ).reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    x_scales = tl.where(four_error < six_error, x_scales_4, x_scales_6)
    return x_e2m1, x_scales


@triton.jit
def if4_quantization_kernel(
    x_block,
    x_amax_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    RBITS: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
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
    else:
        encode_scale = tl.div_rn(E2M1_MAX_VALUE * E4M3_MAX_VALUE, x_amax)
        x_scales_hp = (
            tl.div_rn(tl.max(x_scale_blocks.abs(), axis=-1), E2M1_MAX_VALUE)
            * encode_scale
        )

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
                SCALE_GROUP_SIZE,
                BLOCK_SIZE_M // SCALE_GROUP_SIZE,
                BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            )
            .permute(1, 0, 2)
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // SCALE_GROUP_SIZE)
        )

    x_scales = convert_to_e4m3_with_rtn(x_scales_hp, MAJOR_COMPUTE_CAPABILITY)

    decode_scale = tl.div_rn(
        1,
        tl.div_rn(E2M1_MAX_VALUE * E4M3_MAX_VALUE, x_amax),
    )

    x_block_scaled = tl.where(
        x_scales.expand_dims(2) != 0,
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

    (x_block_scaled_b1, x_block_scaled_b2) = x_block_scaled.reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // 2,
        2,
    ).split()

    x_fp, x_fp_fp16 = convert_to_e2m1x2_and_quantized_fp16(
        x_block_scaled_b1,
        x_block_scaled_b2,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        ROUND_STYLE,
        SCALE_GROUP_SIZE,
        RBITS,
        MAJOR_COMPUTE_CAPABILITY,
    )

    # Int and fp need different amounts of randomness because int gets scaled by 7/6
    if ROUND_STYLE == ROUND_STYLE_STOCHASTIC:
        rbits = (
            tl.rand(
                2,
                tl.arange(0, BLOCK_SIZE_M)[:, None, None] * BLOCK_SIZE_N
                + tl.arange(0, BLOCK_SIZE_N // SCALE_GROUP_SIZE)[None, :, None]
                * SCALE_GROUP_SIZE
                + tl.arange(0, SCALE_GROUP_SIZE)[None, None, :],
            )
            - 0.5
        ) * (6 / 7)

        x_block_scaled = tl.where(x_block_scaled < 0, -1, 1) * tl.abs(
            tl.abs(x_block_scaled) + rbits,
        )

    x_int_hp = tl.extra.cuda.libdevice.rint(tl.clamp(x_block_scaled * (7 / 6), -7, 7))
    (x_int_b1, x_int_b2) = (
        (x_int_hp.to(tl.int8).to(tl.uint8, bitcast=True) & 0xF)
        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
        .split()
    )

    x_int = (x_int_b2 << 4) | x_int_b1
    x_int_dequantized = tl.div_rn(
        x_int_hp
        * (6 / 7)
        * convert_e4m3_to_high_precision(
            x_scales,
            x_amax.dtype,
            MAJOR_COMPUTE_CAPABILITY,
        ).expand_dims(2)
        * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_VALUE,
    )

    x_fp_dequantized = tl.div_rn(
        x_fp_fp16.to(x_amax.dtype)
        * convert_e4m3_to_high_precision(
            x_scales,
            x_amax.dtype,
            MAJOR_COMPUTE_CAPABILITY,
        ).expand_dims(2)
        * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_VALUE,
    )

    diff_fp = x_fp_dequantized - x_scale_blocks
    diff_int = x_int_dequantized - x_scale_blocks

    if SCALE_RULE == SCALE_RULE_ABS_MAX:
        fp_error = tl.max(tl.abs(diff_fp), axis=-1)
        int_error = tl.max(tl.abs(diff_int), axis=-1)
    elif SCALE_RULE == SCALE_RULE_MAE:
        fp_error = tl.sum(tl.abs(diff_fp), axis=-1)
        int_error = tl.sum(tl.abs(diff_int), axis=-1)
    elif SCALE_RULE == SCALE_RULE_MSE:
        fp_error = tl.sum(diff_fp * diff_fp, axis=-1)
        int_error = tl.sum(diff_int * diff_int, axis=-1)

    if BLOCK_SCALE_2D:
        fp_error = fp_error.reshape(
            BLOCK_SIZE_M // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
        ).permute(0, 2, 1)

        int_error = int_error.reshape(
            BLOCK_SIZE_M // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
        ).permute(0, 2, 1)

        if SCALE_RULE == SCALE_RULE_ABS_MAX:
            fp_error = tl.max(fp_error, axis=-1)
            int_error = tl.max(int_error, axis=-1)
        elif SCALE_RULE == SCALE_RULE_MAE or SCALE_RULE == SCALE_RULE_MSE:
            fp_error = tl.sum(fp_error, axis=-1)
            int_error = tl.sum(int_error, axis=-1)

        fp_error = (
            fp_error.expand_dims(0)
            .broadcast_to(
                SCALE_GROUP_SIZE,
                BLOCK_SIZE_M // SCALE_GROUP_SIZE,
                BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            )
            .permute(1, 0, 2)
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // SCALE_GROUP_SIZE)
        )
        int_error = (
            int_error.expand_dims(0)
            .broadcast_to(
                SCALE_GROUP_SIZE,
                BLOCK_SIZE_M // SCALE_GROUP_SIZE,
                BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            )
            .permute(1, 0, 2)
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // SCALE_GROUP_SIZE)
        )

    x_values = tl.where(
        (int_error < fp_error).expand_dims(2),
        x_int.reshape(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE // 2,
        ),
        x_fp.reshape(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE // 2,
        ),
    ).reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    x_scales = tl.where(int_error < fp_error, x_scales + 128, x_scales)

    return x_values, x_scales


@triton.jit
def quantization_kernel(
    x_desc,
    x_amax_ptr,
    x_e2m1_desc,
    x_sf_desc,
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
    RBITS: tl.constexpr,
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
                x_e2m1, x_scales = block_scaled_quantization_kernel(
                    x_block,
                    x_amax_ptr,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    MAX_QUANTIZED_VALUE,
                    MAX_SCALE_FACTOR,
                    QUANTIZED_VALUE_TYPE,
                    ROUND_STYLE,
                    SCALE_TYPE,
                    SCALE_GROUP_SIZE,
                    BLOCK_SCALE_2D,
                    RBITS,
                    MAJOR_COMPUTE_CAPABILITY,
                )
            elif SCALE_TYPE == SCALE_TYPE_NV_IF:
                x_e2m1, x_scales = if4_quantization_kernel(
                    x_block,
                    x_amax_ptr,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    ROUND_STYLE,
                    SCALE_GROUP_SIZE,
                    SCALE_RULE,
                    BLOCK_SCALE_2D,
                    RBITS,
                    MAJOR_COMPUTE_CAPABILITY,
                )
            else:
                x_e2m1, x_scales = nvfp4_fouroversix_quantization_kernel(
                    x_block,
                    x_amax_ptr,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    ROUND_STYLE,
                    SCALE_TYPE,
                    SCALE_GROUP_SIZE,
                    SCALE_RULE,
                    BLOCK_SCALE_2D,
                    RBITS,
                    MAJOR_COMPUTE_CAPABILITY,
                )

            x_e2m1_desc.store(
                [
                    pid_m * BLOCK_SIZE_M + tile_m * TILE_SIZE_M,
                    (pid_n * BLOCK_SIZE_N + tile_n * TILE_SIZE_N)
                    // QUANTIZED_VALUE_PACKING_FACTOR,
                ],
                x_e2m1,
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
