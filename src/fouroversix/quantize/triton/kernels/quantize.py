from __future__ import annotations

import torch
import triton
import triton.language as tl
from fouroversix.quantize.quantized_tensor import from_blocked
from fouroversix.utils import (
    DataType,
    QuantizedValueType,
    RoundStyle,
    ScaleRule,
    ScaleType,
    device_supports_cvt_rn_e2m1x2,
    device_supports_cvt_rs_e2m1x4,
)
from triton.tools.tensor_descriptor import TensorDescriptor

from .fp4 import convert_to_e2m1x2, convert_to_e2m1x2_and_quantized_fp16
from .rht import rht_kernel

E2M1_MAX_VALUE = tl.constexpr(6)
E2M1_MAX_FOUR = tl.constexpr(4)
E4M3_MAX_VALUE = tl.constexpr(448)
E4M3_MAX_FOUROVERSIX = tl.constexpr(256)
SCALE_MEGABLOCK_SIZE = tl.constexpr(512)

DATA_TYPE_IF4 = tl.constexpr(DataType.if4.value)
DATA_TYPE_MXFP4 = tl.constexpr(DataType.mxfp4.value)
DATA_TYPE_NVFP4 = tl.constexpr(DataType.nvfp4.value)

ROUND_STYLE_NEAREST = tl.constexpr(RoundStyle.nearest.value)
ROUND_STYLE_STOCHASTIC = tl.constexpr(RoundStyle.stochastic.value)

SCALE_RULE_ABS_MAX = tl.constexpr(ScaleRule.abs_max.value)
SCALE_RULE_MAE = tl.constexpr(ScaleRule.mae.value)
SCALE_RULE_MSE = tl.constexpr(ScaleRule.mse.value)
SCALE_RULE_STATIC_4 = tl.constexpr(ScaleRule.static_4.value)
SCALE_RULE_STATIC_6 = tl.constexpr(ScaleRule.static_6.value)

SCALE_TYPE_MX = tl.constexpr(ScaleType.mx.value)
SCALE_TYPE_NV = tl.constexpr(ScaleType.nv.value)
SCALE_TYPE_NV_IF = tl.constexpr(ScaleType.nv_if.value)

QUANTIZED_VALUE_TYPE_FP4 = tl.constexpr(QuantizedValueType.fp4.value)
QUANTIZED_VALUE_TYPE_IF4 = tl.constexpr(QuantizedValueType.if4.value)
QUANTIZED_VALUE_TYPE_INT4 = tl.constexpr(QuantizedValueType.int4.value)


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

        x_scales = x_scales_hp.to(tl.float8e4nv)

        decode_scale = tl.div_rn(
            1,
            tl.div_rn(MAX_QUANTIZED_VALUE * MAX_SCALE_FACTOR, x_amax),
        )

        x_block_scaled = tl.where(
            x_scales.expand_dims(2).cast(tl.uint8, bitcast=True) != 0,
            x_scale_blocks
            * tl.div_rn(1, decode_scale * x_scales.to(x_amax.dtype).expand_dims(2)),
            0,
        )

        return x_block_scaled, x_scales, x_amax


@triton.jit
def block_scaled_quantization_kernel(
    x_block,
    x_amax_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    RBITS: tl.constexpr,
    USE_BLACKWELL_CVT_RN_INSTRUCTIONS: tl.constexpr,
    USE_BLACKWELL_CVT_RS_INSTRUCTIONS: tl.constexpr,
) -> None:
    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP4:
        x_block_scaled, x_scales, _ = prepare_inputs_for_block_scaling(
            x_block,
            x_amax_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SCALE_2D,
            E2M1_MAX_VALUE if SCALE_RULE == SCALE_RULE_STATIC_6 else E2M1_MAX_FOUR,
            E4M3_MAX_VALUE if SCALE_TYPE == SCALE_TYPE_NV else None,
            SCALE_TYPE,
            SCALE_GROUP_SIZE,
            None,
        )

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
        USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
        USE_BLACKWELL_CVT_RS_INSTRUCTIONS,
    )

    return x_e2m1, x_scales.reshape(4, 32, 4).permute(1, 0, 2).ravel()


@triton.jit
def nvfp4_fouroversix_quantization_kernel(
    x_block,
    x_amax_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    RBITS: tl.constexpr,
    USE_BLACKWELL_CVT_RN_INSTRUCTIONS: tl.constexpr,
    USE_BLACKWELL_CVT_RS_INSTRUCTIONS: tl.constexpr,
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
        USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
        USE_BLACKWELL_CVT_RS_INSTRUCTIONS,
    )

    x_dequantized_6 = tl.div_rn(
        x_fp16_6.to(x_amax.dtype) * x_scales_6.to(x_amax.dtype).expand_dims(2) * x_amax,
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
        USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
        USE_BLACKWELL_CVT_RS_INSTRUCTIONS,
    )

    x_dequantized_4 = tl.div_rn(
        x_fp16_4.to(x_amax.dtype) * x_scales_4.to(x_amax.dtype).expand_dims(2) * x_amax,
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
        six_error = six_error.reshape(8, 16, 4).permute(0, 2, 1)
        four_error = four_error.reshape(8, 16, 4).permute(0, 2, 1)

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

    x_scales = (
        tl.where(four_error < six_error, x_scales_4, x_scales_6)
        .reshape(4, 32, 4)
        .permute(1, 0, 2)
        .ravel()
    )

    return x_e2m1, x_scales


@triton.jit
def if4_quantization_kernel(
    x_block,
    x_amax_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    RBITS: tl.constexpr,
    USE_BLACKWELL_CVT_RN_INSTRUCTIONS: tl.constexpr,
    USE_BLACKWELL_CVT_RS_INSTRUCTIONS: tl.constexpr,
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

    x_scales = x_scales_hp.to(tl.float8e4nv)

    decode_scale = tl.div_rn(
        1,
        tl.div_rn(E2M1_MAX_VALUE * E4M3_MAX_VALUE, x_amax),
    )

    x_block_scaled = tl.where(
        x_scales.expand_dims(2).to(x_amax.dtype) != 0,
        x_scale_blocks
        * tl.div_rn(1, decode_scale * x_scales.to(x_amax.dtype).expand_dims(2)),
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
        USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
        USE_BLACKWELL_CVT_RS_INSTRUCTIONS,
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
        x_int_hp * (6 / 7) * x_scales.to(x_amax.dtype).expand_dims(2) * x_amax,
        E2M1_MAX_VALUE * E4M3_MAX_VALUE,
    )

    x_fp_dequantized = tl.div_rn(
        x_fp_fp16.to(x_amax.dtype) * x_scales.to(x_amax.dtype).expand_dims(2) * x_amax,
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
        fp_error = fp_error.reshape(8, 16, 4).permute(0, 2, 1)
        int_error = int_error.reshape(8, 16, 4).permute(0, 2, 1)

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

    x_scales = (
        tl.where(
            int_error < fp_error,
            x_scales.to(tl.uint8, bitcast=True) + 128,
            x_scales.to(tl.uint8, bitcast=True),
        )
        .reshape(4, 32, 4)
        .permute(1, 0, 2)
        .ravel()
    )

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
    TRANSPOSE: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_TYPE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    RBITS: tl.constexpr,
    USE_BLACKWELL_CVT_RN_INSTRUCTIONS: tl.constexpr,
    USE_BLACKWELL_CVT_RS_INSTRUCTIONS: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_block_offset = pid_m * BLOCK_SIZE_M
    n_block_offset = pid_n * BLOCK_SIZE_N

    # Load [B, B] block from A or A^T
    if not TRANSPOSE:
        x_block = x_desc.load([m_block_offset, n_block_offset])
    else:
        x_block = x_desc.load([n_block_offset, m_block_offset]).T

    x_block = x_block.to(tl.float32)

    if SCALE_RULE == SCALE_RULE_STATIC_6 or SCALE_RULE == SCALE_RULE_STATIC_4:
        x_e2m1, x_scales = block_scaled_quantization_kernel(
            x_block,
            x_amax_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            QUANTIZED_VALUE_TYPE,
            ROUND_STYLE,
            SCALE_TYPE,
            SCALE_GROUP_SIZE,
            SCALE_RULE,
            BLOCK_SCALE_2D,
            RBITS,
            USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
            USE_BLACKWELL_CVT_RS_INSTRUCTIONS,
        )
    elif SCALE_TYPE == SCALE_TYPE_NV_IF:
        x_e2m1, x_scales = if4_quantization_kernel(
            x_block,
            x_amax_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            QUANTIZED_VALUE_TYPE,
            ROUND_STYLE,
            SCALE_TYPE,
            SCALE_GROUP_SIZE,
            SCALE_RULE,
            BLOCK_SCALE_2D,
            RBITS,
            USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
            USE_BLACKWELL_CVT_RS_INSTRUCTIONS,
        )
    else:
        x_e2m1, x_scales = nvfp4_fouroversix_quantization_kernel(
            x_block,
            x_amax_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            QUANTIZED_VALUE_TYPE,
            ROUND_STYLE,
            SCALE_TYPE,
            SCALE_GROUP_SIZE,
            SCALE_RULE,
            BLOCK_SCALE_2D,
            RBITS,
            USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
            USE_BLACKWELL_CVT_RS_INSTRUCTIONS,
        )

    e2m1_n_block_offset = pid_n * BLOCK_SIZE_N // 2
    x_e2m1_desc.store([m_block_offset, e2m1_n_block_offset], x_e2m1)

    scale_block_offset = (pid_m * tl.num_programs(1) + pid_n) * SCALE_MEGABLOCK_SIZE
    x_sf_desc.store([scale_block_offset], x_scales)


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

    if x_amax is None:
        x_amax = (
            x.abs().max().float()
            if dtype.scale_type != ScaleType.mx
            else torch.ones(1, device=x.device, dtype=torch.float32)
        )

    padded_m = M + (block_size_m - M % block_size_m) % block_size_m
    padded_n = N + (block_size_n - N % block_size_n) % block_size_n

    x_e2m1 = torch.empty((padded_m, padded_n // 2), device=x.device, dtype=torch.uint8)
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
            block_size_m if not transpose else block_size_n,
            block_size_n if not transpose else block_size_m,
        ],
    )
    x_e2m1_desc = TensorDescriptor.from_tensor(
        x_e2m1,
        block_shape=[block_size_m, block_size_n // 2],
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
            block_shape=[block_size_m, block_size_n],
        )

        rht_kernel[grid](
            x_desc,
            h_desc,
            x_rht_desc,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
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
        TRANSPOSE=transpose,
        QUANTIZED_VALUE_TYPE=dtype.quantized_value_type.value,
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

    if dtype.scale_type == ScaleType.mx:
        x_sf = x_sf.view(torch.float8_e8m0fnu)
    elif dtype == DataType.if4:
        x_sf = from_blocked(x_sf, (padded_m, padded_n // dtype.block_size)).view(
            torch.float8_e4m3fn,
        )

    return x_e2m1, x_sf, x_amax
