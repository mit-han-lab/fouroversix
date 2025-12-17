from __future__ import annotations

import itertools
import os
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl
from fouroversix.quantize.reference import get_nvfp4_tensor_scale
from fouroversix.utils import AdaptiveBlockScalingRule
from triton.tools.tensor_descriptor import TensorDescriptor

if TYPE_CHECKING:
    from fouroversix.utils import FP4Format, RoundStyle

E2M1_MAX_VALUE = tl.constexpr(6)
E4M3_MIN_POSITIVE_VALUE = tl.constexpr(0.009765625)
E4M3_MAX_VALUE = tl.constexpr(448)
FOUROVERSIX_AUTOTUNE = os.getenv("FOUROVERSIX_AUTOTUNE", "0") == "1"
SCALE_MEGABLOCK_SIZE = tl.constexpr(512)

SCALE_RULE_ABS_MAX = tl.constexpr(AdaptiveBlockScalingRule.abs_max.value)
SCALE_RULE_ALWAYS_4 = tl.constexpr(AdaptiveBlockScalingRule.always_4.value)
SCALE_RULE_ALWAYS_6 = tl.constexpr(AdaptiveBlockScalingRule.always_6.value)
SCALE_RULE_L1_NORM = tl.constexpr(AdaptiveBlockScalingRule.l1_norm.value)
SCALE_RULE_MSE = tl.constexpr(AdaptiveBlockScalingRule.mse.value)


@triton.jit
def fp32_to_scaled_fp4_kernel_four_over_six(  # noqa: C901, PLR0912, PLR0915
    x_block,
    norm_constant_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    scale_rule: tl.constexpr,
) -> None:
    norm_constant = tl.load(norm_constant_ptr)

    if BLOCK_SCALE_2D:
        x_scale_blocks = (
            x_block.reshape(8, 16, 4, 16).permute(0, 2, 1, 3).reshape(8, 4, 256)
        )
    else:
        x_scale_blocks = x_block.reshape(128, 4, 16)

    # Calculate six blocks
    x_scales_hp_6 = tl.max(x_scale_blocks.abs(), axis=-1) / tl.maximum(
        E2M1_MAX_VALUE * norm_constant,
        1e-12,
    )
    x_scales_hp_6 = tl.clamp(x_scales_hp_6, E4M3_MIN_POSITIVE_VALUE, E4M3_MAX_VALUE)
    x_scales_6 = x_scales_hp_6.to(tl.float8e4nv)

    if BLOCK_SCALE_2D:
        x_scale_blocks = x_block.reshape(128, 4, 16)
        x_scales_6 = (
            x_scales_6[None, :, :]
            .broadcast_to(16, 8, 4)
            .permute(1, 0, 2)
            .reshape(128, 4)
        )

    x_scales_hp_6 = x_scales_6.to(norm_constant.dtype) * norm_constant
    (x_block_scaled_6_b1, x_block_scaled_6_b2) = (
        (x_scale_blocks / tl.maximum(x_scales_hp_6[:, :, None], 1e-12))
        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
        .split()
    )

    if BLOCK_SCALE_2D:
        x_scale_blocks = (
            x_block.reshape(8, 16, 4, 16).permute(0, 2, 1, 3).reshape(8, 4, 256)
        )

    # Calculate four blocks
    x_scales_hp_4 = tl.max(x_scale_blocks.abs(), axis=-1) / tl.maximum(
        (4 * norm_constant.to(tl.bfloat16)).to(tl.float32),
        1e-12,
    )
    x_scales_hp_4 = tl.clamp(x_scales_hp_4, E4M3_MIN_POSITIVE_VALUE, E4M3_MAX_VALUE)
    x_scales_4 = x_scales_hp_4.to(tl.float8e4nv)

    if BLOCK_SCALE_2D:
        x_scale_blocks = x_block.reshape(128, 4, 16)
        x_scales_4 = (
            x_scales_4[None, :, :]
            .broadcast_to(16, 8, 4)
            .permute(1, 0, 2)
            .reshape(128, 4)
        )

    x_scales_hp_4 = x_scales_4.to(norm_constant.dtype) * norm_constant
    (x_block_scaled_4_b1, x_block_scaled_4_b2) = (
        (x_scale_blocks / tl.maximum(x_scales_hp_4[:, :, None], 1e-12))
        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
        .split()
    )

    if ROUND_STYLE == "nearest":
        (x_e2m1_6, x_e2m1_4, x_fp16x2_6, x_fp16x2_4) = tl.inline_asm_elementwise(
            asm="""
                {
                .reg .b8 byte0, byte1, byte2, byte3;

                cvt.rn.satfinite.e2m1x2.f32 byte0, $28, $20;
                cvt.rn.f16x2.e2m1x2 $4, byte0;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $29, $21;
                cvt.rn.f16x2.e2m1x2 $5, byte1;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $30, $22;
                cvt.rn.f16x2.e2m1x2 $6, byte2;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $31, $23;
                cvt.rn.f16x2.e2m1x2 $7, byte3;
                mov.b32 $0, {byte0, byte1, byte2, byte3};

                cvt.rn.satfinite.e2m1x2.f32 byte0, $32, $24;
                cvt.rn.f16x2.e2m1x2 $8, byte0;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $33, $25;
                cvt.rn.f16x2.e2m1x2 $9, byte1;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $34, $26;
                cvt.rn.f16x2.e2m1x2 $10, byte2;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $35, $27;
                cvt.rn.f16x2.e2m1x2 $11, byte3;
                mov.b32 $1, {byte0, byte1, byte2, byte3};

                cvt.rn.satfinite.e2m1x2.f32 byte0, $44, $36;
                cvt.rn.f16x2.e2m1x2 $12, byte0;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $45, $37;
                cvt.rn.f16x2.e2m1x2 $13, byte1;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $46, $38;
                cvt.rn.f16x2.e2m1x2 $14, byte2;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $47, $39;
                cvt.rn.f16x2.e2m1x2 $15, byte3;
                mov.b32 $2, {byte0, byte1, byte2, byte3};

                cvt.rn.satfinite.e2m1x2.f32 byte0, $48, $40;
                cvt.rn.f16x2.e2m1x2 $16, byte0;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $49, $41;
                cvt.rn.f16x2.e2m1x2 $17, byte1;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $50, $42;
                cvt.rn.f16x2.e2m1x2 $18, byte2;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $51, $43;
                cvt.rn.f16x2.e2m1x2 $19, byte3;
                mov.b32 $3, {byte0, byte1, byte2, byte3};
                }
                """,
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[
                x_block_scaled_6_b1.to(tl.float32),
                x_block_scaled_6_b2.to(tl.float32),
                x_block_scaled_4_b1.to(tl.float32),
                x_block_scaled_4_b2.to(tl.float32),
            ],
            dtype=(tl.uint8, tl.uint8, tl.uint32, tl.uint32),
            is_pure=True,
            pack=8,
        )
    elif ROUND_STYLE == "stochastic":
        rbits = tl.rand(
            0,
            tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N // 2
            + tl.arange(0, BLOCK_SIZE_N // 2)[None, :],
        ).cast(tl.uint32, bitcast=True)
        (x_e2m1_6, x_e2m1_4, x_fp16x2_6, x_fp16x2_4) = tl.inline_asm_elementwise(
            asm="""
                {
                .reg .b16 tmp0, tmp1;
                .reg .b8 byte0, byte1;

                cvt.rs.satfinite.e2m1x4.f32 tmp0, {$29, $21, $28, $20}, $52;
                mov.b16 {byte1, byte0}, tmp0;
                cvt.rn.f16x2.e2m1x2 $4, byte0;
                cvt.rn.f16x2.e2m1x2 $5, byte1;
                cvt.rs.satfinite.e2m1x4.f32 tmp1, {$31, $23, $30, $22}, $53;
                mov.b16 {byte1, byte0}, tmp1;
                cvt.rn.f16x2.e2m1x2 $6, byte0;
                cvt.rn.f16x2.e2m1x2 $7, byte1;
                mov.b32 $0, {tmp0, tmp1};

                cvt.rs.satfinite.e2m1x4.f32 tmp0, {$33, $25, $32, $24}, $54;
                mov.b16 {byte1, byte0}, tmp0;
                cvt.rn.f16x2.e2m1x2 $8, byte0;
                cvt.rn.f16x2.e2m1x2 $9, byte1;
                cvt.rs.satfinite.e2m1x4.f32 tmp1, {$35, $27, $34, $26}, $55;
                mov.b16 {byte1, byte0}, tmp1;
                cvt.rn.f16x2.e2m1x2 $10, byte0;
                cvt.rn.f16x2.e2m1x2 $11, byte1;
                mov.b32 $1, {tmp0, tmp1};

                cvt.rs.satfinite.e2m1x4.f32 tmp0, {$45, $37, $44, $36}, $56;
                mov.b16 {byte1, byte0}, tmp0;
                cvt.rn.f16x2.e2m1x2 $12, byte0;
                cvt.rn.f16x2.e2m1x2 $13, byte1;
                cvt.rs.satfinite.e2m1x4.f32 tmp1, {$47, $39, $46, $38}, $57;
                mov.b16 {byte1, byte0}, tmp1;
                cvt.rn.f16x2.e2m1x2 $14, byte0;
                cvt.rn.f16x2.e2m1x2 $15, byte1;
                mov.b32 $2, {tmp0, tmp1};

                cvt.rs.satfinite.e2m1x4.f32 tmp0, {$49, $41, $48, $40}, $58;
                mov.b16 {byte1, byte0}, tmp0;
                cvt.rn.f16x2.e2m1x2 $16, byte0;
                cvt.rn.f16x2.e2m1x2 $17, byte1;
                cvt.rs.satfinite.e2m1x4.f32 tmp1, {$51, $43, $50, $42}, $59;
                mov.b16 {byte1, byte0}, tmp1;
                cvt.rn.f16x2.e2m1x2 $18, byte0;
                cvt.rn.f16x2.e2m1x2 $19, byte1;
                mov.b32 $3, {tmp0, tmp1};
                }
                """,
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[
                x_block_scaled_6_b1.to(tl.float32),
                x_block_scaled_6_b2.to(tl.float32),
                x_block_scaled_4_b1.to(tl.float32),
                x_block_scaled_4_b2.to(tl.float32),
                rbits,
            ],
            dtype=(tl.uint8, tl.uint8, tl.uint32, tl.uint32),
            is_pure=True,
            pack=8,
        )

    x_fp16_6_lo = (x_fp16x2_6 & 0xFFFF).cast(tl.uint16).cast(tl.float16, bitcast=True)
    x_fp16_6_hi = (x_fp16x2_6 >> 16).cast(tl.uint16).cast(tl.float16, bitcast=True)
    x_hp_6 = (
        tl.join(x_fp16_6_lo, x_fp16_6_hi).reshape(128, 4, 16).to(norm_constant.dtype)
    )

    if scale_rule == SCALE_RULE_ABS_MAX:
        six_error = tl.max(
            tl.abs(x_hp_6 * x_scales_hp_6[:, :, None] - x_scale_blocks),
            axis=-1,
        )
    elif scale_rule == SCALE_RULE_ALWAYS_4:
        six_error = tl.full(x_scales_hp_6.shape, 1, tl.float16)
    elif scale_rule == SCALE_RULE_L1_NORM:
        six_error = tl.sum(
            tl.abs(x_hp_6 * x_scales_hp_6[:, :, None] - x_scale_blocks),
            axis=-1,
        )
    elif scale_rule == SCALE_RULE_MSE:
        six_error = tl.sum(
            (x_hp_6 * x_scales_hp_6[:, :, None] - x_scale_blocks)
            * (x_hp_6 * x_scales_hp_6[:, :, None] - x_scale_blocks),
            axis=-1,
        )

    x_fp16_4_lo = (x_fp16x2_4 & 0xFFFF).cast(tl.uint16).cast(tl.float16, bitcast=True)
    x_fp16_4_hi = (x_fp16x2_4 >> 16).cast(tl.uint16).cast(tl.float16, bitcast=True)
    x_hp_4 = (
        tl.join(x_fp16_4_lo, x_fp16_4_hi).reshape(128, 4, 16).to(norm_constant.dtype)
    )

    if scale_rule == SCALE_RULE_ABS_MAX:
        four_error = tl.max(
            tl.abs(x_hp_4 * x_scales_hp_4[:, :, None] - x_scale_blocks),
            axis=-1,
        )
    elif scale_rule == SCALE_RULE_ALWAYS_4:
        four_error = tl.full(x_scales_hp_4.shape, 0, tl.float16)
    elif scale_rule == SCALE_RULE_L1_NORM:
        four_error = tl.sum(
            tl.abs(x_hp_4 * x_scales_hp_4[:, :, None] - x_scale_blocks),
            axis=-1,
        )
    elif scale_rule == SCALE_RULE_MSE:
        four_error = tl.sum(
            (x_hp_4 * x_scales_hp_4[:, :, None] - x_scale_blocks)
            * (x_hp_4 * x_scales_hp_4[:, :, None] - x_scale_blocks),
            axis=-1,
        )

    x_e2m1 = tl.where(
        (four_error < six_error)[:, :, None],
        x_e2m1_4.reshape(128, 4, 8),
        x_e2m1_6.reshape(128, 4, 8),
    ).reshape(128, 32)
    x_scales = (
        tl.where(
            four_error < six_error,
            x_scales_4,
            x_scales_6,
        )
        .reshape(4, 32, 4)
        .permute(1, 0, 2)
        .ravel()
    )

    return x_e2m1, x_scales


@triton.jit
def fp32_to_scaled_fp4_kernel(
    x_block,
    norm_constant_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    FP4_FORMAT: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
) -> None:
    # Get scale factors
    if FP4_FORMAT == "mxfp4":
        x_scale_blocks = x_block.reshape(128, 4, 32)
        x_scales = tl.max(x_scale_blocks.abs(), axis=-1) / E2M1_MAX_VALUE

        # Use the 8-bit exponent as the scale factor, and then add one in order to
        # round up
        x_scales = ((x_scales.cast(tl.uint32, bitcast=True) >> 23) & 0xFF).to(
            tl.uint8,
        ) + 1

        # Convert the rounded-up scale factor back to a 32-bit float
        x_scales_hp = (x_scales.cast(tl.uint32) << 23).cast(
            x_block.dtype,
            bitcast=True,
        )
    elif FP4_FORMAT == "nvfp4":
        norm_constant = tl.load(norm_constant_ptr)

        if BLOCK_SCALE_2D:
            x_scale_blocks = (
                x_block.reshape(8, 16, 4, 16).permute(0, 2, 1, 3).reshape(8, 4, 256)
            )
        else:
            x_scale_blocks = x_block.reshape(128, 4, 16)

        x_scales_hp = tl.max(x_scale_blocks.abs(), axis=-1) / tl.maximum(
            E2M1_MAX_VALUE * norm_constant,
            1e-12,
        )
        x_scales_hp = tl.clamp(x_scales_hp, E4M3_MIN_POSITIVE_VALUE, E4M3_MAX_VALUE)
        x_scales = x_scales_hp.to(tl.float8e4nv)

        if BLOCK_SCALE_2D:
            x_scale_blocks = x_block.reshape(128, 4, 16)
            x_scales = (
                x_scales[None, :, :]
                .broadcast_to(16, 8, 4)
                .permute(1, 0, 2)
                .reshape(128, 4)
            )

        x_scales_hp = x_scales.to(norm_constant.dtype) * norm_constant

    # Reshape and pack into the NVIDIA layout
    x_scales = x_scales.reshape(4, 32, 4).permute(1, 0, 2).ravel()

    # Store into c_e2m1 [M, K // 2]
    (x_block_scaled_b1, x_block_scaled_b2) = (
        (x_scale_blocks / tl.maximum(x_scales_hp[:, :, None], 1e-12))
        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
        .split()
    )

    if ROUND_STYLE == "nearest":
        x_e2m1 = tl.inline_asm_elementwise(
            asm="""
                {
                .reg .b8 byte0, byte1, byte2, byte3;
                cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
                mov.b32 $0, {byte0, byte1, byte2, byte3};
                }
                """,
            constraints="=r,r,r,r,r,r,r,r,r",
            args=[x_block_scaled_b1.to(tl.float32), x_block_scaled_b2.to(tl.float32)],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )
    elif ROUND_STYLE == "stochastic":
        rbits = tl.rand(
            0,
            tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N // 2
            + tl.arange(0, BLOCK_SIZE_N // 2)[None, :],
        ).cast(tl.uint32, bitcast=True)

        x_e2m1 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 tmp0, tmp1;
            cvt.rs.satfinite.e2m1x4.f32 tmp0, {$6, $2, $5, $1}, $9;
            cvt.rs.satfinite.e2m1x4.f32 tmp1, {$8, $4, $7, $3}, $10;
            mov.b32 $0, {tmp0, tmp1};
            }
            """,
            constraints="=r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[
                x_block_scaled_b1.to(tl.float32),
                x_block_scaled_b2.to(tl.float32),
                rbits,
            ],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )

    return x_e2m1, x_scales


@triton.jit
def fp4_quantization_kernel(
    x_ptr,
    norm_constant_ptr,
    x_e2m1_ptr,
    x_sf_ptr,
    stride_xm,
    stride_xn,
    # Meta-parameters
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    FP4_FORMAT: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    scale_rule: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)

    for block_i in range(GROUP_SIZE_M * GROUP_SIZE_N):
        m = block_i // GROUP_SIZE_N
        n = block_i % GROUP_SIZE_N

        offs_m = (
            pid_m * BLOCK_SIZE_M * GROUP_SIZE_M
            + m * BLOCK_SIZE_M
            + tl.arange(0, BLOCK_SIZE_M)
        )
        offs_n = (
            pid_n * BLOCK_SIZE_N * GROUP_SIZE_N
            + n * BLOCK_SIZE_N
            + tl.arange(0, BLOCK_SIZE_N)
        )

        # Load [B, B] block from A or A^T
        if not TRANSPOSE:
            x_block_ptrs = (
                x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
            )
            x_block = tl.load(
                x_block_ptrs,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
                other=0.0,
            )
        else:
            x_block_ptrs = (
                x_ptr + offs_n[:, None] * stride_xm + offs_m[None, :] * stride_xn
            )
            x_block = tl.load(
                x_block_ptrs,
                mask=(offs_n[:, None] < M) & (offs_m[None, :] < N),
                other=0.0,
            ).T

        x_block = x_block.to(tl.float32)

        if scale_rule != SCALE_RULE_ALWAYS_6:
            x_e2m1, x_scales = fp32_to_scaled_fp4_kernel_four_over_six(
                x_block,
                norm_constant_ptr,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                ROUND_STYLE,
                BLOCK_SCALE_2D,
                scale_rule,
            )
        else:
            x_e2m1, x_scales = fp32_to_scaled_fp4_kernel(
                x_block,
                norm_constant_ptr,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                FP4_FORMAT,
                ROUND_STYLE,
                BLOCK_SCALE_2D,
            )

        offs_n_e2m1 = (
            pid_n * BLOCK_SIZE_N * GROUP_SIZE_N // 2
            + n * BLOCK_SIZE_N // 2
            + tl.arange(0, BLOCK_SIZE_N // 2)
        )
        x_e2m1_ptrs = x_e2m1_ptr + offs_m[:, None] * (N // 2) + offs_n_e2m1[None, :]
        tl.store(x_e2m1_ptrs, x_e2m1)

        scale_block_offset = (
            (pid_m * GROUP_SIZE_M + m) * (tl.num_programs(1) * GROUP_SIZE_N)
            + (pid_n * GROUP_SIZE_N + n)
        ) * SCALE_MEGABLOCK_SIZE + tl.arange(0, SCALE_MEGABLOCK_SIZE)
        x_sf_ptrs = x_sf_ptr + scale_block_offset
        tl.store(x_sf_ptrs, x_scales)


autotuned_fp4_quantization_kernel = triton.autotune(
    configs=[
        triton.Config(
            {"GROUP_SIZE_M": group_size_m, "GROUP_SIZE_N": group_size_n},
            num_stages=None,
            num_warps=None,
        )
        for group_size_m, group_size_n in itertools.product(
            [1, 2, 4, 8, 16, 32, 64, 128, 256],
            [1, 2, 4, 8, 16, 32, 64, 128, 256],
        )
    ],
    key=[
        "M",
        "N",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "GROUP_SIZE_M",
        "GROUP_SIZE_N",
        "TRANSPOSE",
        "FP4_FORMAT",
        "ROUND_STYLE",
    ],
    prune_configs_by={
        "early_config_prune": lambda configs, args, **kwargs: (  # noqa: ARG005
            filter(
                lambda config: (
                    (kwargs["M"] // kwargs["BLOCK_SIZE_M"])
                    % config.kwargs["GROUP_SIZE_M"]
                    == 0
                    and (kwargs["N"] // kwargs["BLOCK_SIZE_N"])
                    % config.kwargs["GROUP_SIZE_N"]
                    == 0
                ),
                configs,
            )
        ),
    },
)(fp4_quantization_kernel)


@triton.jit
def fp4_quantization_with_rht_kernel(
    x_desc,
    norm_constant_ptr,
    h_desc,
    x_e2m1_desc,
    x_sf_desc,
    # Meta-parameters
    # TODO(jack): Update RHT kernel to support unpadded dimensions
    M: tl.constexpr,  # noqa: ARG001
    N: tl.constexpr,  # noqa: ARG001
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    FP4_FORMAT: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    scale_rule: tl.constexpr,
) -> None:
    HAD_BLOCK_SIZE: tl.constexpr = h_desc.block_shape[0]

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load H [B, B]
    h_block = h_desc.load([0, 0])

    for block_i in range(GROUP_SIZE_M * GROUP_SIZE_N):
        m = block_i // GROUP_SIZE_N
        n = block_i % GROUP_SIZE_N

        m_block_offset = pid_m * BLOCK_SIZE_M * GROUP_SIZE_M + m * BLOCK_SIZE_M
        n_block_offset = pid_n * BLOCK_SIZE_N * GROUP_SIZE_N + n * BLOCK_SIZE_N

        # Load [B, B] block from A or A^T
        if not TRANSPOSE:
            x_block = x_desc.load([m_block_offset, n_block_offset])
        else:
            x_block = x_desc.load([n_block_offset, m_block_offset]).T

        x_block = tl.dot(
            x_block.reshape(
                BLOCK_SIZE_M * BLOCK_SIZE_N // HAD_BLOCK_SIZE,
                HAD_BLOCK_SIZE,
            ),
            h_block,
        ).reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)

        if scale_rule != SCALE_RULE_ALWAYS_6:
            x_e2m1, x_scales = fp32_to_scaled_fp4_kernel_four_over_six(
                x_block,
                norm_constant_ptr,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                ROUND_STYLE,
                BLOCK_SCALE_2D,
                scale_rule,
            )
        else:
            x_e2m1, x_scales = fp32_to_scaled_fp4_kernel(
                x_block,
                norm_constant_ptr,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                FP4_FORMAT,
                ROUND_STYLE,
                BLOCK_SCALE_2D,
            )

        offs_m_e2m1 = pid_m * BLOCK_SIZE_M * GROUP_SIZE_M + m * BLOCK_SIZE_M
        offs_n_e2m1 = pid_n * BLOCK_SIZE_N * GROUP_SIZE_N // 2 + n * BLOCK_SIZE_N // 2
        x_e2m1_desc.store([offs_m_e2m1, offs_n_e2m1], x_e2m1)

        scale_block_offset = (
            (pid_m * GROUP_SIZE_M + m) * (tl.num_programs(1) * GROUP_SIZE_N)
            + (pid_n * GROUP_SIZE_N + n)
        ) * SCALE_MEGABLOCK_SIZE
        x_sf_desc.store([scale_block_offset], x_scales)


autotuned_fp4_quantization_with_rht_kernel = triton.autotune(
    configs=[
        triton.Config(
            {"GROUP_SIZE_M": group_size_m, "GROUP_SIZE_N": group_size_n},
            num_stages=None,
            num_warps=None,
        )
        for group_size_m, group_size_n in itertools.product(
            [1, 2, 4, 8, 16, 32, 64, 128, 256],
            [1, 2, 4, 8, 16, 32, 64, 128, 256],
        )
    ],
    key=[
        "M",
        "N",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "GROUP_SIZE_M",
        "GROUP_SIZE_N",
        "TRANSPOSE",
        "FP4_FORMAT",
        "ROUND_STYLE",
        "BLOCK_SCALE_2D",
        "scale_rule",
    ],
    prune_configs_by={
        "early_config_prune": lambda configs, args, **kwargs: (  # noqa: ARG005
            filter(
                lambda config: (
                    (kwargs["M"] // kwargs["BLOCK_SIZE_M"])
                    % config.kwargs["GROUP_SIZE_M"]
                    == 0
                    and (kwargs["N"] // kwargs["BLOCK_SIZE_N"])
                    % config.kwargs["GROUP_SIZE_N"]
                    == 0
                ),
                configs,
            )
        ),
    },
)(fp4_quantization_with_rht_kernel)


def quantize_to_fp4(  # noqa: C901, PLR0912
    x: torch.Tensor,
    norm_constant: torch.Tensor | None = None,
    had: torch.Tensor | None = None,
    *,
    fp4_format: FP4Format = "nvfp4",
    round_style: RoundStyle = "nearest",
    scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    block_scale_2d: bool = False,
    transpose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if transpose:
        N, M = x.shape
    else:
        M, N = x.shape

    if fp4_format == "mxfp4":
        block_size_m = 128
        block_size_n = 128
        scale_block_size = 32
        scale_dtype = torch.uint8

        if norm_constant is None:
            norm_constant = torch.ones(1, device=x.device, dtype=torch.float32)
    elif fp4_format == "nvfp4":
        block_size_m = 128
        block_size_n = 64
        scale_block_size = 16
        scale_dtype = torch.float8_e4m3fn

        if norm_constant is None:
            norm_constant = get_nvfp4_tensor_scale(x, scale_rule=scale_rule)

    padded_m = M + (block_size_m - M % block_size_m) % block_size_m
    padded_n = N + (block_size_n - N % block_size_n) % block_size_n

    x_e2m1 = torch.empty((padded_m, padded_n // 2), device=x.device, dtype=torch.uint8)
    x_sf = torch.empty(
        padded_m * padded_n // scale_block_size,
        device=x.device,
        dtype=scale_dtype,
    )

    grid = lambda meta: (  # noqa: E731
        padded_m // block_size_m // meta["GROUP_SIZE_M"],
        padded_n // block_size_n // meta["GROUP_SIZE_N"],
    )

    if had is None:
        (
            autotuned_fp4_quantization_kernel
            if FOUROVERSIX_AUTOTUNE
            else fp4_quantization_kernel
        )[grid](
            x,
            norm_constant,
            x_e2m1,
            x_sf,
            x.stride(0),
            x.stride(1),
            M=M,
            N=N,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            GROUP_SIZE_M=1,
            GROUP_SIZE_N=max(
                2**x for x in range(7) if (N // block_size_n) % (2**x) == 0
            ),
            TRANSPOSE=transpose,
            FP4_FORMAT=fp4_format,
            ROUND_STYLE=round_style,
            BLOCK_SCALE_2D=block_scale_2d,
            scale_rule=scale_rule.value,
            num_warps=1,
            num_stages=3,
        )
    else:
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

        x_desc = TensorDescriptor.from_tensor(
            x,
            block_shape=(
                [block_size_n, block_size_m]
                if transpose
                else [block_size_m, block_size_n]
            ),
        )
        h_desc = TensorDescriptor.from_tensor(
            had,
            block_shape=[had_block_size, had_block_size],
        )
        x_e2m1_desc = TensorDescriptor.from_tensor(
            x_e2m1,
            block_shape=[block_size_m, block_size_n // 2],
        )
        x_sf_desc = TensorDescriptor.from_tensor(
            x_sf,
            block_shape=[SCALE_MEGABLOCK_SIZE.value],
        )

        (
            autotuned_fp4_quantization_with_rht_kernel
            if FOUROVERSIX_AUTOTUNE
            else fp4_quantization_with_rht_kernel
        )[grid](
            x_desc,
            norm_constant,
            h_desc,
            x_e2m1_desc,
            x_sf_desc,
            M=M,
            N=N,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            GROUP_SIZE_M=1,
            GROUP_SIZE_N=max(
                2**x for x in range(7) if (N // block_size_n) % (2**x) == 0
            ),
            TRANSPOSE=transpose,
            FP4_FORMAT=fp4_format,
            ROUND_STYLE=round_style,
            BLOCK_SCALE_2D=block_scale_2d,
            scale_rule=scale_rule.value,
            num_warps=4,
            num_stages=3 if fp4_format == "nvfp4" else 1,
        )

    if fp4_format == "mxfp4":
        x_sf = x_sf.view(torch.float8_e8m0fnu)

    return x_e2m1, x_sf, norm_constant
