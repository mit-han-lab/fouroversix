from __future__ import annotations

import torch
import triton
import triton.language as tl
from fouroversix.quantize.reference import (
    E2M1_MAX_VALUE,
    E4M3_MAX_VALUE,
    E4M3_MIN_POSITIVE_NORMAL,
)
from fouroversix.utils import AdaptiveBlockScalingRule, FP4Format, RoundStyle
from triton.tools.tensor_descriptor import TensorDescriptor

E2M1_MAX_VALUE = tl.constexpr(E2M1_MAX_VALUE)
E2M1_MAX_FOUR = tl.constexpr(4)
E4M3_MAX_VALUE = tl.constexpr(E4M3_MAX_VALUE)
E4M3_MAX_FOUROVERSIX = tl.constexpr(256)
E4M3_MIN_POSITIVE_NORMAL = tl.constexpr(E4M3_MIN_POSITIVE_NORMAL)
SCALE_MEGABLOCK_SIZE = tl.constexpr(512)

FP4_FORMAT_MXFP4 = tl.constexpr(FP4Format.mxfp4.value)
FP4_FORMAT_NVFP4 = tl.constexpr(FP4Format.nvfp4.value)

ROUND_STYLE_NEAREST = tl.constexpr(RoundStyle.nearest.value)
ROUND_STYLE_STOCHASTIC = tl.constexpr(RoundStyle.stochastic.value)

SCALE_RULE_ABS_MAX = tl.constexpr(AdaptiveBlockScalingRule.abs_max.value)
SCALE_RULE_ALWAYS_4 = tl.constexpr(AdaptiveBlockScalingRule.always_4.value)
SCALE_RULE_ALWAYS_6 = tl.constexpr(AdaptiveBlockScalingRule.always_6.value)
SCALE_RULE_L1_NORM = tl.constexpr(AdaptiveBlockScalingRule.l1_norm.value)
SCALE_RULE_MSE = tl.constexpr(AdaptiveBlockScalingRule.mse.value)


@triton.jit
def rht_kernel(
    x_desc,
    h_desc,
    y_desc,
    # Meta-parameters
    # TODO(jack): Update RHT kernel to support unpadded dimensions
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TRANSPOSE: tl.constexpr,
) -> None:
    HAD_BLOCK_SIZE: tl.constexpr = h_desc.block_shape[0]

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load H [B, B]
    h_block = h_desc.load([0, 0])

    m_block_offset = pid_m * BLOCK_SIZE_M
    n_block_offset = pid_n * BLOCK_SIZE_N

    if not TRANSPOSE:
        x_block = x_desc.load([m_block_offset, n_block_offset])
    else:
        x_block = x_desc.load([n_block_offset, m_block_offset]).T

    y_block = tl.dot(
        x_block.reshape(
            BLOCK_SIZE_M * BLOCK_SIZE_N // HAD_BLOCK_SIZE,
            HAD_BLOCK_SIZE,
        ),
        h_block,
    ).reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)

    y_desc.store([m_block_offset, n_block_offset], y_block)


@triton.jit
def fp32_to_scaled_fp4_kernel_fouroversix(  # noqa: C901, PLR0912
    x_block,
    x_amax_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    SCALE_RULE: tl.constexpr,
) -> None:
    x_amax = tl.load(x_amax_ptr)
    x_scale_blocks = x_block.reshape(128, 4, 16)

    # Calculate six blocks
    if SCALE_RULE == SCALE_RULE_ALWAYS_6 or SCALE_RULE == SCALE_RULE_ALWAYS_4:  # noqa: SIM109, PLR1714
        x_scales_hp = tl.max(x_scale_blocks.abs(), axis=-1) * E4M3_MAX_VALUE / x_amax
    else:
        x_scales_hp = (
            tl.max(x_scale_blocks.abs(), axis=-1) * E4M3_MAX_FOUROVERSIX / x_amax
        )

    if BLOCK_SCALE_2D:
        x_scales_hp = (
            tl.max(
                x_scales_hp.reshape(8, 16, 4).permute(0, 2, 1),
                axis=-1,
            )
            .expand_dims(0)
            .broadcast_to(16, 8, 4)
            .permute(1, 0, 2)
            .reshape(128, 4)
        )

    x_scales_6 = x_scales_hp.to(tl.float8e4nv)
    x_scales_4 = (x_scales_hp * (6 / 4)).to(tl.float8e4nv)

    if SCALE_RULE == SCALE_RULE_ALWAYS_4:
        x_scales_6 = (x_scales_hp * (4 / 6)).to(tl.float8e4nv)
        x_scales_4 = x_scales_hp.to(tl.float8e4nv)

    if SCALE_RULE == SCALE_RULE_ALWAYS_6 or SCALE_RULE == SCALE_RULE_ALWAYS_4:  # noqa: SIM109, PLR1714
        (x_block_scaled_6_b1, x_block_scaled_6_b2) = (
            tl.where(
                x_scales_6.expand_dims(2).to(x_amax.dtype) != 0,
                (x_scale_blocks * E2M1_MAX_VALUE * E4M3_MAX_VALUE)
                / (x_amax * x_scales_6.to(x_amax.dtype).expand_dims(2)),
                0,
            )
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
            .split()
        )

        (x_block_scaled_4_b1, x_block_scaled_4_b2) = (
            tl.where(
                x_scales_4.expand_dims(2).to(x_amax.dtype) != 0,
                (x_scale_blocks * E2M1_MAX_FOUR * E4M3_MAX_VALUE)
                / (x_amax * x_scales_4.to(x_amax.dtype).expand_dims(2)),
                0,
            )
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
            .split()
        )
    else:
        (x_block_scaled_6_b1, x_block_scaled_6_b2) = (
            tl.where(
                x_scales_6.expand_dims(2).to(x_amax.dtype) != 0,
                (x_scale_blocks * E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX)
                / (x_amax * x_scales_6.to(x_amax.dtype).expand_dims(2)),
                0,
            )
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
            .split()
        )

        (x_block_scaled_4_b1, x_block_scaled_4_b2) = (
            tl.where(
                x_scales_4.expand_dims(2).to(x_amax.dtype) != 0,
                (x_scale_blocks * E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX)
                / (x_amax * x_scales_4.to(x_amax.dtype).expand_dims(2)),
                0,
            )
            .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
            .split()
        )

    if ROUND_STYLE == ROUND_STYLE_NEAREST:
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
                x_block_scaled_6_b1,
                x_block_scaled_6_b2,
                x_block_scaled_4_b1,
                x_block_scaled_4_b2,
            ],
            dtype=(tl.uint8, tl.uint8, tl.uint32, tl.uint32),
            is_pure=True,
            pack=8,
        )
    elif ROUND_STYLE == ROUND_STYLE_STOCHASTIC:
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
                x_block_scaled_6_b1,
                x_block_scaled_6_b2,
                x_block_scaled_4_b1,
                x_block_scaled_4_b2,
                rbits,
            ],
            dtype=(tl.uint8, tl.uint8, tl.uint32, tl.uint32),
            is_pure=True,
            pack=8,
        )

    x_fp16_6_lo = (
        (x_fp16x2_6 & 0xFFFF)
        .cast(tl.uint16)
        .cast(tl.float16, bitcast=True)
        .cast(x_amax.dtype)
    )
    x_fp16_6_hi = (
        (x_fp16x2_6 >> 16)
        .cast(tl.uint16)
        .cast(tl.float16, bitcast=True)
        .cast(x_amax.dtype)
    )
    x_hp_6 = tl.join(x_fp16_6_lo, x_fp16_6_hi).reshape(128, 4, 16)

    # HACK: Add a fake data dependency barrier to prevent Triton from reordering
    # instructions in a way that causes slight numerical differences to the PyTorch
    # implementation.
    x_dequantized_6 = x_hp_6 * x_scales_6.to(x_amax.dtype).expand_dims(2) * x_amax / (
        E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX
    ) + 0 * tl.program_id(0)

    x_fp16_4_lo = (
        (x_fp16x2_4 & 0xFFFF)
        .cast(tl.uint16)
        .cast(tl.float16, bitcast=True)
        .cast(x_amax.dtype)
    )
    x_fp16_4_hi = (
        (x_fp16x2_4 >> 16)
        .cast(tl.uint16)
        .cast(tl.float16, bitcast=True)
        .cast(x_amax.dtype)
    )
    x_hp_4 = tl.join(x_fp16_4_lo, x_fp16_4_hi).reshape(128, 4, 16)

    # HACK: Add a fake data dependency barrier to prevent Triton from reordering
    # instructions in a way that causes slight numerical differences to the PyTorch
    # implementation.
    x_dequantized_4 = x_hp_4 * x_scales_4.to(x_amax.dtype).expand_dims(2) * x_amax / (
        E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX
    ) + 0 * tl.program_id(0)

    if SCALE_RULE == SCALE_RULE_ALWAYS_6:
        six_error = tl.full((128, 4), 0 * tl.program_id(0), dtype=tl.int32)
        four_error = tl.sum(
            (x_dequantized_4 - x_scale_blocks) * (x_dequantized_4 - x_scale_blocks),
            axis=-1,
        )
    elif SCALE_RULE == SCALE_RULE_ALWAYS_4:
        six_error = tl.sum(
            (x_dequantized_6 - x_scale_blocks) * (x_dequantized_6 - x_scale_blocks),
            axis=-1,
        )
        four_error = tl.full((128, 4), 0 * tl.program_id(0), dtype=tl.int32)
    elif SCALE_RULE == SCALE_RULE_ABS_MAX:
        six_error = tl.max(
            tl.abs(x_dequantized_6 - x_scale_blocks),
            axis=-1,
        )
        four_error = tl.max(
            tl.abs(x_dequantized_4 - x_scale_blocks),
            axis=-1,
        )
    elif SCALE_RULE == SCALE_RULE_L1_NORM:
        six_error = tl.sum(
            tl.abs(x_dequantized_6 - x_scale_blocks),
            axis=-1,
        )
        four_error = tl.sum(
            tl.abs(x_dequantized_4 - x_scale_blocks),
            axis=-1,
        )
    elif SCALE_RULE == SCALE_RULE_MSE:
        six_error = tl.sum(
            (x_dequantized_6 - x_scale_blocks) * (x_dequantized_6 - x_scale_blocks),
            axis=-1,
        )
        four_error = tl.sum(
            (x_dequantized_4 - x_scale_blocks) * (x_dequantized_4 - x_scale_blocks),
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
def fp4_quantization_kernel(
    x_desc,
    x_amax_ptr,
    x_e2m1_desc,
    x_sf_desc,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    FP4_FORMAT: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
    SCALE_RULE: tl.constexpr,
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

    x_e2m1, x_scales = fp32_to_scaled_fp4_kernel_fouroversix(
        x_block.to(tl.float32),
        x_amax_ptr,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        ROUND_STYLE,
        BLOCK_SCALE_2D,
        SCALE_RULE,
    )

    e2m1_n_block_offset = pid_n * BLOCK_SIZE_N // 2
    x_e2m1_desc.store([m_block_offset, e2m1_n_block_offset], x_e2m1)

    scale_block_offset = (pid_m * tl.num_programs(1) + pid_n) * SCALE_MEGABLOCK_SIZE
    x_sf_desc.store([scale_block_offset], x_scales)


def quantize_to_fp4(  # noqa: C901
    x: torch.Tensor,
    x_amax: torch.Tensor | None = None,
    had: torch.Tensor | None = None,
    *,
    fp4_format: FP4Format = FP4Format.nvfp4,
    round_style: RoundStyle = RoundStyle.nearest,
    scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    block_scale_2d: bool = False,
    transpose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if transpose:
        N, M = x.shape
    else:
        M, N = x.shape

    if fp4_format == FP4Format.mxfp4:
        block_size_m = 128
        block_size_n = 128
        scale_block_size = 32
        scale_dtype = torch.uint8

        if x_amax is None:
            x_amax = torch.ones(1, device=x.device, dtype=torch.float32)
    elif fp4_format == FP4Format.nvfp4:
        block_size_m = 128
        block_size_n = 64
        scale_block_size = 16
        scale_dtype = torch.float8_e4m3fn

        if x_amax is None:
            x_amax = x.abs().max().float()

    padded_m = M + (block_size_m - M % block_size_m) % block_size_m
    padded_n = N + (block_size_n - N % block_size_n) % block_size_n

    x_e2m1 = torch.empty((padded_m, padded_n // 2), device=x.device, dtype=torch.uint8)
    x_sf = torch.empty(
        padded_m * padded_n // scale_block_size,
        device=x.device,
        dtype=scale_dtype,
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

        x_rht = torch.empty((M, N), device=x.device, dtype=x.dtype)

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

    fp4_quantization_kernel[grid](
        x_rht_desc if had is not None else x_desc,
        x_amax,
        x_e2m1_desc,
        x_sf_desc,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        TRANSPOSE=transpose,
        FP4_FORMAT=fp4_format.value,
        ROUND_STYLE=round_style.value,
        BLOCK_SCALE_2D=block_scale_2d,
        SCALE_RULE=scale_rule.value,
    )

    if fp4_format == FP4Format.mxfp4:
        x_sf = x_sf.view(torch.float8_e8m0fnu)

    return x_e2m1, x_sf, x_amax
