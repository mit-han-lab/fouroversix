import triton
import triton.language as tl

from .constants import SM_100, SM_110, SM_120
from .fp import (
    convert_low_precision_fp_to_high_precision,
    convert_to_low_precision_fp_with_rtn,
)


@triton.jit
def _convert_to_e3m2x2_with_rtn(
    x_block_scaled,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    if (
        MAJOR_COMPUTE_CAPABILITY == SM_100
        or MAJOR_COMPUTE_CAPABILITY == SM_110
        or MAJOR_COMPUTE_CAPABILITY == SM_120
    ):
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 tmp0, tmp1;
            cvt.rn.satfinite.e3m2x2.f32 tmp0, $2, $1;
            cvt.rn.satfinite.e3m2x2.f32 tmp1, $4, $3;
            mov.b32 $0, {tmp0, tmp1};
            }
            """,
            constraints="=r,r,r,r,r",
            args=[x_block_scaled],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )

    return convert_to_low_precision_fp_with_rtn(
        x_block_scaled,
        EXP_BIAS=3,
        EXP_BITS=3,
        MANTISSA_BITS=2,
        MAX_EXP=4,
        MAX_VALUE=28,
        MIN_NORMAL_EXP=-2,
    )


@triton.jit
def convert_to_e3m2x2(
    x_block_scaled,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    return _convert_to_e3m2x2_with_rtn(
        x_block_scaled,
        MAJOR_COMPUTE_CAPABILITY,
    )


@triton.jit
def convert_e3m2x2_to_fp16(
    x_block,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    if (
        MAJOR_COMPUTE_CAPABILITY == SM_100
        or MAJOR_COMPUTE_CAPABILITY == SM_110
        or MAJOR_COMPUTE_CAPABILITY == SM_120
    ):
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 tmp0, tmp1;
            mov.b32 {tmp0, tmp1}, $2;
            cvt.rn.f16x2.e3m2x2 $0, tmp0;
            cvt.rn.f16x2.e3m2x2 $1, tmp1;
            }
            """,
            constraints="=r,=r,r",
            args=[x_block],
            dtype=tl.float16,
            is_pure=True,
            pack=4,
        )

    return convert_low_precision_fp_to_high_precision(
        x_block,
        EXP_BIAS=1,
        EXP_BITS=3,
        MANTISSA_BITS=2,
    )


@triton.jit
def _convert_to_e2m3x2_with_rtn(
    x_block_scaled,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    if (
        MAJOR_COMPUTE_CAPABILITY == SM_100
        or MAJOR_COMPUTE_CAPABILITY == SM_110
        or MAJOR_COMPUTE_CAPABILITY == SM_120
    ):
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 tmp0, tmp1;
            cvt.rn.satfinite.e2m3x2.f32 tmp0, $2, $1;
            cvt.rn.satfinite.e2m3x2.f32 tmp1, $4, $3;
            mov.b32 $0, {tmp0, tmp1};
            }
            """,
            constraints="=r,r,r,r,r",
            args=[x_block_scaled],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )

    return convert_to_low_precision_fp_with_rtn(
        x_block_scaled,
        EXP_BIAS=1,
        EXP_BITS=2,
        MANTISSA_BITS=3,
        MAX_EXP=2,
        MAX_VALUE=7.5,
        MIN_NORMAL_EXP=0,
    )


@triton.jit
def _convert_to_e2m3x2_and_dequantized_fp16_with_rtn(
    x_block_scaled,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    x_e2m3 = _convert_to_e2m3x2_with_rtn(
        x_block_scaled,
        MAJOR_COMPUTE_CAPABILITY,
    )

    sign = (x_e2m3 >> 5) & 1
    exp = (x_e2m3 >> 3) & 0x3
    mant = x_e2m3 & 0x7

    sign = sign.to(tl.float32)
    exp = exp.to(tl.float32)
    mant = mant.to(tl.float32)

    # Identify subnormals
    is_subnormal = exp == 0
    is_zero = x_e2m3 == 0

    # Normal numbers
    val_norm = (1 + mant / (1 << 3)) * tl.exp2(exp - 1)

    # Subnormals
    val_sub = (mant / (1 << 3)) * tl.exp2(1.0 - 1)

    val = tl.where(is_subnormal, val_sub, val_norm)
    val = tl.where(sign == 1, -val, val)
    val = tl.where(is_zero, 0, val)

    return x_e2m3, val.to(tl.float16)


@triton.jit
def _convert_to_e3m2x2_and_dequantized_fp16_with_rtn(
    x_block_scaled,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    x_e3m2 = _convert_to_e3m2x2_with_rtn(
        x_block_scaled,
        MAJOR_COMPUTE_CAPABILITY,
    )

    sign = (x_e3m2 >> 5) & 1
    exp = (x_e3m2 >> 2) & 0x7
    mant = x_e3m2 & 0x3

    sign = sign.to(tl.float32)
    exp = exp.to(tl.float32)
    mant = mant.to(tl.float32)

    # Identify subnormals
    is_subnormal = exp == 0
    is_zero = x_e3m2 == 0

    # Normal numbers
    val_norm = (1 + mant / (1 << 2)) * tl.exp2(exp - 3)

    # Subnormals
    val_sub = (mant / (1 << 2)) * tl.exp2(1.0 - 3)

    val = tl.where(is_subnormal, val_sub, val_norm)
    val = tl.where(sign == 1, -val, val)
    val = tl.where(is_zero, 0, val)

    return x_e3m2, val.to(tl.float16)


@triton.jit
def convert_to_e2m3x2(
    x_block_scaled,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    return _convert_to_e2m3x2_with_rtn(
        x_block_scaled,
        MAJOR_COMPUTE_CAPABILITY,
    )


@triton.jit
def convert_to_e2m3x2_and_dequantized_fp16(
    x_block_scaled,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    return _convert_to_e2m3x2_and_dequantized_fp16_with_rtn(
        x_block_scaled,
        MAJOR_COMPUTE_CAPABILITY,
    )


@triton.jit
def convert_to_e3m2x2_and_dequantized_fp16(
    x_block_scaled,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    return _convert_to_e3m2x2_and_dequantized_fp16_with_rtn(
        x_block_scaled,
        MAJOR_COMPUTE_CAPABILITY,
    )


@triton.jit
def convert_e2m3x2_to_fp16(
    x_block,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    if (
        MAJOR_COMPUTE_CAPABILITY == SM_100
        or MAJOR_COMPUTE_CAPABILITY == SM_110
        or MAJOR_COMPUTE_CAPABILITY == SM_120
    ):
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 tmp0, tmp1;
            mov.b32 {tmp0, tmp1}, $2;
            cvt.rn.f16x2.e2m3x2 $0, tmp0;
            cvt.rn.f16x2.e2m3x2 $1, tmp1;
            }
            """,
            constraints="=r,=r,r",
            args=[x_block],
            dtype=tl.float16,
            is_pure=True,
            pack=4,
        )

    return convert_low_precision_fp_to_high_precision(
        x_block,
        EXP_BIAS=1,
        EXP_BITS=2,
        MANTISSA_BITS=3,
    )
