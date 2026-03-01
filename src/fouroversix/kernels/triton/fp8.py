import triton
import triton.language as tl

from .constants import SM_80

E4M3_MAX_VALUE = tl.constexpr(448)
EXP_BIAS = tl.constexpr(7)
MANTISSA_BITS = tl.constexpr(3)
MIN_NORMAL_EXP = tl.constexpr(-6)
MAX_EXP = tl.constexpr(8)
MIN_SAFE_UNBIASED_EXP = tl.constexpr(-10)


@triton.jit
def convert_to_e4m3_with_rtn(x, MAJOR_COMPUTE_CAPABILITY: tl.constexpr) -> tl.tensor:
    if MAJOR_COMPUTE_CAPABILITY > SM_80:
        return x.to(tl.float8e4nv).cast(tl.uint8, bitcast=True)

    # Sign
    sign = tl.where(x < 0, 1, 0).to(tl.uint8)
    x_abs = tl.minimum(tl.abs(x), E4M3_MAX_VALUE)

    # Avoid log2(0)
    is_zero = x_abs == 0
    x_safe = tl.where(is_zero, 1.0, x_abs)

    # Compute exponent
    exp = tl.floor(tl.log2(x_safe))

    # Clamp exponent
    exp = tl.clamp(exp, MIN_SAFE_UNBIASED_EXP, MAX_EXP)

    # Normals
    is_normal = exp >= MIN_NORMAL_EXP

    mant = x_safe / tl.exp2(exp) - 1
    mant_q = tl.extra.cuda.libdevice.rint(mant * (1 << MANTISSA_BITS))
    overflow = mant_q == (1 << MANTISSA_BITS)

    mant_q = tl.where(overflow, 0, mant_q)
    exp = tl.where(overflow, exp + 1, exp)
    exp_q = exp + EXP_BIAS

    # For subnormals, value = mantissa * 2^(1 - bias)
    # => mantissa = x / 2^(1 - bias)
    sub_scale = tl.exp2(1.0 - EXP_BIAS)
    mant_sub = x_safe / sub_scale

    mant_sub_q = tl.extra.cuda.libdevice.rint(mant_sub)
    mant_sub_q = tl.minimum(mant_sub_q, (1 << MANTISSA_BITS) - 1)

    # Select normal or subnormal
    mantissa = tl.where(is_normal, mant_q, mant_sub_q)
    exponent = tl.where(is_normal, exp_q, 0)

    # Handle zero
    mantissa = tl.where(is_zero, 0, mantissa).to(tl.uint8)
    exponent = tl.where(is_zero, 0, exponent).to(tl.uint8)

    return (sign << 7) | (exponent << MANTISSA_BITS) | mantissa


@triton.jit
def convert_e4m3_to_fp32(fp8, MAJOR_COMPUTE_CAPABILITY: tl.constexpr) -> tl.tensor:
    if MAJOR_COMPUTE_CAPABILITY > SM_80:
        return fp8.cast(tl.float8e4nv, bitcast=True).to(tl.float32)

    sign = (fp8 >> 7) & 1
    exp = (fp8 >> MANTISSA_BITS) & 0xF
    mant = fp8 & 0x7

    sign = sign.to(tl.float32)
    exp = exp.to(tl.float32)
    mant = mant.to(tl.float32)

    # Identify subnormals
    is_subnormal = exp == 0
    is_zero = fp8 == 0

    # Normal numbers
    val_norm = (1 + mant / (1 << MANTISSA_BITS)) * tl.exp2(exp - EXP_BIAS)

    # Subnormals
    val_sub = (mant / (1 << MANTISSA_BITS)) * tl.exp2(1.0 - EXP_BIAS)

    val = tl.where(is_subnormal, val_sub, val_norm)
    val = tl.where(sign == 1, -val, val)

    return tl.where(is_zero, 0, val)
