import triton
import triton.language as tl

MIN_SAFE_UNBIASED_EXP = tl.constexpr(-10)


@triton.jit
def convert_to_low_precision_fp_with_rtn(
    x,
    EXP_BIAS: tl.constexpr,
    EXP_BITS: tl.constexpr,
    MANTISSA_BITS: tl.constexpr,
    MAX_EXP: tl.constexpr,
    MAX_VALUE: tl.constexpr,
    MIN_NORMAL_EXP: tl.constexpr,
) -> tl.tensor:
    # Sign
    sign = tl.where(x < 0, 1, 0).to(tl.uint8)
    x_abs = tl.minimum(tl.abs(x), MAX_VALUE)

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
    mant_sub = x_safe * tl.exp2(EXP_BIAS + MANTISSA_BITS - 1.0)
    mant_sub_q = tl.extra.cuda.libdevice.rint(mant_sub)
    mant_sub_q = tl.minimum(mant_sub_q, (1 << MANTISSA_BITS) - 1)

    # Select normal or subnormal
    mantissa = tl.where(is_normal, mant_q, mant_sub_q)
    exponent = tl.where(is_normal, exp_q, 0)

    # Handle zero
    mantissa = tl.where(is_zero, 0, mantissa).to(tl.uint8)
    exponent = tl.where(is_zero, 0, exponent).to(tl.uint8)

    return (sign << (EXP_BITS + MANTISSA_BITS)) | (exponent << MANTISSA_BITS) | mantissa


@triton.jit
def convert_low_precision_fp_to_high_precision(
    x,
    EXP_BIAS: tl.constexpr,
    EXP_BITS: tl.constexpr,
    MANTISSA_BITS: tl.constexpr,
) -> tl.tensor:
    sign = ((x >> (EXP_BITS + MANTISSA_BITS)) & 1).to(tl.float32)
    exponent = ((x >> MANTISSA_BITS) & ((1 << EXP_BITS) - 1)).to(tl.float32)
    mantissa = (x & ((1 << MANTISSA_BITS) - 1)).to(tl.float32)

    is_subnormal = exponent == 0
    is_zero = x == 0

    val_norm = (1 + mantissa / (1 << MANTISSA_BITS)) * tl.exp2(exponent - EXP_BIAS)
    val_sub = mantissa / (1 << MANTISSA_BITS) * tl.exp2(1.0 - EXP_BIAS)

    val = tl.where(is_subnormal, val_sub, val_norm)
    val = tl.where(sign == 1, -val, val)
    val = tl.where(is_zero, 0, val)

    return val.to(tl.float32)
