import triton
import triton.language as tl

from .constants import SM_80
from .fp import (
    convert_low_precision_fp_to_high_precision,
    convert_to_low_precision_fp_with_rtn,
)

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

    return convert_to_low_precision_fp_with_rtn(
        x,
        EXP_BIAS=7,
        EXP_BITS=4,
        MANTISSA_BITS=3,
        MAX_EXP=8,
        MAX_VALUE=448,
        MIN_NORMAL_EXP=-6,
    )


@triton.jit
def convert_e4m3_to_high_precision(
    fp8,
    TO_DTYPE: tl.dtype,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> tl.tensor:
    if MAJOR_COMPUTE_CAPABILITY > SM_80:
        return fp8.cast(tl.float8e4nv, bitcast=True).to(TO_DTYPE)

    return convert_low_precision_fp_to_high_precision(
        fp8,
        EXP_BIAS=7,
        EXP_BITS=4,
        MANTISSA_BITS=3,
    ).to(TO_DTYPE)
