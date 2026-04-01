import triton
import triton.language as tl
from fouroversix.utils import DataType

from .constants import (
    IF6_E2M3_INT_EXPANSION_FACTOR,
    IF6_E3M2_INT_EXPANSION_FACTOR,
)
from .fp6 import convert_e2m3x2_to_fp16, convert_e3m2x2_to_fp16

IF6_GROUP_SIZE = tl.constexpr(DataType.if6_e2m3.block_size)


@triton.jit
def convert_if6_e2m3_to_fp16(
    values,
    scale_factors,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    RETURN_FP: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    fp_values = convert_e2m3x2_to_fp16(values, MAJOR_COMPUTE_CAPABILITY).reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // IF6_GROUP_SIZE,
        IF6_GROUP_SIZE,
    )

    if RETURN_FP:
        return fp_values

    int_magnitude = (values & 0x1F).to(tl.int8)
    int_sign = ((values >> 5) & 0x1).to(tl.int8)
    int_values = int_magnitude * (1 - 2 * int_sign)

    int_values = int_values.to(tl.float16).reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // IF6_GROUP_SIZE,
        IF6_GROUP_SIZE,
    )

    real_values = tl.where(
        (scale_factors.to(tl.uint8, bitcast=True) >= 128).expand_dims(2),
        int_values * IF6_E2M3_INT_EXPANSION_FACTOR,
        fp_values,
    )

    return real_values.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)


@triton.jit
def convert_if6_e3m2_to_fp16(
    values,
    scale_factors,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    RETURN_FP: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    fp_values = convert_e3m2x2_to_fp16(values, MAJOR_COMPUTE_CAPABILITY).reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // IF6_GROUP_SIZE,
        IF6_GROUP_SIZE,
    )

    if RETURN_FP:
        return fp_values

    int_magnitude = (values & 0x1F).to(tl.int8)
    int_sign = ((values >> 5) & 0x1).to(tl.int8)
    int_values = int_magnitude * (1 - 2 * int_sign)

    int_values = int_values.to(tl.float16).reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // IF6_GROUP_SIZE,
        IF6_GROUP_SIZE,
    )

    real_values = tl.where(
        (scale_factors.to(tl.uint8, bitcast=True) >= 128).expand_dims(2),
        int_values * IF6_E3M2_INT_EXPANSION_FACTOR,
        fp_values,
    )

    return real_values.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)
