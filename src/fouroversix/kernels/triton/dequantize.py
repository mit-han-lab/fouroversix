import triton
import triton.language as tl

from .constants import (
    QUANTIZED_VALUE_TYPE_FP4,
    QUANTIZED_VALUE_TYPE_FP6_E2M3,
    QUANTIZED_VALUE_TYPE_FP6_E3M2,
    QUANTIZED_VALUE_TYPE_IF4,
)
from .fp6 import convert_e2m3x2_to_fp16, convert_e3m2x2_to_fp16
from .if4 import convert_if4_to_fp32


@triton.jit
def dequantize_to_fp16_kernel(
    values,
    scale_factors,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP4:
        dequantized_values = convert_if4_to_fp32(
            values,
            scale_factors,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            RETURN_FP=True,
            MAJOR_COMPUTE_CAPABILITY=MAJOR_COMPUTE_CAPABILITY,
        ).to(tl.float16)
    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP6_E2M3:
        dequantized_values = convert_e2m3x2_to_fp16(
            values,
            MAJOR_COMPUTE_CAPABILITY=MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP6_E3M2:
        dequantized_values = convert_e3m2x2_to_fp16(
            values,
            MAJOR_COMPUTE_CAPABILITY=MAJOR_COMPUTE_CAPABILITY,
        )
    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF4:
        dequantized_values = convert_if4_to_fp32(
            values,
            scale_factors,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            RETURN_FP=False,
            MAJOR_COMPUTE_CAPABILITY=MAJOR_COMPUTE_CAPABILITY,
        )

    return dequantized_values


@triton.jit
def dequantize_with_tensor_descriptors(
    values_desc,
    scale_factors_desc,
    output_desc,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    QUANTIZED_VALUE_PACKING_FACTOR: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    values = values_desc.load(
        [
            tl.program_id(0) * BLOCK_SIZE_M,
            tl.program_id(1) * BLOCK_SIZE_N // QUANTIZED_VALUE_PACKING_FACTOR,
        ],
    )

    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF4:
        scale_factors = scale_factors_desc.load(
            [
                tl.program_id(0) * BLOCK_SIZE_M,
                tl.program_id(1) * BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            ],
        )
    else:
        scale_factors = None

    output = dequantize_to_fp16_kernel(
        values,
        scale_factors,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        QUANTIZED_VALUE_TYPE,
        MAJOR_COMPUTE_CAPABILITY,
    )

    output_desc.store(
        [tl.program_id(0) * BLOCK_SIZE_M, tl.program_id(1) * BLOCK_SIZE_N],
        output.to(OUT_DTYPE),
    )
