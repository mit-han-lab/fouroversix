import triton
import triton.language as tl

from .constants import (
    QUANTIZED_VALUE_TYPE_FP6_E2M3,
    QUANTIZED_VALUE_TYPE_FP6_E3M2,
)
from .fp6 import convert_e2m3x2_to_fp16, convert_e3m2x2_to_fp16


@triton.jit
def dequantize_kernel(
    values_desc,
    output_desc,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANTIZED_VALUE_TYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    USE_BLACKWELL_CVT_RN_INSTRUCTIONS: tl.constexpr,
) -> None:
    values = values_desc.load(
        [tl.program_id(0) * BLOCK_SIZE_M, tl.program_id(1) * BLOCK_SIZE_N],
    )

    if QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP6_E2M3:
        dequantized_values = convert_e2m3x2_to_fp16(
            values,
            USE_BLACKWELL_CVT_RN_INSTRUCTIONS=USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP6_E3M2:
        dequantized_values = convert_e3m2x2_to_fp16(
            values,
            USE_BLACKWELL_CVT_RN_INSTRUCTIONS=USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
        )

    output_desc.store(
        [tl.program_id(0) * BLOCK_SIZE_M, tl.program_id(1) * BLOCK_SIZE_N],
        dequantized_values.to(OUT_DTYPE),
    )
