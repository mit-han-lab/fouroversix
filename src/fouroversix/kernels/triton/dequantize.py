import triton
import triton.language as tl

from .constants import (
    QUANTIZED_VALUE_TYPE_FP3,
    QUANTIZED_VALUE_TYPE_FP4,
    QUANTIZED_VALUE_TYPE_FP6_E2M3,
    QUANTIZED_VALUE_TYPE_FP6_E3M2,
    QUANTIZED_VALUE_TYPE_IF4,
    QUANTIZED_VALUE_TYPE_IF6_E2M3,
    QUANTIZED_VALUE_TYPE_IF6_E3M2,
    QUANTIZED_VALUE_TYPE_INT3,
    QUANTIZED_VALUE_TYPE_INT4,
    QUANTIZED_VALUE_TYPE_INT6,
    SCALE_MEGABLOCK_SIZE,
    SCALE_TYPE_NV_IF,
)
from .fp3 import convert_e2m0_to_fp16
from .fp6 import convert_e2m3x2_to_fp16, convert_e3m2x2_to_fp16
from .if4 import convert_if4_to_fp32
from .if6 import convert_if6_e2m3_to_fp16, convert_if6_e3m2_to_fp16
from .int3 import convert_int3_to_fp16
from .int4 import convert_int4_to_fp32
from .int6 import convert_int6_to_fp16


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
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP6_E2M3:
        dequantized_values = convert_e2m3x2_to_fp16(
            values,
            MAJOR_COMPUTE_CAPABILITY=MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP6_E3M2:
        dequantized_values = convert_e3m2x2_to_fp16(
            values,
            MAJOR_COMPUTE_CAPABILITY=MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF4:
        dequantized_values = convert_if4_to_fp32(
            values,
            scale_factors,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            RETURN_FP=False,
            MAJOR_COMPUTE_CAPABILITY=MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF6_E2M3:
        dequantized_values = convert_if6_e2m3_to_fp16(
            values,
            scale_factors,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            RETURN_FP=False,
            MAJOR_COMPUTE_CAPABILITY=MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_IF6_E3M2:
        dequantized_values = convert_if6_e3m2_to_fp16(
            values,
            scale_factors,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            RETURN_FP=False,
            MAJOR_COMPUTE_CAPABILITY=MAJOR_COMPUTE_CAPABILITY,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_INT4:
        dequantized_values = convert_int4_to_fp32(
            values,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_INT6:
        dequantized_values = convert_int6_to_fp16(
            values,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_FP3:
        dequantized_values = convert_e2m0_to_fp16(
            values,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )
    elif QUANTIZED_VALUE_TYPE == QUANTIZED_VALUE_TYPE_INT3:
        dequantized_values = convert_int3_to_fp16(
            values,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
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
    SCALE_FACTOR_TYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    values = values_desc.load(
        [
            tl.program_id(0) * BLOCK_SIZE_M,
            tl.program_id(1) * BLOCK_SIZE_N // QUANTIZED_VALUE_PACKING_FACTOR,
        ],
    )

    if SCALE_FACTOR_TYPE == SCALE_TYPE_NV_IF:
        scale_factors = scale_factors_desc.load(
            [
                (tl.program_id(0) * tl.num_programs(1) + tl.program_id(1))
                * SCALE_MEGABLOCK_SIZE,
            ],
        )

        scale_factors = scale_factors.reshape(32, 4, 4).permute(1, 0, 2).reshape(128, 4)
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
