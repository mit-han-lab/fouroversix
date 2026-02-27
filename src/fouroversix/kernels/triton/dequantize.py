import triton
import triton.language as tl

from .fp6 import convert_e3m2x2_to_fp16


@triton.jit
def dequantize_kernel(
    values_desc,
    output_desc,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    USE_BLACKWELL_CVT_RN_INSTRUCTIONS: tl.constexpr,
) -> None:
    values = values_desc.load(
        [tl.program_id(0) * BLOCK_SIZE_M, tl.program_id(1) * BLOCK_SIZE_N],
    )

    output_desc.store(
        [tl.program_id(0) * BLOCK_SIZE_M, tl.program_id(1) * BLOCK_SIZE_N],
        convert_e3m2x2_to_fp16(
            values,
            USE_BLACKWELL_CVT_RN_INSTRUCTIONS=USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
        ),
    )
