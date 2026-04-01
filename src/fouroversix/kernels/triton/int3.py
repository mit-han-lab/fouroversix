import triton
import triton.language as tl

INT3_MAX_VALUE = tl.constexpr(3)


@triton.jit
def convert_to_int3_and_quantized_fp16(
    values,
    VALUE_EXPANSION_FACTOR: tl.constexpr,
) -> tuple[tl.tensor, tl.tensor]:
    """Quantize to INT3 (sign-magnitude, 3 bits) and return both packed and FP16."""
    x_int_hp = tl.clamp(
        tl.extra.cuda.libdevice.rint(values * VALUE_EXPANSION_FACTOR),
        -INT3_MAX_VALUE,
        INT3_MAX_VALUE,
    ).to(tl.int8)

    sign = tl.where(x_int_hp < 0, 1, 0).to(tl.uint8)
    abs_val = tl.abs(x_int_hp).to(tl.uint8) & 0x3

    return (sign << 2) | abs_val, x_int_hp.to(tl.float16)


@triton.jit
def convert_int3_to_fp16(
    values,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
) -> tl.tensor:
    """Dequantize INT3 values stored as uint8 back to FP16."""
    sign = ((values >> 2) & 1).to(tl.float16)
    magnitude = (values & 0x3).to(tl.float16)
    return tl.where(sign != 0, -magnitude, magnitude).reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
