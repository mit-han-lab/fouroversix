import triton
import triton.language as tl

INT6_MAX_VALUE = tl.constexpr(31)


@triton.jit
def convert_to_int6_and_quantized_fp16(
    values,
    VALUE_EXPANSION_FACTOR: tl.constexpr,
) -> None:
    x_int_hp = tl.clamp(
        tl.extra.cuda.libdevice.rint(values * VALUE_EXPANSION_FACTOR),
        -INT6_MAX_VALUE,
        INT6_MAX_VALUE,
    ).to(tl.int8)

    sign = tl.where(x_int_hp < 0, 1, 0).to(tl.uint8)
    abs_val = tl.abs(x_int_hp).to(tl.uint8) & 0x1F

    return (sign << 5) | abs_val, x_int_hp.to(tl.float16)


@triton.jit
def convert_int6_to_fp16(
    values,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
) -> tl.tensor:
    sign = ((values >> 5) & 1).to(tl.float16)
    magnitude = (values & 0x1F).to(tl.float16)
    return tl.where(sign != 0, -magnitude, magnitude).reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
