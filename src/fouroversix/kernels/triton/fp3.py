import triton
import triton.language as tl

E2M0_MAX_VALUE = tl.constexpr(4)


@triton.jit
def convert_to_e2m0(
    values,
) -> tl.tensor:
    """
    Quantize scaled values to E2M0 (FP3) format.

    E2M0 representable values: 0, 1, 2, 4
    Stored as uint8: (sign << 2) | code, where code maps 0->0, 1->1, 2->2, 3->4.
    """
    abs_val = tl.abs(values)
    sign = tl.where(values < 0, 1, 0).to(tl.uint8)

    code = tl.where(
        abs_val <= 0.5,
        0,
        tl.where(
            abs_val <= 1.5,
            1,
            tl.where(abs_val <= 3.0, 2, 3),
        ),
    ).to(tl.uint8)

    return (sign << 2) | code


@triton.jit
def convert_to_e2m0_and_dequantized_fp16(
    values,
) -> tuple[tl.tensor, tl.tensor]:
    """Quantize to E2M0 and also return dequantized FP16 values."""
    abs_val = tl.abs(values)
    sign = tl.where(values < 0, 1, 0).to(tl.uint8)

    code = tl.where(
        abs_val <= 0.5,
        0,
        tl.where(
            abs_val <= 1.5,
            1,
            tl.where(abs_val <= 3.0, 2, 3),
        ),
    ).to(tl.uint8)

    packed = (sign << 2) | code

    fp_value = tl.where(
        code == 0,
        0,
        tl.where(code == 1, 1, tl.where(code == 2, 2, 4)),
    ).to(tl.float16) * tl.where(sign == 1, -1, 1).to(tl.float16)

    return packed, fp_value


@triton.jit
def convert_e2m0_to_fp16(
    values,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
) -> tl.tensor:
    """Dequantize E2M0 (FP3) values stored as uint8 back to FP16."""
    sign = ((values >> 2) & 1).to(tl.float16)
    code = (values & 0x3).to(tl.uint8)

    magnitude = tl.where(
        code == 0,
        0,
        tl.where(code == 1, 1, tl.where(code == 2, 2, 4)),
    ).to(tl.float16)

    return tl.where(sign != 0, -magnitude, magnitude).reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
