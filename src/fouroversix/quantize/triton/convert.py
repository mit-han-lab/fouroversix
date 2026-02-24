import triton
import triton.language as tl


@triton.jit
def convert_to_packed_fp4_kernel(
    x_block_scaled_b1,
    x_block_scaled_b2,
    RETURN_HIGH_PRECISION_VALUES: tl.constexpr,
) -> None:
    abs_b1 = tl.abs(x_block_scaled_b1)
    abs_b2 = tl.abs(x_block_scaled_b2)

    sign_b1 = tl.where(x_block_scaled_b1 >= 0, 0, 1).to(tl.uint8)
    sign_b2 = tl.where(x_block_scaled_b2 >= 0, 0, 1).to(tl.uint8)

    value_b1 = tl.where(
        abs_b1 <= 0.25,
        0,
        tl.where(
            abs_b1 < 0.75,
            1,
            tl.where(
                abs_b1 <= 1.25,
                2,
                tl.where(
                    abs_b1 < 1.75,
                    3,
                    tl.where(
                        abs_b1 <= 2.5,
                        4,
                        tl.where(abs_b1 < 3.5, 5, tl.where(abs_b1 <= 5, 6, 7)),
                    ),
                ),
            ),
        ),
    ).to(tl.uint8)

    value_b2 = tl.where(
        abs_b2 <= 0.25,
        0,
        tl.where(
            abs_b2 < 0.75,
            1,
            tl.where(
                abs_b2 <= 1.25,
                2,
                tl.where(
                    abs_b2 < 1.75,
                    3,
                    tl.where(
                        abs_b2 <= 2.5,
                        4,
                        tl.where(abs_b2 < 3.5, 5, tl.where(abs_b2 <= 5, 6, 7)),
                    ),
                ),
            ),
        ),
    ).to(tl.uint8)

    result = (sign_b2 << 7) | (value_b2 << 4) | (sign_b1 << 3) | value_b1

    if RETURN_HIGH_PRECISION_VALUES:
        dequantized_b1 = tl.where(
            value_b1 == 0,
            0,
            tl.where(
                value_b1 == 1,
                0.5,
                tl.where(
                    value_b1 == 2,
                    1,
                    tl.where(
                        value_b1 == 3,
                        1.5,
                        tl.where(
                            value_b1 == 4,
                            2,
                            tl.where(
                                value_b1 == 5,
                                3,
                                tl.where(value_b1 == 6, 4, 6),
                            ),
                        ),
                    ),
                ),
            ),
        ).to(tl.float32) * tl.where(x_block_scaled_b1 >= 0, 1, -1)

        dequantized_b2 = tl.where(
            value_b2 == 0,
            0,
            tl.where(
                value_b2 == 1,
                0.5,
                tl.where(
                    value_b2 == 2,
                    1,
                    tl.where(
                        value_b2 == 3,
                        1.5,
                        tl.where(
                            value_b2 == 4,
                            2,
                            tl.where(
                                value_b2 == 5,
                                3,
                                tl.where(value_b2 == 6, 4, 6),
                            ),
                        ),
                    ),
                ),
            ),
        ).to(tl.float32) * tl.where(x_block_scaled_b2 >= 0, 1, -1)

        return result, tl.join(dequantized_b1, dequantized_b2).reshape(128, 4, 16)

    return result
