import triton
import triton.language as tl


@triton.jit
def _convert_to_unpacked_fp4(x_block_scaled_b1, x_block_scaled_b2) -> None:
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

    return sign_b1, value_b1, sign_b2, value_b2


@triton.jit
def convert_to_e2m1x2(
    x_block_scaled_b1,
    x_block_scaled_b2,
    USE_BLACKWELL_CVT_RN_INSTRUCTIONS: tl.constexpr,
) -> None:
    if USE_BLACKWELL_CVT_RN_INSTRUCTIONS:
        return tl.inline_asm_elementwise(
            asm="""
                {
                .reg .b8 byte0, byte1, byte2, byte3;
                cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
                mov.b32 $0, {byte0, byte1, byte2, byte3};
                }
                """,
            constraints="=r,r,r,r,r,r,r,r,r",
            args=[x_block_scaled_b1, x_block_scaled_b2],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )

    sign_b1, value_b1, sign_b2, value_b2 = _convert_to_unpacked_fp4(
        x_block_scaled_b1,
        x_block_scaled_b2,
    )

    return (sign_b2 << 7) | (value_b2 << 4) | (sign_b1 << 3) | value_b1


@triton.jit
def convert_to_e2m1x2_and_quantized_fp16(
    x_block_scaled_b1,
    x_block_scaled_b2,
    USE_BLACKWELL_CVT_RN_INSTRUCTIONS: tl.constexpr,
) -> None:
    if USE_BLACKWELL_CVT_RN_INSTRUCTIONS:
        (x_e2m1, x_fp16x2) = tl.inline_asm_elementwise(
            asm="""
                {
                .reg .b8 byte0, byte1, byte2, byte3;
                cvt.rn.satfinite.e2m1x2.f32 byte0, $9, $5;
                cvt.rn.f16x2.e2m1x2 $1, byte0;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $10, $6;
                cvt.rn.f16x2.e2m1x2 $2, byte1;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $11, $7;
                cvt.rn.f16x2.e2m1x2 $3, byte2;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $12, $8;
                cvt.rn.f16x2.e2m1x2 $4, byte3;
                mov.b32 $0, {byte0, byte1, byte2, byte3};
                }
                """,
            constraints="=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r",
            args=[x_block_scaled_b1, x_block_scaled_b2],
            dtype=(tl.uint8, tl.uint32),
            is_pure=True,
            pack=4,
        )

        x_fp16x2_lo = (x_fp16x2 & 0xFFFF).cast(tl.uint16).cast(tl.float16, bitcast=True)
        x_fp16x2_hi = (x_fp16x2 >> 16).cast(tl.uint16).cast(tl.float16, bitcast=True)
        x_fp16 = tl.join(x_fp16x2_lo, x_fp16x2_hi).reshape(128, 4, 16)

        return x_e2m1, x_fp16

    sign_b1, value_b1, sign_b2, value_b2 = _convert_to_unpacked_fp4(
        x_block_scaled_b1,
        x_block_scaled_b2,
    )

    x_packed = (sign_b2 << 7) | (value_b2 << 4) | (sign_b1 << 3) | value_b1

    x_fp16_b1 = tl.where(
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
    ).to(tl.float16) * tl.where(x_block_scaled_b1 >= 0, 1, -1)

    x_fp16_b2 = tl.where(
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
    ).to(tl.float16) * tl.where(x_block_scaled_b2 >= 0, 1, -1)

    return x_packed, tl.join(x_fp16_b1, x_fp16_b2).reshape(128, 4, 16)
