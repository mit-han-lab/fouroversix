import triton
import triton.language as tl

from .constants import (
    ROUND_STYLE_STOCHASTIC,
    ROUND_STYLE_STOCHASTIC_UNBIASED,
    SM_100,
    SM_110,
    SM_120,
)


@triton.jit
def _convert_to_unpacked_fp4_slow(x_block_scaled_b1, x_block_scaled_b2) -> None:
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
def _convert_packed_fp4_to_fp16_slow(
    sign_b1,
    value_b1,
    sign_b2,
    value_b2,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
) -> None:
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
    ).to(tl.float16) * tl.where(sign_b1 == 1, -1, 1)

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
    ).to(tl.float16) * tl.where(sign_b2 == 1, -1, 1)

    return tl.join(x_fp16_b1, x_fp16_b2).reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // SCALE_GROUP_SIZE,
        SCALE_GROUP_SIZE,
    )


@triton.jit
def _create_rbits_for_cvt_rs(
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
) -> tl.tensor:
    return tl.randint(
        tl.load(rbits_ptr),
        tl.arange(0, BLOCK_SIZE_M)[:, None, None] * BLOCK_SIZE_N
        + tl.arange(0, BLOCK_SIZE_N // SCALE_GROUP_SIZE)[None, :, None]
        * SCALE_GROUP_SIZE
        + tl.arange(0, SCALE_GROUP_SIZE // 2)[None, None, :],
    ).cast(tl.uint32, bitcast=True)


@triton.jit
def _add_fake_randomness_for_stochastic_rounding(
    x_block,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    SEED: tl.constexpr,
) -> tl.tensor:
    rbits = (
        tl.rand(
            SEED,
            tl.arange(0, BLOCK_SIZE_M)[:, None, None] * BLOCK_SIZE_N
            + tl.arange(0, BLOCK_SIZE_N // SCALE_GROUP_SIZE)[None, :, None]
            * SCALE_GROUP_SIZE
            + tl.arange(0, SCALE_GROUP_SIZE // 2)[None, None, :],
        )
        - 0.5
    )

    return tl.where(x_block < 0, -1, 1) * tl.abs(
        tl.where(
            tl.abs(x_block) < 2,
            tl.extra.cuda.libdevice.rint(2 * tl.abs(x_block) + rbits) / 2,
            tl.where(
                tl.abs(x_block) < 4,
                tl.extra.cuda.libdevice.rint(tl.abs(x_block) + rbits),
                tl.clamp(
                    2 * tl.extra.cuda.libdevice.rint(tl.abs(x_block) / 2 + rbits),
                    0,
                    6,
                ),
            ),
        ),
    )


@triton.jit
def _convert_to_e2m1x2_with_rtn(
    x_block_scaled_b1,
    x_block_scaled_b2,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    if (
        MAJOR_COMPUTE_CAPABILITY == SM_100
        or MAJOR_COMPUTE_CAPABILITY == SM_110
        or MAJOR_COMPUTE_CAPABILITY == SM_120
    ):
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

    sign_b1, value_b1, sign_b2, value_b2 = _convert_to_unpacked_fp4_slow(
        x_block_scaled_b1,
        x_block_scaled_b2,
    )

    return (sign_b2 << 7) | (value_b2 << 4) | (sign_b1 << 3) | value_b1


@triton.jit
def _convert_to_e2m1x2_with_sr(
    x_block_scaled_b1,
    x_block_scaled_b2,
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    if MAJOR_COMPUTE_CAPABILITY == SM_100:
        rbits = _create_rbits_for_cvt_rs(
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            SCALE_GROUP_SIZE,
        )

        return tl.inline_asm_elementwise(
            asm="""
                {
                .reg .b16 tmp0, tmp1;
                cvt.rs.satfinite.e2m1x4.f32 tmp0, {$6, $2, $5, $1}, $9;
                cvt.rs.satfinite.e2m1x4.f32 tmp1, {$8, $4, $7, $3}, $10;
                mov.b32 $0, {tmp0, tmp1};
                }
                """,
            constraints="=r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[x_block_scaled_b1, x_block_scaled_b2, rbits],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )

    # Add fake randomness since we can't use cvt.rs
    seed = tl.load(rbits_ptr)

    x_block_scaled_b1 = _add_fake_randomness_for_stochastic_rounding(
        x_block_scaled_b1,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SCALE_GROUP_SIZE,
        seed,
    )

    x_block_scaled_b2 = _add_fake_randomness_for_stochastic_rounding(
        x_block_scaled_b2,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SCALE_GROUP_SIZE,
        seed + 1,
    )

    return _convert_to_e2m1x2_with_rtn(
        x_block_scaled_b1,
        x_block_scaled_b2,
        MAJOR_COMPUTE_CAPABILITY,
    )


@triton.jit
def convert_to_e2m1x2(
    x_block_scaled,
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    x_block_scaled_b1, x_block_scaled_b2 = x_block_scaled.reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // SCALE_GROUP_SIZE,
        SCALE_GROUP_SIZE // 2,
        2,
    ).split()

    if (
        ROUND_STYLE == ROUND_STYLE_STOCHASTIC
        or ROUND_STYLE == ROUND_STYLE_STOCHASTIC_UNBIASED
    ):
        return _convert_to_e2m1x2_with_sr(
            x_block_scaled_b1,
            x_block_scaled_b2,
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            SCALE_GROUP_SIZE,
            MAJOR_COMPUTE_CAPABILITY,
        )

    return _convert_to_e2m1x2_with_rtn(
        x_block_scaled_b1,
        x_block_scaled_b2,
        MAJOR_COMPUTE_CAPABILITY,
    )


@triton.jit
def _convert_to_e2m1x2_and_quantized_fp16_with_rtn(
    x_block_scaled_b1,
    x_block_scaled_b2,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    if (
        MAJOR_COMPUTE_CAPABILITY == SM_100
        or MAJOR_COMPUTE_CAPABILITY == SM_110
        or MAJOR_COMPUTE_CAPABILITY == SM_120
    ):
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
        x_fp16 = tl.join(x_fp16x2_lo, x_fp16x2_hi).reshape(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
        )

        return x_e2m1, x_fp16

    sign_b1, value_b1, sign_b2, value_b2 = _convert_to_unpacked_fp4_slow(
        x_block_scaled_b1,
        x_block_scaled_b2,
    )

    x_e2m1 = (sign_b2 << 7) | (value_b2 << 4) | (sign_b1 << 3) | value_b1
    x_fp16 = _convert_packed_fp4_to_fp16_slow(
        sign_b1,
        value_b1,
        sign_b2,
        value_b2,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SCALE_GROUP_SIZE,
    )

    return x_e2m1, x_fp16


@triton.jit
def _convert_to_e2m1x2_and_quantized_fp16_with_sr(
    x_block_scaled_b1,
    x_block_scaled_b2,
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    if MAJOR_COMPUTE_CAPABILITY == SM_100:
        rbits = _create_rbits_for_cvt_rs(
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            SCALE_GROUP_SIZE,
        )

        (x_e2m1, x_fp16x2) = tl.inline_asm_elementwise(
            asm="""
                {
                .reg .b16 tmp0, tmp1;
                .reg .b8 byte0, byte1;

                cvt.rs.satfinite.e2m1x4.f32 tmp0, {$10, $6, $9, $5}, $13;
                mov.b16 {byte0, byte1}, tmp0;
                cvt.rn.f16x2.e2m1x2 $1, byte0;
                cvt.rn.f16x2.e2m1x2 $2, byte1;
                cvt.rs.satfinite.e2m1x4.f32 tmp1, {$12, $8, $11, $7}, $14;
                mov.b16 {byte0, byte1}, tmp1;
                cvt.rn.f16x2.e2m1x2 $3, byte0;
                cvt.rn.f16x2.e2m1x2 $4, byte1;
                mov.b32 $0, {tmp0, tmp1};
                }
                """,
            constraints="=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r",
            args=[x_block_scaled_b1, x_block_scaled_b2, rbits],
            dtype=(tl.uint8, tl.uint32),
            is_pure=True,
            pack=4,
        )

        x_fp16x2_lo = (x_fp16x2 & 0xFFFF).cast(tl.uint16).cast(tl.float16, bitcast=True)
        x_fp16x2_hi = (x_fp16x2 >> 16).cast(tl.uint16).cast(tl.float16, bitcast=True)
        x_fp16 = tl.join(x_fp16x2_lo, x_fp16x2_hi).reshape(
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // SCALE_GROUP_SIZE,
            SCALE_GROUP_SIZE,
        )

        return x_e2m1, x_fp16

    # Add fake randomness since we can't use cvt.rs
    seed = tl.load(rbits_ptr)

    x_block_scaled_b1 = _add_fake_randomness_for_stochastic_rounding(
        x_block_scaled_b1,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SCALE_GROUP_SIZE,
        seed,
    )

    x_block_scaled_b2 = _add_fake_randomness_for_stochastic_rounding(
        x_block_scaled_b2,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SCALE_GROUP_SIZE,
        seed + 1,
    )

    return _convert_to_e2m1x2_and_quantized_fp16_with_rtn(
        x_block_scaled_b1,
        x_block_scaled_b2,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SCALE_GROUP_SIZE,
        MAJOR_COMPUTE_CAPABILITY,
    )


@triton.jit
def convert_to_e2m1x2_and_quantized_fp16(
    x_block_scaled,
    rbits_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    x_block_scaled_b1, x_block_scaled_b2 = x_block_scaled.reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // SCALE_GROUP_SIZE,
        SCALE_GROUP_SIZE // 2,
        2,
    ).split()

    if (
        ROUND_STYLE == ROUND_STYLE_STOCHASTIC
        or ROUND_STYLE == ROUND_STYLE_STOCHASTIC_UNBIASED
    ):
        return _convert_to_e2m1x2_and_quantized_fp16_with_sr(
            x_block_scaled_b1,
            x_block_scaled_b2,
            rbits_ptr,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            SCALE_GROUP_SIZE,
            MAJOR_COMPUTE_CAPABILITY,
        )

    return _convert_to_e2m1x2_and_quantized_fp16_with_rtn(
        x_block_scaled_b1,
        x_block_scaled_b2,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SCALE_GROUP_SIZE,
        MAJOR_COMPUTE_CAPABILITY,
    )
