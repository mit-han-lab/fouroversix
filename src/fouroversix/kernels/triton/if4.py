import triton
import triton.language as tl
from fouroversix.utils import DataType

from .constants import IF4_INT_EXPANSION_FACTOR, SM_100, SM_110, SM_120

IF4_GROUP_SIZE = tl.constexpr(DataType.if4.block_size)


@triton.jit
def convert_if4_to_fp32(
    values,
    scale_factors,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    RETURN_FP: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
) -> None:
    if (
        MAJOR_COMPUTE_CAPABILITY == SM_100
        or MAJOR_COMPUTE_CAPABILITY == SM_110
        or MAJOR_COMPUTE_CAPABILITY == SM_120
    ):
        (fp_values_1, fp_values_2) = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b8 byte0, byte1, byte2, byte3;
            .reg .b16 tmp0, tmp1;
            .reg .b32 result;

            mov.b32 {byte0, byte1, byte2, byte3}, $8;

            cvt.rn.f16x2.e2m1x2 result, byte0;
            mov.b32 {tmp0, tmp1}, result;
            cvt.f32.f16 $0, tmp0;
            cvt.f32.f16 $4, tmp1;

            cvt.rn.f16x2.e2m1x2 result, byte1;
            mov.b32 {tmp0, tmp1}, result;
            cvt.f32.f16 $1, tmp0;
            cvt.f32.f16 $5, tmp1;

            cvt.rn.f16x2.e2m1x2 result, byte2;
            mov.b32 {tmp0, tmp1}, result;
            cvt.f32.f16 $2, tmp0;
            cvt.f32.f16 $6, tmp1;

            cvt.rn.f16x2.e2m1x2 result, byte3;
            mov.b32 {tmp0, tmp1}, result;
            cvt.f32.f16 $3, tmp0;
            cvt.f32.f16 $7, tmp1;
            }
            """,
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,r",
            args=[values],
            dtype=(tl.float32, tl.float32),
            is_pure=True,
            pack=4,
        )
    else:
        sign_1 = tl.where(((values >> 3) & 1) == 1, -1, 1)
        sign_2 = tl.where(((values >> 7) & 1) == 1, -1, 1)

        value_1 = values & 0x7
        value_2 = (values >> 4) & 0x7

        dequantized_value_1 = tl.where(
            value_1 == 0,
            0,
            tl.where(
                value_1 == 1,
                0.5,
                tl.where(
                    value_1 == 2,
                    1,
                    tl.where(
                        value_1 == 3,
                        1.5,
                        tl.where(
                            value_1 == 4,
                            2,
                            tl.where(
                                value_1 == 5,
                                3,
                                tl.where(value_1 == 6, 4, 6),
                            ),
                        ),
                    ),
                ),
            ),
        ).to(tl.float32)

        dequantized_value_2 = tl.where(
            value_2 == 0,
            0,
            tl.where(
                value_2 == 1,
                0.5,
                tl.where(
                    value_2 == 2,
                    1,
                    tl.where(
                        value_2 == 3,
                        1.5,
                        tl.where(
                            value_2 == 4,
                            2,
                            tl.where(
                                value_2 == 5,
                                3,
                                tl.where(value_2 == 6, 4, 6),
                            ),
                        ),
                    ),
                ),
            ),
        ).to(tl.float32)

        fp_values_1 = dequantized_value_1 * sign_1
        fp_values_2 = dequantized_value_2 * sign_2

    fp_values = tl.join(fp_values_1, fp_values_2).reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // IF4_GROUP_SIZE,
        IF4_GROUP_SIZE,
    )

    if RETURN_FP:
        return fp_values.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)

    (int_values_1, int_values_2) = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 byte0, byte1, byte2, byte3;

        mov.b32 {byte0, byte1, byte2, byte3}, $8;
        .reg .b32 tmp0, tmp1, tmp2, tmp3;

        cvt.u32.u8 tmp0, byte0;
        cvt.u32.u8 tmp1, byte1;
        cvt.u32.u8 tmp2, byte2;
        cvt.u32.u8 tmp3, byte3;

        bfe.s32 $0, tmp0, 0, 4;
        cvt.rn.f32.s32 $0, $0;

        bfe.s32 $4, tmp0, 4, 4;
        cvt.rn.f32.s32 $4, $4;

        bfe.s32 $1, tmp1, 0, 4;
        cvt.rn.f32.s32 $1, $1;

        bfe.s32 $5, tmp1, 4, 4;
        cvt.rn.f32.s32 $5, $5;

        bfe.s32 $2, tmp2, 0, 4;
        cvt.rn.f32.s32 $2, $2;

        bfe.s32 $6, tmp2, 4, 4;
        cvt.rn.f32.s32 $6, $6;

        bfe.s32 $3, tmp3, 0, 4;
        cvt.rn.f32.s32 $3, $3;

        bfe.s32 $7, tmp3, 4, 4;
        cvt.rn.f32.s32 $7, $7;
        }
        """,
        constraints="=r,=r,=r,=r,=r,=r,=r,=r,r",
        args=[values],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=4,
    )

    int_values = tl.join(int_values_1, int_values_2).reshape(
        BLOCK_SIZE_M,
        BLOCK_SIZE_N // IF4_GROUP_SIZE,
        IF4_GROUP_SIZE,
    )

    real_values = tl.where(
        (scale_factors.to(tl.uint8, bitcast=True) >= 128).expand_dims(2),
        int_values * IF4_INT_EXPANSION_FACTOR,
        fp_values,
    )

    return real_values.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)
