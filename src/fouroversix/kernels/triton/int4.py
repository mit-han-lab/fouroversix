import triton
import triton.language as tl

from .constants import ROUND_STYLE_STOCHASTIC


@triton.jit
def convert_to_int4(
    x_block_scaled,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROUND_STYLE: tl.constexpr,
) -> tl.tensor:
    if ROUND_STYLE == ROUND_STYLE_STOCHASTIC:
        rbits = (
            tl.rand(
                2,
                tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N
                + tl.arange(0, BLOCK_SIZE_N)[None, :],
            )
            - 0.5
        )

        x_block_scaled = tl.where(x_block_scaled < 0, -1, 1) * tl.abs(
            tl.abs(x_block_scaled) + rbits,
        )

    x_int_hp = tl.extra.cuda.libdevice.rint(tl.clamp(x_block_scaled, -7, 7))
    (x_int_b1, x_int_b2) = (
        (x_int_hp.to(tl.int8).to(tl.uint8, bitcast=True) & 0xF)
        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)
        .split()
    )
    return (x_int_b2 << 4) | x_int_b1


@triton.jit
def convert_int4_to_fp32(
    values,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
) -> None:
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

    return tl.join(int_values_1, int_values_2).reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)
