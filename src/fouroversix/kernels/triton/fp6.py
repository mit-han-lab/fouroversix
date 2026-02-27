import triton
import triton.language as tl


@triton.jit  # noqa: RET503
def convert_to_e3m2x2_with_rtn(
    x_block_scaled,
    USE_BLACKWELL_CVT_RN_INSTRUCTIONS: tl.constexpr,
) -> None:
    if USE_BLACKWELL_CVT_RN_INSTRUCTIONS:
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 tmp0, tmp1;
            cvt.rn.satfinite.e3m2x2.f32 tmp0, $2, $1;
            cvt.rn.satfinite.e3m2x2.f32 tmp1, $4, $3;
            mov.b32 $0, {tmp0, tmp1};
            }
            """,
            constraints="=r,r,r,r,r",
            args=[x_block_scaled],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )


@triton.jit
def convert_to_e3m2x2(
    x_block_scaled,
    USE_BLACKWELL_CVT_RN_INSTRUCTIONS: tl.constexpr,
) -> None:
    return convert_to_e3m2x2_with_rtn(
        x_block_scaled,
        USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
    )


@triton.jit  # noqa: RET503
def convert_e3m2x2_to_fp16(
    x_block,
    USE_BLACKWELL_CVT_RN_INSTRUCTIONS: tl.constexpr,
) -> None:
    if USE_BLACKWELL_CVT_RN_INSTRUCTIONS:
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 tmp0, tmp1;
            mov.b32 {tmp0, tmp1}, $2;
            cvt.rn.f16x2.e3m2x2 $0, tmp0;
            cvt.rn.f16x2.e3m2x2 $1, tmp1;
            }
            """,
            constraints="=r,=r,r",
            args=[x_block],
            dtype=tl.float16,
            is_pure=True,
            pack=4,
        )
