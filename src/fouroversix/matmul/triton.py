import torch
import triton
import triton.language as tl
from .backend import MatmulBackendBase
from fouroversix.quantize import QuantizedTensor
from fouroversix.utils import SM_100, SM_120, DataType

DEVICE_IS_BLACKWELL = tl.constexpr(
    torch.cuda.get_device_capability()[0] in {SM_100, SM_120}
)
Q_BLOCK_SIZE = tl.constexpr(16)


@triton.jit
def dequantize_if4_kernel(
    values,
    scale_factors,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
) -> None:
    if DEVICE_IS_BLACKWELL:
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
        BLOCK_SIZE_M, BLOCK_SIZE_N // Q_BLOCK_SIZE, Q_BLOCK_SIZE
    )

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
        BLOCK_SIZE_N // Q_BLOCK_SIZE,
        Q_BLOCK_SIZE,
    )

    real_values = tl.where(
        (scale_factors < 0.0).expand_dims(2),
        int_values,
        fp_values,
    )

    return real_values.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)


@triton.jit
def matmul_kernel(
    a_values_ptr,
    a_sf_ptr,
    a_amax_ptr,
    b_values_ptr,
    b_sf_ptr,
    b_amax_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am_values: tl.constexpr,
    stride_am_sf: tl.constexpr,
    stride_ak_values: tl.constexpr,
    stride_ak_sf: tl.constexpr,
    stride_bn_values: tl.constexpr,
    stride_bn_sf: tl.constexpr,
    stride_bk_values: tl.constexpr,
    stride_bk_sf: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k_values = tl.arange(0, BLOCK_SIZE_K // 2)
    offs_k_sf = tl.arange(0, BLOCK_SIZE_K // Q_BLOCK_SIZE)
    a_value_ptrs = (
        a_values_ptr
        + offs_am[:, None] * stride_am_values
        + offs_k_values[None, :] * stride_ak_values
    )
    a_sf_ptrs = (
        a_sf_ptr + offs_am[:, None] * stride_am_sf + offs_k_sf[None, :] * stride_ak_sf
    )
    b_value_ptrs = (
        b_values_ptr
        + offs_bn[:, None] * stride_bn_values
        + offs_k_values[None, :] * stride_bk_values
    )
    b_sf_ptrs = (
        b_sf_ptr + offs_bn[:, None] * stride_bn_sf + offs_k_sf[None, :] * stride_bk_sf
    )

    alpha = tl.load(a_amax_ptr) * tl.load(b_amax_ptr) / (6 * 6 * 448 * 448)
    accumulator = tl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N),
        dtype=tl.float32,
    )

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a_values = tl.load(
            a_value_ptrs,
            mask=offs_k_values[None, :] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        a_sf = tl.load(
            a_sf_ptrs,
            mask=offs_k_sf[None, :] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        b_values = tl.load(
            b_value_ptrs,
            mask=offs_k_values[None, :] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        b_sf = tl.load(
            b_sf_ptrs,
            mask=offs_k_sf[None, :] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )

        a_real_values = dequantize_if4_kernel(
            a_values,
            a_sf,
            BLOCK_SIZE_M,
            BLOCK_SIZE_K,
        )
        b_real_values = dequantize_if4_kernel(
            b_values,
            b_sf,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )

        a_sf = tl.where(
            a_sf < 0.0,
            -a_sf.cast(tl.float32) * 6 / 7,
            a_sf,
        )
        b_sf = tl.where(
            b_sf < 0.0,
            -b_sf.cast(tl.float32) * 6 / 7,
            b_sf,
        )

        a_real_values = (
            a_real_values.reshape(
                BLOCK_SIZE_M,
                BLOCK_SIZE_K // Q_BLOCK_SIZE,
                Q_BLOCK_SIZE,
            )
            * a_sf.expand_dims(2)
        ).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)

        b_real_values = (
            b_real_values.reshape(
                BLOCK_SIZE_N,
                BLOCK_SIZE_K // Q_BLOCK_SIZE,
                Q_BLOCK_SIZE,
            )
            * b_sf.expand_dims(2)
        ).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)

        accumulator += tl.dot(
            a_real_values.to(tl.float16), b_real_values.to(tl.float16).T
        )

        a_value_ptrs += BLOCK_SIZE_K // 2 * stride_ak_values
        a_sf_ptrs += BLOCK_SIZE_K // Q_BLOCK_SIZE * stride_ak_sf
        b_value_ptrs += BLOCK_SIZE_K // 2 * stride_bk_values
        b_sf_ptrs += BLOCK_SIZE_K // Q_BLOCK_SIZE * stride_bk_sf

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, (accumulator * alpha).cast(tl.bfloat16), mask=c_mask)


class TritonMatmulBackend(MatmulBackendBase):
    """
    The Triton matrix multiplication backend. Uses Triton kernels to perform fast
    FP4 matrix multiplication. Requires a Blackwell GPU.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the Triton backend is available on the current machine."""
        return True

    @classmethod
    def is_supported(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
    ) -> bool:
        """Return True if the Triton backend supports the given inputs and output data type."""

        if not super().is_supported(input, other, out_dtype=out_dtype):
            return False

        return input.dtype == DataType.if4 and input.device.type == "cuda"

    @classmethod
    def fp4_matmul(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
    ) -> torch.Tensor:
        """Perform a matrix multiplication (`a @ b.T`) between two quantized tensors using the Triton backend."""
        M = input.original_shape[0]
        N = other.original_shape[0]
        K = input.original_shape[1]

        input_scale_factors = input.scale_factors.reshape(
            input.padded_shape[0], input.padded_shape[1] // input.dtype.block_size()
        )
        other_scale_factors = other.scale_factors.reshape(
            other.padded_shape[0], other.padded_shape[1] // other.dtype.block_size()
        )
        output = torch.empty(
            (M, N),
            device=input.values.device,
            dtype=out_dtype.torch_dtype(),
        )

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )  # noqa: E731

        matmul_kernel[grid](
            input.values,
            input_scale_factors,
            input.amax,
            other.values,
            other_scale_factors,
            other.amax,
            output,
            M,
            N,
            K,
            input.values.stride(0),
            input_scale_factors.stride(0),
            input.values.stride(1),
            input_scale_factors.stride(1),
            other.values.stride(0),
            other_scale_factors.stride(0),
            other.values.stride(1),
            other_scale_factors.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=6,
        )

        return output
