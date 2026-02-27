import triton
import triton.language as tl
from fouroversix.quantize.triton.kernels.if4 import (
    Q_BLOCK_SIZE,
    convert_from_if4_to_fp32,
)
from fouroversix.quantize.triton.kernels.quantize import DATA_TYPE_NVFP4


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
    A_E2M1_MAX_VALUE: tl.constexpr,
    B_E2M1_MAX_VALUE: tl.constexpr,
    A_E4M3_MAX_VALUE: tl.constexpr,
    B_E4M3_MAX_VALUE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DTYPE: tl.constexpr,
    INTERMEDIATE_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    USE_BLACKWELL_CVT_RN_INSTRUCTIONS: tl.constexpr,
) -> None:
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

    alpha = (
        tl.load(a_amax_ptr)
        * tl.load(b_amax_ptr)
        / (A_E2M1_MAX_VALUE * B_E2M1_MAX_VALUE * A_E4M3_MAX_VALUE * B_E4M3_MAX_VALUE)
    )
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

        a_real_values = convert_from_if4_to_fp32(
            a_values,
            a_sf,
            BLOCK_SIZE_M,
            BLOCK_SIZE_K,
            RETURN_FP=DTYPE == DATA_TYPE_NVFP4,
            USE_BLACKWELL_CVT_RN_INSTRUCTIONS=USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
        )
        b_real_values = convert_from_if4_to_fp32(
            b_values,
            b_sf,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            RETURN_FP=DTYPE == DATA_TYPE_NVFP4,
            USE_BLACKWELL_CVT_RN_INSTRUCTIONS=USE_BLACKWELL_CVT_RN_INSTRUCTIONS,
        )

        a_sf = tl.where(
            a_sf < 0.0,
            -a_sf.to(INTERMEDIATE_DTYPE),
            a_sf.to(INTERMEDIATE_DTYPE),
        )
        b_sf = tl.where(
            b_sf < 0.0,
            -b_sf.to(INTERMEDIATE_DTYPE),
            b_sf.to(INTERMEDIATE_DTYPE),
        )

        a_real_values = (
            a_real_values.to(INTERMEDIATE_DTYPE).reshape(
                BLOCK_SIZE_M,
                BLOCK_SIZE_K // Q_BLOCK_SIZE,
                Q_BLOCK_SIZE,
            )
            * a_sf.expand_dims(2)
        ).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)

        b_real_values = (
            b_real_values.to(INTERMEDIATE_DTYPE).reshape(
                BLOCK_SIZE_N,
                BLOCK_SIZE_K // Q_BLOCK_SIZE,
                Q_BLOCK_SIZE,
            )
            * b_sf.expand_dims(2)
        ).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)

        accumulator += tl.dot(a_real_values, b_real_values.T)

        a_value_ptrs += BLOCK_SIZE_K // 2 * stride_ak_values
        a_sf_ptrs += BLOCK_SIZE_K // Q_BLOCK_SIZE * stride_ak_sf
        b_value_ptrs += BLOCK_SIZE_K // 2 * stride_bk_values
        b_sf_ptrs += BLOCK_SIZE_K // Q_BLOCK_SIZE * stride_bk_sf

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, (accumulator * alpha).to(OUT_DTYPE), mask=c_mask)
