import triton
import triton.language as tl

from .constants import (
    ROUND_STYLE_STOCHASTIC_UNBIASED,
    SCALE_TYPE_NV_IF,
    UNBIASED_SR_ADJUSTMENT_FACTOR_RCP,
)
from .dequantize import dequantize_to_fp16_kernel
from .fp8 import convert_e4m3_to_high_precision
from .if4 import IF4_GROUP_SIZE


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": block_size_m,
                "BLOCK_SIZE_N": block_size_n,
                "BLOCK_SIZE_K": block_size_k,
                "GROUP_SIZE_M": 8,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_size_m, block_size_n, block_size_k, num_warps, num_stages in [
            (256, 128, 128, 4, 3),
            (128, 128, 32, 4, 3),
            (64, 64, 32, 2, 4),
            (32, 32, 32, 2, 4),
        ]
    ],
    key=[
        "INPUT_QUANTIZED_VALUE_PACKING_FACTOR",
        "INPUT_SCALE_GROUP_SIZE",
        "OTHER_QUANTIZED_VALUE_PACKING_FACTOR",
        "OTHER_SCALE_GROUP_SIZE",
        "INTERMEDIATE_DTYPE",
        "M_N_K_BUCKET",
    ],
)
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
    INPUT_QUANTIZED_VALUE_TYPE: tl.constexpr,
    INPUT_QUANTIZED_VALUE_PACKING_FACTOR: tl.constexpr,
    INPUT_QUANTIZED_VALUE_MAX: tl.constexpr,
    INPUT_SCALE_TYPE: tl.constexpr,
    INPUT_SCALE_FACTOR_MAX: tl.constexpr,
    INPUT_SCALE_GROUP_SIZE: tl.constexpr,
    INPUT_ROUND_STYLE: tl.constexpr,
    OTHER_QUANTIZED_VALUE_TYPE: tl.constexpr,
    OTHER_QUANTIZED_VALUE_PACKING_FACTOR: tl.constexpr,
    OTHER_QUANTIZED_VALUE_MAX: tl.constexpr,
    OTHER_SCALE_TYPE: tl.constexpr,
    OTHER_SCALE_FACTOR_MAX: tl.constexpr,
    OTHER_SCALE_GROUP_SIZE: tl.constexpr,
    OTHER_ROUND_STYLE: tl.constexpr,
    INTERMEDIATE_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    MAJOR_COMPUTE_CAPABILITY: tl.constexpr,
    M_N_K_BUCKET: tl.constexpr,  # noqa: ARG001
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
    offs_ak_values = tl.arange(0, BLOCK_SIZE_K // INPUT_QUANTIZED_VALUE_PACKING_FACTOR)
    offs_ak_sf = tl.arange(0, BLOCK_SIZE_K // INPUT_SCALE_GROUP_SIZE)
    offs_bk_values = tl.arange(0, BLOCK_SIZE_K // OTHER_QUANTIZED_VALUE_PACKING_FACTOR)
    offs_bk_sf = tl.arange(0, BLOCK_SIZE_K // OTHER_SCALE_GROUP_SIZE)
    a_value_ptrs = (
        a_values_ptr
        + offs_am[:, None] * stride_am_values
        + offs_ak_values[None, :] * stride_ak_values
    )
    a_sf_ptrs = (
        a_sf_ptr + offs_am[:, None] * stride_am_sf + offs_ak_sf[None, :] * stride_ak_sf
    )
    b_value_ptrs = (
        b_values_ptr
        + offs_bn[:, None] * stride_bn_values
        + offs_bk_values[None, :] * stride_bk_values
    )
    b_sf_ptrs = (
        b_sf_ptr + offs_bn[:, None] * stride_bn_sf + offs_bk_sf[None, :] * stride_bk_sf
    )

    INPUT_SR_SCALE: tl.constexpr = (
        UNBIASED_SR_ADJUSTMENT_FACTOR_RCP
        if INPUT_ROUND_STYLE == ROUND_STYLE_STOCHASTIC_UNBIASED
        else 1
    )
    OTHER_SR_SCALE: tl.constexpr = (
        UNBIASED_SR_ADJUSTMENT_FACTOR_RCP
        if OTHER_ROUND_STYLE == ROUND_STYLE_STOCHASTIC_UNBIASED
        else 1
    )

    alpha = (
        tl.load(a_amax_ptr)
        * tl.load(b_amax_ptr)
        * INPUT_SR_SCALE
        * OTHER_SR_SCALE
        / (
            INPUT_QUANTIZED_VALUE_MAX
            * OTHER_QUANTIZED_VALUE_MAX
            * INPUT_SCALE_FACTOR_MAX
            * OTHER_SCALE_FACTOR_MAX
        )
    )
    accumulator = tl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N),
        dtype=tl.float32,
    )

    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        a_values = tl.load(
            a_value_ptrs,
            mask=offs_ak_values[None, :] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        a_sf = tl.load(
            a_sf_ptrs,
            mask=offs_ak_sf[None, :] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        b_values = tl.load(
            b_value_ptrs,
            mask=offs_bk_values[None, :] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        b_sf = tl.load(
            b_sf_ptrs,
            mask=offs_bk_sf[None, :] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )

        a_real_values = dequantize_to_fp16_kernel(
            a_values,
            a_sf,
            BLOCK_SIZE_M,
            BLOCK_SIZE_K,
            QUANTIZED_VALUE_TYPE=INPUT_QUANTIZED_VALUE_TYPE,
            MAJOR_COMPUTE_CAPABILITY=MAJOR_COMPUTE_CAPABILITY,
        )
        b_real_values = dequantize_to_fp16_kernel(
            b_values,
            b_sf,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            QUANTIZED_VALUE_TYPE=OTHER_QUANTIZED_VALUE_TYPE,
            MAJOR_COMPUTE_CAPABILITY=MAJOR_COMPUTE_CAPABILITY,
        )

        a_sf = convert_e4m3_to_high_precision(
            a_sf,
            INTERMEDIATE_DTYPE,
            MAJOR_COMPUTE_CAPABILITY,
        )
        b_sf = convert_e4m3_to_high_precision(
            b_sf,
            INTERMEDIATE_DTYPE,
            MAJOR_COMPUTE_CAPABILITY,
        )

        if INPUT_SCALE_TYPE == SCALE_TYPE_NV_IF:
            a_sf = tl.where(a_sf < 0.0, -a_sf, a_sf)

        if OTHER_SCALE_TYPE == SCALE_TYPE_NV_IF:
            b_sf = tl.where(b_sf < 0.0, -b_sf, b_sf)

        a_real_values = (
            a_real_values.to(INTERMEDIATE_DTYPE).reshape(
                BLOCK_SIZE_M,
                BLOCK_SIZE_K // IF4_GROUP_SIZE,
                IF4_GROUP_SIZE,
            )
            * a_sf.expand_dims(2)
        ).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)

        b_real_values = (
            b_real_values.to(INTERMEDIATE_DTYPE).reshape(
                BLOCK_SIZE_N,
                BLOCK_SIZE_K // IF4_GROUP_SIZE,
                IF4_GROUP_SIZE,
            )
            * b_sf.expand_dims(2)
        ).reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)

        accumulator += tl.dot(a_real_values, b_real_values.T)

        a_value_ptrs += BLOCK_SIZE_K // 2 * stride_ak_values
        a_sf_ptrs += BLOCK_SIZE_K // IF4_GROUP_SIZE * stride_ak_sf
        b_value_ptrs += BLOCK_SIZE_K // 2 * stride_bk_values
        b_sf_ptrs += BLOCK_SIZE_K // IF4_GROUP_SIZE * stride_bk_sf

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, (accumulator * alpha).to(OUT_DTYPE), mask=c_mask)
