"""
Triton kernel for fused unpack-transpose-repack of packed FP4 nibbles.

Operates on 2D-block-scaled NVFP4 tensors where each 16x16 tile shares a
single scale factor, making nibble transpose lossless.

The kernel decomposes the full BLOCK_M x BLOCK_N code transpose into four
(BLOCK_M//2 x BLOCK_N//2) sub-matrix transposes by separating even/odd
source rows and low/high nibbles, avoiding any intermediate full-size buffer.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def transpose_packed_fp4_kernel(
    src_ptr,
    dst_ptr,
    rows,  # noqa: ARG001 -- passed by grid config, used implicitly via strides
    cols,  # noqa: ARG001
    src_stride_row,
    dst_stride_row,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """
    Transpose packed FP4 nibbles on GPU.

    Each program handles a ``BLOCK_M x BLOCK_N`` tile of unpacked codes.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_base = pid_m * BLOCK_M
    packed_col_base = pid_n * (BLOCK_N // 2)

    packed_col_offsets = tl.arange(0, BLOCK_N // 2)

    even_row_offsets = tl.arange(0, BLOCK_M // 2) * 2
    odd_row_offsets = even_row_offsets + 1

    src_even_ptrs = (
        src_ptr
        + (row_base + even_row_offsets[:, None]) * src_stride_row
        + (packed_col_base + packed_col_offsets[None, :])
    )
    src_odd_ptrs = (
        src_ptr
        + (row_base + odd_row_offsets[:, None]) * src_stride_row
        + (packed_col_base + packed_col_offsets[None, :])
    )
    packed_even = tl.load(src_even_ptrs)
    packed_odd = tl.load(src_odd_ptrs)

    low_even = packed_even & 0xF
    high_even = (packed_even >> 4) & 0xF
    low_odd = packed_odd & 0xF
    high_odd = (packed_odd >> 4) & 0xF

    low_even_t = tl.trans(low_even)
    low_odd_t = tl.trans(low_odd)
    high_even_t = tl.trans(high_even)
    high_odd_t = tl.trans(high_odd)

    result_even = (low_odd_t << 4) | low_even_t
    result_odd = (high_odd_t << 4) | high_even_t

    dst_packed_col_base = pid_m * (BLOCK_M // 2)
    dst_packed_col_offsets = tl.arange(0, BLOCK_M // 2)

    dst_even_row_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N // 2) * 2
    dst_odd_row_offsets = dst_even_row_offsets + 1

    dst_even_ptrs = (
        dst_ptr
        + dst_even_row_offsets[:, None] * dst_stride_row
        + (dst_packed_col_base + dst_packed_col_offsets[None, :])
    )
    dst_odd_ptrs = (
        dst_ptr
        + dst_odd_row_offsets[:, None] * dst_stride_row
        + (dst_packed_col_base + dst_packed_col_offsets[None, :])
    )

    tl.store(dst_even_ptrs, result_even)
    tl.store(dst_odd_ptrs, result_odd)
