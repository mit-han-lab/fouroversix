"""Host-side launcher for the Triton FP4 nibble transpose kernel."""

from __future__ import annotations

import torch

from .transpose import transpose_packed_fp4_kernel

BLOCK_M = 16
BLOCK_N = 16


def transpose_packed_fp4(
    values: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """
    Transpose packed FP4 nibbles using a Triton kernel.

    Parameters
    ----------
    values : torch.Tensor
        Packed FP4 data with shape ``[rows, cols // 2]``, dtype ``uint8``.
    rows : int
        Number of rows (unpacked dimension).
    cols : int
        Number of columns (unpacked dimension).

    Returns
    -------
    torch.Tensor
        Transposed packed FP4 data with shape ``[cols, rows // 2]``, dtype ``uint8``.

    """
    if rows % BLOCK_M != 0 or cols % BLOCK_N != 0:
        msg = (
            f"rows ({rows}) and cols ({cols}) must be multiples of "
            f"BLOCK_M ({BLOCK_M}) and BLOCK_N ({BLOCK_N})"
        )
        raise ValueError(msg)

    dst = torch.empty(cols, rows // 2, dtype=torch.uint8, device=values.device)

    grid = (rows // BLOCK_M, cols // BLOCK_N)
    transpose_packed_fp4_kernel[grid](
        values,
        dst,
        rows,
        cols,
        values.stride(0),
        dst.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return dst
