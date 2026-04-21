"""
Lossless transpose for 2D-block-scaled packed FP4 tensors.

When ``block_scale_2d=True`` is used during quantization, each 16x16 tile shares
a single scale factor.  Transposing the tile doesn't change the scale, so the
packed FP4 codes can be rearranged without re-quantization.

Three backends are supported for the nibble transpose (auto-selected):
  1. **Triton** -- preferred when available, JIT-compiled, no build step.
  2. **CUDA**   -- requires building the C++ extension.
  3. **PyTorch** -- pure-PyTorch fallback, always available.
"""

from __future__ import annotations

from enum import Enum, auto

import torch
from fouroversix.quantize.dequantize_utils import from_blocked
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.quantize.utils import to_blocked

_triton_available: bool
try:
    from fouroversix.kernels.triton.ops_transpose import (
        transpose_packed_fp4 as _triton_transpose,
    )

    _triton_available = True
except ImportError:
    _triton_available = False

_cuda_available: bool
try:
    _cuda_transpose = torch.ops.fouroversix.transpose_packed_fp4
    _cuda_available = True
except (AttributeError, RuntimeError):
    _cuda_available = False


class TransposeBackend(Enum):
    """Backend selection for the nibble transpose kernel."""

    auto_select = auto()
    triton = auto()
    cuda = auto()
    pytorch = auto()


# ---------------------------------------------------------------------------
# Pure-PyTorch reference helpers
# ---------------------------------------------------------------------------


def _unpack_nibbles_raw(x: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed FP4 to raw 4-bit codes (uint8, values 0..15)."""
    low = x & 0xF
    high = (x >> 4) & 0xF
    return torch.stack([low, high], dim=-1).reshape(x.shape[0], x.shape[1] * 2)


def _pack_nibbles_raw(codes: torch.Tensor) -> torch.Tensor:
    """Pack raw 4-bit codes (uint8) back into the fouroversix byte layout."""
    rows, cols = codes.shape
    codes = codes.reshape(rows, cols // 2, 2)
    low = codes[:, :, 0]
    high = codes[:, :, 1]
    return (high << 4) | low


def _transpose_values_pytorch(
    values: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Transpose packed FP4 nibbles using PyTorch ops."""
    codes = _unpack_nibbles_raw(values)
    codes_2d = codes[:rows, :cols]
    codes_t = codes_2d.T.contiguous()
    return _pack_nibbles_raw(codes_t)


# ---------------------------------------------------------------------------
# Scale grid transpose (small, always PyTorch)
# ---------------------------------------------------------------------------


def _transpose_scales(
    qt: QuantizedTensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Transpose the 2D block-scale grid and return it in the original layout."""
    block_size = qt.dtype.block_size

    scale_shape = (rows, cols // block_size)
    if qt.scale_factors_are_in_blackwell_layout:
        scales_2d = from_blocked(qt.scale_factors, scale_shape)
    else:
        scales_2d = qt.scale_factors.reshape(scale_shape)

    tile_rows = rows // block_size
    tile_cols = cols // block_size
    tile_grid = scales_2d.reshape(tile_rows, block_size, tile_cols).select(1, 0)
    tile_grid_t = tile_grid.T.contiguous()

    new_tile_rows, new_tile_cols = tile_grid_t.shape
    scales_t_2d = (
        tile_grid_t.unsqueeze(1)
        .expand(new_tile_rows, block_size, new_tile_cols)
        .reshape(cols, rows // block_size)
    )

    if qt.scale_factors_are_in_blackwell_layout:
        return to_blocked(scales_t_2d)
    return scales_t_2d.flatten()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transpose_quantized_tensor(
    qt: QuantizedTensor,
    backend: TransposeBackend = TransposeBackend.auto_select,
) -> QuantizedTensor:
    """
    Transpose a ``QuantizedTensor`` that was quantized with ``block_scale_2d=True``.

    This is a lossless operation: the FP4 codes are rearranged (nibble shuffle)
    and the scale factor grid is transposed.  No re-quantization is performed.

    Parameters
    ----------
    qt : QuantizedTensor
        A tensor produced with ``block_scale_2d=True``.
    backend : TransposeBackend
        Which kernel backend to use.  ``auto`` tries Triton > CUDA > PyTorch.

    Returns
    -------
    QuantizedTensor
        The transposed matrix.

    """
    rows, cols = qt.padded_shape

    values_t = _dispatch_transpose_values(qt.values, rows, cols, backend)

    scales_t = _transpose_scales(qt, rows, cols)

    new_original_shape = (qt.original_shape[1], qt.original_shape[0])
    new_padded_shape = (cols, rows)

    return QuantizedTensor(
        values_t,
        scales_t,
        qt.amax,
        qt.dtype,
        new_original_shape,
        qt.scale_rule,
        qt.round_style,
        new_padded_shape,
        scale_factors_are_in_blackwell_layout=qt.scale_factors_are_in_blackwell_layout,
    )


def _dispatch_transpose_values(
    values: torch.Tensor,
    rows: int,
    cols: int,
    backend: TransposeBackend,
) -> torch.Tensor:
    """Select the best available backend for the nibble transpose."""
    if backend == TransposeBackend.auto_select:
        if values.is_cuda and _triton_available:
            return _triton_transpose(values, rows, cols)
        if values.is_cuda and _cuda_available:
            return _cuda_transpose(values, rows, cols)
        return _transpose_values_pytorch(values, rows, cols)

    if backend == TransposeBackend.triton:
        if not _triton_available:
            msg = "Triton backend requested but triton is not importable"
            raise RuntimeError(msg)
        return _triton_transpose(values, rows, cols)

    if backend == TransposeBackend.cuda:
        if not _cuda_available:
            msg = (
                "CUDA backend requested but fouroversix._C is not built "
                "(install without SKIP_CUDA_BUILD=1)"
            )
            raise RuntimeError(msg)
        return _cuda_transpose(values, rows, cols)

    return _transpose_values_pytorch(values, rows, cols)
