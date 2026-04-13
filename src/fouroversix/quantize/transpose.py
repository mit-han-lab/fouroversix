"""
Lossless transpose for 2D-block-scaled packed FP4 tensors.

When ``block_scale_2d=True`` is used during quantization, each 16x16 tile shares
a single scale factor.  Transposing the tile doesn't change the scale, so the
packed FP4 codes can be rearranged without re-quantization.
"""

from __future__ import annotations

import torch
from fouroversix.quantize.dequantize_utils import from_blocked
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.quantize.utils import to_blocked


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


def transpose_quantized_tensor(qt: QuantizedTensor) -> QuantizedTensor:
    """
    Transpose a ``QuantizedTensor`` that was quantized with ``block_scale_2d=True``.

    This is a lossless operation: the FP4 codes are rearranged (nibble shuffle)
    and the scale factor grid is transposed.  No re-quantization is performed.

    The returned ``QuantizedTensor`` has ``original_shape`` and ``padded_shape``
    with rows and columns swapped, and its scale factors are in the Blackwell
    blocked layout for the transposed geometry.

    Args:
        qt: A ``QuantizedTensor`` produced with ``block_scale_2d=True``.

    Returns:
        A new ``QuantizedTensor`` representing the transposed matrix.

    """
    rows, cols = qt.padded_shape
    block_size = qt.dtype.block_size

    codes = _unpack_nibbles_raw(qt.values)
    codes_2d = codes[:rows, :cols]
    codes_t = codes_2d.T.contiguous()
    values_t = _pack_nibbles_raw(codes_t)

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
        scales_t_blocked = to_blocked(scales_t_2d)
    else:
        scales_t_blocked = scales_t_2d.flatten()

    new_original_shape = (qt.original_shape[1], qt.original_shape[0])
    new_padded_shape = (cols, rows)

    return QuantizedTensor(
        values_t,
        scales_t_blocked,
        qt.amax,
        qt.dtype,
        new_original_shape,
        qt.scale_rule,
        qt.round_style,
        new_padded_shape,
        scale_factors_are_in_blackwell_layout=qt.scale_factors_are_in_blackwell_layout,
    )
