from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .backend import MatmulBackend, QuantizeBackend
from .utils import AdaptiveBlockScalingRule, DataType, FP4Format, RoundStyle

if TYPE_CHECKING:
    import torch


def fp4_matmul(
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    *,
    backend: MatmulBackend | None = None,
    a_e2m1: torch.Tensor | None = None,
    a_sf: torch.Tensor | None = None,
    a_normconst: torch.Tensor | None = None,
    b_e2m1: torch.Tensor | None = None,
    b_sf: torch.Tensor | None = None,
    b_normconst: torch.Tensor | None = None,
    a_quantize_kwargs: dict[str, Any] | None = None,
    b_quantize_kwargs: dict[str, Any] | None = None,
    fp4_format: FP4Format = FP4Format.nvfp4,
    out_dtype: DataType = DataType.bfloat16,
    out_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    if a is None and (a_e2m1 is None or a_sf is None):
        msg = "If a is None, a_e2m1 and a_sf must be provided"
        raise ValueError(msg)

    if b is None and (b_e2m1 is None or b_sf is None):
        msg = "If b is None, b_e2m1 and b_sf must be provided"
        raise ValueError(msg)

    if a_quantize_kwargs is None:
        a_quantize_kwargs = {}

    if b_quantize_kwargs is None:
        b_quantize_kwargs = {}

    if a_e2m1 is None or a_sf is None:
        a_e2m1, a_sf, a_normconst = quantize_to_fp4(a, **a_quantize_kwargs)

    if b_e2m1 is None or b_sf is None:
        b_e2m1, b_sf, b_normconst = quantize_to_fp4(b, **b_quantize_kwargs)

    kwargs = {
        "fp4_format": fp4_format,
        "out_dtype": out_dtype,
        "out_shape": out_shape,
    }

    if backend is None:
        backend = MatmulBackend.auto_select()
    elif not backend.is_available():
        msg = f"Backend {backend} is not available"
        raise ValueError(msg)

    return backend.fp4_matmul(
        a_e2m1,
        a_sf,
        a_normconst,
        b_e2m1,
        b_sf,
        b_normconst,
        **kwargs,
    )


def quantize_to_fp4(
    x: torch.Tensor,
    *,
    backend: QuantizeBackend | None = None,
    scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    block_scale_2d: bool = False,
    had: torch.Tensor | None = None,
    fp4_format: FP4Format = FP4Format.nvfp4,
    round_style: RoundStyle = RoundStyle.nearest,
    transpose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    kwargs = {
        "scale_rule": scale_rule,
        "block_scale_2d": block_scale_2d,
        "had": had,
        "fp4_format": fp4_format,
        "round_style": round_style,
        "transpose": transpose,
    }

    if backend is None:
        backend = QuantizeBackend.auto_select(x, **kwargs)
    elif not backend.is_supported(x, **kwargs):
        msg = f"Backend {backend} does not support the given parameters"
        raise ValueError(msg)

    return backend.quantize_to_fp4(x, **kwargs)
