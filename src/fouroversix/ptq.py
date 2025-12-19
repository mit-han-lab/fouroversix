from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn  # noqa: PLR0402

from .frontend import fp4_matmul, quantize_to_fp4
from .utils import AdaptiveBlockScalingRule, DataType, FP4Format

if TYPE_CHECKING:
    from collections.abc import Callable

    from .backend import MatmulBackend


def build_forward(
    *,
    device: str,
    dtype: DataType,
    fp4_format: FP4Format,
    a_scale_rule: AdaptiveBlockScalingRule,
    w_scale_rule: AdaptiveBlockScalingRule,
    w_scale_2d: bool,
    matmul_backend: MatmulBackend | None,
    a_quantize_kwargs: dict[str, Any],
    w_quantize_kwargs: dict[str, Any],
    **kwargs: dict[str, Any],  # noqa: ARG001
) -> Callable:
    def forward(
        self,  # noqa: ANN001
        input: tuple[torch.Tensor, ...],  # noqa: A002
    ) -> torch.Tensor:
        if not hasattr(self, "weight_e2m1"):
            self.weight_e2m1, self.weight_sf, self.weight_normconst = quantize_to_fp4(
                self.weight,
                scale_rule=w_scale_rule,
                block_scale_2d=w_scale_2d,
                fp4_format=fp4_format,
                **w_quantize_kwargs,
            )
            del self.weight

        out_n = (
            self.weight_e2m1.shape[0]
            if hasattr(self, "weight_e2m1") and self.weight_e2m1 is not None
            else self.weight.shape[0]
        )

        out = torch.empty(
            *input.shape[:-1],
            out_n,
            device=device,
            dtype=dtype.torch(),
        )

        # Slow bmm
        for i in range(input.shape[0]):
            out[i] = fp4_matmul(
                input[i],
                backend=matmul_backend,
                b_e2m1=self.weight_e2m1,
                b_sf=self.weight_sf,
                b_normconst=self.weight_normconst,
                fp4_format=fp4_format,
                out_dtype=dtype,
                out_shape=(input.shape[1], out_n),
                a_quantize_kwargs={
                    "scale_rule": a_scale_rule,
                    "fp4_format": fp4_format,
                    **a_quantize_kwargs,
                },
                b_quantize_kwargs={
                    "fp4_format": fp4_format,
                },
            )

        if hasattr(self, "bias") and self.bias is not None:
            out = out + self.bias

        return out

    return forward


def apply_ptq(
    model: nn.Module,
    *,
    exclude_layers: list[str] | None = None,
    device: str = "cuda",
    dtype: DataType = DataType.bfloat16,
    fp4_format: FP4Format = FP4Format.nvfp4,
    a_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    w_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    w_scale_2d: bool = False,
    matmul_backend: MatmulBackend | None = None,
    a_quantize_kwargs: dict[str, Any] | None = None,
    w_quantize_kwargs: dict[str, Any] | None = None,
    build_forward_fn: Callable | None = None,
    **kwargs: dict[str, Any],
) -> None:
    if exclude_layers is None:
        exclude_layers = ["lm_head"]

    if a_quantize_kwargs is None:
        a_quantize_kwargs = {}

    if w_quantize_kwargs is None:
        w_quantize_kwargs = {}

    if build_forward_fn is None:
        build_forward_fn = build_forward

    for name, module in model.named_modules():
        if name in exclude_layers or not isinstance(module, nn.Linear):
            continue

        module.forward = types.MethodType(
            build_forward_fn(
                device=device,
                dtype=dtype,
                fp4_format=fp4_format,
                a_scale_rule=a_scale_rule,
                w_scale_rule=w_scale_rule,
                w_scale_2d=w_scale_2d,
                matmul_backend=matmul_backend,
                a_quantize_kwargs=a_quantize_kwargs,
                w_quantize_kwargs=w_quantize_kwargs,
                **kwargs,
            ),
            module,
        )
