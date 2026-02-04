from __future__ import annotations

from typing import Any

import torch.nn as nn  # noqa: PLR0402
from fouroversix.utils import AdaptiveBlockScalingRule, FP4Format

from .fp4_linear import FP4Linear


def quantize_model(
    model: nn.Module,
    *,
    exclude_layers: list[str] | None = None,
    fp4_format: FP4Format = FP4Format.nvfp4,
    activation_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    weight_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    w_scale_2d: bool = False,
    linear_cls: type[FP4Linear] | None = None,
    **kwargs: dict[str, Any],
) -> None:
    if exclude_layers is None:
        exclude_layers = ["lm_head"]

    if linear_cls is None:
        linear_cls = FP4Linear

    for name, module in model.named_modules():
        if name in exclude_layers or not isinstance(module, nn.Linear):
            continue

        fp4_linear = linear_cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            fp4_format=fp4_format,
            activation_scale_rule=activation_scale_rule,
            weight_scale_rule=weight_scale_rule,
            w_scale_2d=w_scale_2d,
            **kwargs,
        )

        fp4_linear.weight = module.weight
        fp4_linear.bias = module.bias
        fp4_linear.apply_ptq()

        model.set_submodule(name, fp4_linear)
