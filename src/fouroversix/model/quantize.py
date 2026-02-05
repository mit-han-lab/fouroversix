from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn  # noqa: PLR0402

from .linear import FourOverSixLinear

if TYPE_CHECKING:
    from .config import FourOverSixLinearConfig


def quantize_model(
    model: nn.Module,
    *,
    exclude_layers: list[str] | None = None,
    linear_cls: type[FourOverSixLinear] | None = None,
    linear_config: FourOverSixLinearConfig | None = None,
) -> None:
    if exclude_layers is None:
        exclude_layers = ["lm_head"]

    if linear_cls is None:
        linear_cls = FourOverSixLinear

    for name, module in model.named_modules():
        if name in exclude_layers or not isinstance(module, nn.Linear):
            continue

        four_over_six_linear = linear_cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            config=linear_config,
        )

        four_over_six_linear.weight = module.weight
        four_over_six_linear.bias = module.bias
        four_over_six_linear.apply_ptq()

        model.set_submodule(name, four_over_six_linear)
