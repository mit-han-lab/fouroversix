"""Utilities for diffusers integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch.nn as nn
from fouroversix.model.modules.linear import FourOverSixLinear

if TYPE_CHECKING:
    from fouroversix.model.config import ModuleQuantizationConfig

logger = logging.getLogger(__name__)


def replace_with_fouroversix_linear(
    model: nn.Module,
    config: ModuleQuantizationConfig,
    modules_to_not_convert: list[str] | None = None,
    current_key_name: str = "",
) -> nn.Module:
    """
    Replace ``nn.Linear`` modules in the model with ``FourOverSixLinear``.

    Recursively walks the module tree and replaces eligible ``nn.Linear`` modules,
    skipping any whose fully-qualified name appears in *modules_to_not_convert*.
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    for name, module in model.named_children():
        full_name = f"{current_key_name}.{name}" if current_key_name else name

        if full_name in modules_to_not_convert:
            continue

        if isinstance(module, nn.Linear):
            quantized = FourOverSixLinear(module, config)
            model._modules[name] = quantized  # noqa: SLF001
        else:
            replace_with_fouroversix_linear(
                module,
                config,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=full_name,
            )

    return model
