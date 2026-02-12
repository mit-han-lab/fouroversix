from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import ModelQuantizationConfig


class QuantizedLayer:
    """Base class for all quantized layers."""

    _registry: ClassVar[dict[type[nn.Module], type[nn.Module]]] = {}

    @classmethod
    def get_cls(
        cls,
        high_precision_cls: type[nn.Module],
    ) -> type[nn.Module] | None:
        """Get the quantized layer for a given high-precision layer."""
        return cls._registry.get(high_precision_cls)

    @classmethod
    def register(
        cls,
        high_precision_cls: type[nn.Module],
        *,
        replace_existing_layers: bool = False,
    ) -> Callable[[type[nn.Module]], type[nn.Module]]:
        """Register a new type of quantized layer."""

        if high_precision_cls in cls._registry and not replace_existing_layers:
            msg = f"High-precision layer {high_precision_cls} is already registered."
            raise ValueError(msg)

        modules_to_delete = []

        for module_cls in cls._registry:
            if issubclass(high_precision_cls, module_cls):
                if replace_existing_layers:
                    modules_to_delete.append(module_cls)
                else:
                    msg = (
                        f"High-precision layer {high_precision_cls} is a subclass of "
                        f"{module_cls}, which is already registered."
                    )
                    raise TypeError(msg)

        for module_cls in modules_to_delete:
            del cls._registry[module_cls]

        def inner_wrapper(
            wrapped_cls: type[nn.Module],
        ) -> type[nn.Module]:
            cls._registry[high_precision_cls] = wrapped_cls
            return wrapped_cls

        return inner_wrapper


def quantize_model(
    model: nn.Module,
    config: ModelQuantizationConfig,
    **kwargs: dict[str, Any],
) -> None:
    for name, module in model.named_modules():
        if (
            name == ""
            or name in config.exclude_layers
            or not isinstance(module, nn.Module)
        ):
            continue

        layer_cls = QuantizedLayer.get_cls(type(module))

        if layer_cls is None:
            continue

        layer = layer_cls(module, config.get_layer_config(name), **kwargs)
        model.set_submodule(name, layer)
