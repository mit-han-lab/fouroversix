from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import ModelQuantizationConfig


class QuantizedModule:
    """Base class for all quantized modules."""

    _registry: ClassVar[dict[type[nn.Module], type[nn.Module]]] = {}

    @classmethod
    def get_cls(
        cls,
        high_precision_cls: type[nn.Module],
    ) -> type[nn.Module] | None:
        """Get the quantized module for a given high-precision module."""
        return cls._registry.get(high_precision_cls)

    @classmethod
    def register(
        cls,
        high_precision_cls: type[nn.Module],
        *,
        replace_existing_modules: bool = False,
    ) -> Callable[[type[nn.Module]], type[nn.Module]]:
        """Register a new type of quantized module."""

        if high_precision_cls in cls._registry and not replace_existing_modules:
            msg = f"High-precision module {high_precision_cls} is already registered."
            raise ValueError(msg)

        modules_to_delete = []

        for module_cls in cls._registry:
            if issubclass(high_precision_cls, module_cls):
                if replace_existing_modules:
                    modules_to_delete.append(module_cls)
                else:
                    msg = (
                        f"High-precision module {high_precision_cls} is a subclass of "
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
    for module_name, module in model.named_modules():
        if (
            module_name == ""
            or module_name in config.modules_to_not_convert
            or not isinstance(module, nn.Module)
        ):
            continue

        module_cls = QuantizedModule.get_cls(type(module))

        if module_cls is None:
            continue

        quantized_module = module_cls(
            module,
            config.get_module_config(module_name),
            **kwargs,
        )
        model.set_submodule(module_name, quantized_module)
