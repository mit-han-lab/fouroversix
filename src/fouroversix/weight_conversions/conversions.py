from typing import TYPE_CHECKING, ClassVar

from transformers import WeightConverter, PretrainedConfig

from collections.abc import Callable

class WeightConversions:
    """Base class for weight conversions for quantized models."""

    _registry: ClassVar[dict[type[PretrainedConfig], list[WeightConverter]]] = {}

    @classmethod
    def register(
        cls,
        model_config_type: type(PretrainedConfig)
    ) -> Callable[[type[PretrainedConfig]], list[WeightConverter]]:

        if model_config_type in cls._registry:
            msg = f"Model with config {model_config_type} is already registered."
            raise ValueError(msg)
        
        def inner_wrapper(
            wrapped_cls: type[PretrainedConfig],
        ) -> list[WeightConverter]:
            weight_conversions = wrapped_cls.get_weight_conversions()
            cls._registry[model_config_type] = weight_conversions
            return weight_conversions

        return inner_wrapper

    @classmethod
    def get_weight_conversions(
        cls,
        model_config_type: type(PretrainedConfig),
    ) -> list[WeightConverter]:
        return cls._registry.get(model_config_type, [])