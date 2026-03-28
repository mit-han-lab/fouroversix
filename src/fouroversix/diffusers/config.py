"""Quantization configuration for diffusers integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from diffusers.quantizers.quantization_config import QuantizationConfigMixin
from fouroversix.model.config import ModelQuantizationConfig, ModuleQuantizationConfig
from fouroversix.utils import (
    DataType,
    MatmulBackend,
    QuantizeBackend,
    ScaleRule,
)


class _QuantMethodStr(str):
    """A ``str`` subclass with a ``.value`` property for diffusers enum compat."""

    __slots__ = ()

    @property
    def value(self) -> str:
        return str(self)


FOUROVERSIX_QUANT_METHOD = _QuantMethodStr("fouroversix")


@dataclass
class FourOverSixConfig(QuantizationConfigMixin):
    """
    Configuration for NVFP4 quantization using Four Over Six with diffusers models.

    Args:
        dtype (str): The quantization data type. One of ``"nvfp4"`` or ``"mxfp4"``.
            Defaults to ``"nvfp4"``.
        scale_rule (str): The fallback block scale selection rule. One of ``"mse"``,
            ``"mae"``, ``"abs_max"``, ``"static_4"``, ``"static_6"``. Defaults to
            ``"mse"``.
        weight_scale_rule (str | None): Scale rule specifically for weights. Falls back
            to ``scale_rule`` if not provided.
        activation_scale_rule (str | None): Scale rule specifically for activations.
            Falls back to ``scale_rule`` if not provided.
        weight_scale_2d (bool): Whether to use 2D block scaling for weights. Should be
            ``True`` if the model is used for training.
        quantize_backend (str | None): Backend for quantization. One of ``"cuda"``,
            ``"pytorch"``, ``"triton"``, or ``None`` for automatic selection.
        matmul_backend (str | None): Backend for FP4 matrix multiplication. One of
            ``"cutlass"``, ``"pytorch"``, or ``None`` for automatic selection.
        output_dtype (str): Output data type for quantized layers. Defaults to
            ``"bfloat16"``.
        modules_to_not_convert (list[str] | None): Module names to skip during
            quantization. Defaults to ``None`` (quantize all eligible modules).

    """

    quant_method: str = field(default_factory=lambda: FOUROVERSIX_QUANT_METHOD)

    dtype: str = "nvfp4"
    scale_rule: str = "mse"
    weight_scale_rule: str | None = None
    activation_scale_rule: str | None = None
    weight_scale_2d: bool = False
    quantize_backend: str | None = None
    matmul_backend: str | None = None
    output_dtype: str = "bfloat16"
    modules_to_not_convert: list[str] | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        DataType(self.dtype)
        ScaleRule(self.scale_rule)

        if self.weight_scale_rule is not None:
            ScaleRule(self.weight_scale_rule)
        if self.activation_scale_rule is not None:
            ScaleRule(self.activation_scale_rule)
        if self.quantize_backend is not None:
            QuantizeBackend(self.quantize_backend)
        if self.matmul_backend is not None:
            MatmulBackend(self.matmul_backend)
        if self.output_dtype is not None:
            DataType(self.output_dtype)

    def to_model_quantization_config(self) -> ModelQuantizationConfig:
        """Convert to a fouroversix ``ModelQuantizationConfig``."""
        kwargs: dict[str, Any] = {
            "dtype": self.dtype,
            "scale_rule": self.scale_rule,
            "weight_scale_2d": self.weight_scale_2d,
            "output_dtype": self.output_dtype,
            "modules_to_not_convert": self.modules_to_not_convert or [],
        }
        if self.weight_scale_rule is not None:
            kwargs["weight_scale_rule"] = self.weight_scale_rule
        if self.activation_scale_rule is not None:
            kwargs["activation_scale_rule"] = self.activation_scale_rule
        if self.quantize_backend is not None:
            kwargs["quantize_backend"] = self.quantize_backend
        if self.matmul_backend is not None:
            kwargs["matmul_backend"] = self.matmul_backend
        return ModelQuantizationConfig(**kwargs)

    def to_module_quantization_config(self) -> ModuleQuantizationConfig:
        """Convert to a fouroversix ``ModuleQuantizationConfig``."""
        kwargs: dict[str, Any] = {
            "dtype": self.dtype,
            "scale_rule": self.scale_rule,
            "weight_scale_2d": self.weight_scale_2d,
            "output_dtype": self.output_dtype,
        }
        if self.weight_scale_rule is not None:
            kwargs["weight_scale_rule"] = self.weight_scale_rule
        if self.activation_scale_rule is not None:
            kwargs["activation_scale_rule"] = self.activation_scale_rule
        if self.quantize_backend is not None:
            kwargs["quantize_backend"] = self.quantize_backend
        if self.matmul_backend is not None:
            kwargs["matmul_backend"] = self.matmul_backend
        return ModuleQuantizationConfig(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary."""
        output = {
            "quant_method": self.quant_method,
            "dtype": self.dtype,
            "scale_rule": self.scale_rule,
            "weight_scale_2d": self.weight_scale_2d,
            "output_dtype": self.output_dtype,
        }
        if self.weight_scale_rule is not None:
            output["weight_scale_rule"] = self.weight_scale_rule
        if self.activation_scale_rule is not None:
            output["activation_scale_rule"] = self.activation_scale_rule
        if self.quantize_backend is not None:
            output["quantize_backend"] = self.quantize_backend
        if self.matmul_backend is not None:
            output["matmul_backend"] = self.matmul_backend
        if self.modules_to_not_convert is not None:
            output["modules_to_not_convert"] = self.modules_to_not_convert
        return output
