"""Diffusers quantizer implementation for Four Over Six."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from diffusers.quantizers.base import DiffusersQuantizer
from fouroversix.model.modules.linear import FourOverSixLinear

from .utils import replace_with_fouroversix_linear

if TYPE_CHECKING:
    from diffusers.models.modeling_utils import ModelMixin

    from .config import FourOverSixConfig

logger = logging.getLogger(__name__)

QUANTIZED_WEIGHT_SUFFIXES = (
    "quantized_weight_values",
    "quantized_weight_scale_factors",
    "quantized_weight_amax",
    "quantized_weight_metadata",
)


class FourOverSixQuantizer(DiffusersQuantizer):
    """
    Diffusers quantizer for NVFP4 quantization via Four Over Six.

    Supports two loading modes:

    - **Quantize on load** (``pre_quantized=False``): loads full-precision weights,
      then quantizes them to FP4 after weight loading.
    - **Pre-quantized** (``pre_quantized=True``): loads previously quantized packed
      ``uint8`` weights directly into ``FourOverSixLinear`` buffers.
    """

    use_keep_in_fp32_modules = False
    requires_calibration = False

    def __init__(self, quantization_config, **kwargs) -> None:  # noqa: ANN001, ANN003
        super().__init__(quantization_config, **kwargs)
        self.quantization_config: FourOverSixConfig = quantization_config

        if quantization_config.modules_to_not_convert is not None:
            self.modules_to_not_convert = list(
                quantization_config.modules_to_not_convert,
            )

    def validate_environment(
        self,
        *args: list[Any],  # noqa: ARG002
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """Check that CUDA is available."""
        if not torch.cuda.is_available():
            logger.warning(
                "No CUDA GPU detected. FourOverSix quantized models will fall back to "
                "the PyTorch reference backend, which is significantly slower.",
            )

    def update_torch_dtype(
        self,
        torch_dtype: torch.dtype,
    ) -> torch.dtype:
        """Ensure bfloat16 is used for the high-precision weights during loading."""

        if torch_dtype is None:
            torch_dtype = torch.bfloat16
            logger.info(
                "Setting torch_dtype to torch.bfloat16 for FourOverSix quantization",
            )

        return torch_dtype

    def update_device_map(
        self,
        device_map: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Return a device map, defaulting to the current CUDA device when no device_map
        is provided.
        """

        if device_map is None and torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}

        return device_map

    def _process_model_before_weight_loading(
        self,
        model: ModelMixin,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        """Replace ``nn.Linear`` modules with ``FourOverSixLinear``."""
        module_config = self.quantization_config.to_module_quantization_config()

        replace_with_fouroversix_linear(
            model,
            config=module_config,
            modules_to_not_convert=self.modules_to_not_convert,
        )

        if hasattr(model, "config"):
            model.config.quantization_config = self.quantization_config.to_dict()

    def _process_model_after_weight_loading(
        self,
        model: ModelMixin,
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        """
        Quantize weights after loading for non-pre-quantized models.

        For pre-quantized models this is a no-op since the packed buffers are
        already populated from the checkpoint.
        """
        if self.pre_quantized:
            return

        for _name, module in model.named_modules():
            if not isinstance(module, FourOverSixLinear):
                continue

            if module.weight is None or module.weight.device == torch.device("meta"):
                continue

            quantized_params = module.get_quantized_parameters(
                "weight",
                module.weight.data,
            )
            for buffer_name, buffer_value in quantized_params.items():
                setattr(module, buffer_name, buffer_value.to(module.weight.device))

            module.weight = None

    def check_if_quantized_param(
        self,
        model: ModelMixin,  # noqa: ARG002
        param_value: torch.Tensor,  # noqa: ARG002
        param_name: str,
        state_dict: dict[str, Any],  # noqa: ARG002
        **kwargs,  # noqa: ARG002, ANN003
    ) -> bool:
        """Return ``True`` if *param_name* is a fouroversix quantized buffer."""
        return any(param_name.endswith(suffix) for suffix in QUANTIZED_WEIGHT_SUFFIXES)

    def create_quantized_param(
        self,
        model: ModelMixin,
        param_value: torch.Tensor,
        param_name: str,
        target_device: torch.device,
        state_dict: dict[str, Any],  # noqa: ARG002
        unexpected_keys: list[str] | None = None,  # noqa: ARG002
        **kwargs,  # noqa: ARG002, ANN003
    ) -> None:
        """Load a pre-quantized buffer into the model."""
        module_name, tensor_name = param_name.rsplit(".", 1)

        parent = model
        for part in module_name.split("."):
            parent = getattr(parent, part)

        new_value = param_value.to(target_device)
        if hasattr(parent, tensor_name):
            setattr(parent, tensor_name, new_value)
        else:
            parent.register_buffer(tensor_name, new_value)

    def check_quantized_param_shape(
        self,
        param_name: str,  # noqa: ARG002
        current_param: torch.Tensor,
        loaded_param: torch.Tensor,
    ) -> bool:
        """Accept any shape match for quantized buffers."""
        return current_param.shape == loaded_param.shape

    def update_missing_keys(
        self,
        model: ModelMixin,
        missing_keys: list[str],
        prefix: str,  # noqa: ARG002
    ) -> list[str]:
        """
        Remove expected missing keys.

        When loading pre-quantized checkpoints the original ``weight`` parameter
        is absent and should not be reported as missing. Conversely, when
        quantizing on the fly the quantized buffers are absent in the original
        checkpoint.
        """
        if self.pre_quantized:
            return [
                k
                for k in missing_keys
                if not k.endswith(".weight") or not _has_fouroversix_parent(model, k)
            ]

        return [
            k
            for k in missing_keys
            if not any(k.endswith(suffix) for suffix in QUANTIZED_WEIGHT_SUFFIXES)
        ]

    @property
    def is_serializable(self) -> bool:
        """Quantized checkpoints can be saved and reloaded."""
        return True

    @property
    def is_trainable(self) -> bool:
        """FourOverSix supports quantization-aware training."""
        return True


def _has_fouroversix_parent(model: nn.Module, param_name: str) -> bool:
    """Return ``True`` if the module owning *param_name* is a ``FourOverSixLinear``."""
    parts = param_name.rsplit(".", 1)
    if len(parts) < 2:  # noqa: PLR2004
        return False
    module_name = parts[0]
    try:
        module = model.get_submodule(module_name)
    except AttributeError:
        return False
    return isinstance(module, FourOverSixLinear)
