from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from fouroversix import QuantizationConfig as FourOverSixConfig
from fouroversix import quantize
from fouroversix.quantize import get_rht_matrix
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import replace_parameter

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig


def _quantize_dequantize_impl(
    x: torch.Tensor,
    dtype: str,
    scale_rule: str,
    rht: bool,
) -> torch.Tensor:
    # Make sure last dimension is divisible by 16 before we reshape
    original_shape = x.shape
    pad = -x.shape[-1] % 16

    if pad > 0:
        x = F.pad(x, (0, pad))

    if rht:
        x = (x.reshape(*x.shape[:-1], -1, 16) @ get_rht_matrix()).reshape_as(x)

    out = quantize(
        x.reshape(-1, x.shape[-1]),
        FourOverSixConfig(
            dtype=dtype,
            pseudo_quantize=True,
            scale_rule=scale_rule,
        ),
    ).reshape_as(x)

    if pad > 0:
        out = out[..., : original_shape[-1]]

    return out.contiguous()


@torch.library.custom_op("fouroversix::quantize_dequantize", mutates_args=())
def _quantize_dequantize(
    x: torch.Tensor,
    dtype: str,
    scale_rule: str,
    rht: bool,
) -> torch.Tensor:
    return _quantize_dequantize_impl(x, dtype, scale_rule, rht)


@_quantize_dequantize.register_fake
def _(
    x: torch.Tensor,
    dtype: str,
    scale_rule: str,
    rht: bool,
) -> torch.Tensor:
    return torch.empty_like(x)


class FourOverSixLinearMethod(UnquantizedLinearMethod):
    """
    Quantize weights and activations with Four Over Six FP4, then
    dequantize back to high precision for the linear operation.
    """

    def __init__(
        self,
        activation_dtype: str,
        activation_scale_rule: str,
        weight_dtype: str,
        weight_scale_rule: str,
        rht: bool,
    ) -> None:
        super().__init__()

        self.activation_dtype = activation_dtype
        self.activation_scale_rule = activation_scale_rule
        self.weight_dtype = weight_dtype
        self.weight_scale_rule = weight_scale_rule
        self.rht = rht

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        replace_parameter(
            layer,
            "weight",
            _quantize_dequantize_impl(
                layer.weight.data,
                self.weight_dtype,
                self.weight_scale_rule,
                self.rht,
            ),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.ops.fouroversix.quantize_dequantize(
            x,
            self.activation_dtype,
            self.activation_scale_rule,
            self.rht,
        )
        return super().apply(layer, x, bias)


class FourOverSixMoEMethod(UnquantizedFusedMoEMethod):
    """
    Pseudo-quantize weights and activations with Four Over Six, then run the
    MoE computation in high precision using the unquantized fused MoE kernel.
    """

    def __init__(
        self,
        moe: FusedMoEConfig,
        activation_dtype: str,
        activation_scale_rule: str,
        weight_dtype: str,
        weight_scale_rule: str,
        rht: bool,
    ) -> None:
        super().__init__(moe)

        self.activation_dtype = activation_dtype
        self.activation_scale_rule = activation_scale_rule
        self.weight_dtype = weight_dtype
        self.weight_scale_rule = weight_scale_rule
        self.rht = rht

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        replace_parameter(
            layer,
            "w13_weight",
            _quantize_dequantize_impl(
                layer.w13_weight.data,
                self.weight_dtype,
                self.weight_scale_rule,
                self.rht,
            ),
        )

        replace_parameter(
            layer,
            "w2_weight",
            _quantize_dequantize_impl(
                layer.w2_weight.data,
                self.weight_dtype,
                self.weight_scale_rule,
                self.rht,
            ),
        )

        super().process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = torch.ops.fouroversix.quantize_dequantize(
            x,
            self.activation_dtype,
            self.activation_scale_rule,
            self.rht,
        )
        return super().apply(layer, x, topk_weights, topk_ids, shared_experts_input)


class FourOverSixQuantizationConfig(QuantizationConfig):
    """Custom quantization config for Four Over Six FP4."""

    def __init__(self) -> None:
        super().__init__()

    def get_supported_act_dtypes(self) -> list:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(
        cls,
        config: dict,  # noqa: ARG003
    ) -> FourOverSixQuantizationConfig:
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,  # noqa: ARG002
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return FourOverSixLinearMethod(
                self.activation_dtype,
                self.activation_scale_rule,
                self.weight_dtype,
                self.weight_scale_rule,
                self.rht,
            )

        if isinstance(layer, FusedMoE):
            return FourOverSixMoEMethod(
                layer.moe_config,
                self.activation_dtype,
                self.activation_scale_rule,
                self.weight_dtype,
                self.weight_scale_rule,
                self.rht,
            )

        return None


@register_quantization_config("fouroversix_mxfp4")
class FourOverSixMXFP4QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "mxfp4"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "mxfp4"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_mxfp4"


@register_quantization_config("fouroversix_mxfp4_rht")
class FourOverSixMXFP4RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "mxfp4"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "mxfp4"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_mxfp4_rht"


@register_quantization_config("fouroversix_nvint4")
class FourOverSixNVINT4QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvint4"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "nvint4"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvint4"


@register_quantization_config("fouroversix_nvint4_rht")
class FourOverSixNVINT4RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvint4"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "nvint4"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvint4_rht"


@register_quantization_config("fouroversix_nvfp4")
class FourOverSixNVFP4QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp4"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "nvfp4"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvfp4"


@register_quantization_config("fouroversix_nvfp4_rht")
class FourOverSixNVFP4RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp4"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "nvfp4"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvfp4_rht"


@register_quantization_config("fouroversix")
class FourOverSixMSEQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp4"
        self.activation_scale_rule = "mse"
        self.rht = False
        self.weight_dtype = "nvfp4"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix"


@register_quantization_config("fouroversix_rht")
class FourOverSixMSERHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp4"
        self.activation_scale_rule = "mse"
        self.rht = True
        self.weight_dtype = "nvfp4"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_rht"


@register_quantization_config("fouroversix_if4")
class FourOverSixIF4QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "if4"
        self.activation_scale_rule = "mse"
        self.rht = False
        self.weight_dtype = "if4"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_if4"


@register_quantization_config("fouroversix_if4_rht")
class FourOverSixIF4RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "if4"
        self.activation_scale_rule = "mse"
        self.rht = True
        self.weight_dtype = "if4"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_if4_rht"


@register_quantization_config("fouroversix_nvfp6_e2m3")
class FourOverSixNVFP6E2M3QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp6_e2m3"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "nvfp6_e2m3"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvfp6_e2m3"


@register_quantization_config("fouroversix_nvfp6_e2m3_rht")
class FourOverSixNVFP6E2M3RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp6_e2m3"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "nvfp6_e2m3"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvfp6_e2m3_rht"


@register_quantization_config("fouroversix_nvfp6_e3m2")
class FourOverSixNVFP6E3M2QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp6_e3m2"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "nvfp6_e3m2"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvfp6_e3m2"


@register_quantization_config("fouroversix_nvfp6_e3m2_rht")
class FourOverSixNVFP6E3M2RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp6_e3m2"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "nvfp6_e3m2"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvfp6_e3m2_rht"


@register_quantization_config("fouroversix_if6_e2m3")
class FourOverSixIF6E2M3QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "if6_e2m3"
        self.activation_scale_rule = "mse"
        self.rht = False
        self.weight_dtype = "if6_e2m3"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_if6_e2m3"


@register_quantization_config("fouroversix_if6_e2m3_rht")
class FourOverSixIF6E2M3RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "if6_e2m3"
        self.activation_scale_rule = "mse"
        self.rht = True
        self.weight_dtype = "if6_e2m3"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_if6_e2m3_rht"


@register_quantization_config("fouroversix_if6_e3m2")
class FourOverSixIF6E3M2QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "if6_e3m2"
        self.activation_scale_rule = "mse"
        self.rht = False
        self.weight_dtype = "if6_e3m2"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_if6_e3m2"


@register_quantization_config("fouroversix_if6_e3m2_rht")
class FourOverSixIF6E3M2RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "if6_e3m2"
        self.activation_scale_rule = "mse"
        self.rht = True
        self.weight_dtype = "if6_e3m2"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_if6_e3m2_rht"


@register_quantization_config("fouroversix_mxfp6_e2m3")
class FourOverSixMXFP6E2M3QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "mxfp6_e2m3"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "mxfp6_e2m3"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_mxfp6_e2m3"


@register_quantization_config("fouroversix_mxfp6_e2m3_rht")
class FourOverSixMXFP6E2M3RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "mxfp6_e2m3"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "mxfp6_e2m3"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_mxfp6_e2m3_rht"


@register_quantization_config("fouroversix_mxfp6_e3m2")
class FourOverSixMXFP6E3M2QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "mxfp6_e3m2"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "mxfp6_e3m2"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_mxfp6_e3m2"


@register_quantization_config("fouroversix_mxfp6_e3m2_rht")
class FourOverSixMXFP6E3M2RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "mxfp6_e3m2"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "mxfp6_e3m2"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_mxfp6_e3m2_rht"


@register_quantization_config("fouroversix_nvint6")
class FourOverSixNVINT6QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvint6"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "nvint6"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvint6"


@register_quantization_config("fouroversix_nvint6_rht")
class FourOverSixNVINT6RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvint6"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "nvint6"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvint6_rht"


@register_quantization_config("fouroversix_mxfp3")
class FourOverSixMXFP3QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "mxfp3"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "mxfp3"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_mxfp3"


@register_quantization_config("fouroversix_mxfp3_rht")
class FourOverSixMXFP3RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "mxfp3"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "mxfp3"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_mxfp3_rht"


@register_quantization_config("fouroversix_mxfp3_bs8")
class FourOverSixMXFP3BS8QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "mxfp3_bs8"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "mxfp3_bs8"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_mxfp3_bs8"


@register_quantization_config("fouroversix_mxfp3_bs8_rht")
class FourOverSixMXFP3BS8RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "mxfp3_bs8"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "mxfp3_bs8"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_mxfp3_bs8_rht"


@register_quantization_config("fouroversix_nvfp3")
class FourOverSixNVFP3QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp3"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "nvfp3"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvfp3"


@register_quantization_config("fouroversix_nvfp3_rht")
class FourOverSixNVFP3RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp3"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "nvfp3"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvfp3_rht"


@register_quantization_config("fouroversix_nvfp3_bs8")
class FourOverSixNVFP3BS8QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp3_bs8"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "nvfp3_bs8"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvfp3_bs8"


@register_quantization_config("fouroversix_nvfp3_bs8_rht")
class FourOverSixNVFP3BS8RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp3_bs8"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "nvfp3_bs8"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvfp3_bs8_rht"


@register_quantization_config("fouroversix_nvint3")
class FourOverSixNVINT3QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvint3"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "nvint3"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvint3"


@register_quantization_config("fouroversix_nvint3_rht")
class FourOverSixNVINT3RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvint3"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "nvint3"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvint3_rht"


@register_quantization_config("fouroversix_nvint3_bs8")
class FourOverSixNVINT3BS8QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvint3_bs8"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "nvint3_bs8"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvint3_bs8"


@register_quantization_config("fouroversix_nvint3_bs8_rht")
class FourOverSixNVINT3BS8RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvint3_bs8"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "nvint3_bs8"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvint3_bs8_rht"


@register_quantization_config("fouroversix_if3")
class FourOverSixIF3QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "if3"
        self.activation_scale_rule = "mse"
        self.rht = False
        self.weight_dtype = "if3"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_if3"


@register_quantization_config("fouroversix_if3_rht")
class FourOverSixIF3RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "if3"
        self.activation_scale_rule = "mse"
        self.rht = True
        self.weight_dtype = "if3"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_if3_rht"


@register_quantization_config("fouroversix_if3_bs8")
class FourOverSixIF3BS8QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "if3_bs8"
        self.activation_scale_rule = "mse"
        self.rht = False
        self.weight_dtype = "if3_bs8"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_if3_bs8"


@register_quantization_config("fouroversix_if3_bs8_rht")
class FourOverSixIF3BS8RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "if3_bs8"
        self.activation_scale_rule = "mse"
        self.rht = True
        self.weight_dtype = "if3_bs8"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_if3_bs8_rht"


@register_quantization_config("fouroversix_mxfp4_bs8")
class FourOverSixMXFP4BS8QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "mxfp4_bs8"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "mxfp4_bs8"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_mxfp4_bs8"


@register_quantization_config("fouroversix_mxfp4_bs8_rht")
class FourOverSixMXFP4BS8RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "mxfp4_bs8"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "mxfp4_bs8"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_mxfp4_bs8_rht"


@register_quantization_config("fouroversix_nvfp4_bs8")
class FourOverSixNVFP4BS8QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp4_bs8"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "nvfp4_bs8"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvfp4_bs8"


@register_quantization_config("fouroversix_nvfp4_bs8_rht")
class FourOverSixNVFP4BS8RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp4_bs8"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "nvfp4_bs8"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvfp4_bs8_rht"


@register_quantization_config("fouroversix_bs8")
class FourOverSixBS8QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp4_bs8"
        self.activation_scale_rule = "mse"
        self.rht = False
        self.weight_dtype = "nvfp4_bs8"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_bs8"


@register_quantization_config("fouroversix_bs8_rht")
class FourOverSixBS8RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvfp4_bs8"
        self.activation_scale_rule = "mse"
        self.rht = True
        self.weight_dtype = "nvfp4_bs8"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_bs8_rht"


@register_quantization_config("fouroversix_nvint4_bs8")
class FourOverSixNVINT4BS8QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvint4_bs8"
        self.activation_scale_rule = "static_6"
        self.rht = False
        self.weight_dtype = "nvint4_bs8"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvint4_bs8"


@register_quantization_config("fouroversix_nvint4_bs8_rht")
class FourOverSixNVINT4BS8RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "nvint4_bs8"
        self.activation_scale_rule = "static_6"
        self.rht = True
        self.weight_dtype = "nvint4_bs8"
        self.weight_scale_rule = "static_6"

    def get_name(self) -> str:
        return "fouroversix_nvint4_bs8_rht"


@register_quantization_config("fouroversix_if4_bs8")
class FourOverSixIF4BS8QuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "if4_bs8"
        self.activation_scale_rule = "mse"
        self.rht = False
        self.weight_dtype = "if4_bs8"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_if4_bs8"


@register_quantization_config("fouroversix_if4_bs8_rht")
class FourOverSixIF4BS8RHTQuantizationConfig(FourOverSixQuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

        self.activation_dtype = "if4_bs8"
        self.activation_scale_rule = "mse"
        self.rht = True
        self.weight_dtype = "if4_bs8"
        self.weight_scale_rule = "mse"

    def get_name(self) -> str:
        return "fouroversix_if4_bs8_rht"


def register() -> None:
    """Register the fouroversix quantization method with vLLM."""
