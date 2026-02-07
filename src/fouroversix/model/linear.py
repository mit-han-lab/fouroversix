import torch
from fouroversix.matmul import fp4_matmul
from fouroversix.quantize import (
    QuantizationConfig,
    QuantizedTensor,
    quantize_to_fp4,
)
from fouroversix.utils import RoundStyle
from torch import nn

from .config import FourOverSixLinearConfig

HBS = 16


class FourOverSixLinearFunction(torch.autograd.Function):
    """Differentiable FP4 linear layer."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        config: FourOverSixLinearConfig,
        input: torch.Tensor,
        weight: torch.Tensor | QuantizedTensor,
        bias: torch.Tensor = None,
    ) -> tuple[torch.Tensor,]:
        """
        Perform an FP4 matrix multiplication. The input is provided in high precision
        and quantized to FP4 prior to the matrix multiplication, while the weight is
        provided in low precision.
        """

        fprop_activation_config = QuantizationConfig(
            backend=config.quantize_backend,
            dtype=config.dtype,
            scale_rule=config.get_activation_scale_rule(),
        )

        fprop_weight_config = QuantizationConfig(
            backend=config.quantize_backend,
            block_scale_2d=config.weight_scale_2d,
            dtype=config.dtype,
            scale_rule=config.get_weight_scale_rule(),
        )

        if isinstance(weight, torch.Tensor):
            ctx.save_for_backward(input, weight, bias)
            weight = quantize_to_fp4(weight, fprop_weight_config)

        ctx.config = config

        out = fp4_matmul(
            input.reshape(-1, input.shape[-1]),
            weight,
            backend=config.matmul_backend,
            input_config=fprop_activation_config,
            out_dtype=config.output_dtype,
        ).reshape(*input.shape[:-1], weight.original_shape[0])

        assert out.dtype == torch.bfloat16  # noqa: S101

        if bias is not None:
            out = out + bias

        return (out,)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Backward pass for the FP4 linear layer."""

        input, weight, bias = ctx.saved_tensors

        assert grad_output.shape[0] == 1  # noqa: S101

        dgrad_grad_config = QuantizationConfig(
            backend=ctx.config.quantize_backend,
            dtype=ctx.config.dtype,
            round_style=RoundStyle.stochastic,
            scale_rule=ctx.config.get_gradient_scale_rule(),
        )

        dgrad_weight_config = QuantizationConfig(
            backend=ctx.config.quantize_backend,
            block_scale_2d=ctx.config.weight_scale_2d,
            dtype=ctx.config.dtype,
            scale_rule=ctx.config.get_weight_scale_rule(),
            transpose=True,
        )

        grad_input = fp4_matmul(
            grad_output[0],
            weight,
            backend=ctx.matmul_backend,
            input_config=dgrad_grad_config,
            other_config=dgrad_weight_config,
            out_dtype=ctx.config.output_dtype,
        ).unsqueeze(0)

        wgrad_grad_config = QuantizationConfig(
            backend=ctx.config.quantize_backend,
            dtype=ctx.config.dtype,
            rht=True,
            round_style=RoundStyle.stochastic,
            scale_rule=ctx.config.get_gradient_scale_rule(),
            transpose=True,
        )

        wgrad_activation_config = QuantizationConfig(
            backend=ctx.config.quantize_backend,
            dtype=ctx.config.dtype,
            rht=True,
            scale_rule=ctx.config.get_activation_scale_rule(),
            transpose=True,
        )

        grad_weight = fp4_matmul(
            grad_output[0],
            input[0],
            backend=ctx.matmul_backend,
            input_config=wgrad_grad_config,
            other_config=wgrad_activation_config,
            out_dtype=ctx.config.output_dtype,
        ).unsqueeze(0)

        grad_bias = (
            grad_output.sum(0) if bias is not None and ctx.needs_input_grad[3] else None
        )

        return (
            None,
            grad_input,
            grad_weight,
            grad_bias,
        )


class FourOverSixLinear(nn.Linear):
    """
    Drop-in replacement for `nn.Linear` that quantizes weights, activations, and
    gradients.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,  # noqa: FBT001, FBT002
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        config: FourOverSixLinearConfig | None = None,
    ) -> None:
        """
        Initialize the FourOverSixLinear layer.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool): Whether to include a bias term.
            device (torch.device): The device to use for the layer.
            dtype (torch.dtype): The data type to use for the layer.
            config (FourOverSixLinearConfig): The quantization configuration to use for
                the layer.

        """

        super().__init__(in_features, out_features, bias, device, dtype)
        self.config = config or FourOverSixLinearConfig()

    def apply_ptq(self) -> None:
        """Apply post-training quantization to this layer."""
        self.quantized_weight  # noqa: B018

    @property
    def quantized_weight(self) -> QuantizedTensor:
        """
        Prepare this layer for post-training quantization by quantizing the weight,
        storing the quantized weight, and deleting the original weight. This should not
        be done if the layer is used for training, as training requires storage of the
        high-precision weight.
        """

        if not hasattr(self, "weight_values"):
            weight_config = QuantizationConfig(
                backend=self.config.quantize_backend,
                block_scale_2d=self.config.weight_scale_2d,
                dtype=self.config.dtype,
                scale_rule=self.config.get_weight_scale_rule(),
            )

            quantized_weight = quantize_to_fp4(self.weight, weight_config)

            if self.config.keep_master_weights:
                return quantized_weight

            self.weight_values = nn.Parameter(
                quantized_weight.values,
                requires_grad=False,
            )
            self.weight_scale_factors = nn.Parameter(
                quantized_weight.scale_factors,
                requires_grad=False,
            )
            self.weight_amax = nn.Parameter(quantized_weight.amax, requires_grad=False)

            self.weight_dtype = quantized_weight.dtype
            self.weight_original_shape = quantized_weight.original_shape
            self.weight_scale_rule = quantized_weight.scale_rule
            self.weight_padded_shape = quantized_weight.padded_shape

        if not hasattr(self, "_quantized_weight"):
            self._quantized_weight = QuantizedTensor(
                self.weight_values.data,
                self.weight_scale_factors.data,
                self.weight_amax.data,
                self.weight_dtype,
                self.weight_original_shape,
                self.weight_scale_rule,
                self.weight_padded_shape,
            )

        return self._quantized_weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FP4 linear layer."""

        (out,) = FourOverSixLinearFunction.apply(
            self.config,
            input,
            self.quantized_weight,
            self.bias,
        )

        return out
