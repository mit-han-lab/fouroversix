"""
Transposable FP4 linear layer using 2D block scaling.

Uses ``block_scale_2d=True`` so that the backward pass can obtain the transposed
weight via a cheap nibble shuffle instead of a full re-quantization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from fouroversix.matmul import quantized_matmul
from fouroversix.quantize import (
    QuantizedTensor,
    quantize,
)
from fouroversix.quantize.transpose import transpose_quantized_tensor
from fouroversix.utils import DataType

if TYPE_CHECKING:
    from fouroversix.model.config import ModuleQuantizationConfig


class TransposableFourOverSixLinearFunction(torch.autograd.Function):
    """Differentiable FP4 linear layer without weight re-quantization."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        config: ModuleQuantizationConfig,
        input: torch.Tensor,
        weight: QuantizedTensor | nn.Parameter | torch.Tensor,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """Perform an FP4 matmul using 2D-block-scaled weights."""
        if isinstance(weight, nn.Parameter):
            ctx.config = config
            ctx.save_for_backward(input, weight, bias)

        fprop_activation_config = config.get_activation_config()
        fprop_weight_config = config.get_weight_config()

        if isinstance(weight, nn.Parameter):
            weight = quantize(weight.data, fprop_weight_config)
        elif isinstance(weight, torch.Tensor):
            weight = quantize(weight, fprop_weight_config)

        out = quantized_matmul(
            input.reshape(-1, input.shape[-1]),
            weight,
            backend=config.matmul_backend,
            input_config=fprop_activation_config,
            out_dtype=config.output_dtype,
        ).reshape(*input.shape[:-1], weight.original_shape[0])

        if bias is not None:
            out = out + bias

        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Backward pass with cheap nibble-transpose for dgrad."""
        input, weight, bias = ctx.saved_tensors

        assert grad_output.shape[0] == 1  # noqa: S101

        # --- dgrad: dX = dY @ W ---
        fprop_weight_config = ctx.config.get_weight_config()
        weight_q = quantize(weight, fprop_weight_config)
        weight_q_t = transpose_quantized_tensor(weight_q)

        dgrad_grad_config = ctx.config.get_gradient_config()

        if ctx.config.disable_dgrad_quantization == "quantize_grad_only":
            from fouroversix.quantize import dequantize

            grad_input = (
                torch.matmul(
                    dequantize(
                        quantize(
                            grad_output.reshape(
                                -1,
                                grad_output.shape[-1],
                            ),
                            dgrad_grad_config,
                        ),
                        torch.float32,
                    ),
                    dequantize(weight_q_t, torch.float32).T,
                )
                .reshape(*grad_output.shape[:-1], weight.shape[1])
                .to(ctx.config.output_dtype.torch_dtype)
            )
        elif ctx.config.disable_dgrad_quantization:
            from fouroversix.quantize import dequantize

            grad_input = torch.matmul(
                grad_output.reshape(-1, grad_output.shape[-1]),
                dequantize(weight_q_t, torch.float32).T,
            ).reshape(*grad_output.shape[:-1], weight.shape[1])
        else:
            grad_input = quantized_matmul(
                grad_output.reshape(-1, grad_output.shape[-1]),
                weight_q_t,
                backend=ctx.config.matmul_backend,
                input_config=dgrad_grad_config,
                out_dtype=ctx.config.output_dtype,
            ).reshape(*grad_output.shape[:-1], weight.shape[1])

        # --- wgrad: dW = dY.T @ X ---
        dtype_kwargs = (
            {"dtype": DataType.nvint4} if ctx.config.wgrad_nvint4 else {}
        )
        wgrad_grad_config = ctx.config.get_gradient_config(
            rht=True,
            transpose=True,
            **dtype_kwargs,
        )
        wgrad_activation_config = ctx.config.get_activation_config(
            rht=True,
            transpose=True,
            **dtype_kwargs,
        )

        if ctx.config.disable_wgrad_quantization:
            grad_weight = torch.matmul(
                grad_output.reshape(-1, grad_output.shape[-1]).T,
                input,
            )
        else:
            grad_weight = quantized_matmul(
                grad_output.reshape(-1, grad_output.shape[-1]),
                input.reshape(-1, input.shape[-1]),
                backend=ctx.config.matmul_backend,
                input_config=wgrad_grad_config,
                other_config=wgrad_activation_config,
                out_dtype=ctx.config.output_dtype,
            )

        grad_bias = (
            grad_output.sum(0)
            if bias is not None and ctx.needs_input_grad[3]
            else None
        )

        return None, grad_input, grad_weight, grad_bias


class TransposableFourOverSixLinear(nn.Linear):
    """
    FP4 linear layer that uses 2D block scaling for lossless weight transpose.

    Instead of re-quantizing the weight with ``transpose=True`` on every backward
    pass, this module transposes the packed nibbles and scale factor grid directly.
    This is mathematically lossless because 16x16 tile scales are invariant to
    transposition.

    Requires ``weight_scale_2d=True`` in the quantization config.
    """

    def __init__(
        self,
        module: nn.Linear,
        config: ModuleQuantizationConfig,
    ) -> None:
        """Initialize from an existing ``nn.Linear`` and a quantization config."""
        if not config.weight_scale_2d:
            msg = (
                "TransposableFourOverSixLinear requires weight_scale_2d=True "
                "in the quantization config"
            )
            raise ValueError(msg)

        super().__init__(
            module.in_features,
            module.out_features,
            module.bias is not None,
            module.weight.device,
            module.weight.dtype,
        )

        self.weight = module.weight
        self.bias = module.bias
        self.config = config

        if not self.config.keep_master_weights:
            self.register_buffer(
                "quantized_weight_values",
                nn.Parameter(
                    torch.zeros(
                        self.out_features,
                        self.in_features
                        // self.config.weight_dtype
                        .quantized_value_type.packing_factor,
                        dtype=torch.uint8,
                    ),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "quantized_weight_scale_factors",
                nn.Parameter(
                    torch.zeros(
                        self.out_features
                        * self.in_features
                        // self.config.weight_dtype.block_size,
                        dtype=(
                            self.config.weight_dtype
                            .scale_type.torch_dtype
                        ),
                    ),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "quantized_weight_amax",
                nn.Parameter(
                    torch.zeros(1, dtype=torch.float32),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "quantized_weight_metadata",
                nn.Parameter(
                    torch.zeros(2 + 2 + 2, dtype=torch.int32),
                    requires_grad=False,
                ),
            )

    @property
    def parameters_to_quantize(self) -> tuple[str, ...]:
        """Return high precision parameters to be quantized and deleted."""
        return ("weight",)

    def get_element_size(self, parameter_name: str) -> float:
        """Get the size of a single element, in bytes, for a parameter."""
        return {"quantized_weight_values": 0.5, "weight": 9 / 16}.get(
            parameter_name,
            getattr(self, parameter_name).element_size(),
        )

    def get_quantized_parameters(
        self,
        parameter_name: str,
        parameter: torch.Tensor,
    ) -> dict[str, Any]:
        """Get the quantized parameters for the layer."""
        if parameter_name == "weight":
            weight_config = self.config.get_weight_config()
            quantized_weight = quantize(parameter, weight_config)

            return {
                "quantized_weight_values": quantized_weight.values,
                "quantized_weight_scale_factors": (
                    quantized_weight.scale_factors
                ),
                "quantized_weight_amax": quantized_weight.amax,
                "quantized_weight_metadata": torch.tensor(
                    [
                        quantized_weight.original_shape[0],
                        quantized_weight.original_shape[1],
                        quantized_weight.padded_shape[0],
                        quantized_weight.padded_shape[1],
                        1,
                        (
                            1
                            if quantized_weight
                            .scale_factors_are_in_blackwell_layout
                            else 0
                        ),
                    ],
                ),
            }

        msg = f"Unsupported high-precision parameter: {parameter_name}"
        raise ValueError(msg)

    def quantized_weight(self) -> QuantizedTensor | nn.Parameter:
        """Return the quantized weight, building it from buffers if needed."""
        if not hasattr(self, "_quantized_weight"):
            if self.config.keep_master_weights:
                return self.weight

            original_shape = tuple(
                self.quantized_weight_metadata.data[:2].tolist(),
            )
            padded_shape = tuple(
                self.quantized_weight_metadata.data[2:4].tolist(),
            )
            scale_factors_are_in_blackwell_layout = (
                self.quantized_weight_metadata.data[5].item() == 1
            )

            self._quantized_weight = QuantizedTensor(
                self.quantized_weight_values.data,
                self.quantized_weight_scale_factors.data,
                self.quantized_weight_amax.data,
                self.config.weight_dtype,
                original_shape,
                self.config.weight_scale_rule,
                self.config.weight_round_style,
                padded_shape,
                scale_factors_are_in_blackwell_layout=(
                    scale_factors_are_in_blackwell_layout
                ),
            )

        return self._quantized_weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for the transposable FP4 linear layer."""
        return TransposableFourOverSixLinearFunction.apply(
            self.config,
            input,
            self.quantized_weight(),
            self.bias,
        )
