from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from fouroversix.matmul import quantized_matmul
from fouroversix.model.config import ModuleQuantizationConfig
from fouroversix.model.quantize import QuantizedModule
from fouroversix.quantize import (
    QuantizedTensor,
    dequantize,
    get_rht_matrix,
    quantize,
)
from fouroversix.utils import DataType

try:
    from fouroversix.kernels.triton.quartet_ii import eden_1x16s_fp4_kernel_wrapper
except ImportError:
    eden_1x16s_fp4_kernel_wrapper = None

QUARTET_II_BACKWARD_SCALE_OVERRIDE = (17 / 16) * 0.93


def rerotate_hadamard(hadamard_matrix: torch.Tensor) -> torch.Tensor:
    signs = (
        torch.randint(
            0,
            2,
            (hadamard_matrix.size(0),),
            device=hadamard_matrix.device,
            dtype=hadamard_matrix.dtype,
        )
        * 2
        - 1
    )
    return hadamard_matrix * signs[None, :]


class FourOverSixLinearFunction(torch.autograd.Function):
    """Differentiable FP4 linear layer."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        config: ModuleQuantizationConfig,
        input: torch.Tensor,
        weight: QuantizedTensor | nn.Parameter | torch.Tensor,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Perform an FP4 matrix multiplication. The input is provided in high precision
        and quantized to FP4 prior to the matrix multiplication, while the weight is
        provided in low precision.
        """

        if isinstance(weight, nn.Parameter):
            ctx.config = config
            ctx.save_for_backward(input, weight, bias)

        if config.disable_fprop_quantization:
            assert isinstance(weight, torch.Tensor)  # noqa: S101
            out = torch.matmul(input, weight.T)
        else:
            fprop_activation_config = config.get_activation_config()
            fprop_weight_config = config.get_weight_config()

            if isinstance(weight, nn.Parameter):
                weight = quantize(weight.data, fprop_weight_config)
            elif isinstance(weight, torch.Tensor):
                weight = quantize(weight, fprop_weight_config)

            if config.pseudo_quantize:
                input_pseudoquant = quantize(
                    input.reshape(-1, input.shape[-1]),
                    fprop_activation_config,
                )
                out = torch.matmul(input_pseudoquant, weight.T).reshape(
                    *input.shape[:-1],
                    weight.shape[0],
                )
            else:
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
        """Backward pass for the FP4 linear layer."""

        input, weight, bias = ctx.saved_tensors

        assert grad_output.shape[0] == 1  # noqa: S101

        if ctx.config.quartet_ii_dtype is not None:
            assert ctx.config.quartet_ii_dtype in {  # noqa: S101
                DataType.nvfp4,
                DataType.if4,
            }

            had = get_rht_matrix()

            if ctx.config.quartet_ii_rerotate_had:
                had = rerotate_hadamard(had)

            e_ht_amax = (
                (grad_output.reshape(-1, had.size(0)) @ had.T).abs().max().float()
            )
            e_ht_fp4, _ = eden_1x16s_fp4_kernel_wrapper(
                grad_output.reshape(-1, grad_output.shape[-1]),
                had.to(grad_output.dtype),
                QUARTET_II_BACKWARD_SCALE_OVERRIDE,
                ctx.config.dtype.block_size,
                e_ht_amax,
                dtype=ctx.config.quartet_ii_dtype,
            )

            weight_fp4 = dequantize(
                quantize(weight, ctx.config.get_weight_config()),
            )
            weight_tht_amax = (
                (weight_fp4.T.reshape(-1, had.size(0)) @ had.T).abs().max().float()
            )
            weight_tht_fp4, _ = eden_1x16s_fp4_kernel_wrapper(
                weight_fp4.T,
                had.to(weight_fp4.dtype),
                QUARTET_II_BACKWARD_SCALE_OVERRIDE,
                ctx.config.dtype.block_size,
                weight_tht_amax,
                dtype=ctx.config.quartet_ii_dtype,
            )

            grad_input = F.linear(
                e_ht_fp4,
                weight_tht_fp4,
                None,
            ).reshape(*grad_output.shape[:-1], weight.shape[1])

            e_tht_amax = (
                (
                    grad_output.reshape(-1, grad_output.shape[-1]).T.reshape(
                        -1,
                        had.size(0),
                    )
                    @ had.T
                )
                .abs()
                .max()
                .float()
            )
            e_tht_fp4, _ = eden_1x16s_fp4_kernel_wrapper(
                grad_output.reshape(-1, grad_output.shape[-1]).T,
                had.to(grad_output.dtype),
                QUARTET_II_BACKWARD_SCALE_OVERRIDE,
                ctx.config.dtype.block_size,
                e_tht_amax,
                dtype=ctx.config.quartet_ii_dtype,
            )

            input_fp4 = dequantize(
                quantize(
                    input.reshape(-1, input.shape[-1]),
                    ctx.config.get_activation_config(),
                ),
            )
            input_tht_amax = (
                (input_fp4.T.reshape(-1, had.size(0)) @ had.T).abs().max().float()
            )
            input_tht_fp4, _ = eden_1x16s_fp4_kernel_wrapper(
                input_fp4.T,
                had.to(input_fp4.dtype),
                QUARTET_II_BACKWARD_SCALE_OVERRIDE,
                ctx.config.dtype.block_size,
                input_tht_amax,
                dtype=ctx.config.quartet_ii_dtype,
            )

            grad_weight = F.linear(
                e_tht_fp4,
                input_tht_fp4,
                None,
            )

            grad_bias = (
                grad_output.sum(0)
                if bias is not None and ctx.needs_input_grad[3]
                else None
            )

            return None, grad_input, grad_weight, grad_bias

        dgrad_grad_config = ctx.config.get_gradient_config()
        dgrad_weight_config = ctx.config.get_weight_config(transpose=True)

        if ctx.config.disable_dgrad_quantization == "quantize_grad_only":
            grad_input = (
                torch.matmul(
                    dequantize(
                        quantize(
                            grad_output.reshape(-1, grad_output.shape[-1]),
                            dgrad_grad_config,
                        ),
                        torch.float32,
                    ),
                    weight,
                )
                .reshape(*grad_output.shape[:-1], weight.shape[1])
                .to(ctx.config.output_dtype.torch_dtype)
            )
        elif ctx.config.disable_dgrad_quantization:
            grad_input = torch.matmul(grad_output, weight)
        elif ctx.config.pseudo_quantize:
            grad_pseudoquant = quantize(
                grad_output.reshape(-1, grad_output.shape[-1]),
                dgrad_grad_config,
            )
            weight_pseudoquant = quantize(
                weight,
                dgrad_weight_config,
            )
            grad_input = torch.matmul(
                grad_pseudoquant,
                weight_pseudoquant.T,
            ).reshape(*grad_output.shape[:-1], weight.shape[1])
        else:
            grad_input = quantized_matmul(
                grad_output.reshape(-1, grad_output.shape[-1]),
                weight if isinstance(weight, torch.Tensor) else weight.data,
                backend=ctx.config.matmul_backend,
                input_config=dgrad_grad_config,
                other_config=dgrad_weight_config,
                out_dtype=ctx.config.output_dtype,
            ).reshape(*grad_output.shape[:-1], weight.shape[1])

        dtype_kwargs = {"dtype": DataType.nvint4} if ctx.config.wgrad_nvint4 else {}
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
        elif ctx.config.pseudo_quantize:
            grad_pseudoquant = quantize(
                grad_output.reshape(-1, grad_output.shape[-1]),
                wgrad_grad_config,
            )
            input_pseudoquant = quantize(
                input.reshape(-1, input.shape[-1]),
                wgrad_activation_config,
            )
            grad_weight = torch.matmul(grad_pseudoquant, input_pseudoquant.T)
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
            grad_output.sum(0) if bias is not None and ctx.needs_input_grad[3] else None
        )

        return None, grad_input, grad_weight, grad_bias


@QuantizedModule.register(nn.Linear)
class FourOverSixLinear(nn.Linear):
    """
    Drop-in replacement for `nn.Linear` that quantizes weights, activations, and
    gradients.
    """

    def __init__(
        self,
        module: nn.Linear,
        config: ModuleQuantizationConfig,
    ) -> None:
        """
        Initialize the FourOverSixLinear layer.

        Args:
            module (nn.Linear): The high-precision module that this quantized layer will
                replace.
            config (ModuleQuantizationConfig): The quantization configuration to use for
                the layer.

        """

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
                        // self.config.weight_dtype.quantized_value_type.packing_factor,
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
                        dtype=self.config.weight_dtype.scale_type.torch_dtype,
                    ),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "quantized_weight_amax",
                nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False),
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

        # quantized_weight_values is packed, so there are 4 bits, or 0.5 bytes, per
        # element. Once quantized, weight will have (8+1)/16 bytes per element (one
        # block of 16 values is 8 bytes of values + 1 byte of scale factors).
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
                "quantized_weight_scale_factors": quantized_weight.scale_factors,
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
                            if quantized_weight.scale_factors_are_in_blackwell_layout
                            else 0
                        ),
                    ],
                ),
            }

        msg = f"Unsupported high-precision parameter: {parameter_name}"
        raise ValueError(msg)

    def quantized_weight(self) -> QuantizedTensor | nn.Parameter:
        """
        Prepare this layer for post-training quantization by quantizing the weight,
        storing the quantized weight, and deleting the original weight. This should not
        be done if the layer is used for training, as training requires storage of the
        high-precision weight.
        """

        if not hasattr(self, "_quantized_weight"):
            if self.config.keep_master_weights:
                return self.weight

            original_shape = tuple(self.quantized_weight_metadata.data[:2].tolist())
            padded_shape = tuple(self.quantized_weight_metadata.data[2:4].tolist())
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
                scale_factors_are_in_blackwell_layout=scale_factors_are_in_blackwell_layout,
            )

        return self._quantized_weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FP4 linear layer."""

        return FourOverSixLinearFunction.apply(
            self.config,
            input,
            self.quantized_weight(),
            self.bias,
        )
