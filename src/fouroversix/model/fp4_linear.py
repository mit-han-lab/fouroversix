from typing import Any

import torch
from fouroversix.backend import MatmulBackend, QuantizeBackend
from fouroversix.frontend import fp4_matmul, quantize_to_fp4
from fouroversix.quantize import FP4Tensor, get_rht_matrix
from fouroversix.utils import AdaptiveBlockScalingRule, FP4Format, RoundStyle
from torch import nn

HBS = 16


class FP4LinearFunction(torch.autograd.Function):
    """Differentiable FP4 linear layer."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor | FP4Tensor,
        bias: torch.Tensor = None,
        fp4_format: FP4Format = FP4Format.nvfp4,
        a_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        w_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        g_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        w_scale_2d: bool = False,  # noqa: FBT001, FBT002
        out_dtype: torch.dtype = torch.bfloat16,
        matmul_backend: MatmulBackend | None = None,
        quantize_backend: QuantizeBackend | None = None,
    ) -> tuple[torch.Tensor,]:
        """
        Perform an FP4 matrix multiplication. The input is provided in high precision
        and quantized to FP4 prior to the matrix multiplication, while the weight is
        provided in low precision.
        """

        if isinstance(weight, torch.Tensor):
            ctx.save_for_backward(input, weight, bias)
            weight = quantize_to_fp4(
                weight,
                backend=quantize_backend,
                block_scale_2d=w_scale_2d,
                fp4_format=fp4_format,
                scale_rule=w_scale_rule,
            )

        ctx.fp4_format = fp4_format
        ctx.a_scale_rule = a_scale_rule
        ctx.w_scale_rule = w_scale_rule
        ctx.g_scale_rule = g_scale_rule
        ctx.w_scale_2d = w_scale_2d
        ctx.out_dtype = out_dtype
        ctx.matmul_backend = matmul_backend
        ctx.quantize_backend = quantize_backend

        assert ctx.a_scale_rule == ctx.w_scale_rule  # noqa: S101

        if ctx.g_scale_rule is not None:
            assert ctx.a_scale_rule == ctx.g_scale_rule  # noqa: S101

        out = fp4_matmul(
            input.reshape(-1, input.shape[-1]),
            weight,
            backend=matmul_backend,
            input_quantize_kwargs={
                "backend": quantize_backend,
                "fp4_format": fp4_format,
                "scale_rule": a_scale_rule,
            },
            out_dtype=out_dtype,
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
        had = get_rht_matrix()

        assert grad_output.shape[0] == 1  # noqa: S101

        grad_input = fp4_matmul(
            grad_output[0],
            weight,
            backend=ctx.matmul_backend,
            input_quantize_kwargs={
                "backend": ctx.quantize_backend,
                "scale_rule": ctx.g_scale_rule,
                "fp4_format": ctx.fp4_format,
                "round_style": RoundStyle.stochastic,
            },
            other_quantize_kwargs={
                "backend": ctx.quantize_backend,
                "scale_rule": ctx.w_scale_rule,
                "fp4_format": ctx.fp4_format,
                "transpose": True,
                "block_scale_2d": ctx.w_scale_2d,
            },
            out_dtype=torch.bfloat16,
        ).unsqueeze(0)

        grad_weight = fp4_matmul(
            grad_output[0],
            input[0],
            backend=ctx.matmul_backend,
            input_quantize_kwargs={
                "backend": ctx.quantize_backend,
                "transpose": True,
                "round_style": RoundStyle.stochastic,
                "scale_rule": ctx.g_scale_rule,
                "fp4_format": ctx.fp4_format,
                "had": had,
            },
            other_quantize_kwargs={
                "backend": ctx.quantize_backend,
                "transpose": True,
                "scale_rule": ctx.a_scale_rule,
                "fp4_format": ctx.fp4_format,
                "had": had,
            },
            out_dtype=torch.bfloat16,
        ).unsqueeze(0)

        grad_bias = (
            grad_output.sum(0) if bias is not None and ctx.needs_input_grad[5] else None
        )

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FP4Linear(nn.Linear):
    """Drop-in replacement for `nn.Linear` that uses FP4 quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,  # noqa: FBT001, FBT002
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        fp4_format: FP4Format = FP4Format.nvfp4,
        a_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        w_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        w_scale_2d: bool = False,
        matmul_backend: MatmulBackend | None = None,
        quantize_backend: QuantizeBackend | None = None,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.fp4_format = fp4_format
        self.a_scale_rule = a_scale_rule
        self.w_scale_rule = w_scale_rule
        self.w_scale_2d = w_scale_2d
        self.out_dtype = torch.bfloat16
        self.matmul_backend = matmul_backend
        self.quantize_backend = quantize_backend

    def apply_ptq(self) -> None:
        """
        Prepare this layer for post-training quantization by quantizing the weight,
        storing the quantized weight, and deleting the original weight. This should not
        be done if the layer is used for training, as training requires storage of the
        high-precision weight.
        """

        weight = self.weight
        del self.weight

        self.weight = self.quantized_weight(weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FP4 linear layer."""

        (out,) = FP4LinearFunction.apply(
            input,
            self.weight,
            self.bias,
            self.fp4_format,
            self.a_scale_rule,
            self.w_scale_rule,
            None,
            self.w_scale_2d,
            self.out_dtype,
            self.matmul_backend,
            self.quantize_backend,
        )

        return out

    def quantized_weight(self, weight: torch.Tensor | None = None) -> FP4Tensor:
        """Compute the quantized weights."""

        if isinstance(weight, torch.Tensor) or isinstance(self.weight, torch.Tensor):
            return quantize_to_fp4(
                weight if isinstance(weight, torch.Tensor) else self.weight,
                scale_rule=self.w_scale_rule,
                block_scale_2d=self.w_scale_2d,
                fp4_format=self.fp4_format,
                backend=self.quantize_backend,
            )

        return self.weight


class TrainableFP4Linear(FP4Linear):
    """
    Drop-in replacement for `nn.Linear` that uses FP4 quantization. This should be
    used instead of `FP4Linear` if the layer is used for training, as training requires
    storage of the high-precision weight and a hadamard matrix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,  # noqa: FBT001, FBT002
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        fp4_format: FP4Format = FP4Format.nvfp4,
        a_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        w_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        g_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        matmul_backend: MatmulBackend | None = None,
        quantize_backend: QuantizeBackend | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
            fp4_format=fp4_format,
            a_scale_rule=a_scale_rule,
            w_scale_rule=w_scale_rule,
            w_scale_2d=True,
            matmul_backend=matmul_backend,
            quantize_backend=quantize_backend,
        )

        self.g_scale_rule = g_scale_rule

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FP4 linear layer."""

        (out,) = FP4LinearFunction.apply(
            input,
            self.weight,
            self.bias,
            self.fp4_format,
            self.a_scale_rule,
            self.w_scale_rule,
            self.g_scale_rule,
            self.w_scale_2d,
            self.out_dtype,
            self.matmul_backend,
            self.quantize_backend,
        )

        return out
