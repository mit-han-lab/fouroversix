from typing import Any

import torch
from fouroversix.backend import MatmulBackend, QuantizeBackend
from fouroversix.frontend import fp4_matmul, quantize_to_fp4
from fouroversix.quantize import get_rht_matrix
from fouroversix.utils import AdaptiveBlockScalingRule, FP4Format, RoundStyle
from torch import nn

HBS = 16


class FP4LinearFunction(torch.autograd.Function):
    """Differentiable FP4 linear layer."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,  # noqa: A002
        weight: torch.Tensor,
        weight_e2m1: torch.Tensor = None,
        weight_sf: torch.Tensor = None,
        weight_amax: torch.Tensor = None,
        bias: torch.Tensor = None,
        had: torch.Tensor = None,
        fp4_format: FP4Format = FP4Format.nvfp4,
        a_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        w_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        g_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        w_scale_2d: bool = False,  # noqa: FBT001, FBT002
        a_quantize_kwargs: dict[str, Any] | None = None,
        w_quantize_kwargs: dict[str, Any] | None = None,
        g_quantize_kwargs: dict[str, Any] | None = None,
        out_dtype: torch.dtype = torch.bfloat16,
        quantize_backend: QuantizeBackend = QuantizeBackend.triton,
    ) -> tuple[torch.Tensor,]:
        """
        Perform an FP4 matrix multiplication. The input is provided in high precision
        and quantized to FP4 prior to the matrix multiplication, while the weight is
        provided in low precision.
        """

        assert input.ndim == 2 or input.ndim == 3, "Input must be 2D or 3D"  # noqa: S101, PLR1714, PLR2004

        ctx.save_for_backward(input, weight, had, bias)
        ctx.fp4_format = fp4_format
        ctx.a_scale_rule = a_scale_rule
        ctx.w_scale_rule = w_scale_rule
        ctx.g_scale_rule = g_scale_rule
        ctx.w_scale_2d = w_scale_2d
        ctx.a_quantize_kwargs = a_quantize_kwargs
        ctx.w_quantize_kwargs = w_quantize_kwargs
        ctx.g_quantize_kwargs = g_quantize_kwargs
        ctx.out_dtype = out_dtype
        ctx.quantize_backend = quantize_backend

        assert ctx.a_scale_rule == ctx.w_scale_rule  # noqa: S101

        if ctx.g_scale_rule is not None:
            assert ctx.a_scale_rule == ctx.g_scale_rule  # noqa: S101

        if had is not None:
            ctx.mark_non_differentiable(had)

        if weight is not None:
            weight_e2m1, weight_sf, weight_amax = quantize_to_fp4(
                weight,
                backend=ctx.quantize_backend,
                fp4_format=fp4_format,
                scale_rule=w_scale_rule,
                block_scale_2d=w_scale_2d,
                **(w_quantize_kwargs or {}),
            )

        fp4_matmul_kwargs = {
            "backend": MatmulBackend.cutlass,
            "b_e2m1": weight_e2m1,
            "b_sf": weight_sf,
            "b_amax": weight_amax,
            "fp4_format": fp4_format,
            "scale_rule": w_scale_rule,
            "out_dtype": out_dtype,
            "a_quantize_kwargs": {
                "backend": ctx.quantize_backend,
                "scale_rule": a_scale_rule,
                "fp4_format": fp4_format,
                **(a_quantize_kwargs or {}),
            },
        }

        if input.ndim == 2:  # noqa: PLR2004
            out = fp4_matmul(
                input[0],
                out_shape=(input.shape[0], weight_e2m1.shape[0]),
                **fp4_matmul_kwargs,
            ).unsqueeze(0)
        elif input.ndim == 3:  # noqa: PLR2004
            out = torch.empty(
                *input.shape[:-1],
                weight_e2m1.shape[0],
                device=input.device,
                dtype=out_dtype,
            )

            # Slow bmm
            for i in range(input.shape[0]):
                out[i] = fp4_matmul(
                    input[i],
                    out_shape=(input.shape[1], weight_e2m1.shape[0]),
                    **fp4_matmul_kwargs,
                )

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

        input, weight, had, bias = ctx.saved_tensors  # noqa: A001

        assert grad_output.shape[0] == 1  # noqa: S101

        grad_input = fp4_matmul(
            grad_output[0],
            weight,
            backend=MatmulBackend.cutlass,
            a_quantize_kwargs={
                "backend": ctx.quantize_backend,
                "scale_rule": ctx.g_scale_rule,
                "fp4_format": ctx.fp4_format,
                "round_style": RoundStyle.stochastic,
                **(ctx.g_quantize_kwargs or {}),
            },
            b_quantize_kwargs={
                "backend": ctx.quantize_backend,
                "scale_rule": ctx.w_scale_rule,
                "fp4_format": ctx.fp4_format,
                "transpose": True,
                "block_scale_2d": ctx.w_scale_2d,
                **(ctx.w_quantize_kwargs or {}),
            },
            scale_rule=ctx.w_scale_rule,
            out_dtype=torch.bfloat16,
            out_shape=(grad_output.shape[1], weight.shape[1]),
        ).unsqueeze(0)

        grad_weight = fp4_matmul(
            grad_output[0],
            input[0],
            backend=MatmulBackend.cutlass,
            a_quantize_kwargs={
                "backend": ctx.quantize_backend,
                "transpose": True,
                "round_style": RoundStyle.stochastic,
                "scale_rule": ctx.g_scale_rule,
                "fp4_format": ctx.fp4_format,
                "had": had,
                **(ctx.g_quantize_kwargs or {}),
            },
            b_quantize_kwargs={
                "backend": ctx.quantize_backend,
                "transpose": True,
                "scale_rule": ctx.a_scale_rule,
                "fp4_format": ctx.fp4_format,
                "had": had,
                **(ctx.a_quantize_kwargs or {}),
            },
            scale_rule=ctx.a_scale_rule,
            out_dtype=torch.bfloat16,
            out_shape=(grad_output.shape[2], input.shape[2]),
        ).unsqueeze(0)

        grad_bias = (
            grad_output.sum(0) if bias is not None and ctx.needs_input_grad[5] else None
        )

        return (
            grad_input,
            grad_weight,
            None,
            None,
            None,
            grad_bias,
            None,
            None,
            None,
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
        a_quantize_kwargs: dict[str, Any] | None = None,
        w_quantize_kwargs: dict[str, Any] | None = None,
        quantize_backend: QuantizeBackend = QuantizeBackend.triton,
        **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.had = None
        self.fp4_format = fp4_format
        self.a_scale_rule = a_scale_rule
        self.w_scale_rule = w_scale_rule
        self.w_scale_2d = w_scale_2d
        self.a_quantize_kwargs = a_quantize_kwargs
        self.w_quantize_kwargs = w_quantize_kwargs
        self.out_dtype = torch.bfloat16
        self.quantize_backend = quantize_backend

    def apply_ptq(self) -> None:
        """
        Prepare this layer for post-training quantization by quantizing the weight,
        storing the quantized weight, and deleting the original weight. This should not
        be done if the layer is used for training, as training requires storage of the
        high-precision weight.
        """

        self.weight_e2m1, self.weight_sf, self.weight_amax = self.quantized_weight()
        del self.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        """Forward pass for the FP4 linear layer."""

        (out,) = FP4LinearFunction.apply(
            input,
            None,
            *self.quantized_weight(),
            self.bias,
            self.had,
            self.fp4_format,
            self.a_scale_rule,
            self.w_scale_rule,
            None,
            self.w_scale_2d,
            self.a_quantize_kwargs,
            self.w_quantize_kwargs,
            None,
            self.out_dtype,
            self.quantize_backend,
        )

        return out

    def quantized_weight(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the quantized weights."""

        if hasattr(self, "weight"):
            return quantize_to_fp4(
                self.weight,
                scale_rule=self.w_scale_rule,
                block_scale_2d=self.w_scale_2d,
                **(self.w_quantize_kwargs or {}),
            )

        return self.weight_e2m1, self.weight_sf, self.weight_amax


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
        a_quantize_kwargs: dict[str, Any] | None = None,
        w_quantize_kwargs: dict[str, Any] | None = None,
        g_quantize_kwargs: dict[str, Any] | None = None,
        quantize_backend: QuantizeBackend = QuantizeBackend.triton,
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
            a_quantize_kwargs=a_quantize_kwargs,
            w_quantize_kwargs=w_quantize_kwargs,
            quantize_backend=quantize_backend,
        )

        self.g_scale_rule = g_scale_rule
        self.g_quantize_kwargs = g_quantize_kwargs
        self.had = get_rht_matrix()

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        """Forward pass for the FP4 linear layer."""

        (out,) = FP4LinearFunction.apply(
            input,
            self.weight,
            None,
            None,
            None,
            self.bias,
            self.had,
            self.fp4_format,
            self.a_scale_rule,
            self.w_scale_rule,
            self.g_scale_rule,
            self.w_scale_2d,
            self.a_quantize_kwargs,
            self.w_quantize_kwargs,
            self.g_quantize_kwargs,
            self.out_dtype,
            self.quantize_backend,
        )

        return out
