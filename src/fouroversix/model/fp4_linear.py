from typing import Any

import torch
from fouroversix.frontend import fp4_matmul, quantize_to_fp4
from fouroversix.utils import AdaptiveBlockScalingRule, DataType, FP4Format
from scipy.linalg import hadamard
from torch import nn

HBS = 16


class FP4LinearFunction(torch.autograd.Function):
    """Differentiable FP4 linear layer."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,  # noqa: A002
        weight_e2m1: torch.Tensor = None,
        weight_sf: torch.Tensor = None,
        weight_normconst: torch.Tensor = None,
        bias: torch.Tensor = None,
        had: torch.Tensor = None,
        fp4_format: FP4Format = FP4Format.nvfp4,
        a_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        w_scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        w_scale_2d: bool = False,  # noqa: FBT001, FBT002
        out_dtype: DataType = DataType.bfloat16,
        a_quantize_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor,]:
        """
        Perform an FP4 matrix multiplication. The input is provided in high precision
        and quantized to FP4 prior to the matrix multiplication, while the weight is
        provided in low precision.
        """

        ctx.save_for_backward(
            input,
            weight_e2m1,
            weight_sf,
            weight_normconst,
            bias,
            had,
        )

        ctx.fp4_format = fp4_format
        ctx.a_scale_rule = a_scale_rule
        ctx.w_scale_rule = w_scale_rule
        ctx.w_scale_2d = w_scale_2d
        ctx.out_dtype = out_dtype
        ctx.a_quantize_kwargs = a_quantize_kwargs

        if had is not None:
            ctx.mark_non_differentiable(had)

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
                b_e2m1=weight_e2m1,
                b_sf=weight_sf,
                b_normconst=weight_normconst,
                fp4_format=fp4_format,
                out_dtype=out_dtype,
                out_shape=(input.shape[1], weight_e2m1.shape[0]),
                a_quantize_kwargs={
                    "scale_rule": a_scale_rule,
                    "fp4_format": fp4_format,
                    **(a_quantize_kwargs or {}),
                },
            )

        if bias is not None:
            out = out + bias

        return (out,)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Backward pass for the FP4 linear layer."""
        # TODO(jack): Add implementation


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
        self.out_dtype = dtype

    def apply_ptq(self) -> None:
        """
        Prepare this layer for post-training quantization by quantizing the weight,
        storing the quantized weight, and deleting the original weight. This should not
        be done if the layer is used for training, as training requires storage of the
        high-precision weight.
        """

        self.weight_e2m1, self.weight_sf, self.weight_normconst = (
            self.quantized_weight()
        )

        del self.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        """Forward pass for the FP4 linear layer."""

        (out,) = FP4LinearFunction.apply(
            input,
            *self.quantized_weight(),
            self.bias,
            self.had,
            self.fp4_format,
            self.a_scale_rule,
            self.w_scale_rule,
            self.w_scale_2d,
            self.out_dtype,
            self.a_quantize_kwargs,
        )

        return out

    def quantized_weight(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the quantized weights."""

        if hasattr(self, "weight"):
            return quantize_to_fp4(
                self.weight,
                scale_rule=self.w_scale_rule,
                **(self.w_quantize_kwargs or {}),
            )

        return self.weight_e2m1, self.weight_sf, self.weight_normconst


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
        a_quantize_kwargs: dict[str, Any] | None = None,
        w_quantize_kwargs: dict[str, Any] | None = None,
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
        )

        self.had_gen = torch.Generator(device=device)
        self.had_gen.manual_seed(41)
        self.had = torch.FloatTensor(hadamard(HBS)).to(dtype=dtype, device=device) / (
            HBS**0.5
        )
        self.had = (
            self.had.T
            * (
                torch.randint(
                    0,
                    2,
                    (HBS,),
                    dtype=dtype,
                    device=device,
                    generator=self.had_gen,
                )
                * 2
                - 1
            )
        ).T.contiguous()
