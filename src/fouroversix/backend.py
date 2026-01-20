from __future__ import annotations

from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from .utils import AdaptiveBlockScalingRule, FP4Format, RoundStyle

SM_100 = 10
SM_110 = 11
SM_120 = 12


class MatmulBackend(str, Enum):
    """
    Backends for matrix multiplication with FP4.

    - `cutlass`: CUTLASS implementation. This requires a Blackwell GPU.
    - `pytorch`: PyTorch implementation which first dequantizes the input tensors to
        bf16 and then performs a bf16 matrix multiplication.
    """

    cutlass = "cutlass"
    pytorch = "pytorch"

    @classmethod
    def auto_select(cls) -> MatmulBackend:
        """Select the fastest backend for the given parameters."""

        for backend in [cls.cutlass, cls.pytorch]:
            if backend.is_available():
                return backend

        msg = "No available backend found"
        raise ValueError(msg)

    def is_available(self) -> bool:
        """Check if the backend is available given the CUDA device and installation."""

        if self == MatmulBackend.cutlass:
            return torch.cuda.is_available() and torch.cuda.get_device_capability()[
                0
            ] in [SM_100, SM_110, SM_120]

        return True

    def fp4_matmul(  # noqa: C901
        self,
        a_e2m1: torch.Tensor,
        a_sf: torch.Tensor,
        a_amax: torch.Tensor,
        b_e2m1: torch.Tensor,
        b_sf: torch.Tensor,
        b_amax: torch.Tensor,
        *,
        fp4_format: FP4Format,
        scale_rule: AdaptiveBlockScalingRule,
        out_dtype: torch.dtype,
        out_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Perform a matrix multiplication with two FP4-quantized tensors. See frontend.py
        for more details.
        """

        if self == MatmulBackend.cutlass:
            from .ops import (
                gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt,
                gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt,
                gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120,
                gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt,
                gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120,
            )

            if fp4_format == FP4Format.mxfp4:
                alpha = torch.ones(1, device=a_e2m1.device, dtype=torch.float32)
            elif fp4_format == FP4Format.nvfp4:
                if scale_rule == AdaptiveBlockScalingRule.always_6:
                    alpha = ((a_amax * b_amax) / (6 * 6 * 448 * 448)).to(torch.float32)
                elif scale_rule == AdaptiveBlockScalingRule.always_4:
                    alpha = ((a_amax * b_amax) / (4 * 4 * 448 * 448)).to(torch.float32)
                else:
                    alpha = ((a_amax * b_amax) / (6 * 6 * 256 * 256)).to(torch.float32)

            gemm_fns = {
                (
                    SM_100,
                    FP4Format.mxfp4,
                    torch.bfloat16,
                ): gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt,
                (
                    SM_100,
                    FP4Format.nvfp4,
                    torch.bfloat16,
                ): gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt,
                (
                    SM_120,
                    FP4Format.nvfp4,
                    torch.bfloat16,
                ): gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120,
                (
                    SM_100,
                    FP4Format.nvfp4,
                    torch.float16,
                ): gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt,
                (
                    SM_120,
                    FP4Format.nvfp4,
                    torch.float16,
                ): gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120,
            }

            gemm_fn = gemm_fns.get(
                (torch.cuda.get_device_capability()[0], fp4_format, out_dtype),
            )

            if gemm_fn is None:
                msg = (
                    "No gemm function found for the given device capability and "
                    f"out_dtype: {torch.cuda.get_device_capability()[0]}, {out_dtype}"
                )
                raise ValueError(msg)

            out = gemm_fn(a_e2m1, b_e2m1, a_sf, b_sf, alpha)

            if out_shape is not None and out.shape != out_shape:
                out = out[: out_shape[0], : out_shape[1]]

            return out

        if self == MatmulBackend.pytorch:
            from .quantize.reference import dequantize_from_fp4

            a = dequantize_from_fp4(
                a_e2m1,
                a_sf,
                dtype=torch.float32,
                fp4_format=fp4_format,
            )
            b = dequantize_from_fp4(
                b_e2m1,
                b_sf,
                dtype=torch.float32,
                fp4_format=fp4_format,
            )

            # Fix mismatched shapes introduced by padding during quantization
            if a.shape[1] > b.shape[1]:
                b = F.pad(b, (0, a.shape[1] - b.shape[1]))
            elif b.shape[1] > a.shape[1]:
                a = F.pad(a, (0, b.shape[1] - a.shape[1]))

            out = ((a @ b.T).float() * a_amax / (6 * 448) * b_amax / (6 * 448)).to(
                out_dtype,
            )

            return out[: out_shape[0], : out_shape[1]] if out_shape is not None else out

        msg = f"Invalid backend: {self}"
        raise ValueError(msg)


class QuantizeBackend(str, Enum):
    """
    Backends for quantizing a tensor to NVFP4 or MXFP4.

    - `cuda`: CUDA implementation. Requires a Blackwell GPU, and currently only supports
        the forward pass for PTQ (no stochastic rounding, no transposed matrices, no
        RHT, no 2D block scaling).
    - `pytorch`: PyTorch implementation.
    - `triton`: Triton implementation. Requires a Blackwell GPU.
    """

    cuda = "cuda"
    pytorch = "pytorch"
    transformer_engine = "transformer_engine"
    triton = "triton"

    @classmethod
    def auto_select(
        cls,
        x: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> QuantizeBackend:
        """Select the fastest backend for the given quantization parameters."""

        for backend in [cls.cuda, cls.triton, cls.pytorch]:
            if backend.is_supported(x, **kwargs):
                return backend

        msg = f"No backend found for the given parameters: {kwargs}"
        raise ValueError(msg)

    def is_available(self) -> bool:
        """Check if the backend can be used given the CUDA device and installation."""

        if self == QuantizeBackend.cuda:
            # TODO(jack, junxian): Re-enable CUDA backend once precision issues are
            # resolved
            return False

        if self == QuantizeBackend.triton:  # noqa: SIM102
            if not torch.cuda.is_available() or torch.cuda.get_device_capability()[
                0
            ] not in [SM_100, SM_110, SM_120]:
                return False

        if self == QuantizeBackend.transformer_engine:
            return torch.cuda.is_available()

        return True

    def is_supported(  # noqa: C901, PLR0911
        self,
        x: torch.Tensor,
        *,
        scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        block_scale_2d: bool = False,
        had: torch.Tensor | None = None,
        fp4_format: FP4Format = FP4Format.nvfp4,
        round_style: RoundStyle = RoundStyle.nearest,
        transpose: bool = False,
    ) -> bool:
        """Check if the backend supports the given quantization parameters."""

        if x.ndim != 2:  # noqa: PLR2004
            return False

        if not self.is_available():
            return False

        if fp4_format == FP4Format.mxfp4 and scale_rule not in (
            AdaptiveBlockScalingRule.always_6,
            AdaptiveBlockScalingRule.always_4,
        ):
            msg = (
                "MXFP4 quantization only supports the `always_6` and `always_4` scale "
                "rules"
            )
            raise ValueError(msg)

        if self == QuantizeBackend.cuda:
            return (
                had is None
                and fp4_format == FP4Format.nvfp4
                and round_style == RoundStyle.nearest
                and not block_scale_2d
                and not transpose
            )

        if self == QuantizeBackend.pytorch:
            return True

        if self == QuantizeBackend.triton:
            if round_style == RoundStyle.stochastic:
                return torch.cuda.get_device_capability()[0] == SM_100

            return True

        if self == QuantizeBackend.transformer_engine:
            if (
                fp4_format != FP4Format.nvfp4
                or scale_rule != AdaptiveBlockScalingRule.always_6
            ):
                return False

            if not transpose and had is not None:
                return False

            if transpose and had is not None and block_scale_2d:  # noqa: SIM103
                return False

            return True

        msg = f"Invalid backend: {self}"
        raise ValueError(msg)

    def quantize_to_fp4(
        self,
        x: torch.Tensor,
        *,
        scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
        block_scale_2d: bool = False,
        had: torch.Tensor | None = None,
        fp4_format: FP4Format = FP4Format.nvfp4,
        round_style: RoundStyle = RoundStyle.nearest,
        transpose: bool = False,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Quantize a tensor to FP4. See frontend.py for more details."""

        if self == QuantizeBackend.cuda:
            from .ops import quantize_to_fp4

            return quantize_to_fp4(
                x,
                fp4_format == FP4Format.nvfp4,
                round_style == RoundStyle.nearest,
                had is not None,
                transpose,
                scale_rule.cuda_id(),
                **kwargs,
            )

        if self == QuantizeBackend.transformer_engine:
            from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

            from .quantize.reference import to_blocked

            q = NVFP4Quantizer(
                with_2d_quantization=block_scale_2d,
                with_rht=had is not None,
                with_post_rht_amax=had is not None,
                stochastic_rounding=round_style == RoundStyle.stochastic,
            )

            out = q.quantize(x)

            if transpose:
                return (
                    out._columnwise_data,  # noqa: SLF001
                    to_blocked(out._columnwise_scale_inv.view(torch.float8_e4m3fn)),  # noqa: SLF001
                    out._amax_columnwise,  # noqa: SLF001
                )
            return (
                out._rowwise_data,  # noqa: SLF001
                to_blocked(out._rowwise_scale_inv.view(torch.float8_e4m3fn)),  # noqa: SLF001
                out._amax_rowwise,  # noqa: SLF001
            )

        if self == QuantizeBackend.triton:
            from .quantize.triton_kernel import quantize_to_fp4

            return quantize_to_fp4(
                x,
                had=had,
                fp4_format=fp4_format,
                round_style=round_style,
                scale_rule=scale_rule,
                block_scale_2d=block_scale_2d,
                transpose=transpose,
                **kwargs,
            )

        if self == QuantizeBackend.pytorch:
            from .quantize.reference import quantize_to_fp4

            rows_div = 128
            cols_div = 64

            if x.shape[0] % rows_div != 0 or x.shape[1] % cols_div != 0:
                x = F.pad(
                    x,
                    (
                        0,
                        cols_div - (x.shape[1] % cols_div),
                        0,
                        rows_div - (x.shape[0] % rows_div),
                    ),
                )

            return quantize_to_fp4(
                x,
                had=had,
                fp4_format=fp4_format,
                round_style=round_style,
                scale_rule=scale_rule,
                block_scale_2d=block_scale_2d,
                transpose=transpose,
                **kwargs,
            )

        msg = f"Invalid backend: {self}"
        raise ValueError(msg)


def quantize_to_fp4(
    x: torch.Tensor,
    *,
    backend: QuantizeBackend | None = None,
    scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    block_scale_2d: bool = False,
    had: torch.Tensor | None = None,
    fp4_format: FP4Format = FP4Format.nvfp4,
    round_style: RoundStyle = RoundStyle.nearest,
    transpose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Quantize a tensor to FP4.

    Args:
        x (torch.Tensor): The input tensor to quantize.
        backend (QuantizeBackend | None): The backend to use. If None, the fastest
            backend that supports the given parameters will be selected automatically.
        scale_rule (AdaptiveBlockScalingRule): The block scale selection
            rule to use, e.g. `AdaptiveBlockScalingRule.always_6` for normal NVFP4
            quantization, or `AdaptiveBlockScalingRule.mse` for 4/6 with MSE selection.
        block_scale_2d (bool): Whether to use 2D block scaling.
        had (torch.Tensor | None): The Hadamard matrix to use.
        fp4_format (FP4Format): The FP4 format to use, either `FP4Format.nvfp4` or
            `FP4Format.mxfp4`.
        round_style (RoundStyle): The rounding style to use, either `RoundStyle.nearest`
            for round-to-nearest, or `RoundStyle.stochastic` for stochastic rounding.
        transpose (bool): Whether to transpose the input tensor before quantization.

    Returns:
        A tuple containing the E2M1 values, the scale factors, and the per-tensor
            normalization constant (if used).

    """

    kwargs = {
        "scale_rule": scale_rule,
        "block_scale_2d": block_scale_2d,
        "had": had,
        "fp4_format": fp4_format,
        "round_style": round_style,
        "transpose": transpose,
    }

    if backend is None:
        backend = QuantizeBackend.auto_select(x, **kwargs)
    elif not backend.is_supported(x, **kwargs):
        msg = f"Backend {backend} does not support the given parameters"
        raise ValueError(msg)

    return backend.quantize_to_fp4(x, **kwargs)
