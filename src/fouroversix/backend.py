from __future__ import annotations

from enum import Enum
from typing import Any

import torch

from .utils import BlockScaleSelectionRule, DataType, FP4Format, RoundStyle


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
    def auto_select(cls, **kwargs: dict[str, Any]) -> MatmulBackend:
        """Select the fastest backend for the given parameters."""

        for backend in [cls.cutlass, cls.pytorch]:
            if backend.is_supported(**kwargs):
                return backend

        msg = f"No backend found for the given parameters: {kwargs}"
        raise ValueError(msg)

    def is_supported(self, **kwargs: dict[str, Any]) -> bool:  # noqa: ARG002
        """Check if the backend supports the given parameters."""

        if self == MatmulBackend.cutlass:
            return (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] == 10  # noqa: PLR2004
            )

        return True

    def fp4_matmul(  # noqa: C901, PLR0912
        self,
        a_e2m1: torch.Tensor,
        a_sf: torch.Tensor,
        a_normconst: torch.Tensor,
        b_e2m1: torch.Tensor,
        b_sf: torch.Tensor,
        b_normconst: torch.Tensor,
        *,
        fp4_format: FP4Format,
        out_dtype: DataType,
        out_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Perform a matrix multiplication with two FP4-quantized tensors.

        Args:
            a_e2m1 (torch.Tensor): The values of the first input tensor in packed E2M1
                format (2 values per byte).
            a_sf (torch.Tensor): The scale factors of the first input tensor.
            a_normconst (torch.Tensor): The per-tensor normalization constant of the
                first input tensor.
            b_e2m1 (torch.Tensor): The values of the second input tensor in packed E2M1
                format (2 values per byte).
            b_sf (torch.Tensor): The scale factors of the second input tensor.
            b_normconst (torch.Tensor): The per-tensor normalization constant of the
                second input tensor.
            fp4_format (FP4Format): The FP4 format of the input tensors, either
                `FP4Format.nvfp4` or `FP4Format.mxfp4`.
            out_dtype (DataType): The data type of the output tensor, either
                `DataType.bfloat16` or `DataType.float16`.
            out_shape (tuple[int, int] | None): The shape of the output tensor. This is
                helpful when the input tensors have shapes that are not multiples of 64,
                but which were padded to multiples of 64 during quantization.

        Returns:
            The output tensor.

        """

        if self == MatmulBackend.cutlass:
            from .ops import (
                gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt,
                gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt,
                gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt,
            )

            if fp4_format == FP4Format.mxfp4:
                alpha = torch.ones(1, device=a_e2m1.device, dtype=torch.float32)
            elif fp4_format == FP4Format.nvfp4:
                alpha = (a_normconst * b_normconst).to(torch.float32)

            if fp4_format == FP4Format.mxfp4:
                if out_dtype == DataType.bfloat16:
                    out = gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt(
                        a_e2m1,
                        b_e2m1,
                        a_sf,
                        b_sf,
                        alpha,
                    )
                else:
                    msg = f"Invalid out_dtype for mxfp4: {out_dtype}"
                    raise ValueError(msg)
            elif fp4_format == FP4Format.nvfp4:
                if out_dtype == DataType.bfloat16:
                    out = gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt(
                        a_e2m1,
                        b_e2m1,
                        a_sf,
                        b_sf,
                        alpha,
                    )
                elif out_dtype == DataType.float16:
                    out = gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt(
                        a_e2m1,
                        b_e2m1,
                        a_sf,
                        b_sf,
                        alpha,
                    )
                else:
                    msg = f"Invalid out_dtype for nvfp4: {out_dtype}"
                    raise ValueError(msg)
            else:
                msg = f"Invalid fp4_format: {fp4_format}"
                raise ValueError(msg)

            if out_shape is not None:
                return out[: out_shape[0], : out_shape[1]]

            return out

        if self == MatmulBackend.pytorch:
            from .quantize.reference import dequantize_from_fp4, from_blocked

            a = dequantize_from_fp4(
                a_e2m1,
                from_blocked(a_sf, (a_e2m1.shape[0], a_e2m1.shape[1] // 8)),
                a_normconst,
            )

            b = dequantize_from_fp4(
                b_e2m1,
                from_blocked(b_sf, (b_e2m1.shape[0], b_e2m1.shape[1] // 8)),
                b_normconst,
            )

            return a @ b.T

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

    def is_supported(
        self,
        x: torch.Tensor,
        *,
        block_scale_selection_rule: BlockScaleSelectionRule = (
            BlockScaleSelectionRule.always_6
        ),
        block_scale_2d: bool = False,
        had: torch.Tensor | None = None,
        fp4_format: FP4Format = FP4Format.nvfp4,
        round_style: RoundStyle = RoundStyle.nearest,
        transpose: bool = False,
    ) -> bool:
        """Check if the backend supports the given quantization parameters."""

        if x.ndim != 2:  # noqa: PLR2004
            return False

        if self == QuantizeBackend.cuda:
            if (
                not torch.cuda.is_available()
                or torch.cuda.get_device_capability()[0] != 10  # noqa: PLR2004
            ):
                return False

            try:
                import fouroversix._C  # noqa: F401
            except ModuleNotFoundError:
                return False

            return (
                had is None
                and fp4_format == "nvfp4"
                and round_style == "nearest"
                and not block_scale_2d
                and not transpose
            )

        if self == QuantizeBackend.pytorch:
            return (
                block_scale_selection_rule == BlockScaleSelectionRule.always_6
            ) and (x.shape[1] % 16 == 0)

        if self == QuantizeBackend.triton:
            return (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] == 10  # noqa: PLR2004
            )

        msg = f"Invalid backend: {self}"
        raise ValueError(msg)

    def quantize_to_fp4(
        self,
        x: torch.Tensor,
        *,
        block_scale_selection_rule: BlockScaleSelectionRule = (
            BlockScaleSelectionRule.always_6
        ),
        block_scale_2d: bool = False,
        had: torch.Tensor | None = None,
        fp4_format: FP4Format = FP4Format.nvfp4,
        round_style: RoundStyle = RoundStyle.nearest,
        transpose: bool = False,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Quantize a tensor to FP4. See `quantize_to_fp4` for more details."""

        if self == QuantizeBackend.cuda:
            from .ops import quantize_to_fp4

            return quantize_to_fp4(
                x,
                fp4_format == FP4Format.nvfp4,
                round_style == RoundStyle.nearest,
                had is not None,
                transpose,
                block_scale_selection_rule.cuda_id(),
                **kwargs,
            )

        if self == QuantizeBackend.triton:
            from .quantize.triton_kernel import quantize_to_fp4

            return quantize_to_fp4(
                x,
                had=had,
                fp4_format=fp4_format,
                round_style=round_style,
                block_scale_selection_rule=block_scale_selection_rule,
                block_scale_2d=block_scale_2d,
                transpose=transpose,
                **kwargs,
            )

        if self == QuantizeBackend.pytorch:
            from .quantize.reference import quantize_to_fp4

            return quantize_to_fp4(
                x,
                had=had,
                fp4_format=fp4_format,
                round_style=round_style,
                block_scale_selection_rule=block_scale_selection_rule,
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
    block_scale_selection_rule: BlockScaleSelectionRule = (
        BlockScaleSelectionRule.always_6
    ),
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
        block_scale_selection_rule (BlockScaleSelectionRule): The block scale selection
            rule to use, e.g. `BlockScaleSelectionRule.always_6` for normal NVFP4
            quantization, or `BlockScaleSelectionRule.mse` for 4/6 with MSE selection.
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
        "block_scale_selection_rule": block_scale_selection_rule,
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
