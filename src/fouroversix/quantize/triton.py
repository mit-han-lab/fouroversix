import functools

import torch
from fouroversix.quantize.backend import QuantizeBackendBase
from fouroversix.quantize.config import QuantizationConfig
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.quantize.utils import get_rht_matrix
from fouroversix.utils import BLACKWELL_SM_IDS, DataType


class TritonQuantizeBackend(QuantizeBackendBase):
    """
    The Triton quantization backend. Supports all parameters required for efficient
    NVFP4 training, including stochastic rounding, the random Hadamard transform,
    transposed inputs, and 2D block scaling. Requires a Blackwell GPU.
    """

    @classmethod
    @functools.lru_cache
    def is_available(cls) -> bool:
        """Return True if the Triton backend is available on the current machine."""
        return torch.cuda.is_available()

    @classmethod
    def can_dequantize_values(cls, tensor: QuantizedTensor) -> bool:
        """
        Return True if the Triton backend can dequantize the values of the given
        quantized tensor.
        """

        # FP6 dequantize kernel only works on Blackwell
        return (
            tensor.dtype
            in {
                DataType.mxfp3,
                DataType.mxfp6_e2m3,
                DataType.mxfp6_e3m2,
                DataType.nvfp3,
                DataType.nvfp3_bs8,
                DataType.nvfp6_e2m3,
                DataType.nvfp6_e3m2,
                DataType.nvint3,
                DataType.nvint3_bs8,
                DataType.nvint6,
                DataType.if3,
                DataType.if3_bs8,
                DataType.if6_e2m3,
                DataType.if6_e3m2,
                DataType.mxfp3_bs8,
            }
            and torch.cuda.get_device_capability()[0] in BLACKWELL_SM_IDS
        )

    @classmethod
    def can_quantize(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        """
        Return True if the Triton backend supports the given input and quantization
        configuration.
        """

        if not super().can_quantize(x, config):
            return False

        return x.device.type == "cuda"

    @classmethod
    def dequantize_values(
        cls,
        tensor: QuantizedTensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Get a high-precision representation of the raw values of a quantized tensor.
        Note that this returns raw values only, and not the fully dequantized tensor.
        """

        if tensor.dtype in {
            DataType.mxfp3,
            DataType.mxfp6_e2m3,
            DataType.mxfp6_e3m2,
            DataType.nvfp3,
            DataType.nvfp3_bs8,
            DataType.nvfp6_e2m3,
            DataType.nvfp6_e3m2,
            DataType.nvint3,
            DataType.nvint3_bs8,
            DataType.nvint6,
            DataType.if3,
            DataType.if3_bs8,
            DataType.if6_e2m3,
            DataType.if6_e3m2,
            DataType.mxfp3_bs8,
        }:
            from fouroversix.kernels.triton import dequantize_values

            return dequantize_values(tensor, dtype=dtype)

        return super().dequantize_values(tensor, dtype=dtype)

    @classmethod
    def quantize(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        """
        Quantize a tensor to FP4 using the Triton backend.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            config (QuantizationConfig): The quantization configuration.

        Returns:
            The quantized tensor.

        """

        from fouroversix.kernels.triton import quantize

        values, scale_factors, amax = quantize(
            x,
            had=get_rht_matrix() if config.rht else None,
            dtype=config.dtype,
            round_style=config.round_style,
            scale_rule=config.scale_rule,
            block_scale_2d=config.block_scale_2d,
            transpose=config.transpose,
            **config.kwargs,
        )

        return QuantizedTensor(
            values,
            scale_factors,
            amax,
            config.dtype,
            (x.shape[1], x.shape[0]) if config.transpose else x.shape,
            config.scale_rule,
            config.round_style,
            scale_factors_are_in_blackwell_layout=config.dtype
            not in {
                DataType.if3,
                DataType.if3_bs8,
                DataType.if4,
                DataType.if4_bs8,
                DataType.nvint3,
                DataType.nvint3_bs8,
                DataType.nvint4,
                DataType.nvint4_bs8,
                DataType.nvint6,
            },
        )

    @classmethod
    def pseudo_quantize(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> torch.Tensor:
        """
        Pseudo-quantize a tensor to FP4 using the Triton backend.

        Args:
            x (torch.Tensor): The input tensor to pseudo-quantize.
            config (QuantizationConfig): The quantization configuration.

        Returns:
            The pseudo-quantized tensor.

        """

        from fouroversix.kernels.triton import pseudo_quantize

        return pseudo_quantize(
            x,
            had=get_rht_matrix() if config.rht else None,
            dtype=config.dtype,
            round_style=config.round_style,
            scale_rule=config.scale_rule,
            block_scale_2d=config.block_scale_2d,
            transpose=config.transpose,
            **config.kwargs,
        )
