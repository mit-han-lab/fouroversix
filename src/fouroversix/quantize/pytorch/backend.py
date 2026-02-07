import torch
import torch.nn.functional as F
from fouroversix.quantize.backend import QuantizeBackendBase
from fouroversix.quantize.config import QuantizationConfig
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.quantize.utils import get_rht_matrix
from fouroversix.utils import DataType

from .reference import quantize_to_fp4


class PyTorchQuantizeBackend(QuantizeBackendBase):
    """
    The PyTorch quantization backend. Supports all quantization options, and can be run
    on non-Blackwell GPUs, but is slow. Should be used primarily as a reference.
    """

    def is_available(self) -> bool:
        """Return True if the PyTorch backend is available on the current machine."""
        return True

    def is_supported(
        self,
        x: torch.Tensor,  # noqa: ARG002
        config: QuantizationConfig,  # noqa: ARG002
    ) -> bool:
        """
        Return True if the PyTorch backend supports the given input and quantization
        configuration.
        """

        return True

    def quantize_to_fp4(
        self,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        """
        Quantize a tensor to FP4 using the PyTorch backend.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            config (QuantizationConfig): The quantization configuration.

        Returns:
            The quantized tensor.

        """

        rows_div = 128
        cols_div = 64 if config.dtype == DataType.nvfp4 else 128

        if x.shape[0] % rows_div != 0 or x.shape[1] % cols_div != 0:
            x = F.pad(
                x,
                (
                    0,
                    (
                        cols_div - (x.shape[1] % cols_div)
                        if x.shape[1] % cols_div > 0
                        else 0
                    ),
                    0,
                    (
                        rows_div - (x.shape[0] % rows_div)
                        if x.shape[0] % rows_div > 0
                        else 0
                    ),
                ),
            )

        values, scale_factors, amax = quantize_to_fp4(
            x,
            had=get_rht_matrix() if config.rht else None,
            fp4_format=config.dtype,
            round_style=config.round_style,
            scale_rule=config.scale_rule,
            block_scale_2d=config.block_scale_2d,
            transpose=config.transpose,
        )

        return QuantizedTensor(
            values,
            scale_factors,
            amax,
            config.dtype,
            (x.shape[1], x.shape[0]) if config.transpose else x.shape,
            config.scale_rule,
        )
