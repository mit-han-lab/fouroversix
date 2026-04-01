import torch
import torch.nn.functional as F
from fouroversix.quantize.backend import QuantizeBackendBase
from fouroversix.quantize.config import QuantizationConfig
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.quantize.utils import get_rht_matrix
from fouroversix.utils import DataType

from .reference import quantize


class PyTorchQuantizeBackend(QuantizeBackendBase):
    """
    The PyTorch quantization backend. Supports all quantization options, and can be run
    on non-Blackwell GPUs, but is slow. Should be used primarily as a reference.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the PyTorch backend is available on the current machine."""
        return True

    @classmethod
    def can_quantize(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> bool:
        """
        Return True if the PyTorch backend supports the given input and quantization
        configuration.
        """

        if config.pseudo_quantize:
            return False

        return super().can_quantize(x, config)

    @classmethod
    def quantize(
        cls,
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

        input_shape = (x.shape[1], x.shape[0]) if config.transpose else x.shape

        rows_div = 128
        cols_div = 4 * config.dtype.block_size

        if input_shape[0] % rows_div != 0 or input_shape[1] % cols_div != 0:
            x = F.pad(
                x,
                (
                    0,
                    (
                        cols_div - (input_shape[1] % cols_div)
                        if input_shape[1] % cols_div > 0
                        else 0
                    ),
                    0,
                    (
                        rows_div - (input_shape[0] % rows_div)
                        if input_shape[0] % rows_div > 0
                        else 0
                    ),
                ),
            )

        if x.device.type == "meta":
            values = torch.zeros(
                input_shape[0],
                input_shape[1] // config.dtype.quantized_value_type.packing_factor,
                device=x.device,
                dtype=torch.uint8,
            )
            scale_factors = torch.zeros(
                input_shape[0] * input_shape[1] // config.dtype.block_size,
                device=x.device,
                dtype=(
                    torch.uint8
                    if config.dtype.scale_type.torch_dtype == torch.float8_e8m0fnu
                    else config.dtype.scale_type.torch_dtype
                ),
            )
            amax = torch.zeros(1, device=x.device, dtype=torch.float32)
        else:
            values, scale_factors, amax = quantize(
                x,
                had=get_rht_matrix() if config.rht else None,
                fp4_format=config.dtype,
                round_style=config.round_style,
                scale_rule=config.scale_rule,
                block_scale_2d=config.block_scale_2d,
                transpose=config.transpose,
                use_blackwell_scale_layout=config.dtype
                not in {DataType.if4, DataType.nvint4},
            )

        return QuantizedTensor(
            values,
            scale_factors,
            amax,
            config.dtype,
            input_shape,
            config.scale_rule,
            config.round_style,
            scale_factors_are_in_blackwell_layout=config.dtype
            not in {DataType.if4, DataType.nvint4},
        )
