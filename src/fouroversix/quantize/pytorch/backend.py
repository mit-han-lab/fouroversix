import torch
import torch.nn.functional as F
from fouroversix.quantize.backend import QuantizeBackendBase
from fouroversix.quantize.config import QuantizationConfig
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.quantize.utils import get_rht_matrix
from fouroversix.utils import DataType

from .reference import quantize_to_fp4


class PyTorchQuantizeBackend(QuantizeBackendBase):
    def is_available(self) -> bool:
        return True

    def is_supported(self, x: torch.Tensor, config: QuantizationConfig) -> bool:
        return True

    def quantize_to_fp4(
        self, x: torch.Tensor, config: QuantizationConfig,
    ) -> QuantizedTensor:
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
