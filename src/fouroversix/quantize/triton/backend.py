import torch
from fouroversix.quantize.backend import QuantizeBackendBase
from fouroversix.quantize.config import QuantizationConfig
from fouroversix.quantize.quantized_tensor import QuantizedTensor
from fouroversix.quantize.utils import get_rht_matrix
from fouroversix.utils import SM_100, SM_110, SM_120, RoundStyle


class TritonQuantizeBackend(QuantizeBackendBase):

    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in {
            SM_100,
            SM_110,
            SM_120,
        }

    @classmethod
    def is_supported(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        if not super().is_supported(x, config):
            return False

        if config.round_style == RoundStyle.stochastic:
            return torch.cuda.get_device_capability()[0] == SM_100

        return True

    @classmethod
    def quantize_to_fp4(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        from .kernel import quantize_to_fp4

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
