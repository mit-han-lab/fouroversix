import torch
from fouroversix.utils import DataType, RoundStyle

from .backend import QuantizeBackendBase
from .config import QuantizationConfig
from .quantized_tensor import QuantizedTensor


class CUDAQuantizeBackend(QuantizeBackendBase):

    @classmethod
    def is_available(self) -> bool:
        # TODO(jack, junxian): Re-enable CUDA backend once precision issues are resolved
        return False

    @classmethod
    def is_supported(self, x: torch.Tensor, config: QuantizationConfig) -> bool:
        if not super().is_supported(x, config):
            return False

        return (
            not config.rht
            and config.dtype == DataType.nvfp4
            and config.round_style == RoundStyle.nearest
            and not config.block_scale_2d
            and not config.transpose
        )

    @classmethod
    def quantize_to_fp4(
        self, x: torch.Tensor, config: QuantizationConfig,
    ) -> QuantizedTensor:
        msg = "The CUDA backend is currently disabled and will be updated soon"
        raise NotImplementedError(msg)
