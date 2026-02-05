import torch
from fouroversix.quantize.utils import to_blocked
from fouroversix.utils import DataType, RoundStyle, ScaleRule

from .backend import QuantizeBackendBase
from .config import QuantizationConfig
from .quantized_tensor import QuantizedTensor


class TransformerEngineQuantizeBackend(QuantizeBackendBase):
    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def is_supported(self, x: torch.Tensor, config: QuantizationConfig) -> bool:
        if not super().is_supported(x, config):
            return False

        if config.dtype != DataType.nvfp4 or config.scale_rule != ScaleRule.static_6:
            return False

        if not config.transpose and config.rht:
            return False

        if config.transpose and config.rht and config.block_scale_2d:  # noqa: SIM103
            return False

        return True

    def quantize_to_fp4(
        self, x: torch.Tensor, config: QuantizationConfig,
    ) -> QuantizedTensor:
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

        q = NVFP4Quantizer(
            with_2d_quantization=config.block_scale_2d,
            with_rht=config.rht,
            with_post_rht_amax=config.rht,
            stochastic_rounding=config.round_style == RoundStyle.stochastic,
        )

        out = q.quantize(x)

        if config.transpose:
            values = out._columnwise_data  # noqa: SLF001
            scale_factors = to_blocked(
                out._columnwise_scale_inv.view(torch.float8_e4m3fn),  # noqa: SLF001
            )
            amax = out._amax_columnwise  # noqa: SLF001
        else:
            values = out._rowwise_data  # noqa: SLF001
            scale_factors = to_blocked(
                out._rowwise_scale_inv.view(torch.float8_e4m3fn),  # noqa: SLF001
            )
            amax = out._amax_rowwise  # noqa: SLF001

        return QuantizedTensor(
            values,
            scale_factors,
            amax,
            config.dtype,
            (x.shape[1], x.shape[0]) if config.transpose else x.shape,
            config.scale_rule,
        )
