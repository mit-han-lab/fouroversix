from abc import ABC, abstractmethod

import torch
from fouroversix.utils import DataType, ScaleRule

from .config import QuantizationConfig
from .quantized_tensor import QuantizedTensor


class QuantizeBackendBase(ABC):

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool: ...

    @classmethod
    @abstractmethod
    def is_supported(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        if not cls.is_available():
            return False

        if x.ndim != 2:
            return False

        if config.dtype not in {DataType.mxfp4, DataType.nvfp4}:
            return False

        if config.dtype == DataType.mxfp4 and config.scale_rule not in {
            ScaleRule.static_6,
            ScaleRule.static_4,
        }:
            msg = (
                "MXFP4 quantization only supports the `static_6` and `static_4` scale "
                "rules"
            )
            raise ValueError(msg)

        return True

    @classmethod
    @abstractmethod
    def quantize_to_fp4(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor: ...
