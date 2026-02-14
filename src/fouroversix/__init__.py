from importlib.metadata import version

from .matmul import fp4_matmul
from .model import (
    FourOverSixLinear,
    ModelQuantizationConfig,
    ModuleQuantizationConfig,
    QuantizedModule,
    quantize_model,
)
from .quantize import QuantizationConfig, QuantizedTensor, quantize_to_fp4
from .utils import DataType, MatmulBackend, QuantizeBackend, RoundStyle, ScaleRule

__version__ = version("fouroversix")

__all__ = [
    "DataType",
    "FourOverSixLinear",
    "MatmulBackend",
    "ModelQuantizationConfig",
    "ModuleQuantizationConfig",
    "QuantizationConfig",
    "QuantizeBackend",
    "QuantizedModule",
    "QuantizedTensor",
    "RoundStyle",
    "ScaleRule",
    "fp4_matmul",
    "quantize_model",
    "quantize_to_fp4",
]
