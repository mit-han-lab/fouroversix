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
from .weight_conversions import WeightConversions

try:
    from .diffusers import FourOverSixConfig
except ImportError:
    FourOverSixConfig = None

__version__ = version("fouroversix")

__all__ = [
    "DataType",
    "FourOverSixConfig",
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
    "WeightConversions",
    "fp4_matmul",
    "quantize_model",
    "quantize_to_fp4",
]
