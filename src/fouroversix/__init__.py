from importlib.metadata import version

from .matmul import fp4_matmul
from .model import FourOverSixLayerConfig, FourOverSixLinear, quantize_model
from .quantize import QuantizationConfig, QuantizedTensor, quantize_to_fp4
from .utils import DataType, MatmulBackend, QuantizeBackend, RoundStyle, ScaleRule

__version__ = version("fouroversix")

__all__ = [
    "DataType",
    "FourOverSixLayerConfig",
    "FourOverSixLinear",
    "MatmulBackend",
    "QuantizationConfig",
    "QuantizeBackend",
    "QuantizedTensor",
    "RoundStyle",
    "ScaleRule",
    "fp4_matmul",
    "quantize_model",
    "quantize_to_fp4",
]
