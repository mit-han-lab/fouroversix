from importlib.metadata import version

from .matmul import fp4_matmul
from .model import FourOverSixLinear, FourOverSixLinearConfig, quantize_model
from .quantize import QuantizationConfig, quantize_to_fp4
from .utils import DataType, MatmulBackend, QuantizeBackend, RoundStyle, ScaleRule

__version__ = version("fouroversix")

__all__ = [
    "DataType",
    "FourOverSixLinear",
    "FourOverSixLinearConfig",
    "MatmulBackend",
    "QuantizationConfig",
    "QuantizeBackend",
    "RoundStyle",
    "ScaleRule",
    "fp4_matmul",
    "quantize_model",
    "quantize_to_fp4",
]
