from importlib.metadata import version

from .backend import MatmulBackend, QuantizeBackend
from .frontend import fp4_matmul, quantize_to_fp4
from .model import quantize_model
from .quantize import FP4Tensor
from .utils import AdaptiveBlockScalingRule, DataType, FP4Format, RoundStyle

__version__ = version("fouroversix")

__all__ = [
    "AdaptiveBlockScalingRule",
    "DataType",
    "FP4Format",
    "FP4Tensor",
    "MatmulBackend",
    "QuantizeBackend",
    "RoundStyle",
    "fp4_matmul",
    "quantize_model",
    "quantize_to_fp4",
]
