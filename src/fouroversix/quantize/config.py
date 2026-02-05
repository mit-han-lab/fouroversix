from dataclasses import dataclass

from fouroversix.utils import DataType, QuantizeBackend, RoundStyle, ScaleRule


@dataclass
class QuantizationConfig:
    backend: QuantizeBackend | None = None
    block_scale_2d: bool = False
    dtype: DataType = DataType.nvfp4
    rht: bool = False
    round_style: RoundStyle = RoundStyle.nearest
    scale_rule: ScaleRule = ScaleRule.mse
    transpose: bool = False
