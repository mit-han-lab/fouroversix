from dataclasses import dataclass

from fouroversix.utils import DataType, MatmulBackend, QuantizeBackend, ScaleRule


@dataclass
class FourOverSixLinearConfig:
    dtype: DataType = DataType.nvfp4
    keep_master_weights: bool = False
    matmul_backend: MatmulBackend | None = None
    quantize_backend: QuantizeBackend | None = None
    output_dtype: DataType = DataType.bfloat16
    scale_rule: ScaleRule = ScaleRule.mse
    weight_scale_2d: bool = False

    @property
    def activation_scale_rule(self) -> ScaleRule:
        return self._activation_scale_rule or self.scale_rule

    @activation_scale_rule.setter
    def activation_scale_rule(self, value: ScaleRule) -> None:
        self._activation_scale_rule = value

    @property
    def gradient_scale_rule(self) -> ScaleRule:
        return self._gradient_scale_rule or self.scale_rule

    @gradient_scale_rule.setter
    def gradient_scale_rule(self, value: ScaleRule) -> None:
        self._gradient_scale_rule = value

    @property
    def weight_scale_rule(self) -> ScaleRule:
        return self._weight_scale_rule or self.scale_rule

    @weight_scale_rule.setter
    def weight_scale_rule(self, value: ScaleRule) -> None:
        self._weight_scale_rule = value
