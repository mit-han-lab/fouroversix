from dataclasses import dataclass

from fouroversix.utils import DataType, MatmulBackend, QuantizeBackend, ScaleRule


@dataclass
class FourOverSixLinearConfig:
    """
    Configuration for the FourOverSixLinear layer.

    Args:
        activation_scale_rule (ScaleRule | None): The scaling rule to use for activation
            tensors. If not provided, `scale_rule` will be used.
        dtype (DataType): The quantization data type to use for the layer. Defaults to
            `DataType.nvfp4`.
        gradient_scale_rule (ScaleRule | None): The scaling rule to use for gradient
            tensors. If not provided, `scale_rule` will be used.
        keep_master_weights (bool): Whether to keep the master weights. Defaults to
            `False`.
        matmul_backend (MatmulBackend | None): The backend to use for matrix
            multiplications. If not provided, a backend will be selected automatically
            based on the available GPU and the specified options.
        quantize_backend (QuantizeBackend | None): The backend to use for quantization.
            If not provided, a backend will be selected automatically based on the
            available GPU and the specified options.
        output_dtype (DataType): The data type to use for the layer's output. Defaults
            to `DataType.bfloat16`.
        scale_rule (ScaleRule): The fallback scaling rule which will be used if any of
            the other scaling rules are not specified.
        weight_scale_2d (bool): Whether to use 2D block scaling for weights. Should be
            set to `True` if the layer is used for training.
        weight_scale_rule (ScaleRule | None): The scaling rule to use for weights. If
            not provided, `scale_rule` will be used.

    """

    activation_scale_rule: ScaleRule | None = None
    dtype: DataType = DataType.nvfp4
    gradient_scale_rule: ScaleRule | None = None
    keep_master_weights: bool = False
    matmul_backend: MatmulBackend | None = None
    quantize_backend: QuantizeBackend | None = None
    output_dtype: DataType = DataType.bfloat16
    scale_rule: ScaleRule = ScaleRule.mse
    weight_scale_2d: bool = False
    weight_scale_rule: ScaleRule | None = None

    def get_activation_scale_rule(self) -> ScaleRule:
        """Return the scaling rule to use for activation tensors."""
        return self.activation_scale_rule or self.scale_rule

    def get_gradient_scale_rule(self) -> ScaleRule:
        """Return the scaling rule to use for gradient tensors."""
        return self.gradient_scale_rule or self.scale_rule

    def get_weight_scale_rule(self) -> ScaleRule:
        """Return the scaling rule to use for weight tensors."""
        return self.weight_scale_rule or self.scale_rule
