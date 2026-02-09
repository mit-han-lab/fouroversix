from dataclasses import dataclass

from fouroversix.quantize import QuantizationConfig
from fouroversix.utils import DataType, MatmulBackend, QuantizeBackend, ScaleRule


@dataclass
class FourOverSixLayerConfig:
    """
    Configuration for quantizing layers with Four Over Six.

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
        output_dtype (DataType): The data type to use for the layer's output. Defaults
            to `DataType.bfloat16`.
        quantize_backend (QuantizeBackend | None): The backend to use for quantization.
            If not provided, a backend will be selected automatically based on the
            available GPU and the specified options.
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
    output_dtype: DataType = DataType.bfloat16
    quantize_backend: QuantizeBackend | None = None
    scale_rule: ScaleRule = ScaleRule.mse
    weight_scale_2d: bool = False
    weight_scale_rule: ScaleRule | None = None

    def __post_init__(self) -> None:
        """Convert string values to enums."""

        if isinstance(self.activation_scale_rule, str):
            self.activation_scale_rule = ScaleRule(self.activation_scale_rule)

        if isinstance(self.dtype, str):
            self.dtype = DataType(self.dtype)

        if isinstance(self.gradient_scale_rule, str):
            self.gradient_scale_rule = ScaleRule(self.gradient_scale_rule)

        if isinstance(self.matmul_backend, str):
            self.matmul_backend = MatmulBackend(self.matmul_backend)

        if isinstance(self.output_dtype, str):
            self.output_dtype = DataType(self.output_dtype)

        if isinstance(self.quantize_backend, str):
            self.quantize_backend = QuantizeBackend(self.quantize_backend)

        if isinstance(self.scale_rule, str):
            self.scale_rule = ScaleRule(self.scale_rule)

        if isinstance(self.weight_scale_rule, str):
            self.weight_scale_rule = ScaleRule(self.weight_scale_rule)

    def get_activation_scale_rule(self) -> ScaleRule:
        """Return the scaling rule to use for activation tensors."""
        return self.activation_scale_rule or self.scale_rule

    def get_gradient_scale_rule(self) -> ScaleRule:
        """Return the scaling rule to use for gradient tensors."""
        return self.gradient_scale_rule or self.scale_rule

    def get_weight_scale_rule(self) -> ScaleRule:
        """Return the scaling rule to use for weight tensors."""
        return self.weight_scale_rule or self.scale_rule

    def get_activation_config(self, **kwargs) -> QuantizationConfig:
        """Return the quantization configuration for the activation tensors."""
        return QuantizationConfig(
            backend=self.quantize_backend,
            dtype=self.dtype,
            scale_rule=self.get_activation_scale_rule(),
            **kwargs,
        )

    def get_gradient_config(self, **kwargs) -> QuantizationConfig:
        """Return the quantization configuration for the gradient tensors."""
        return QuantizationConfig(
            backend=self.quantize_backend,
            dtype=self.dtype,
            scale_rule=self.get_gradient_scale_rule(),
            **kwargs,
        )

    def get_weight_config(self, **kwargs) -> QuantizationConfig:
        """Return the quantization configuration for the weight tensors."""
        return QuantizationConfig(
            backend=self.quantize_backend,
            dtype=self.dtype,
            scale_rule=self.get_weight_scale_rule(),
            **kwargs,
        )
