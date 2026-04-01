from enum import Enum

import torch

SM_80 = 8
SM_100 = 10
SM_110 = 11
SM_120 = 12

BLACKWELL_SM_IDS = {SM_100, SM_110, SM_120}


class DataType(str, Enum):
    """Data types."""

    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"
    mxfp3 = "mxfp3"
    mxfp3_bs8 = "mxfp3_bs8"
    mxfp4 = "mxfp4"
    mxfp4_bs8 = "mxfp4_bs8"
    mxfp6_e2m3 = "mxfp6_e2m3"
    mxfp6_e3m2 = "mxfp6_e3m2"
    nvfp3 = "nvfp3"
    nvfp3_bs8 = "nvfp3_bs8"
    nvfp4 = "nvfp4"
    nvfp4_bs8 = "nvfp4_bs8"
    nvfp6_e2m3 = "nvfp6_e2m3"
    nvfp6_e3m2 = "nvfp6_e3m2"
    nvint3 = "nvint3"
    nvint3_bs8 = "nvint3_bs8"
    nvint4 = "nvint4"
    nvint4_bs8 = "nvint4_bs8"
    nvint6 = "nvint6"
    if3 = "if3"
    if3_bs8 = "if3_bs8"
    if4 = "if4"
    if4_bs8 = "if4_bs8"
    if6_e2m3 = "if6_e2m3"
    if6_e3m2 = "if6_e3m2"

    @property
    def block_size(self) -> int | None:
        """Return the block size if this a block-scaled format, or `None` otherwise."""

        return {
            DataType.mxfp3: 32,
            DataType.mxfp3_bs8: 8,
            DataType.mxfp4: 32,
            DataType.mxfp4_bs8: 8,
            DataType.mxfp6_e2m3: 32,
            DataType.mxfp6_e3m2: 32,
            DataType.nvfp3: 16,
            DataType.nvfp3_bs8: 8,
            DataType.nvfp4: 16,
            DataType.nvfp4_bs8: 8,
            DataType.nvfp6_e2m3: 16,
            DataType.nvfp6_e3m2: 16,
            DataType.nvint3: 16,
            DataType.nvint3_bs8: 8,
            DataType.nvint4: 16,
            DataType.nvint4_bs8: 8,
            DataType.nvint6: 16,
            DataType.if3: 16,
            DataType.if3_bs8: 8,
            DataType.if4: 16,
            DataType.if4_bs8: 8,
            DataType.if6_e2m3: 16,
            DataType.if6_e3m2: 16,
        }.get(self)

    @property
    def quantized_value_type(self) -> "QuantizedValueType | None":
        """
        Return the quantized value type if this a block-scaled format, or `None`
        otherwise.
        """

        return {
            DataType.mxfp3: QuantizedValueType.fp3,
            DataType.mxfp3_bs8: QuantizedValueType.fp3,
            DataType.mxfp4: QuantizedValueType.fp4,
            DataType.mxfp4_bs8: QuantizedValueType.fp4,
            DataType.mxfp6_e2m3: QuantizedValueType.fp6_e2m3,
            DataType.mxfp6_e3m2: QuantizedValueType.fp6_e3m2,
            DataType.nvfp3: QuantizedValueType.fp3,
            DataType.nvfp3_bs8: QuantizedValueType.fp3,
            DataType.nvfp4: QuantizedValueType.fp4,
            DataType.nvfp4_bs8: QuantizedValueType.fp4,
            DataType.nvfp6_e2m3: QuantizedValueType.fp6_e2m3,
            DataType.nvfp6_e3m2: QuantizedValueType.fp6_e3m2,
            DataType.nvint3: QuantizedValueType.int3,
            DataType.nvint3_bs8: QuantizedValueType.int3,
            DataType.nvint4: QuantizedValueType.int4,
            DataType.nvint4_bs8: QuantizedValueType.int4,
            DataType.nvint6: QuantizedValueType.int6,
            DataType.if3: QuantizedValueType.if3,
            DataType.if3_bs8: QuantizedValueType.if3,
            DataType.if4: QuantizedValueType.if4,
            DataType.if4_bs8: QuantizedValueType.if4,
            DataType.if6_e2m3: QuantizedValueType.if6_e2m3,
            DataType.if6_e3m2: QuantizedValueType.if6_e3m2,
        }.get(self)

    @property
    def scale_type(self) -> "ScaleType | None":
        """Return the scale type if this a block-scaled format, or `None` otherwise."""

        return {
            DataType.mxfp3: ScaleType.mx,
            DataType.mxfp3_bs8: ScaleType.mx,
            DataType.mxfp4: ScaleType.mx,
            DataType.mxfp4_bs8: ScaleType.mx,
            DataType.mxfp6_e2m3: ScaleType.mx,
            DataType.mxfp6_e3m2: ScaleType.mx,
            DataType.nvfp3: ScaleType.nv,
            DataType.nvfp3_bs8: ScaleType.nv,
            DataType.nvfp4: ScaleType.nv,
            DataType.nvfp4_bs8: ScaleType.nv,
            DataType.nvfp6_e2m3: ScaleType.nv,
            DataType.nvfp6_e3m2: ScaleType.nv,
            DataType.nvint3: ScaleType.nv,
            DataType.nvint3_bs8: ScaleType.nv,
            DataType.nvint4: ScaleType.nv,
            DataType.nvint4_bs8: ScaleType.nv,
            DataType.nvint6: ScaleType.nv,
            DataType.if3: ScaleType.nv_if,
            DataType.if3_bs8: ScaleType.nv_if,
            DataType.if4: ScaleType.nv_if,
            DataType.if4_bs8: ScaleType.nv_if,
            DataType.if6_e2m3: ScaleType.nv_if,
            DataType.if6_e3m2: ScaleType.nv_if,
        }.get(self)

    @property
    def supported_scale_rules(self) -> set["ScaleRule"]:
        """Return the scale rules that are allowed for this data type."""

        if self in {DataType.if3, DataType.if3_bs8, DataType.if4, DataType.if4_bs8, DataType.if6_e2m3, DataType.if6_e3m2}:
            return {scale_rule for scale_rule in ScaleRule if not scale_rule.is_static}

        if self in {
            DataType.nvfp3,
            DataType.nvfp3_bs8,
            DataType.nvint3,
            DataType.nvint3_bs8,
            DataType.nvint4,
            DataType.nvint4_bs8,
            DataType.nvint6,
            DataType.nvfp6_e2m3,
            DataType.nvfp6_e3m2,
        }:
            return {ScaleRule.static_6}

        if self in {
            DataType.mxfp3,
            DataType.mxfp3_bs8,
            DataType.mxfp4,
            DataType.mxfp4_bs8,
            DataType.mxfp6_e2m3,
            DataType.mxfp6_e3m2,
        }:
            return {scale_rule for scale_rule in ScaleRule if scale_rule.is_static}

        if self in {DataType.nvfp4, DataType.nvfp4_bs8}:
            return set(ScaleRule)

        return set()

    @property
    def torch_dtype(self) -> torch.dtype | None:
        """
        Return the corresponding torch.dtype if one is available, or `None`
        otherwise.
        """

        return {
            DataType.bfloat16: torch.bfloat16,
            DataType.float16: torch.float16,
            DataType.float32: torch.float32,
        }.get(self)


class MatmulBackend(str, Enum):
    """
    Backends for matrix multiplication with FP4.

    - `cutlass`: CUTLASS implementation. This requires a Blackwell GPU.
    - `pytorch`: PyTorch implementation which first dequantizes the input tensors to
        FP32 and then performs an FP32 matrix multiplication.
    """

    cutlass = "cutlass"
    triton = "triton"
    pytorch = "pytorch"


class QuantizeBackend(str, Enum):
    """
    Backends for quantizing a tensor to NVFP4 or MXFP4.

    - `cuda`: CUDA implementation. Requires a Blackwell GPU, and currently only supports
        the forward pass for PTQ (no stochastic rounding, no transposed matrices, no
        RHT, no 2D block scaling).
    - `pytorch`: PyTorch implementation.
    - `triton`: Triton implementation. Requires a Blackwell GPU.
    """

    cuda = "cuda"
    pytorch = "pytorch"
    transformer_engine = "transformer_engine"
    triton = "triton"


class RoundStyle(str, Enum):
    """
    Rounding styles for quantization.

    - `nearest`: Round to the nearest FP4 value.
    - `stochastic`: Round to the nearest FP4 value after applying random noise to each
        value.
    - `stochastic_unbiased`: Round to the nearest FP4 value after applying random noise
        to each value, but scaling by 16/17 to make it unbiased. See Quartet II for
        details: https://arxiv.org/abs/2601.22813.
    """

    nearest = "nearest"
    stochastic = "stochastic"
    stochastic_unbiased = "stochastic_unbiased"

    @property
    def adjustment_factor(self) -> float:
        """Return the adjustment factor for the rounding style."""
        return 16 / 17 if self == RoundStyle.stochastic_unbiased else 1

    @property
    def is_stochastic(self) -> bool:
        """Return True if the rounding style is stochastic, False otherwise."""
        return self in {RoundStyle.stochastic, RoundStyle.stochastic_unbiased}


class ScaleRule(str, Enum):
    """
    Block scale selection rules for NVFP4 quantization.

    - `abs_max`: Between 4 and 6, select the block scale that minimizes the maximum
        absolute quantization error.
    - `static_4`: Select 4 for all blocks.
    - `static_6`: Select 6 for all blocks (normal NVFP4 quantization).
    - `mae`: Between 4 and 6, select the block scale that minimizes the mean absolute
        quantization error.
    - `mse`: Between 4 and 6, select the block scale that minimizes the mean squared
        quantization error.
    """

    abs_max = "abs_max"
    mae = "mae"
    mse = "mse"
    static_4 = "static_4"
    static_6 = "static_6"

    @property
    def cuda_id(self) -> int:
        """ID for the rule in the CUDA implementation."""

        return {
            ScaleRule.abs_max: 4,
            ScaleRule.mae: 2,
            ScaleRule.mse: 3,
            ScaleRule.static_4: 1,
            ScaleRule.static_6: 0,
        }[self]

    @property
    def is_static(self) -> bool:
        """Return True if the rule is static, False otherwise."""
        return self in {ScaleRule.static_4, ScaleRule.static_6}


class ScaleType(str, Enum):
    """
    Scale types for quantization.

    - `mx`: E8M0 scale factors, as is done in MX-style quantization formats.
    - `nv`: E4M3 scale factors, as in NVFP4.
    - `nv_if`: E4M3, but using the sign bit as an indicator for IF formats.
    """

    mx = "mx"
    nv = "nv"
    nv_if = "nv_if"

    def get_maximum_value(self, scale_rule: "ScaleRule") -> int | None:
        """Return the maximum value for the scale type."""
        return {
            ScaleType.mx: None,
            ScaleType.nv: 448 if scale_rule.is_static else 256,
            ScaleType.nv_if: 448,
        }.get(self)

    @property
    def torch_dtype(self) -> torch.dtype | None:
        """
        Return the corresponding torch.dtype if one is available, or `None`
        otherwise.
        """

        return {
            ScaleType.mx: torch.float8_e8m0fnu,
            ScaleType.nv: torch.float8_e4m3fn,
            ScaleType.nv_if: torch.float8_e4m3fn,
        }.get(self)


class QuantizedValueType(str, Enum):
    """
    Allowed types for quantized values.

    - `fp4`: FP4.
    - `fp6_e2m3`: FP6 (E2M3).
    - `fp6_e3m2`: FP6 (E3M2).
    - `if4`: IF4.
    - `if6_e2m3`: IF6 (E2M3).
    - `if6_e3m2`: IF6 (E3M2).
    """

    fp3 = "fp3"
    fp4 = "fp4"
    fp6_e2m3 = "fp6_e2m3"
    fp6_e3m2 = "fp6_e3m2"
    if3 = "if3"
    if4 = "if4"
    if6_e2m3 = "if6_e2m3"
    if6_e3m2 = "if6_e3m2"
    int3 = "int3"
    int4 = "int4"
    int6 = "int6"

    def get_maximum_value(self, scale_rule: "ScaleRule") -> int:
        """Return the maximum value for the quantized value type."""

        if scale_rule == ScaleRule.static_4:
            return 4

        return {
            QuantizedValueType.fp3: 4,
            QuantizedValueType.fp4: 6,
            QuantizedValueType.fp6_e2m3: 7.5,
            QuantizedValueType.fp6_e3m2: 28,
            QuantizedValueType.if3: 4,
            QuantizedValueType.if4: 6,
            QuantizedValueType.int3: 3,
            QuantizedValueType.int4: 7,
            QuantizedValueType.int6: 31,
            QuantizedValueType.if6_e2m3: 7.5,
            QuantizedValueType.if6_e3m2: 28,
        }.get(self)

    @property
    def packing_factor(self) -> int:
        """Return the packing factor for the quantized value type."""

        return {
            QuantizedValueType.fp3: 1,
            QuantizedValueType.fp4: 2,
            QuantizedValueType.fp6_e2m3: 1,
            QuantizedValueType.fp6_e3m2: 1,
            QuantizedValueType.if3: 1,
            QuantizedValueType.if4: 2,
            QuantizedValueType.int3: 1,
            QuantizedValueType.int4: 2,
            QuantizedValueType.int6: 1,
            QuantizedValueType.if6_e2m3: 1,
            QuantizedValueType.if6_e3m2: 1,
        }.get(self)
