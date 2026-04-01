import triton.language as tl
from fouroversix.kernels.constants import (
    IF3_INT_EXPANSION_FACTOR,
    IF3_INT_EXPANSION_FACTOR_RCP,
    IF4_INT_EXPANSION_FACTOR,
    IF4_INT_EXPANSION_FACTOR_RCP,
    IF6_E2M3_INT_EXPANSION_FACTOR,
    IF6_E2M3_INT_EXPANSION_FACTOR_RCP,
    IF6_E3M2_INT_EXPANSION_FACTOR,
    IF6_E3M2_INT_EXPANSION_FACTOR_RCP,
)
from fouroversix.utils import QuantizedValueType, RoundStyle, ScaleRule, ScaleType

E2M1_MAX_VALUE = tl.constexpr(6)
E2M1_MAX_FOUR = tl.constexpr(4)
E2M3_MAX_VALUE = tl.constexpr(7.5)
E3M2_MAX_VALUE = tl.constexpr(28)
E4M3_MAX_VALUE = tl.constexpr(448)
E4M3_MAX_FOUROVERSIX = tl.constexpr(256)
SCALE_MEGABLOCK_SIZE = tl.constexpr(512)

IF3_INT_EXPANSION_FACTOR = tl.constexpr(IF3_INT_EXPANSION_FACTOR)
IF3_INT_EXPANSION_FACTOR_RCP = tl.constexpr(IF3_INT_EXPANSION_FACTOR_RCP)
IF4_INT_EXPANSION_FACTOR = tl.constexpr(IF4_INT_EXPANSION_FACTOR)
IF4_INT_EXPANSION_FACTOR_RCP = tl.constexpr(IF4_INT_EXPANSION_FACTOR_RCP)
IF6_E2M3_INT_EXPANSION_FACTOR = tl.constexpr(IF6_E2M3_INT_EXPANSION_FACTOR)
IF6_E2M3_INT_EXPANSION_FACTOR_RCP = tl.constexpr(IF6_E2M3_INT_EXPANSION_FACTOR_RCP)
IF6_E3M2_INT_EXPANSION_FACTOR = tl.constexpr(IF6_E3M2_INT_EXPANSION_FACTOR)
IF6_E3M2_INT_EXPANSION_FACTOR_RCP = tl.constexpr(IF6_E3M2_INT_EXPANSION_FACTOR_RCP)

ROUND_STYLE_NEAREST = tl.constexpr(RoundStyle.nearest.value)
ROUND_STYLE_STOCHASTIC = tl.constexpr(RoundStyle.stochastic.value)
ROUND_STYLE_STOCHASTIC_UNBIASED = tl.constexpr(RoundStyle.stochastic_unbiased.value)

UNBIASED_SR_ADJUSTMENT_FACTOR = tl.constexpr(
    RoundStyle.stochastic_unbiased.adjustment_factor,
)
UNBIASED_SR_ADJUSTMENT_FACTOR_RCP = tl.constexpr(
    1 / RoundStyle.stochastic_unbiased.adjustment_factor,
)

SCALE_RULE_ABS_MAX = tl.constexpr(ScaleRule.abs_max.value)
SCALE_RULE_MAE = tl.constexpr(ScaleRule.mae.value)
SCALE_RULE_MSE = tl.constexpr(ScaleRule.mse.value)
SCALE_RULE_STATIC_4 = tl.constexpr(ScaleRule.static_4.value)
SCALE_RULE_STATIC_6 = tl.constexpr(ScaleRule.static_6.value)

SCALE_TYPE_MX = tl.constexpr(ScaleType.mx.value)
SCALE_TYPE_NV = tl.constexpr(ScaleType.nv.value)
SCALE_TYPE_NV_IF = tl.constexpr(ScaleType.nv_if.value)

SM_80 = tl.constexpr(8)
SM_100 = tl.constexpr(10)
SM_110 = tl.constexpr(11)
SM_120 = tl.constexpr(12)

QUANTIZED_VALUE_TYPE_FP3 = tl.constexpr(QuantizedValueType.fp3.value)
QUANTIZED_VALUE_TYPE_FP4 = tl.constexpr(QuantizedValueType.fp4.value)
QUANTIZED_VALUE_TYPE_FP6_E2M3 = tl.constexpr(QuantizedValueType.fp6_e2m3.value)
QUANTIZED_VALUE_TYPE_FP6_E3M2 = tl.constexpr(QuantizedValueType.fp6_e3m2.value)
QUANTIZED_VALUE_TYPE_IF3 = tl.constexpr(QuantizedValueType.if3.value)
QUANTIZED_VALUE_TYPE_IF4 = tl.constexpr(QuantizedValueType.if4.value)
QUANTIZED_VALUE_TYPE_IF6_E2M3 = tl.constexpr(QuantizedValueType.if6_e2m3.value)
QUANTIZED_VALUE_TYPE_IF6_E3M2 = tl.constexpr(QuantizedValueType.if6_e3m2.value)
QUANTIZED_VALUE_TYPE_INT3 = tl.constexpr(QuantizedValueType.int3.value)
QUANTIZED_VALUE_TYPE_INT4 = tl.constexpr(QuantizedValueType.int4.value)
QUANTIZED_VALUE_TYPE_INT6 = tl.constexpr(QuantizedValueType.int6.value)
