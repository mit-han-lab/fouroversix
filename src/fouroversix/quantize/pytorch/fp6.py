from __future__ import annotations

import torch

E2M3_MAX_VALUE = 7.5
E3M2_MAX_VALUE = 28

# E2M3: 2-bit exponent, 3-bit mantissa, exp_bias = 1
E2M3_EXP_BIAS = 1
E2M3_MANTISSA_BITS = 3
E2M3_MIN_NORMAL_EXP = 0
E2M3_MAX_EXP = 2

# E3M2: 3-bit exponent, 2-bit mantissa, exp_bias = 3
E3M2_EXP_BIAS = 3
E3M2_MANTISSA_BITS = 2
E3M2_MIN_NORMAL_EXP = -2
E3M2_MAX_EXP = 4


def _convert_to_fp6_with_rtn(
    x: torch.Tensor,
    *,
    max_value: float,
    exp_bias: int,
    mantissa_bits: int,
    min_normal_exp: int,
    max_exp: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert float values to a 6-bit floating point format with round-to-nearest.

    Mirrors the approach of convert_to_e4m3_with_rtn in the Triton backend.
    Returns both the uint8-encoded values and the fake-quantized float values.
    """

    sign = torch.where(x < 0, 1, 0).to(torch.uint8)
    x_abs = x.abs().clamp(max=max_value)

    is_zero = x_abs == 0
    x_safe = torch.where(is_zero, 1, x_abs)

    # Compute exponent
    exp = torch.floor(torch.log2(x_safe))
    exp = exp.clamp(min=-10, max=max_exp)

    is_normal = exp >= min_normal_exp

    # Normal numbers: value = (1 + mantissa / 2^m) * 2^exp
    mant = x_safe / torch.exp2(exp) - 1
    mant_q = torch.round(mant * (1 << mantissa_bits))
    overflow = mant_q == (1 << mantissa_bits)
    mant_q = torch.where(overflow, 0, mant_q)
    exp = torch.where(overflow, exp + 1, exp)
    exp_q = exp + exp_bias

    # Subnormals
    mant_sub_q = (x_safe * 2 ** (exp_bias + mantissa_bits - 1)).round()

    # Select normal or subnormal
    mantissa = torch.where(is_normal, mant_q, mant_sub_q)
    exponent = torch.where(is_normal, exp_q, 0)

    # Handle zero
    mantissa = torch.where(is_zero, 0, mantissa).to(
        torch.uint8,
    )
    exponent = torch.where(is_zero, 0, exponent).to(
        torch.uint8,
    )

    return (
        (sign << (mantissa_bits + max_exp.bit_length()))
        | (exponent << mantissa_bits)
        | mantissa
    )


def convert_to_e2m3_with_rtn(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert float values to E2M3 format. Returns (uint8 codes, fake-quantized)."""
    return _convert_to_fp6_with_rtn(
        x,
        max_value=E2M3_MAX_VALUE,
        exp_bias=E2M3_EXP_BIAS,
        mantissa_bits=E2M3_MANTISSA_BITS,
        min_normal_exp=E2M3_MIN_NORMAL_EXP,
        max_exp=E2M3_MAX_EXP,
    )


def convert_to_e3m2_with_rtn(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert float values to E3M2 format. Returns (uint8 codes, fake-quantized)."""
    return _convert_to_fp6_with_rtn(
        x,
        max_value=E3M2_MAX_VALUE,
        exp_bias=E3M2_EXP_BIAS,
        mantissa_bits=E3M2_MANTISSA_BITS,
        min_normal_exp=E3M2_MIN_NORMAL_EXP,
        max_exp=E3M2_MAX_EXP,
    )


def _convert_fp6_to_high_precision(
    quantized: torch.Tensor,
    *,
    exp_bias: int,
    mantissa_bits: int,
    max_exp: int,
) -> None:
    sign = ((quantized >> (mantissa_bits + max_exp.bit_length())) & 0x1).to(
        torch.float32,
    )
    exponent = ((quantized >> mantissa_bits) & ((1 << max_exp.bit_length()) - 1)).to(
        torch.float32,
    )
    mantissa = (quantized & ((1 << mantissa_bits) - 1)).to(torch.float32)

    is_subnormal = exponent == 0
    is_zero = quantized == 0

    # Normals
    val_norm = (1 + mantissa / (1 << mantissa_bits)) * torch.exp2(exponent - exp_bias)

    # Subnormals
    val_sub = (mantissa / (1 << mantissa_bits)) * (2 ** (1.0 - exp_bias))

    val = torch.where(is_subnormal, val_sub, val_norm)
    val = torch.where(sign == 1, -val, val)
    return torch.where(is_zero, 0, val)


def convert_e2m3_to_high_precision(
    quantized: torch.Tensor,
) -> torch.Tensor:
    return _convert_fp6_to_high_precision(
        quantized,
        exp_bias=E2M3_EXP_BIAS,
        mantissa_bits=E2M3_MANTISSA_BITS,
        max_exp=E2M3_MAX_EXP,
    )


def convert_e3m2_to_high_precision(
    quantized: torch.Tensor,
) -> torch.Tensor:
    return _convert_fp6_to_high_precision(
        quantized,
        exp_bias=E3M2_EXP_BIAS,
        mantissa_bits=E3M2_MANTISSA_BITS,
        max_exp=E3M2_MAX_EXP,
    )
