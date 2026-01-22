from __future__ import annotations

from typing import Literal

import torch
from fouroversix.utils import AdaptiveBlockScalingRule, FP4Format, RoundStyle

MIN_ALLOWED_NORM_CONSTANT = 1e-12
E2M1_MAX_VALUE = 6
E2M1_MAX_FOUR = 4
E4M3_MAX_VALUE = 448
E4M3_MAX_FOUROVERSIX = 256
E4M3_MIN_POSITIVE_NORMAL = 0.015625

ScaleFactorsSimulationMode = Literal["high_precision"] | None
ValueSimulationMode = (
    Literal[
        "all_in_high_precision",
        "nonzeros_in_high_precision",
        "zeros_in_high_precision",
        "greater_than_threshold_in_high_precision",
        "less_than_threshold_in_high_precision",
        "nvint4",
    ]
    | None
)


def fake_quantize_positive_to_e2m1(
    x: torch.Tensor,
    *,
    stochastic_rounding: bool = False,
) -> torch.Tensor:
    if stochastic_rounding:
        rbits = torch.rand_like(x) - 0.5
        step1 = torch.round(2 * x + rbits) / 2
        step2 = torch.round(x + rbits)
        step3 = 2 * torch.round(x / 2 + rbits)
        step3[step3 > E2M1_MAX_VALUE] = E2M1_MAX_VALUE
    else:
        step1 = torch.round(2 * x) / 2
        step2 = torch.round(x)
        step3 = 2 * torch.round(x / 2)

    mask1 = x < 2  # noqa: PLR2004
    mask2 = x < 4  # noqa: PLR2004

    return step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)


def quantize_bf16_to_unpacked_fp4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16

    bx = x.view(torch.int16)
    s = (bx >> 15) & 0x1
    e = (bx >> 7) & 0xFF
    m = bx & 0x7F
    is_zero = (e == 0) & (m == 0)

    # Default mantissa bit (for 1.5, 3.0, 6.0)
    m = (m >> 6) & 1
    is_half = (e == 126) & (m == 0)  # noqa: PLR2004
    m = torch.where(is_half, torch.tensor(1, dtype=torch.int16, device=x.device), m)

    # Exponent mapping
    # exp=126 -> E=0 (subnormals)
    # exp=127 -> E=1
    # exp=128 -> E=2
    # exp=129 -> E=3
    e = e - 126
    e = torch.where(is_zero, torch.tensor(0, dtype=torch.int16, device=x.device), e)

    # Zero always M=0
    m = torch.where(is_zero, torch.tensor(0, dtype=torch.int16, device=x.device), m)

    code = (s << 3) | (e << 1) | m
    return code.to(torch.uint8)


def pack_unpacked_fp4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8

    dim = 1
    size_along_dim = x.size(dim)
    new_size_along_dim = (size_along_dim + 1) // 2

    # If the size is odd, we pad the data along dim with zeros at the end
    if size_along_dim % 2 != 0:
        pad_sizes = [0] * (2 * x.ndim)
        pad_index = (x.ndim - dim - 1) * 2 + 1
        pad_sizes[pad_index] = 1
        x = torch.nn.functional.pad(x, pad_sizes, mode="constant", value=0)

    new_shape = list(x.shape)
    new_shape[dim] = new_size_along_dim
    new_shape.insert(dim + 1, 2)  # packed dimension of length 2
    x = x.reshape(*new_shape)

    low = x.select(dim + 1, 0)
    high = x.select(dim + 1, 1)
    return (high << 4) | low


def quantize_bf16_to_scaled_fp4(  # noqa: C901
    x: torch.Tensor,
    x_amax: torch.Tensor | None = None,
    *,
    block_size: int,
    scale_dtype: torch.dtype,
    return_block_selections: bool = False,
    scale_factors_simulation_mode: ScaleFactorsSimulationMode = None,
    scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    block_scale_2d: bool = False,
    stochastic_rounding: bool = True,
    # TODO(jack): Reimplement simulations
    values_simulation_mode: ValueSimulationMode = None,  # noqa: ARG001
    values_simulation_threshold: float | None = None,  # noqa: ARG001
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]
):
    if block_scale_2d:
        assert x.ndim == 2  # noqa: PLR2004
        assert x.shape[1] % block_size == 0

        x_scale_blocks = (
            x.reshape(-1, block_size, x.shape[1] // block_size, block_size)
            .permute(0, 2, 1, 3)
            .reshape(-1, block_size**2)
            .float()
        )
    else:
        x_scale_blocks = x.reshape(-1, block_size).float()

    x_scales_hp = (
        x_scale_blocks.abs().max(axis=-1).values
        * (
            E4M3_MAX_VALUE
            if scale_rule
            in (AdaptiveBlockScalingRule.always_4, AdaptiveBlockScalingRule.always_6)
            else E4M3_MAX_FOUROVERSIX
        )
        / x_amax
    )

    x_scales_6 = x_scales_hp.to(scale_dtype)
    x_scales_4 = (x_scales_hp * (6 / 4)).to(scale_dtype)

    if scale_rule == AdaptiveBlockScalingRule.always_4:
        x_scales_6 = (x_scales_hp * (4 / 6)).to(scale_dtype)
        x_scales_4 = x_scales_hp.to(scale_dtype)

    x_block_scaled_6 = torch.where(
        x_scales_6.unsqueeze(1) != 0,
        (
            x_scale_blocks
            * E2M1_MAX_VALUE
            * (
                E4M3_MAX_VALUE
                if scale_rule
                in (
                    AdaptiveBlockScalingRule.always_6,
                    AdaptiveBlockScalingRule.always_4,
                )
                else E4M3_MAX_FOUROVERSIX
            )
        )
        / (x_amax * x_scales_6.unsqueeze(1).to(x_amax.dtype)),
        0,
    )
    x_block_scaled_4 = torch.where(
        x_scales_4.unsqueeze(1) != 0,
        (
            x_scale_blocks
            * (
                E2M1_MAX_FOUR
                if scale_rule
                in (
                    AdaptiveBlockScalingRule.always_6,
                    AdaptiveBlockScalingRule.always_4,
                )
                else E2M1_MAX_VALUE
            )
            * (
                E4M3_MAX_VALUE
                if scale_rule
                in (
                    AdaptiveBlockScalingRule.always_6,
                    AdaptiveBlockScalingRule.always_4,
                )
                else E4M3_MAX_FOUROVERSIX
            )
        )
        / (x_amax * x_scales_4.unsqueeze(1).to(x_amax.dtype)),
        0,
    )

    x_e2m1_6 = x_scale_blocks.sign() * fake_quantize_positive_to_e2m1(
        x_block_scaled_6.abs(),
        stochastic_rounding=stochastic_rounding,
    )
    x_e2m1_4 = x_scale_blocks.sign() * fake_quantize_positive_to_e2m1(
        x_block_scaled_4.abs(),
        stochastic_rounding=stochastic_rounding,
    )

    x_dequantized_6 = (
        x_e2m1_6
        * x_scales_6.unsqueeze(1).to(x_amax.dtype)
        * x_amax
        / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX)
    )
    x_dequantized_4 = (
        x_e2m1_4
        * x_scales_4.unsqueeze(1).to(x_amax.dtype)
        * x_amax
        / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX)
    )

    if scale_rule == AdaptiveBlockScalingRule.abs_max:
        x_error_4 = (
            (x_dequantized_4 - x_scale_blocks)
            .abs()
            .reshape(-1, block_size)
            .max(dim=-1)
            .values
        )
        x_error_6 = (
            (x_dequantized_6 - x_scale_blocks)
            .abs()
            .reshape(-1, block_size)
            .max(dim=-1)
            .values
        )
    elif scale_rule == AdaptiveBlockScalingRule.l1_norm:
        x_error_4 = (
            (x_dequantized_4 - x_scale_blocks).abs().reshape(-1, block_size).sum(dim=-1)
        )
        x_error_6 = (
            (x_dequantized_6 - x_scale_blocks).abs().reshape(-1, block_size).sum(dim=-1)
        )
    elif scale_rule == AdaptiveBlockScalingRule.mse:
        x_error_4 = (
            ((x_dequantized_4 - x_scale_blocks) ** 2)
            .reshape(-1, block_size * block_size if block_scale_2d else block_size)
            .sum(dim=-1)
        )
        x_error_6 = (
            ((x_dequantized_6 - x_scale_blocks) ** 2)
            .reshape(-1, block_size * block_size if block_scale_2d else block_size)
            .sum(dim=-1)
        )

    if scale_rule == AdaptiveBlockScalingRule.always_4:
        x_quantized = x_e2m1_4
        scales = x_scales_4
    elif scale_rule == AdaptiveBlockScalingRule.always_6:
        x_quantized = x_e2m1_6
        scales = x_scales_6
    else:
        select_4 = (x_error_4 < x_error_6)[:, None]
        x_quantized = torch.where(
            select_4,
            x_e2m1_4.reshape(
                -1,
                block_size * block_size if block_scale_2d else block_size,
            ),
            x_e2m1_6.reshape(
                -1,
                block_size * block_size if block_scale_2d else block_size,
            ),
        )
        scales = torch.where(
            select_4,
            x_scales_4.reshape(-1, 1),
            x_scales_6.reshape(-1, 1),
        )

    if block_scale_2d:
        x_quantized = (
            x_quantized.reshape(
                -1,
                x.shape[1] // block_size,
                block_size,
                block_size,
            )
            .permute(0, 2, 1, 3)
            .reshape_as(x_quantized)
        )

        scales = (
            scales.reshape(1, x.shape[0] // block_size, x.shape[1] // block_size)
            .broadcast_to(
                block_size,
                x.shape[0] // block_size,
                x.shape[1] // block_size,
            )
            .reshape(block_size, x.shape[0] // block_size, x.shape[1] // block_size)
            .permute(1, 0, 2)
        )

    x_quantized = pack_unpacked_fp4(
        quantize_bf16_to_unpacked_fp4(x_quantized.bfloat16().reshape_as(x)),
    )
    reshaped_scales = to_blocked(
        scales.reshape(
            x.shape[0],
            x.shape[1] // block_size,
        ),
    )

    if scale_factors_simulation_mode != "high_precision":
        reshaped_scales = reshaped_scales.to(scale_dtype)

    outputs = (x_quantized, reshaped_scales, x_amax)

    if return_block_selections:
        outputs = (*outputs, select_4)

    return outputs


def quantize_to_fp4(
    x: torch.Tensor,
    x_amax: torch.Tensor | None = None,
    had: torch.Tensor | None = None,
    *,
    block_scale_2d: bool = False,
    fp4_format: FP4Format = FP4Format.nvfp4,
    return_block_selections: bool = False,
    round_style: RoundStyle = RoundStyle.nearest,
    scale_factors_simulation_mode: ScaleFactorsSimulationMode = None,
    scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    transpose: bool = False,
    values_simulation_mode: ValueSimulationMode = None,
    values_simulation_threshold: float | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]
):
    if transpose:
        x = x.T

    if had is not None:
        x = (x.reshape(-1, had.shape[0]) @ had).reshape_as(x)

    if x_amax is None:
        x_amax = (
            torch.ones(1, device=x.device, dtype=x.dtype)
            if fp4_format == FP4Format.mxfp4
            else x.abs().max().float()
        )

    return quantize_bf16_to_scaled_fp4(
        x,
        x_amax,
        block_size=fp4_format.block_size(),
        scale_dtype=fp4_format.scale_dtype(),
        stochastic_rounding=round_style == RoundStyle.stochastic,
        block_scale_2d=block_scale_2d,
        scale_factors_simulation_mode=scale_factors_simulation_mode,
        scale_rule=scale_rule,
        values_simulation_mode=values_simulation_mode,
        values_simulation_threshold=values_simulation_threshold,
        return_block_selections=return_block_selections,
    )


def to_blocked(a: torch.Tensor) -> torch.Tensor:
    return (
        a.view(a.shape[0] // 128, 128, a.shape[1] // 4, 4)
        .transpose(1, 2)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1, 32, 16)
        .flatten()
    )
