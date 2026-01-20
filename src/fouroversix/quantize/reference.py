from __future__ import annotations

from typing import Literal

import torch
from fouroversix.utils import AdaptiveBlockScalingRule, FP4Format, RoundStyle

MIN_ALLOWED_NORM_CONSTANT = 1e-12
E2M1_MAX_VALUE = 6
E4M3_MAX_VALUE = 448
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


def get_nvfp4_tensor_scale(
    x: torch.Tensor,
    *,
    scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
) -> torch.Tensor:
    # 384 is the largest E4M3 value for which the value * (4/6) can be represented
    # perfectly in E4M3 (256).
    scale = {
        AdaptiveBlockScalingRule.always_4: 4 * E4M3_MAX_VALUE,
        AdaptiveBlockScalingRule.always_6: E2M1_MAX_VALUE * E4M3_MAX_VALUE,
    }.get(scale_rule, 384 * 4)

    return (x.abs().max().float() / scale).clamp(min=MIN_ALLOWED_NORM_CONSTANT)


def quantize_bf16_to_scaled_fp4(  # noqa: C901, PLR0912
    x: torch.Tensor,
    block_size: int,
    scale_dtype: torch.dtype,
    *,
    norm_constant: torch.Tensor | None = None,
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

    # Compute values and scale factors when blocks are scaled to 6
    if scale_rule != AdaptiveBlockScalingRule.always_4:
        x_scales_hp_6 = (
            x_scale_blocks.abs().max(axis=-1).values / (E2M1_MAX_VALUE * norm_constant)
        ).clamp(min=E4M3_MIN_POSITIVE_NORMAL, max=E4M3_MAX_VALUE)
        x_scales_6 = x_scales_hp_6.to(scale_dtype)
        x_scales_hp_6 = x_scales_6.float() * norm_constant
        x_block_scaled_6 = x_scale_blocks / x_scales_hp_6[:, None]
        x_quantized_6 = x_scale_blocks.sign() * fake_quantize_positive_to_e2m1(
            x_block_scaled_6.abs(),
            stochastic_rounding=stochastic_rounding,
        )
        x_dequantized_6 = x_quantized_6 * x_scales_hp_6[:, None]

    if scale_rule != AdaptiveBlockScalingRule.always_6:
        x_scales_hp_4 = (
            x_scale_blocks.abs().max(axis=-1).values / (4 * norm_constant)
        ).clamp(min=E4M3_MIN_POSITIVE_NORMAL, max=E4M3_MAX_VALUE)
        x_scales_4 = x_scales_hp_4.to(scale_dtype)
        x_scales_hp_4 = x_scales_4.float() * norm_constant
        x_block_scaled_4 = x_scale_blocks / x_scales_hp_4[:, None]
        x_quantized_4 = x_scale_blocks.sign() * fake_quantize_positive_to_e2m1(
            x_block_scaled_4.abs(),
            stochastic_rounding=stochastic_rounding,
        )
        x_dequantized_4 = x_quantized_4 * x_scales_hp_4[:, None]

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
        x_quantized = x_quantized_4
        scales = x_scales_4
    elif scale_rule == AdaptiveBlockScalingRule.always_6:
        x_quantized = x_quantized_6
        scales = x_scales_6
    else:
        select_4 = (x_error_4 < x_error_6)[:, None]
        x_quantized = torch.where(
            select_4,
            x_quantized_4.reshape(
                -1,
                block_size * block_size if block_scale_2d else block_size,
            ),
            x_quantized_6.reshape(
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

    outputs = (x_quantized, reshaped_scales, norm_constant)

    if return_block_selections:
        outputs = (*outputs, select_4)

    return outputs


def quantize_to_fp4(
    x: torch.Tensor,
    norm_constant: torch.Tensor | None = None,
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

    if norm_constant is None:
        norm_constant = (
            torch.ones(1, device=x.device, dtype=x.dtype)
            if fp4_format == FP4Format.mxfp4
            else get_nvfp4_tensor_scale(x, scale_rule=scale_rule)
        )

    return quantize_bf16_to_scaled_fp4(
        x,
        fp4_format.block_size(),
        fp4_format.scale_dtype(),
        stochastic_rounding=round_style == RoundStyle.stochastic,
        block_scale_2d=block_scale_2d,
        norm_constant=norm_constant,
        scale_factors_simulation_mode=scale_factors_simulation_mode,
        scale_rule=scale_rule,
        values_simulation_mode=values_simulation_mode,
        values_simulation_threshold=values_simulation_threshold,
        return_block_selections=return_block_selections,
    )


def convert_e2m1_to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    sign = (x >> 3) & 0x1
    exponent = (x >> 1) & 0x3
    mantissa = x & 0x1

    # Make adjustments
    new_exponent = torch.where(
        (exponent == 0) & (mantissa == 0),
        0,
        (exponent + 6) & 0xF,
    )
    new_mantissa = torch.where(exponent == 0, 0, mantissa << 2)

    return ((sign << 7) | (new_exponent << 3) | new_mantissa).view(torch.float8_e4m3fn)


def convert_e2m1_to_fp8_e8m0(x: torch.Tensor) -> torch.Tensor:
    e = (x >> 1) & 0x3
    m = x & 0x1

    # There might be a better way to do this but I'm feeling lazy right now
    return torch.where(
        (e == 3) & (m == 1),  # noqa: PLR2004
        torch.tensor(130, dtype=torch.uint8),
        torch.where(
            e == 3,  # noqa: PLR2004
            torch.tensor(129, dtype=torch.uint8),
            torch.where(
                (e == 2) & (m == 1),  # noqa: PLR2004
                torch.tensor(129, dtype=torch.uint8),
                torch.where(
                    e == 2,  # noqa: PLR2004
                    torch.tensor(128, dtype=torch.uint8),
                    torch.where(
                        (e == 1) & (m == 1),
                        torch.tensor(128, dtype=torch.uint8),
                        torch.where(
                            e == 1,
                            torch.tensor(127, dtype=torch.uint8),
                            torch.where(
                                (e == 0) & (m == 1),
                                torch.tensor(126, dtype=torch.uint8),
                                torch.tensor(0, dtype=torch.uint8),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).view(torch.float8_e8m0fnu)


def unpack_packed_fp4(
    x: torch.Tensor,
    to_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    if to_dtype == torch.float8_e4m3fn:
        convert_function = convert_e2m1_to_fp8_e4m3
    elif to_dtype == torch.float8_e8m0fnu:
        convert_function = convert_e2m1_to_fp8_e8m0
    else:
        msg = f"Unsupported dtype: {to_dtype}"
        raise ValueError(msg)

    high = (x >> 4) & 0xF
    low = x & 0xF
    return torch.stack(
        [convert_function(low), convert_function(high)],
        dim=-1,
    ).reshape(x.shape[0], x.shape[1] * 2)


def dequantize_from_fp4(
    x: torch.Tensor,
    scales: torch.Tensor,
    x_amax: torch.Tensor | None = None,
    *,
    scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    dtype: torch.dtype = torch.bfloat16,
    fp4_format: FP4Format = FP4Format.nvfp4,
    values_simulation_mode: ValueSimulationMode = None,
) -> torch.Tensor:
    if values_simulation_mode is not None:
        values = x
    else:
        values = unpack_packed_fp4(x, to_dtype=fp4_format.scale_dtype()).to(
            dtype,
        )

    scales = from_blocked(
        scales,
        (x.shape[0], x.shape[1] // fp4_format.block_size() * 2),
    )

    result = values * scales.to(
        dtype,
    ).repeat_interleave(fp4_format.block_size(), -1)

    if fp4_format == FP4Format.mxfp4:
        high = (x >> 4) & 0xF
        low = x & 0xF
        values = torch.stack([low, high], dim=-1).reshape(x.shape[0], x.shape[1] * 2)
        x_sign = torch.where(
            ((values >> 3) & 0x1) == 0,
            torch.tensor(1, dtype=dtype),
            torch.tensor(-1, dtype=dtype),
        )
        result = result * x_sign
    elif fp4_format == FP4Format.nvfp4:
        if x_amax is not None:
            if scale_rule == AdaptiveBlockScalingRule.always_6:
                result = (result.to(torch.float32) * x_amax / (6 * 448)).to(dtype)
            elif scale_rule == AdaptiveBlockScalingRule.always_4:
                result = (result.to(torch.float32) * x_amax / (4 * 448)).to(dtype)
            else:
                result = (result.to(torch.float32) * x_amax / (6 * 256)).to(dtype)

    return result


def fake_quantize_to_fp4(
    x: torch.Tensor,
    norm_constant: torch.Tensor | None = None,
    had: torch.Tensor | None = None,
    *,
    fp4_format: FP4Format = FP4Format.nvfp4,
    round_style: RoundStyle = RoundStyle.nearest,
    transpose: bool = False,
    scale_factors_simulation_mode: ScaleFactorsSimulationMode = None,
    scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    values_simulation_mode: ValueSimulationMode = None,
    values_simulation_threshold: float | None = None,
    return_block_selections: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    out_e2m1, out_sf, out_normconst, *out_extras = quantize_to_fp4(
        x,
        norm_constant,
        had,
        fp4_format=fp4_format,
        round_style=round_style,
        transpose=transpose,
        scale_factors_simulation_mode=scale_factors_simulation_mode,
        scale_rule=scale_rule,
        values_simulation_mode=values_simulation_mode,
        values_simulation_threshold=values_simulation_threshold,
        return_block_selections=return_block_selections,
    )

    result = dequantize_from_fp4(
        out_e2m1,
        out_sf,
        out_normconst,
        fp4_format=fp4_format,
        values_simulation_mode=values_simulation_mode,
    )

    return (result, *out_extras) if len(out_extras) > 0 else result


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def from_blocked(a: torch.Tensor, orig_shape: tuple[int, int]) -> torch.Tensor:
    rows, cols = orig_shape
    return (
        a.view(-1, 32, 4, 4)
        .transpose(1, 2)
        .reshape(-1, ceil_div(cols, 4), 128, 4)
        .transpose(1, 2)
        .reshape(rows, cols)
    )


def to_blocked(a: torch.Tensor) -> torch.Tensor:
    return (
        a.view(ceil_div(a.shape[0], 128), 128, ceil_div(a.shape[1], 4), 4)
        .transpose(1, 2)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1, 32, 16)
        .flatten()
    )
