from __future__ import annotations

import torch
from fouroversix.kernels.constants import (
    IF4_INT_EXPANSION_FACTOR,
    IF4_INT_EXPANSION_FACTOR_RCP,
    IF6_E2M3_INT_EXPANSION_FACTOR,
    IF6_E2M3_INT_EXPANSION_FACTOR_RCP,
    IF6_E3M2_INT_EXPANSION_FACTOR,
    IF6_E3M2_INT_EXPANSION_FACTOR_RCP,
)
from fouroversix.quantize.utils import to_blocked
from fouroversix.utils import DataType, RoundStyle, ScaleRule, ScaleType

from .fp6 import (
    convert_e2m3_to_high_precision,
    convert_e3m2_to_high_precision,
    convert_to_e2m3_with_rtn,
    convert_to_e3m2_with_rtn,
)

E2M1_MAX_VALUE = 6
E2M1_MAX_FOUR = 4
INT4_MAX_VALUE = 7
INT6_MAX_VALUE = 31
E4M3_MAX_VALUE = 448
E4M3_MAX_FOUROVERSIX = 256


def fake_quantize_to_e2m1(
    x: torch.Tensor,
    *,
    round_style: RoundStyle = RoundStyle.nearest,
) -> torch.Tensor:
    if round_style == RoundStyle.nearest:
        step1 = torch.round(2 * x.abs()) / 2
        step2 = torch.round(x.abs())
        step3 = 2 * torch.round(x.abs() / 2)
    elif round_style.is_stochastic:
        rbits = torch.rand_like(x.abs()) - 0.5
        step1 = torch.round(2 * x.abs() + rbits) / 2
        step2 = torch.round(x.abs() + rbits)
        step3 = 2 * torch.round(x.abs() / 2 + rbits)
        step3[step3 > E2M1_MAX_VALUE] = E2M1_MAX_VALUE

    mask1 = x.abs() < 2  # noqa: PLR2004
    mask2 = x.abs() < 4  # noqa: PLR2004

    return x.sign() * (
        step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)
    )


def fake_quantize_to_int4(
    x: torch.Tensor,
    *,
    round_style: RoundStyle = RoundStyle.nearest,
) -> torch.Tensor:
    if round_style.is_stochastic:
        rbits = torch.rand_like(x) - 0.5
        return x.sign() * (x.abs() + rbits).clamp(min=-7, max=7).round()

    return x.clamp(min=-7, max=7).round()


def fake_quantize_to_int6(
    x: torch.Tensor,
    *,
    round_style: RoundStyle = RoundStyle.nearest,
) -> torch.Tensor:
    if round_style.is_stochastic:
        rbits = torch.rand_like(x) - 0.5
        return x.sign() * (x.abs() + rbits).clamp(max=INT6_MAX_VALUE).round()

    return x.clamp(min=-INT6_MAX_VALUE, max=INT6_MAX_VALUE).round()


def quantize_bf16_to_unpacked_int4(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(min=-7, max=7).to(torch.int8).view(torch.uint8) & 0xF


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


def quantize_to_nvint4(
    x_scale_blocks: torch.Tensor,
    x_amax: torch.Tensor,
    round_style: RoundStyle,
) -> tuple[torch.Tensor, torch.Tensor]:
    return compute_nv_scale_factors(
        x_scale_blocks,
        x_amax,
        fp4_format=DataType.nvint4,
        scale_rule=ScaleRule.static_6,
        round_style=round_style,
    )


def quantize_to_mxfp4(
    x_scale_blocks: torch.Tensor,
    *,
    scale_rule: ScaleRule = ScaleRule.mse,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert scale_rule in {ScaleRule.static_6, ScaleRule.static_4}

    x_scales_hp = x_scale_blocks.abs().max(
        axis=-1,
    ).values / DataType.mxfp4.quantized_value_type.get_maximum_value(scale_rule)

    x_scales_e8m0_u32 = x_scales_hp.view(torch.int32)

    # Use the 8-bit exponent as the scale factor
    x_scales_e8m0 = ((x_scales_e8m0_u32 >> 23) & 0xFF).to(torch.uint8)

    # Add one in order to round up
    x_scales = torch.where(
        (x_scales_e8m0_u32 & 0x7FFFFF) == 0,
        x_scales_e8m0,
        x_scales_e8m0 + 1,
    )

    # Convert the rounded-up scale factor back to a 32-bit float
    x_scales_hp = (x_scales.to(torch.int32) << 23).view(torch.float32)

    x_block_scaled = torch.where(
        x_scales.unsqueeze(1) != 0,
        x_scale_blocks / x_scales_hp.unsqueeze(1),
        0,
    )

    return x_block_scaled, x_scales.view(torch.float8_e8m0fnu)


def compute_nv_scale_factors(
    x_scale_blocks: torch.Tensor,
    x_amax: torch.Tensor,
    *,
    fp4_format: DataType,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
    scale_expansion_factor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scale factors for NV and NV_IF scale types."""
    max_quantized_value = fp4_format.quantized_value_type.get_maximum_value(scale_rule)
    max_scale_factor = fp4_format.scale_type.get_maximum_value(scale_rule)

    if x_amax == 0:
        x_scales_hp = torch.zeros(
            *x_scale_blocks.shape[:-1],
            dtype=x_amax.dtype,
            device=x_amax.device,
        )
    else:
        encode_scale = (
            torch.tensor(
                max_quantized_value * max_scale_factor * round_style.adjustment_factor,
                dtype=x_amax.dtype,
                device=x_amax.device,
            )
            / x_amax
        )
        x_scales_hp = (
            x_scale_blocks.abs().max(axis=-1).values
            / torch.tensor(
                max_quantized_value * round_style.adjustment_factor,
                dtype=x_amax.dtype,
                device=x_amax.device,
            )
            * encode_scale
        )

    if scale_expansion_factor is not None:
        x_scales_hp = x_scales_hp * scale_expansion_factor

    x_scales = x_scales_hp.to(torch.float8_e4m3fn)

    decode_scale = 1 / (
        torch.tensor(
            max_quantized_value * max_scale_factor * round_style.adjustment_factor,
            dtype=x_amax.dtype,
            device=x_amax.device,
        )
        / x_amax
    )
    x_block_scaled = torch.where(
        x_scales.unsqueeze(1) != 0,
        x_scale_blocks * (1 / (decode_scale * x_scales.to(x_amax.dtype).unsqueeze(1))),
        0,
    )

    return x_block_scaled, x_scales


def quantize_to_nvfp4(
    x_scale_blocks: torch.Tensor,
    x_amax: torch.Tensor,
    *,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
    scale_expansion_factor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return compute_nv_scale_factors(
        x_scale_blocks,
        x_amax,
        fp4_format=DataType.nvfp4,
        scale_rule=scale_rule,
        round_style=round_style,
        scale_expansion_factor=scale_expansion_factor,
    )


def select_fouroversix(
    x_scale_blocks: torch.Tensor,
    x_block_scaled_6: torch.Tensor,
    scales_6: torch.Tensor,
    x_block_scaled_4: torch.Tensor,
    scales_4: torch.Tensor,
    x_amax: torch.Tensor,
    *,
    scale_rule: ScaleRule = ScaleRule.mse,
    round_style: RoundStyle = RoundStyle.nearest,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_fake_quantized_6 = fake_quantize_to_e2m1(
        x_block_scaled_6,
        round_style=round_style,
    )
    x_fake_quantized_4 = fake_quantize_to_e2m1(
        x_block_scaled_4,
        round_style=round_style,
    )

    x_dequantized_6 = (
        x_fake_quantized_6.to(x_amax.dtype)
        * scales_6.unsqueeze(1).to(x_amax.dtype)
        * x_amax
        / torch.tensor(
            E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX * round_style.adjustment_factor,
            dtype=x_amax.dtype,
            device=x_amax.device,
        )
    )
    x_dequantized_4 = (
        x_fake_quantized_4.to(x_amax.dtype)
        * scales_4.unsqueeze(1).to(x_amax.dtype)
        * x_amax
        / torch.tensor(
            E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX * round_style.adjustment_factor,
            dtype=x_amax.dtype,
            device=x_amax.device,
        )
    )

    if scale_rule == ScaleRule.abs_max:
        x_error_4 = (x_dequantized_4 - x_scale_blocks).abs().max(axis=-1).values
        x_error_6 = (x_dequantized_6 - x_scale_blocks).abs().max(axis=-1).values
    elif scale_rule == ScaleRule.mae:
        x_error_4 = (x_dequantized_4 - x_scale_blocks).abs().sum(axis=-1)
        x_error_6 = (x_dequantized_6 - x_scale_blocks).abs().sum(axis=-1)
    elif scale_rule == ScaleRule.mse:
        x_error_4 = ((x_dequantized_4 - x_scale_blocks) ** 2).sum(axis=-1)
        x_error_6 = ((x_dequantized_6 - x_scale_blocks) ** 2).sum(axis=-1)

    select_4 = (x_error_4 < x_error_6).unsqueeze(1)
    x_fake_quantized = torch.where(
        select_4,
        x_fake_quantized_4.reshape(x_scale_blocks.shape[0], -1),
        x_fake_quantized_6.reshape(x_scale_blocks.shape[0], -1),
    )
    scales = torch.where(
        select_4,
        scales_4.reshape(-1, 1),
        scales_6.reshape(-1, 1),
    )

    return x_fake_quantized, scales


def select_intfloat(
    x_scale_blocks: torch.Tensor,
    x_block_scaled: torch.Tensor,
    scales: torch.Tensor,
    x_amax: torch.Tensor,
    *,
    fp4_format: DataType = DataType.if4,
    scale_rule: ScaleRule = ScaleRule.mse,
    round_style: RoundStyle = RoundStyle.nearest,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_quantized_value = fp4_format.quantized_value_type.get_maximum_value(
        ScaleRule.static_6,
    )
    max_scale_value = fp4_format.scale_type.get_maximum_value(ScaleRule.static_6)

    if fp4_format == DataType.if4:
        int_expansion_factor = IF4_INT_EXPANSION_FACTOR
        x_fake_quantized_fp = fake_quantize_to_e2m1(
            x_block_scaled,
            round_style=round_style,
        )
        x_fake_quantized_int = fake_quantize_to_int4(
            x_block_scaled * IF4_INT_EXPANSION_FACTOR_RCP,
            round_style=round_style,
        )
        x_quantized_int = None
    elif fp4_format == DataType.if6_e2m3:
        int_expansion_factor = IF6_E2M3_INT_EXPANSION_FACTOR
        x_quantized_fp = convert_to_e2m3_with_rtn(x_block_scaled)
        x_fake_quantized_fp = convert_e2m3_to_high_precision(x_quantized_fp)
        x_fake_quantized_int = fake_quantize_to_int6(
            x_block_scaled * IF6_E2M3_INT_EXPANSION_FACTOR_RCP,
            round_style=round_style,
        )
        x_quantized_int = (
            torch.where(x_fake_quantized_int < 0, 1, 0).to(torch.uint8) << 5
        ) | (torch.abs(x_fake_quantized_int).to(torch.uint8) & 0x1F)
    elif fp4_format == DataType.if6_e3m2:
        int_expansion_factor = IF6_E3M2_INT_EXPANSION_FACTOR
        x_quantized_fp = convert_to_e3m2_with_rtn(x_block_scaled)
        x_fake_quantized_fp = convert_e3m2_to_high_precision(x_quantized_fp)
        x_fake_quantized_int = fake_quantize_to_int6(
            x_block_scaled * IF6_E3M2_INT_EXPANSION_FACTOR_RCP,
            round_style=round_style,
        )
        x_quantized_int = (
            torch.where(x_fake_quantized_int < 0, 1, 0).to(torch.uint8) << 5
        ) | (torch.abs(x_fake_quantized_int).to(torch.uint8) & 0x1F)

    x_dequantized_fp = (
        x_fake_quantized_fp * scales.unsqueeze(1).to(x_amax.dtype) * x_amax
    ) / (max_quantized_value * max_scale_value * round_style.adjustment_factor)

    x_dequantized_int = (
        (
            x_fake_quantized_int.to(x_amax.dtype)
            * scales.unsqueeze(1).to(x_amax.dtype)
            * x_amax
        )
        * int_expansion_factor
    ) / (max_quantized_value * max_scale_value * round_style.adjustment_factor)

    if scale_rule == ScaleRule.abs_max:
        x_error_fp = (x_dequantized_fp - x_scale_blocks).abs().max(axis=-1).values
        x_error_int = (x_dequantized_int - x_scale_blocks).abs().max(axis=-1).values
    elif scale_rule == ScaleRule.mae:
        x_error_fp = (x_dequantized_fp - x_scale_blocks).abs().sum(axis=-1)
        x_error_int = (x_dequantized_int - x_scale_blocks).abs().sum(axis=-1)
    elif scale_rule == ScaleRule.mse:
        x_error_fp = ((x_dequantized_fp - x_scale_blocks) ** 2).sum(axis=-1)
        x_error_int = ((x_dequantized_int - x_scale_blocks) ** 2).sum(axis=-1)

    select_int = x_error_int < x_error_fp

    if x_quantized_int is None:
        x_quantized = None
    else:
        x_quantized = torch.where(
            select_int.unsqueeze(1),
            x_quantized_int.reshape(x_scale_blocks.shape[0], -1),
            x_quantized_fp.reshape(x_scale_blocks.shape[0], -1),
        )

    x_fake_quantized = torch.where(
        select_int.unsqueeze(1),
        x_fake_quantized_int.reshape(x_scale_blocks.shape[0], -1),
        x_fake_quantized_fp.reshape(x_scale_blocks.shape[0], -1),
    )

    # If int is selected, keep track of this by setting the sign bit of the scale
    # factor
    scales_with_if_indicator = torch.where(
        select_int,
        (scales.view(torch.uint8) + 128).view(torch.float8_e4m3fn),
        scales,
    )

    return x_quantized, x_fake_quantized, scales_with_if_indicator


def generic_block_scaled_quantization(
    x_scale_blocks: torch.Tensor,
    x_amax: torch.Tensor,
    *,
    fp4_format: DataType,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize using a static scale rule. Handles MXFP4, NVFP4, and NVINT4."""

    if fp4_format == DataType.mxfp4:
        x_block_scaled, scales = quantize_to_mxfp4(
            x_scale_blocks,
            scale_rule=scale_rule,
        )
        x_fake_quantized = fake_quantize_to_e2m1(
            x_block_scaled,
            round_style=round_style,
        )
    elif fp4_format == DataType.nvint4:
        x_block_scaled, scales = quantize_to_nvint4(
            x_scale_blocks,
            x_amax,
            round_style,
        )
        x_fake_quantized = fake_quantize_to_int4(
            x_block_scaled,
            round_style=round_style,
        )
    elif fp4_format == DataType.nvfp6_e2m3:
        x_block_scaled, scales = compute_nv_scale_factors(
            x_scale_blocks,
            x_amax,
            fp4_format=fp4_format,
            scale_rule=scale_rule,
            round_style=round_style,
        )
        x_fake_quantized = convert_to_e2m3_with_rtn(x_block_scaled)
    elif fp4_format == DataType.nvfp6_e3m2:
        x_block_scaled, scales = compute_nv_scale_factors(
            x_scale_blocks,
            x_amax,
            fp4_format=fp4_format,
            scale_rule=scale_rule,
            round_style=round_style,
        )
        x_fake_quantized = convert_to_e3m2_with_rtn(x_block_scaled)
    else:
        x_block_scaled, scales = quantize_to_nvfp4(
            x_scale_blocks,
            x_amax,
            scale_rule=scale_rule,
            round_style=round_style,
        )
        x_fake_quantized = fake_quantize_to_e2m1(
            x_block_scaled,
            round_style=round_style,
        )

    return x_fake_quantized, scales


def intfloat_block_scaled_quantization(
    x_scale_blocks: torch.Tensor,
    x_amax: torch.Tensor,
    *,
    fp4_format: DataType = DataType.if4,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize using intfloat format, selecting between int and float per block."""

    x_block_scaled, scales = compute_nv_scale_factors(
        x_scale_blocks,
        x_amax,
        fp4_format=fp4_format,
        scale_rule=ScaleRule.static_6,
        round_style=round_style,
    )

    return select_intfloat(
        x_scale_blocks,
        x_block_scaled,
        scales,
        x_amax,
        fp4_format=fp4_format,
        scale_rule=scale_rule,
        round_style=round_style,
    )


def nvfp4_fouroversix_block_scaled_quantization(
    x_scale_blocks: torch.Tensor,
    x_amax: torch.Tensor,
    *,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize NVFP4 with adaptive 4/6 block scale selection."""

    x_block_scaled_6, scales_6 = quantize_to_nvfp4(
        x_scale_blocks,
        x_amax,
        scale_rule=scale_rule,
        round_style=round_style,
    )

    x_block_scaled_4, scales_4 = quantize_to_nvfp4(
        x_scale_blocks,
        x_amax,
        scale_rule=scale_rule,
        round_style=round_style,
        scale_expansion_factor=1.5,
    )

    return select_fouroversix(
        x_scale_blocks,
        x_block_scaled_6,
        scales_6,
        x_block_scaled_4,
        scales_4,
        x_amax,
        scale_rule=scale_rule,
        round_style=round_style,
    )


def quantize(  # noqa: C901, PLR0912
    x: torch.Tensor,
    x_amax: torch.Tensor | None = None,
    had: torch.Tensor | None = None,
    *,
    block_scale_2d: bool = False,
    fp4_format: DataType = DataType.nvfp4,
    round_style: RoundStyle = RoundStyle.nearest,
    scale_rule: ScaleRule = ScaleRule.mse,
    transpose: bool = False,
    use_blackwell_scale_layout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if transpose:
        x = x.T

    if had is not None:
        x = (x.reshape(-1, had.shape[0]) @ had.to(x.dtype)).reshape_as(x)

    if x_amax is None:
        x_amax = (
            torch.ones(1, device=x.device, dtype=torch.float32)
            if fp4_format == DataType.mxfp4
            else x.abs().max().float()
        )

    if block_scale_2d:
        assert x.ndim == 2  # noqa: PLR2004
        assert x.shape[1] % fp4_format.block_size == 0

        x_scale_blocks = (
            x.reshape(
                -1,
                fp4_format.block_size,
                x.shape[1] // fp4_format.block_size,
                fp4_format.block_size,
            )
            .permute(0, 2, 1, 3)
            .reshape(-1, fp4_format.block_size**2)
            .float()
        )
    else:
        x_scale_blocks = x.reshape(-1, fp4_format.block_size).float()

    if scale_rule in {ScaleRule.static_6, ScaleRule.static_4}:
        x_fake_quantized, scales = generic_block_scaled_quantization(
            x_scale_blocks,
            x_amax,
            fp4_format=fp4_format,
            scale_rule=scale_rule,
            round_style=round_style,
        )
    elif fp4_format.scale_type == ScaleType.nv_if:
        x_quantized, x_fake_quantized, scales = intfloat_block_scaled_quantization(
            x_scale_blocks,
            x_amax,
            fp4_format=fp4_format,
            scale_rule=scale_rule,
            round_style=round_style,
        )
    else:
        x_fake_quantized, scales = nvfp4_fouroversix_block_scaled_quantization(
            x_scale_blocks,
            x_amax,
            scale_rule=scale_rule,
            round_style=round_style,
        )

    if block_scale_2d:
        x_fake_quantized = x_fake_quantized.reshape(
            -1,
            x.shape[1] // fp4_format.block_size,
            fp4_format.block_size,
            fp4_format.block_size,
        ).permute(0, 2, 1, 3)

        scales = (
            scales.reshape(
                1,
                x.shape[0] // fp4_format.block_size,
                x.shape[1] // fp4_format.block_size,
            )
            .broadcast_to(
                fp4_format.block_size,
                x.shape[0] // fp4_format.block_size,
                x.shape[1] // fp4_format.block_size,
            )
            .permute(1, 0, 2)
        )

    if fp4_format == DataType.if4:
        x_quantized = pack_unpacked_fp4(
            torch.where(
                scales.unsqueeze(-1).view(torch.uint8) >= 128,  # noqa: PLR2004
                quantize_bf16_to_unpacked_int4(x_fake_quantized.bfloat16()),
                quantize_bf16_to_unpacked_fp4(x_fake_quantized.bfloat16()),
            ).reshape_as(x),
        )
    elif fp4_format in {DataType.if6_e2m3, DataType.if6_e3m2}:
        x_quantized = x_quantized.reshape_as(x)
    elif fp4_format == DataType.nvint4:
        x_quantized = pack_unpacked_fp4(
            quantize_bf16_to_unpacked_int4(
                x_fake_quantized.bfloat16().reshape_as(x),
            ),
        )
    elif fp4_format in {DataType.nvfp6_e2m3, DataType.nvfp6_e3m2}:
        x_quantized = x_fake_quantized.reshape_as(x)
    else:
        x_quantized = pack_unpacked_fp4(
            quantize_bf16_to_unpacked_fp4(
                x_fake_quantized.bfloat16().reshape_as(x),
            ),
        )

    if use_blackwell_scale_layout:
        reshaped_scales = to_blocked(
            scales.reshape(
                x.shape[0],
                x.shape[1] // fp4_format.block_size,
            ),
        )
    else:
        reshaped_scales = scales.flatten()

    return x_quantized, reshaped_scales, x_amax
