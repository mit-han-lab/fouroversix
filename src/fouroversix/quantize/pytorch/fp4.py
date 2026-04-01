from __future__ import annotations

import torch
from fouroversix.utils import RoundStyle

E2M1_MAX_VALUE = 6


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


def quantize_bf16_to_unpacked_fp4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16  # noqa: S101

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
    assert x.dtype == torch.uint8  # noqa: S101

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
