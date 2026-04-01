from __future__ import annotations

import torch
from fouroversix.utils import RoundStyle

INT4_MAX_VALUE = 7


def fake_quantize_to_int4(
    x: torch.Tensor,
    *,
    round_style: RoundStyle = RoundStyle.nearest,
) -> torch.Tensor:
    if round_style.is_stochastic:
        rbits = torch.rand_like(x) - 0.5
        return x.sign() * (x.abs() + rbits).clamp(min=-7, max=7).round()

    return x.clamp(min=-7, max=7).round()


def quantize_bf16_to_unpacked_int4(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(min=-7, max=7).to(torch.int8).view(torch.uint8) & 0xF
