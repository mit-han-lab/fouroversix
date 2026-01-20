import pytest
import torch
from fouroversix import (
    AdaptiveBlockScalingRule,
    QuantizeBackend,
    quantize_to_fp4,
)
from fouroversix.quantize import get_rht_matrix


@pytest.mark.parametrize("block_scale_2d", [True, False])
@pytest.mark.parametrize("rht", [True, False])
@pytest.mark.parametrize(
    "scale_rule",
    [
        AdaptiveBlockScalingRule.always_6,
        AdaptiveBlockScalingRule.always_4,
        AdaptiveBlockScalingRule.mse,
        AdaptiveBlockScalingRule.l1_norm,
        AdaptiveBlockScalingRule.abs_max,
    ],
)
def test_rht(
    *,
    block_scale_2d: bool,
    rht: bool,
    scale_rule: AdaptiveBlockScalingRule,
) -> None:
    torch.manual_seed(0)

    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    had = get_rht_matrix() if rht else None

    y_e2m1_identity, y_sf_identity, y_normconst_identity = quantize_to_fp4(
        x,
        had=had,
        backend=QuantizeBackend.triton,
        block_scale_2d=block_scale_2d,
        scale_rule=scale_rule,
    )
    y_e2m1_transpose, y_sf_transpose, y_normconst_transpose = quantize_to_fp4(
        x.T.contiguous(),
        had=had,
        transpose=True,
        backend=QuantizeBackend.triton,
        block_scale_2d=block_scale_2d,
        scale_rule=scale_rule,
    )

    assert torch.allclose(y_normconst_identity, y_normconst_transpose)
    assert torch.allclose(y_sf_identity.bfloat16(), y_sf_transpose.bfloat16())
    assert torch.allclose(y_e2m1_identity, y_e2m1_transpose)
