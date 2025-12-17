import pytest
import torch
from fouroversix import (
    AdaptiveBlockScalingRule,
    FP4Format,
    QuantizeBackend,
    RoundStyle,
    quantize_to_fp4,
)
from scipy.linalg import hadamard


@pytest.mark.parametrize("input_type", ["zeros", "ones", "randn"])
@pytest.mark.parametrize("input_shape", [(1024, 1024)])
@pytest.mark.parametrize(
    ("backend_a", "backend_b"),
    [
        (QuantizeBackend.cuda, QuantizeBackend.triton),
        (QuantizeBackend.cuda, QuantizeBackend.pytorch),
        (QuantizeBackend.triton, QuantizeBackend.pytorch),
    ],
)
@pytest.mark.parametrize("block_scale_2d", ["block_scale_2d", "no_block_scale_2d"])
@pytest.mark.parametrize("fp4_format", [FP4Format.nvfp4, FP4Format.mxfp4])
@pytest.mark.parametrize("had", ["had", "no_had"])
@pytest.mark.parametrize(
    "scale_rule",
    [
        AdaptiveBlockScalingRule.abs_max,
        AdaptiveBlockScalingRule.l1_norm,
        AdaptiveBlockScalingRule.mse,
        AdaptiveBlockScalingRule.always_4,
        AdaptiveBlockScalingRule.always_6,
    ],
)
@pytest.mark.parametrize("round_style", [RoundStyle.nearest, RoundStyle.stochastic])
@pytest.mark.parametrize("transpose", ["transpose", "no_transpose"])
def test_correctness(
    input_type: str,
    input_shape: tuple[int, int],
    backend_a: QuantizeBackend,
    backend_b: QuantizeBackend,
    *,
    block_scale_2d: str,
    fp4_format: FP4Format,
    had: str,
    round_style: RoundStyle,
    scale_rule: AdaptiveBlockScalingRule,
    transpose: str,
) -> None:
    block_scale_2d = block_scale_2d == "block_scale_2d"
    had = had == "had"
    transpose = transpose == "transpose"

    if input_type == "zeros":
        x = torch.zeros(*input_shape, dtype=torch.bfloat16, device="cuda")
    elif input_type == "ones":
        x = torch.ones(*input_shape, dtype=torch.bfloat16, device="cuda")
    elif input_type == "randn":
        x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    else:
        msg = f"Invalid input type: {input_type}"
        raise ValueError(msg)

    kwargs = {
        "block_scale_2d": block_scale_2d,
        "fp4_format": fp4_format,
        "had": (
            torch.tensor(hadamard(16) / (16**0.5), dtype=torch.bfloat16, device="cuda")
            if had
            else None
        ),
        "round_style": round_style,
        "scale_rule": scale_rule,
        "transpose": transpose,
    }
    x_e2m1_a, x_sf_a, x_normconst_a = quantize_to_fp4(
        x,
        backend=backend_a,
        **kwargs,
    )
    x_e2m1_b, x_sf_b, x_normconst_b = quantize_to_fp4(
        x,
        backend=backend_b,
        **kwargs,
    )
    assert torch.allclose(x_normconst_a, x_normconst_b)
    assert torch.allclose(x_e2m1_a, x_e2m1_b)
    assert torch.allclose(x_sf_a.bfloat16(), x_sf_b.bfloat16())
