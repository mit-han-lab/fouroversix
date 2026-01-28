import pytest
import torch
from fouroversix import (
    AdaptiveBlockScalingRule,
    FP4Format,
    QuantizeBackend,
    RoundStyle,
    quantize_to_fp4,
)
from fouroversix.quantize.rht_utils import get_rht_matrix


@pytest.mark.parametrize("input_type", ["zeros", "ones", "rand01", "randn", "fixed"])
@pytest.mark.parametrize(
    "input_shape",
    [(1024, 1024), (1024, 512), (512, 1024), (8192, 8192), (8192, 4096), (4096, 8192)],
)
@pytest.mark.parametrize("random_seed", range(10))
@pytest.mark.parametrize(
    ("backend_a", "backend_b"),
    [
        (QuantizeBackend.transformer_engine, QuantizeBackend.triton),
        (QuantizeBackend.cuda, QuantizeBackend.triton),
        (QuantizeBackend.cuda, QuantizeBackend.pytorch),
        (QuantizeBackend.triton, QuantizeBackend.pytorch),
    ],
)
@pytest.mark.parametrize("block_scale_2d", ["block_scale_2d", "no_block_scale_2d"])
@pytest.mark.parametrize("fp4_format", [FP4Format.nvfp4])
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
def test_backend_outputs_are_consistent(
    input_type: str,
    input_shape: tuple[int, int],
    random_seed: int,
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
    torch.set_printoptions(precision=10)

    if not backend_a.is_available() or not backend_b.is_available():
        pytest.skip("Backend is not available")

    block_scale_2d = block_scale_2d == "block_scale_2d"
    had = had == "had"
    transpose = transpose == "transpose"

    torch.manual_seed(random_seed)

    if input_type == "zeros":
        x = torch.zeros(*input_shape, dtype=torch.bfloat16, device="cuda")
    elif input_type == "ones":
        x = torch.ones(*input_shape, dtype=torch.bfloat16, device="cuda")
    elif input_type == "rand01":
        x = torch.randint(0, 2, input_shape, dtype=int, device="cuda").to(
            torch.bfloat16,
        )
    elif input_type == "randn":
        x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
    elif input_type == "fixed":
        x = torch.zeros(*input_shape, dtype=torch.bfloat16, device="cuda")
        x[0, :16] = torch.tensor(
            [
                0.3125000000,
                0.3671875000,
                2.0468750000,
                -0.4863281250,
                0.6640625000,
                -0.2001953125,
                0.7070312500,
                -1.5000000000,
                -0.5742187500,
                0.0639648438,
                -1.4921875000,
                -0.3417968750,
                -0.3828125000,
                -0.9492187500,
                0.2929687500,
                1.5390625000,
            ],
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        x[0, 16] = 4.9062500000
    elif input_type == "fixed2":
        x = torch.zeros(*input_shape, dtype=torch.bfloat16, device="cuda")
        x[0, :3] = torch.tensor(
            [
                0.1767578125,
                -1.3203125000,
                2.1875000000,
            ],
            device="cuda:0",
            dtype=torch.bfloat16,
        )
        x[0, 16] = 4.875
    else:
        msg = f"Invalid input type: {input_type}"
        raise ValueError(msg)

    kwargs = {
        "block_scale_2d": block_scale_2d,
        "fp4_format": fp4_format,
        "had": get_rht_matrix() if had else None,
        "round_style": round_style,
        "scale_rule": scale_rule,
        "transpose": transpose,
    }

    if not backend_a.is_supported(x, **kwargs) or not backend_b.is_supported(
        x,
        **kwargs,
    ):
        pytest.skip("Backend is not supported")

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

    if input_type == "fixed":
        x_sf_a = x_sf_a[0:2]
        x_sf_b = x_sf_b[0:2]
        x_e2m1_a = x_e2m1_a[0, 0:8]
        x_e2m1_b = x_e2m1_b[0, 0:8]

    assert torch.allclose(x_normconst_a, x_normconst_b)
    assert torch.allclose(x_sf_a.bfloat16(), x_sf_b.bfloat16())
    assert torch.allclose(x_e2m1_a, x_e2m1_b)


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
def test_zeros(scale_rule: AdaptiveBlockScalingRule) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.zeros(1024, 1024, dtype=torch.bfloat16, device=device)
    x_e2m1, x_sf, x_normconst = quantize_to_fp4(
        x,
        backend=QuantizeBackend.pytorch,
        scale_rule=scale_rule,
    )

    x_e2m1_expected = torch.zeros(1024, 512, dtype=torch.uint8, device=device)
    x_sf_expected = torch.full(
        (1024 * 1024 // 16,),
        0,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    x_normconst_expected = torch.tensor(
        0,
        dtype=torch.bfloat16,
        device=device,
    )

    assert torch.allclose(x_normconst, x_normconst_expected)
    assert torch.allclose(x_sf.bfloat16(), x_sf_expected.bfloat16())
    assert torch.allclose(x_e2m1, x_e2m1_expected)
