import itertools

import pytest
import torch
from fouroversix import (
    AdaptiveBlockScalingRule,
    FP4Format,
    QuantizeBackend,
    RoundStyle,
    quantize_to_fp4,
)
from fouroversix.quantize import get_rht_matrix

MSE_L1_NORM_MISMATCH_TOLERANCE = 1e-3
NUM_RANDOM_SEEDS = 10


@pytest.mark.parametrize("input_type", ["zeros", "ones", "rand01", "randn"])
@pytest.mark.parametrize(
    "input_shape",
    [(1024, 1024), (1024, 512), (512, 1024), (8192, 8192), (8192, 4096), (4096, 8192)],
)
@pytest.mark.parametrize(
    ("backend_a", "backend_b"),
    [
        (backend_a, backend_b)
        for backend_a, backend_b in itertools.product(
            [
                QuantizeBackend.cuda,
                QuantizeBackend.triton,
                QuantizeBackend.pytorch,
                QuantizeBackend.transformer_engine,
            ],
            [
                QuantizeBackend.cuda,
                QuantizeBackend.triton,
                QuantizeBackend.pytorch,
                QuantizeBackend.transformer_engine,
            ],
        )
        if backend_a != backend_b
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

    kwargs = {
        "block_scale_2d": block_scale_2d,
        "fp4_format": fp4_format,
        "had": get_rht_matrix() if had else None,
        "round_style": round_style,
        "scale_rule": scale_rule,
        "transpose": transpose,
    }

    if block_scale_2d or had or transpose or round_style == RoundStyle.stochastic:
        pytest.xfail("This test is currently not targeting FP4 training features")

    for random_seed in range(NUM_RANDOM_SEEDS):
        print(f"Testing with random seed: {random_seed}")
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
        else:
            msg = f"Invalid input type: {input_type}"
            raise ValueError(msg)

        if not backend_a.is_supported(x, **kwargs) or not backend_b.is_supported(
            x,
            **kwargs,
        ):
            pytest.skip("Backend is not supported")

        quantized_a = quantize_to_fp4(x, backend=backend_a, **kwargs)
        quantized_b = quantize_to_fp4(x, backend=backend_b, **kwargs)

        assert torch.allclose(quantized_a.amax, quantized_b.amax)

        sf_a = quantized_a.scale_factors.bfloat16()
        sf_b = quantized_b.scale_factors.bfloat16()

        if scale_rule in {
            AdaptiveBlockScalingRule.always_6,
            AdaptiveBlockScalingRule.always_4,
            AdaptiveBlockScalingRule.abs_max,
        }:
            assert torch.allclose(sf_a, sf_b)
            assert torch.allclose(quantized_a.e2m1_values, quantized_b.e2m1_values)
        else:
            # When computing 4/6 with the MSE and L1 norm scale rules, computing the
            # errors requires summing the errors in each block of 16 values. This
            # operation executes differently (elements are summed in different orders,
            # and floating-point addition is not associative) in PyTorch and Triton, and
            # can not be easily made deterministic in a way that allows for good
            # performance. As a result, we allow a small number of mismatches between
            # the scale factors and e2m1 values for these two rules. Fortunately,
            # abs_max does not involve a summation, so we can use it to test the
            # correctness of the rest of the 4/6 implementation.

            scale_factors_mismatch_prop = (sf_a != sf_b).sum() / sf_a.numel()
            assert scale_factors_mismatch_prop < MSE_L1_NORM_MISMATCH_TOLERANCE

            e2m1_values_mismatch_prop = (
                quantized_a.e2m1_values != quantized_b.e2m1_values
            ).sum() / quantized_a.e2m1_values.numel()
            assert e2m1_values_mismatch_prop < MSE_L1_NORM_MISMATCH_TOLERANCE


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
