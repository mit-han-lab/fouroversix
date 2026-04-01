import itertools
from typing import Any

import pytest
import torch
from fouroversix import (
    DataType,
    QuantizationConfig,
    QuantizeBackend,
    RoundStyle,
    ScaleRule,
    dequantize,
    quantize,
)
from fouroversix.quantize import from_blocked
from fouroversix.quantize.frontend import AVAILABLE_BACKENDS
from fouroversix.utils import SM_80, SM_100, SM_120

MAE_MSE_MISMATCH_TOLERANCE = 1e-3
NUM_RANDOM_SEEDS = 10


@pytest.mark.parametrize("input_type", ["zeros", "ones", "rand01", "randn"])
@pytest.mark.parametrize(
    "input_shape",
    [(1024, 1024), (1024, 512), (512, 1024)],
)
@pytest.mark.parametrize(
    ("backend_a", "kwargs_a", "backend_b", "kwargs_b"),
    [
        (a[0], a[1], b[0], b[1])
        for a, b in itertools.chain(
            itertools.combinations(
                [
                    (QuantizeBackend.cuda, {}),
                    (QuantizeBackend.triton, {}),
                    (QuantizeBackend.pytorch, {}),
                    (QuantizeBackend.transformer_engine, {}),
                ],
                r=2,
            ),
            [
                (
                    (QuantizeBackend.triton, {}),
                    (QuantizeBackend.triton, {"major_compute_capability": SM_120}),
                ),
                (
                    (QuantizeBackend.triton, {}),
                    (QuantizeBackend.triton, {"major_compute_capability": SM_100}),
                ),
                (
                    (QuantizeBackend.triton, {}),
                    (QuantizeBackend.triton, {"major_compute_capability": SM_80}),
                ),
            ],
        )
    ],
)
@pytest.mark.parametrize("block_scale_2d", ["block_scale_2d", "no_block_scale_2d"])
@pytest.mark.parametrize(
    "dtype",
    [
        DataType.if4,
        DataType.if6_e2m3,
        DataType.if6_e3m2,
        DataType.mxfp4,
        DataType.nvfp4,
        DataType.nvfp6_e2m3,
        DataType.nvfp6_e3m2,
        DataType.nvint4,
    ],
)
@pytest.mark.parametrize("rht", ["rht", "no_rht"])
@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
@pytest.mark.parametrize("round_style", [RoundStyle.nearest])
@pytest.mark.parametrize("transpose", ["transpose", "no_transpose"])
def test_backend_outputs_are_consistent(  # noqa: C901, PLR0912, PLR0915
    input_type: str,
    input_shape: tuple[int, int],
    backend_a: QuantizeBackend,
    kwargs_a: dict[str, Any],
    backend_b: QuantizeBackend,
    kwargs_b: dict[str, Any],
    *,
    block_scale_2d: str,
    dtype: DataType,
    rht: str,
    round_style: RoundStyle,
    scale_rule: ScaleRule,
    transpose: str,
) -> None:
    torch.set_printoptions(precision=10)

    if (
        kwargs_a.get("major_compute_capability") is not None
        or kwargs_b.get("major_compute_capability") is not None
    ) and torch.cuda.get_device_capability()[0] != SM_100:
        pytest.skip("Can only simulate different major compute capabilities on SM_100")

    if dtype in {
        DataType.nvfp6_e2m3,
        DataType.nvfp6_e3m2,
        DataType.if6_e2m3,
        DataType.if6_e3m2,
    } and (
        kwargs_a.get("major_compute_capability") is not None
        or kwargs_b.get("major_compute_capability") is not None
    ):
        pytest.skip("No simulation is allowed for NVFP6 right now")

    backend_a_cls = AVAILABLE_BACKENDS[backend_a]
    backend_b_cls = AVAILABLE_BACKENDS[backend_b]

    if not backend_a_cls.is_available() or not backend_b_cls.is_available():
        pytest.skip("Backend is not available")

    config_a = QuantizationConfig(
        backend=backend_a,
        block_scale_2d=block_scale_2d == "block_scale_2d",
        dtype=dtype,
        kwargs=kwargs_a,
        rht=rht == "rht",
        round_style=round_style,
        scale_rule=scale_rule,
        transpose=transpose == "transpose",
    )

    config_b = QuantizationConfig(
        backend=backend_b,
        block_scale_2d=block_scale_2d == "block_scale_2d",
        dtype=dtype,
        kwargs=kwargs_b,
        rht=rht == "rht",
        round_style=round_style,
        scale_rule=scale_rule,
        transpose=transpose == "transpose",
    )

    if round_style.is_stochastic:
        pytest.xfail("This test is not currently targeting stochastic rounding")

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

        if not backend_a_cls.can_quantize(
            x,
            config_a,
        ) or not backend_b_cls.can_quantize(
            x,
            config_b,
        ):
            pytest.skip("Backend is not supported")

        quantized_a = quantize(x.clone(), config_a)
        quantized_b = quantize(x.clone(), config_b)

        if not torch.allclose(quantized_a.amax, quantized_b.amax):
            print("Backends A and B have different amax values!")
            print(f"{backend_a}: {quantized_a.amax}")
            print(f"{backend_b}: {quantized_b.amax}")
            pytest.fail("Backends A and B have different amax values!")

        if quantized_a.scale_factors_are_in_blackwell_layout:
            sf_a = from_blocked(
                quantized_a.scale_factors.bfloat16(),
                (input_shape[0], input_shape[1] // dtype.block_size),
            )
        else:
            sf_a = quantized_a.scale_factors.bfloat16().reshape(
                input_shape[0],
                input_shape[1] // dtype.block_size,
            )

        if quantized_b.scale_factors_are_in_blackwell_layout:
            sf_b = from_blocked(
                quantized_b.scale_factors.bfloat16(),
                (input_shape[0], input_shape[1] // dtype.block_size),
            )
        else:
            sf_b = quantized_b.scale_factors.bfloat16().reshape(
                input_shape[0],
                input_shape[1] // dtype.block_size,
            )

        quantized_parameters_per_group = (
            dtype.block_size // dtype.quantized_value_type.packing_factor
        )

        # When computing 4/6 with the MAE and MSE scale rules, computing the errors
        # requires summing the errors in each block of 16 values. This operation
        # differently (elements are summed in different orders, and floating-point
        # addition is not associative) in PyTorch and Triton, and can not be easily made
        # deterministic in a way that allows for good performance. As a result, we allow
        # a small number of mismatches between the scale factors and values for these
        # two rules. Fortunately, abs_max does not involve a summation, so we can use it
        # to test the correctness of the rest of the 4/6 implementation.
        scale_factors_mismatch_prop = (sf_a != sf_b).sum() / sf_a.numel()

        if (
            scale_rule in {ScaleRule.static_6, ScaleRule.static_4, ScaleRule.abs_max}
            and scale_factors_mismatch_prop > 0
        ) or scale_factors_mismatch_prop >= MAE_MSE_MISMATCH_TOLERANCE:
            print(
                "Backends A and B have different scale factors! "
                f"{scale_factors_mismatch_prop:.2%} mismatch",
            )

            [i, *_], [j, *_] = torch.where(sf_a != sf_b)
            print(i, j)
            print(backend_a)
            print("amax", quantized_a.amax)
            print("sf", sf_a[i, j])
            print(
                "e2m1",
                quantized_a.values[
                    i,
                    quantized_parameters_per_group
                    * j : quantized_parameters_per_group
                    * (j + 1),
                ],
            )
            print(backend_b)
            print("sf", sf_b[i, j])
            print(
                "e2m1",
                quantized_b.values[
                    i,
                    quantized_parameters_per_group
                    * j : quantized_parameters_per_group
                    * (j + 1),
                ],
            )
            print("original")
            print("x", x[i, dtype.block_size * j : dtype.block_size * (j + 1)])
            pytest.fail("Backends A and B have different scale factors!")

        values_mismatch_prop = (
            quantized_a.values != quantized_b.values
        ).sum() / quantized_a.values.numel()

        if (
            scale_rule in {ScaleRule.static_6, ScaleRule.static_4, ScaleRule.abs_max}
            and values_mismatch_prop > 0
        ) or values_mismatch_prop >= MAE_MSE_MISMATCH_TOLERANCE:
            print(
                "Backends A and B have different e2m1 values! "
                f"{values_mismatch_prop:.2%} mismatch",
            )

            [i, *_], [j, *_] = torch.where(
                quantized_a.values != quantized_b.values,
            )
            print(i, j)
            print("amax", quantized_a.amax)
            print("sf", sf_a[i, j // quantized_parameters_per_group])
            print(backend_a)
            print(
                "e2m1",
                quantized_a.values[
                    i,
                    quantized_parameters_per_group
                    * (
                        j // quantized_parameters_per_group
                    ) : quantized_parameters_per_group
                    * (j // quantized_parameters_per_group + 1),
                ],
            )
            print(backend_b)
            print(
                "e2m1",
                quantized_b.values[
                    i,
                    quantized_parameters_per_group
                    * (
                        j // quantized_parameters_per_group
                    ) : quantized_parameters_per_group
                    * (j // quantized_parameters_per_group + 1),
                ],
            )
            print("original")
            print(
                "x",
                x[
                    i,
                    dtype.block_size
                    * (j // quantized_parameters_per_group) : dtype.block_size
                    * (j // quantized_parameters_per_group + 1),
                ],
            )
            pytest.fail("Backends A and B have different e2m1 values!")


def test_stochastic_rounding() -> None:
    test_cases = [
        ({"backend": "transformer_engine", "scale_rule": "static_6"}, 141, None),
        ({"backend": "triton", "scale_rule": "static_6"}, 137, 158),
        ({"backend": "triton", "scale_rule": "mse"}, 122, 139),
        ({"backend": "triton", "dtype": "nvint4", "scale_rule": "static_6"}, 125, 135),
        ({"backend": "triton", "dtype": "if4", "scale_rule": "mse"}, 110, 122),
        (
            {
                "backend": "triton",
                "scale_rule": "static_6",
                "kwargs": {"major_compute_capability": SM_120},
            },
            141,
            158,
        ),
        (
            {
                "backend": "triton",
                "scale_rule": "mse",
                "kwargs": {"major_compute_capability": SM_120},
            },
            126,
            139,
        ),
        (
            {
                "backend": "triton",
                "scale_rule": "static_6",
                "kwargs": {"major_compute_capability": SM_80},
            },
            141,
            158,
        ),
        (
            {
                "backend": "triton",
                "scale_rule": "mse",
                "kwargs": {"major_compute_capability": SM_80},
            },
            126,
            139,
        ),
        ({"backend": "pytorch", "scale_rule": "static_6"}, 137, 158),
        ({"backend": "pytorch", "scale_rule": "mse"}, 122, 139),
    ]

    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")

    for kwargs, expected_dist, expected_dist_unbiased in test_cases:
        config = QuantizationConfig(round_style=RoundStyle.stochastic, **kwargs)
        out = quantize(x, config)
        x_dequantized = dequantize(out)
        dist = torch.dist(x_dequantized, x)
        print(kwargs, expected_dist, dist)
        assert abs((dist - expected_dist) / expected_dist) < 0.05  # noqa: PLR2004

        if expected_dist_unbiased is None:
            continue

        config = QuantizationConfig(
            round_style=RoundStyle.stochastic_unbiased,
            **kwargs,
        )
        out = quantize(x, config)
        x_dequantized = dequantize(out)
        dist = torch.dist(x_dequantized, x)
        print(kwargs, expected_dist_unbiased, dist)
        assert (
            abs((dist - expected_dist_unbiased) / expected_dist_unbiased)
            < 0.05  # noqa: PLR2004
        )


@pytest.mark.parametrize(
    "input_shape",
    [(1024, 1024), (1024, 512), (512, 1024)],
)
@pytest.mark.parametrize(
    "dtype",
    [
        DataType.nvfp4,
        DataType.if4,
        DataType.mxfp4,
        DataType.nvint4,
    ],
)
@pytest.mark.parametrize(
    "scale_rule",
    [
        ScaleRule.abs_max,
        ScaleRule.mae,
        ScaleRule.mse,
        ScaleRule.static_4,
        ScaleRule.static_6,
    ],
)
def test_pseudo_quantize(
    input_shape: tuple[int, int],
    dtype: DataType,
    scale_rule: ScaleRule,
) -> None:
    if scale_rule not in dtype.supported_scale_rules:
        pytest.skip(f"Scale rule {scale_rule} not supported for dtype {dtype}")

    backend = QuantizeBackend.triton

    if not AVAILABLE_BACKENDS[backend].is_available():
        pytest.skip("Triton backend is not available")

    for _ in range(NUM_RANDOM_SEEDS):
        x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")

        config_roundtrip = QuantizationConfig(
            backend=backend,
            dtype=dtype,
            scale_rule=scale_rule,
        )

        if not AVAILABLE_BACKENDS[backend].can_quantize(x, config_roundtrip):
            pytest.skip("Backend does not support this configuration")

        expected = dequantize(
            quantize(x.clone(), config_roundtrip),
            dtype=x.dtype,
            intermediate_dtype=torch.float32,
        )

        config_pseudo = QuantizationConfig(
            backend=backend,
            dtype=dtype,
            scale_rule=scale_rule,
            pseudo_quantize=True,
        )

        result = quantize(x.clone(), config_pseudo)
        print(torch.dist(result, expected))

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.bfloat16
        assert result.shape == x.shape
        assert torch.equal(result, expected)
