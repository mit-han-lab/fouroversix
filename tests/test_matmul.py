import itertools

import pytest
import torch
from fouroversix import (
    DataType,
    MatmulBackend,
    QuantizationConfig,
    ScaleRule,
    quantize,
    quantized_matmul,
)
from fouroversix.matmul.frontend import AVAILABLE_BACKENDS

MISMATCH_TOLERANCE = 1e-3
NUM_RANDOM_SEEDS = 10


@pytest.mark.parametrize(("m", "n", "k"), itertools.product([256, 512, 1024], repeat=3))
@pytest.mark.parametrize(
    ("backend_a", "backend_b"),
    itertools.combinations(
        [
            MatmulBackend.cutlass,
            MatmulBackend.triton,
            MatmulBackend.pytorch,
        ],
        r=2,
    ),
)
@pytest.mark.parametrize("dtype", [DataType.if4, DataType.nvfp4, DataType.nvint4])
@pytest.mark.parametrize(
    "scale_rule",
    [ScaleRule.static_6, ScaleRule.static_4, ScaleRule.mse],
)
def test_matmul(
    m: int,
    n: int,
    k: int,
    backend_a: MatmulBackend,
    backend_b: MatmulBackend,
    dtype: DataType,
    scale_rule: ScaleRule,
) -> None:
    torch.set_printoptions(precision=10)

    if scale_rule not in dtype.supported_scale_rules:
        pytest.skip(f"Scale rule {scale_rule} is not supported for dtype {dtype}")

    backend_a_cls = AVAILABLE_BACKENDS[backend_a]
    backend_b_cls = AVAILABLE_BACKENDS[backend_b]

    if not backend_a_cls.is_available() or not backend_b_cls.is_available():
        pytest.skip("Backend is not available")

    for random_seed in range(NUM_RANDOM_SEEDS):
        print(f"Testing with random seed: {random_seed}")
        torch.manual_seed(random_seed)

        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        y = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

        config = QuantizationConfig(dtype=dtype, scale_rule=scale_rule)
        x_quantized = quantize(x, config)
        y_quantized = quantize(y, config)

        if not backend_a_cls.is_supported(
            x_quantized,
            y_quantized,
            out_dtype=DataType.bfloat16,
        ) or not backend_b_cls.is_supported(
            x_quantized,
            y_quantized,
            out_dtype=DataType.bfloat16,
        ):
            pytest.skip("Backend is not supported")

        out_a = quantized_matmul(x_quantized, y_quantized, backend=backend_a)
        out_b = quantized_matmul(x_quantized, y_quantized, backend=backend_b)

        values_mismatch_prop = (out_a != out_b).sum() / out_a.numel()

        if values_mismatch_prop > MISMATCH_TOLERANCE:
            print(f"Values mismatch: {values_mismatch_prop:.2%}")
            print(f"Distance: {torch.dist(out_a, out_b)}")
            print(f"Out A: {out_a}")
            print(f"Out B: {out_b}")
            pytest.fail("Values mismatch")
