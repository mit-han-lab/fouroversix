"""Tests for transpose_quantized_tensor and TransposableFourOverSixLinear."""

import pytest
import torch
from fouroversix import (
    QuantizationConfig,
    QuantizeBackend,
    ScaleRule,
    dequantize,
    quantize,
)
from fouroversix.quantize.transpose import transpose_quantized_tensor

# ---------------------------------------------------------------------------
# transpose_quantized_tensor correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "cols"),
    [(128, 128), (256, 512), (512, 256), (256, 256)],
)
@pytest.mark.parametrize("scale_rule", [ScaleRule.static_6, ScaleRule.mse])
def test_transpose_matches_quantize_with_transpose_flag(
    rows: int,
    cols: int,
    scale_rule: ScaleRule,
) -> None:
    """Nibble-transposing a 2D-block-scaled tensor must match quantizing W.T."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    w = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")

    config = QuantizationConfig(
        backend=QuantizeBackend.pytorch,
        block_scale_2d=True,
        scale_rule=scale_rule,
    )
    config_t = QuantizationConfig(
        backend=QuantizeBackend.pytorch,
        block_scale_2d=True,
        transpose=True,
        scale_rule=scale_rule,
    )

    qt = quantize(w, config)
    qt_t_ref = quantize(w, config_t)

    qt_t = transpose_quantized_tensor(qt)

    assert qt_t.original_shape == qt_t_ref.original_shape, (
        f"shape mismatch: {qt_t.original_shape} vs {qt_t_ref.original_shape}"
    )
    assert qt_t.padded_shape == qt_t_ref.padded_shape, (
        f"padded shape mismatch: {qt_t.padded_shape} vs {qt_t_ref.padded_shape}"
    )

    deq_t = dequantize(qt_t, dtype=torch.float32)
    deq_t_ref = dequantize(qt_t_ref, dtype=torch.float32)

    r, c = qt_t.original_shape
    deq_t = deq_t[:r, :c]
    deq_t_ref = deq_t_ref[:r, :c]

    torch.testing.assert_close(deq_t, deq_t_ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("rows", "cols"),
    [(128, 128), (256, 512)],
)
def test_transpose_roundtrip_lossless(rows: int, cols: int) -> None:
    """Transposing twice must recover the original tensor exactly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(123)
    w = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")

    config = QuantizationConfig(
        backend=QuantizeBackend.pytorch,
        block_scale_2d=True,
    )
    qt = quantize(w, config)
    qt_tt = transpose_quantized_tensor(transpose_quantized_tensor(qt))

    deq_orig = dequantize(qt, dtype=torch.float32)
    deq_tt = dequantize(qt_tt, dtype=torch.float32)

    r, c = qt.original_shape
    torch.testing.assert_close(
        deq_orig[:r, :c],
        deq_tt[:r, :c],
        atol=0,
        rtol=0,
    )


# ---------------------------------------------------------------------------
# TransposableFourOverSixLinear
# ---------------------------------------------------------------------------


def test_transposable_linear_requires_2d_scales() -> None:
    """Must raise if weight_scale_2d is False."""
    from fouroversix.model.config import ModuleQuantizationConfig
    from fouroversix.model.modules.transposable_linear import (
        TransposableFourOverSixLinear,
    )

    module = torch.nn.Linear(128, 256)
    config = ModuleQuantizationConfig(
        weight_scale_2d=False,
        keep_master_weights=True,
    )

    with pytest.raises(ValueError, match="weight_scale_2d=True"):
        TransposableFourOverSixLinear(module, config)


def test_transposable_linear_forward_matches_original() -> None:
    """Forward output should match FourOverSixLinear with same config."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from fouroversix.model.config import ModuleQuantizationConfig
    from fouroversix.model.modules.linear import FourOverSixLinear
    from fouroversix.model.modules.transposable_linear import (
        TransposableFourOverSixLinear,
    )

    torch.manual_seed(0)
    base = torch.nn.Linear(256, 128, bias=True).cuda().to(torch.bfloat16)

    config = ModuleQuantizationConfig(
        weight_scale_2d=True,
        keep_master_weights=True,
        quantize_backend=QuantizeBackend.pytorch,
    )

    orig = FourOverSixLinear(base, config).cuda()
    trans = TransposableFourOverSixLinear(base, config).cuda()

    x = torch.randn(
        1, 128, 256,
        dtype=torch.bfloat16,
        device="cuda",
    )

    out_orig = orig(x)
    out_trans = trans(x)

    torch.testing.assert_close(out_orig, out_trans, atol=0, rtol=0)


def test_transposable_linear_backward_runs() -> None:
    """Backward pass must complete and produce gradients."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from fouroversix.model.config import ModuleQuantizationConfig
    from fouroversix.model.modules.transposable_linear import (
        TransposableFourOverSixLinear,
    )

    torch.manual_seed(0)
    base = torch.nn.Linear(256, 128, bias=True).cuda().to(torch.bfloat16)

    config = ModuleQuantizationConfig(
        weight_scale_2d=True,
        keep_master_weights=True,
        quantize_backend=QuantizeBackend.pytorch,
    )

    layer = TransposableFourOverSixLinear(base, config).cuda()
    x = torch.randn(
        1, 128, 256,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    out = layer(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert x.grad.shape == x.shape
    assert layer.weight.grad.shape == layer.weight.shape


def test_transposable_linear_backward_matches_original() -> None:
    """Backward gradients should closely match FourOverSixLinear."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from fouroversix.model.config import ModuleQuantizationConfig
    from fouroversix.model.modules.linear import FourOverSixLinear
    from fouroversix.model.modules.transposable_linear import (
        TransposableFourOverSixLinear,
    )

    torch.manual_seed(0)
    base = torch.nn.Linear(256, 128, bias=False).cuda().to(torch.bfloat16)

    config = ModuleQuantizationConfig(
        weight_scale_2d=True,
        keep_master_weights=True,
        quantize_backend=QuantizeBackend.pytorch,
    )

    orig = FourOverSixLinear(base, config).cuda()
    trans = TransposableFourOverSixLinear(base, config).cuda()

    x1 = torch.randn(
        1, 128, 256,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    x2 = x1.clone().detach().requires_grad_(True)  # noqa: FBT003

    out1 = orig(x1)
    out1.sum().backward()

    out2 = trans(x2)
    out2.sum().backward()

    torch.testing.assert_close(
        x1.grad,
        x2.grad,
        atol=1e-2,
        rtol=1e-2,
        msg="input gradients diverge",
    )
    torch.testing.assert_close(
        orig.weight.grad,
        trans.weight.grad,
        atol=0,
        rtol=0,
        msg="weight gradients diverge",
    )
