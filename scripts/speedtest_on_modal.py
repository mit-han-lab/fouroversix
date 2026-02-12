from __future__ import annotations

from .resources import Dependency, app, get_image

img = get_image(dependencies=[Dependency.transformer_engine, Dependency.fouroversix])

with img.imports():
    import torch
    import torch.utils.benchmark as benchmark
    from fouroversix import QuantizationConfig, QuantizeBackend, RoundStyle, ScaleRule
    from fouroversix.quantize.frontend import AVAILABLE_BACKENDS


@app.function(image=img, cpu=4, memory=8 * 1024, gpu="B200")
def run_tests(
    *,
    block_scale_2d: bool = False,
    input_shape: str = "1024,1024",
    repeats: int = 100,
    rht: bool = False,
    round_style: str = "nearest",
    scale_rule: str = "mse",
    transpose: bool = False,
) -> None:
    """Test speed on a B200 on Modal."""

    input_shape = tuple(int(dim.strip()) for dim in input_shape.split(","))
    x = torch.randn(input_shape, dtype=torch.bfloat16, device="cuda")

    print("Testing with config:")
    print(f"- block_scale_2d: {block_scale_2d}")
    print(f"- input_shape: {input_shape}")
    print(f"- rht: {rht}")
    print(f"- round_style: {round_style}")
    print(f"- scale_rule: {scale_rule}")
    print(f"- transpose: {transpose}")
    print()

    for backend in [
        QuantizeBackend.cuda,
        QuantizeBackend.transformer_engine,
        QuantizeBackend.triton,
        QuantizeBackend.pytorch,
    ]:
        config = QuantizationConfig(
            backend=backend,
            block_scale_2d=block_scale_2d,
            rht=rht,
            round_style=RoundStyle(round_style),
            scale_rule=ScaleRule(scale_rule),
            transpose=transpose,
        )

        backend_cls = AVAILABLE_BACKENDS[backend]
        print(f"{backend.value}: ", end="")

        if not backend_cls.is_available():
            print("Not available")
            continue

        if not backend_cls.is_supported(x, config):
            print("Not supported")
            continue

        config = QuantizationConfig(
            backend=backend,
            rht=rht,
            round_style=RoundStyle(round_style),
            scale_rule=ScaleRule(scale_rule),
        )

        t = benchmark.Timer(
            setup="from fouroversix import quantize_to_fp4",
            stmt="quantize_to_fp4(x, config)",
            globals={"x": x, "config": config},
        )

        print(f"{t.timeit(repeats).mean * 1000:.4f}ms")
