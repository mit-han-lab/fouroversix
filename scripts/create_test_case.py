from .resources import app, get_image

img = get_image()

with img.imports():
    import torch
    from fouroversix import AdaptiveBlockScalingRule, QuantizeBackend, quantize_to_fp4
    from fouroversix.quantize.reference import from_blocked


@app.function(image=img, gpu="B200")
def create_test_case(
    backend_a: str = "transformer_engine",
    backend_b: str = "triton",
    scale_rule: str = "always_6",
) -> None:
    torch.random.manual_seed(4)
    torch.cuda.manual_seed(4)
    torch.set_printoptions(precision=10)

    backend_a = QuantizeBackend(backend_a)
    backend_b = QuantizeBackend(backend_b)
    scale_rule = AdaptiveBlockScalingRule(scale_rule)

    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    x_e2m1_a, x_sf_a, x_normconst_a = quantize_to_fp4(
        x,
        backend=backend_a,
        scale_rule=scale_rule,
    )
    x_e2m1_b, x_sf_b, x_normconst_b = quantize_to_fp4(
        x,
        backend=backend_b,
        scale_rule=scale_rule,
    )
    x_sf_a = from_blocked(x_sf_a.bfloat16(), (1024, 64))
    x_sf_b = from_blocked(x_sf_b.bfloat16(), (1024, 64))

    print(f"x absmax: {x.abs().max()}")

    if not torch.allclose(x_normconst_a, x_normconst_b):
        print("Backends A and B have different norm constants!")
        print(f"{backend_a}: {x_normconst_a}")
        print(f"{backend_b}: {x_normconst_b}")
        return

    if not torch.allclose(x_sf_a.bfloat16(), x_sf_b.bfloat16()):
        mismatch_prop = (x_sf_a != x_sf_b).sum() / x_sf_a.numel()
        print(
            "Backends A and B have different scale factors! "
            f"{mismatch_prop:.2%} mismatch",
        )

        [i, *_], [j, *_] = torch.where(x_sf_a != x_sf_b)
        print(backend_a)
        print("sf", x_sf_a[i, j])
        print("e2m1", x_e2m1_a[i, 8 * j : 8 * (j + 1)])
        print(backend_b)
        print("sf", x_sf_b[i, j])
        print("e2m1", x_e2m1_b[i, 8 * j : 8 * (j + 1)])
        print("original")
        print("x", x[i, 16 * j : 16 * (j + 1)])
        return

    if not torch.allclose(x_e2m1_a, x_e2m1_b):
        mismatch_prop = (x_e2m1_a != x_e2m1_b).sum() / x_e2m1_a.numel()
        print(
            "Backends A and B have different e2m1 values! "
            f"{mismatch_prop:.2%} mismatch",
        )

        [i, *_], [j, *_] = torch.where(x_e2m1_a != x_e2m1_b)
        print(i, j)
        print("normconst", x_normconst_a)
        print("sf", x_sf_a[i, j // 8])
        print(backend_a)
        print("e2m1", x_e2m1_a[i, 8 * (j // 8) : 8 * (j // 8 + 1)])
        print(backend_b)
        print("e2m1", x_e2m1_b[i, 8 * (j // 8) : 8 * (j // 8 + 1)])
        print("original")
        print("x", x[i, 16 * (j // 8) : 16 * (j // 8 + 1)])
        return
