import pytest
import torch
from fouroversix import AdaptiveBlockScalingRule, FP4Format, FP4Tensor


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("fp4_format", [FP4Format.mxfp4])
@pytest.mark.parametrize("original_shape", [(2880, 2880)])
def test_dims(device: str, fp4_format: FP4Format, original_shape: tuple[int, int]):
    torch.manual_seed(0)

    x_e2m1 = torch.randint(
        0,
        256,
        (original_shape[0], original_shape[1] // 2),
        dtype=torch.uint8,
        device=device,
    )
    x_sf = torch.randint(
        0,
        128,
        (original_shape[0], original_shape[1] // fp4_format.block_size()),
        dtype=torch.uint8,
        device=device,
    ).view(fp4_format.scale_dtype())
    x_amax = torch.full((1,), 1, device=device, dtype=torch.float32)

    x_fp4 = FP4Tensor(
        x_e2m1,
        x_sf,
        x_amax,
        fp4_format,
        original_shape,
        AdaptiveBlockScalingRule.always_6,
    )

    print(x_fp4.dequantize())
