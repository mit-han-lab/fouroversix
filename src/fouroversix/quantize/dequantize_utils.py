import torch


def from_blocked(a: torch.Tensor, orig_shape: tuple[int, int]) -> torch.Tensor:
    rows, cols = orig_shape
    return (
        a.view(-1, 32, 4, 4)
        .transpose(1, 2)
        .reshape(-1, cols // 4, 128, 4)
        .transpose(1, 2)
        .reshape(rows, cols)
    )


def convert_e2m1_to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    sign = (x >> 3) & 0x1
    exponent = (x >> 1) & 0x3
    mantissa = x & 0x1

    # Make adjustments
    new_exponent = torch.where(
        (exponent == 0) & (mantissa == 0),
        0,
        (exponent + 6) & 0xF,
    )
    new_mantissa = torch.where(exponent == 0, 0, mantissa << 2)

    return ((sign << 7) | (new_exponent << 3) | new_mantissa).view(torch.float8_e4m3fn)


def unpack_packed_fp4(
    x: torch.Tensor,
    to_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    if to_dtype == torch.float8_e4m3fn:
        convert_function = convert_e2m1_to_fp8_e4m3
    else:
        msg = f"Unsupported dtype: {to_dtype}"
        raise ValueError(msg)

    high = (x >> 4) & 0xF
    low = x & 0xF

    return torch.stack(
        [convert_function(low), convert_function(high)],
        dim=-1,
    ).reshape(x.shape[0], x.shape[1] * 2)


def unpack_packed_if4(
    x: torch.Tensor,
    scale_factors: torch.Tensor,
    to_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    high = (x >> 4) & 0xF
    low = x & 0xF

    x_unpacked = torch.stack(
        [low, high],
        dim=-1,
    ).reshape(x.shape[0], x.shape[1] * 2 // 16, 16)

    return torch.where(
        (scale_factors.view(torch.uint8) >= 128).unsqueeze(2),  # noqa: PLR2004
        ((x_unpacked.to(torch.int8) << 4) >> 4).to(to_dtype) * (6 / 7),
        convert_e2m1_to_fp8_e4m3(x_unpacked).to(to_dtype),
    ).reshape(x.shape[0], x.shape[1] * 2)


def unpack_packed_int4(
    x: torch.Tensor,
    to_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    high = (x >> 4) & 0xF
    low = x & 0xF

    x_unpacked = torch.stack(
        [low, high],
        dim=-1,
    ).reshape(x.shape[0], x.shape[1] * 2 // 16, 16)

    return (
        ((x_unpacked.to(torch.int8) << 4) >> 4)
        .to(to_dtype)
        .reshape(x.shape[0], x.shape[1] * 2)
    )
