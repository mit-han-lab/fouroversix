import torch
from fouroversix.utils import AdaptiveBlockScalingRule, FP4Format


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def from_blocked(a: torch.Tensor, orig_shape: tuple[int, int]) -> torch.Tensor:
    rows, cols = orig_shape
    return (
        a.view(-1, 32, 4, 4)
        .transpose(1, 2)
        .reshape(-1, ceil_div(cols, 4), 128, 4)
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


def convert_e2m1_to_fp8_e8m0(x: torch.Tensor) -> torch.Tensor:
    e = (x >> 1) & 0x3
    m = x & 0x1

    # There might be a better way to do this but I'm feeling lazy right now
    return torch.where(
        (e == 3) & (m == 1),  # noqa: PLR2004
        torch.tensor(130, dtype=torch.uint8),
        torch.where(
            e == 3,  # noqa: PLR2004
            torch.tensor(129, dtype=torch.uint8),
            torch.where(
                (e == 2) & (m == 1),  # noqa: PLR2004
                torch.tensor(129, dtype=torch.uint8),
                torch.where(
                    e == 2,  # noqa: PLR2004
                    torch.tensor(128, dtype=torch.uint8),
                    torch.where(
                        (e == 1) & (m == 1),
                        torch.tensor(128, dtype=torch.uint8),
                        torch.where(
                            e == 1,
                            torch.tensor(127, dtype=torch.uint8),
                            torch.where(
                                (e == 0) & (m == 1),
                                torch.tensor(126, dtype=torch.uint8),
                                torch.tensor(0, dtype=torch.uint8),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).view(torch.float8_e8m0fnu)


def unpack_packed_fp4(
    x: torch.Tensor,
    to_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    if to_dtype == torch.float8_e4m3fn:
        convert_function = convert_e2m1_to_fp8_e4m3
    elif to_dtype == torch.float8_e8m0fnu:
        convert_function = convert_e2m1_to_fp8_e8m0
    else:
        msg = f"Unsupported dtype: {to_dtype}"
        raise ValueError(msg)

    high = (x >> 4) & 0xF
    low = x & 0xF
    return torch.stack(
        [convert_function(low), convert_function(high)],
        dim=-1,
    ).reshape(x.shape[0], x.shape[1] * 2)


class FP4Tensor:
    """A quantized FP4 tensor."""

    e2m1_values: torch.Tensor
    scale_factors: torch.Tensor
    amax: torch.Tensor

    fp4_format: FP4Format
    original_shape: tuple[int, int]
    scale_rule: AdaptiveBlockScalingRule

    def __init__(
        self,
        e2m1_values: torch.Tensor,
        scale_factors: torch.Tensor,
        amax: torch.Tensor,
        fp4_format: FP4Format,
        original_shape: tuple[int, int],
        scale_rule: AdaptiveBlockScalingRule,
    ) -> None:
        self.e2m1_values = e2m1_values
        self.scale_factors = scale_factors
        self.amax = amax
        self.fp4_format = fp4_format
        self.original_shape = original_shape
        self.scale_rule = scale_rule

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Return a high-precision tensor with the dequantized values."""

        values = unpack_packed_fp4(
            self.e2m1_values,
            to_dtype=self.fp4_format.scale_dtype(),
        ).to(dtype)

        scales = from_blocked(
            self.scale_factors,
            (
                self.e2m1_values.shape[0],
                self.e2m1_values.shape[1] // self.fp4_format.block_size() * 2,
            ),
        )

        result = values * scales.to(
            dtype,
        ).repeat_interleave(self.fp4_format.block_size(), -1)

        if self.fp4_format == FP4Format.mxfp4:
            high = (self.e2m1_values >> 4) & 0xF
            low = self.e2m1_values & 0xF
            values = torch.stack([low, high], dim=-1).reshape(
                self.e2m1_values.shape[0],
                self.e2m1_values.shape[1] * 2,
            )
            x_sign = torch.where(
                ((values >> 3) & 0x1) == 0,
                torch.tensor(1, dtype=dtype),
                torch.tensor(-1, dtype=dtype),
            )
            result = result * x_sign
        elif self.fp4_format == FP4Format.nvfp4:
            if self.amax is not None:
                result = (
                    result.to(torch.float32)
                    * self.amax
                    / self.scale_rule.get_maximum_allowed_quantized_value()
                ).to(dtype)

        return result
