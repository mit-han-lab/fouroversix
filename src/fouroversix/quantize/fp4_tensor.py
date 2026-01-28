import torch
import torch.nn.functional as F
from fouroversix.utils import AdaptiveBlockScalingRule, FP4Format

from .reference import to_blocked


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


class FP4Tensor:
    """A quantized FP4 tensor."""

    e2m1_values: torch.Tensor
    scale_factors: torch.Tensor
    amax: torch.Tensor

    fp4_format: FP4Format
    original_shape: tuple[int, int]
    scale_rule: AdaptiveBlockScalingRule

    padded_shape: tuple[int, int]

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

        rows_div = 128
        # The scale factor layout requires 4 blocks along the K dimension for both
        # MXFP4 and NVFP4. See:
        # https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts
        cols_div = 4 * fp4_format.block_size()

        self.padded_shape = (
            original_shape[0] + (rows_div - original_shape[0] % rows_div) % rows_div,
            original_shape[1] + (cols_div - original_shape[1] % cols_div) % cols_div,
        )

        expected_packed_elements = self.padded_shape[0] * self.padded_shape[1] // 2
        expected_scale_factors = expected_packed_elements * 2 // fp4_format.block_size()

        if self.e2m1_values.numel() != expected_packed_elements:
            self.e2m1_values = F.pad(
                self.e2m1_values,
                (
                    0,
                    # Divide by 2 because these are packed values
                    self.padded_shape[1] // 2 - self.e2m1_values.shape[1],
                    0,
                    self.padded_shape[0] - self.e2m1_values.shape[0],
                ),
            )

        # If the scale factors are 1D, we assume that they are already in the
        # correct layout for Blackwell. See:
        # https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts
        if (
            self.scale_factors.ndim > 1
            and self.scale_factors.numel() != expected_scale_factors
        ):
            self.scale_factors = F.pad(
                self.scale_factors,
                (
                    0,
                    (
                        self.padded_shape[1] // fp4_format.block_size()
                        - self.scale_factors.shape[1]
                    ),
                    0,
                    self.padded_shape[0] - self.scale_factors.shape[0],
                ),
                value=0 if fp4_format == FP4Format.nvfp4 else 1,
            )

            self.scale_factors = to_blocked(self.scale_factors)

        if self.e2m1_values.numel() != expected_packed_elements:
            msg = (
                f"Expected {expected_packed_elements} e2m1 values, got "
                f"{self.e2m1_values.numel()}"
            )
            raise ValueError(msg)

        if self.scale_factors.numel() != expected_scale_factors:
            msg = (
                f"Expected {expected_scale_factors} scale factors, got "
                f"{self.scale_factors.numel()}"
            )
            raise ValueError(msg)

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Return a high-precision tensor with the dequantized values."""

        values = unpack_packed_fp4(self.e2m1_values).to(dtype)
        scales = from_blocked(
            self.scale_factors,
            (
                self.padded_shape[0],
                self.padded_shape[1] // self.fp4_format.block_size(),
            ),
        )

        result = values * scales.to(dtype).repeat_interleave(
            self.fp4_format.block_size(),
            -1,
        )

        if self.fp4_format == FP4Format.nvfp4 and self.amax is not None:
            result = (
                result.to(torch.float32)
                * self.amax
                / (
                    self.scale_rule.max_allowed_e2m1_value()
                    * self.scale_rule.max_allowed_e4m3_value()
                )
            ).to(dtype)

        if result.shape != self.original_shape:
            result = result[: self.original_shape[0], : self.original_shape[1]]

        return result

    @property
    def device(self) -> torch.device:
        """Get device of the values in this tensor."""
        return self.e2m1_values.device
