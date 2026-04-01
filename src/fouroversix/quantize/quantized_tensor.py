from dataclasses import dataclass

import torch
import torch.nn.functional as F
from fouroversix.utils import DataType, RoundStyle, ScaleRule

from .utils import to_blocked


@dataclass
class QuantizedTensor:
    """A quantized tensor."""

    values: torch.Tensor
    scale_factors: torch.Tensor
    amax: torch.Tensor

    dtype: DataType
    original_shape: tuple[int, int]
    scale_rule: ScaleRule
    round_style: RoundStyle
    padded_shape: tuple[int, int]

    scale_factors_are_in_blackwell_layout: bool

    def __init__(  # noqa: C901
        self,
        values: torch.Tensor,
        scale_factors: torch.Tensor,
        amax: torch.Tensor,
        dtype: DataType,
        original_shape: tuple[int, int],
        scale_rule: ScaleRule,
        round_style: RoundStyle,
        padded_shape: tuple[int, int] | None = None,
        *,
        scale_factors_are_in_blackwell_layout: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(dtype, str):
            dtype = DataType(dtype)

        if isinstance(original_shape, torch.Size):
            original_shape = tuple(original_shape)

        if isinstance(scale_rule, str):
            scale_rule = ScaleRule(scale_rule)

        if isinstance(round_style, str):
            round_style = RoundStyle(round_style)

        if isinstance(padded_shape, torch.Size):
            padded_shape = tuple(padded_shape)

        self.dtype = dtype
        self.original_shape = original_shape
        self.scale_rule = scale_rule
        self.round_style = round_style
        self.padded_shape = padded_shape
        self.scale_factors_are_in_blackwell_layout = (
            scale_factors_are_in_blackwell_layout
        )

        if self.padded_shape is None:
            rows_div = 128

            # The scale factor layout requires 4 blocks along the K dimension for both
            # MXFP4 and NVFP4. See:
            # https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts
            cols_div = 4 * dtype.block_size

            self.padded_shape = (
                original_shape[0]
                + (rows_div - original_shape[0] % rows_div) % rows_div,
                original_shape[1]
                + (cols_div - original_shape[1] % cols_div) % cols_div,
            )

            packing_factor = dtype.quantized_value_type.packing_factor
            expected_packed_elements = (
                self.padded_shape[0] * self.padded_shape[1] // packing_factor
            )
            expected_scale_factors = (
                expected_packed_elements * packing_factor // dtype.block_size
            )

            if values.numel() != expected_packed_elements:
                values = F.pad(
                    values,
                    (
                        0,
                        self.padded_shape[1] // packing_factor - values.shape[1],
                        0,
                        self.padded_shape[0] - values.shape[0],
                    ),
                )

            # If the scale factors are 1D, we assume that they are already in the
            # correct layout for Blackwell. See:
            # https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts
            if (
                scale_factors.ndim > 1
                and scale_factors.numel() != expected_scale_factors
            ):
                scale_factors = F.pad(
                    scale_factors,
                    (
                        0,
                        (
                            self.padded_shape[1] // dtype.block_size
                            - scale_factors.shape[1]
                        ),
                        0,
                        self.padded_shape[0] - scale_factors.shape[0],
                    ),
                    value=0 if dtype == DataType.nvfp4 else 1,
                )

                if self.scale_factors_are_in_blackwell_layout:
                    scale_factors = to_blocked(scale_factors)

            if values.numel() != expected_packed_elements:
                msg = (
                    f"Expected {expected_packed_elements} e2m1 values, got "
                    f"{values.numel()}"
                )
                raise ValueError(msg)

            if scale_factors.numel() != expected_scale_factors:
                msg = (
                    f"Expected {expected_scale_factors} scale factors, got "
                    f"{scale_factors.numel()}"
                )
                raise ValueError(msg)

        self.values = values
        self.scale_factors = scale_factors
        self.amax = amax

    @property
    def device(self) -> torch.device:
        """Get device of the values in this tensor."""
        return self.values.device
