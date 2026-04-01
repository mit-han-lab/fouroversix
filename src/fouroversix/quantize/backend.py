from abc import ABC, abstractmethod

import torch
from fouroversix.utils import DataType, ScaleType

from .config import QuantizationConfig
from .dequantize_utils import (
    from_blocked,
    unpack_packed_fp4,
    unpack_packed_if4,
    unpack_packed_int4,
)
from .quantized_tensor import QuantizedTensor


class QuantizeBackendBase(ABC):
    """Base class for all quantization backends."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if the backend is available on the current machine."""
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @classmethod
    @abstractmethod
    def can_dequantize(cls, tensor: QuantizedTensor) -> bool:
        """Return True if the backend can dequantize the given quantized tensor."""

        if not cls.is_available():
            return False

        return cls.can_dequantize_values(tensor)

    @classmethod
    @abstractmethod
    def can_dequantize_values(cls, tensor: QuantizedTensor) -> bool:
        """
        Return True if the backend can dequantize the values of the given quantized
        tensor.
        """

        if not cls.is_available():
            return False

        return tensor.dtype not in {
            DataType.mxfp3,
            DataType.mxfp6_e2m3,
            DataType.mxfp6_e3m2,
            DataType.nvfp3,
            DataType.nvfp3_bs8,
            DataType.mxfp3_bs8,
            DataType.nvfp6_e2m3,
            DataType.nvfp6_e3m2,
            DataType.nvint3,
            DataType.nvint3_bs8,
            DataType.nvint6,
            DataType.if3,
            DataType.if3_bs8,
            DataType.if6_e2m3,
            DataType.if6_e3m2,
        }

    @classmethod
    @abstractmethod
    def can_quantize(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        """
        Return True if the backend supports the given input and quantization
        configuration.
        """

        if not cls.is_available():
            return False

        if x.ndim != 2:  # noqa: PLR2004
            return False

        if config.scale_rule not in config.dtype.supported_scale_rules:
            return False

        if config.dtype in {DataType.if6_e2m3, DataType.if6_e3m2}:
            # TODO(jack): Fix later, no idea why tests are failing in these cases
            return not config.block_scale_2d and not config.transpose

        return True

    @classmethod
    def dequantize(
        cls,
        tensor: QuantizedTensor,
        dtype: torch.dtype = torch.bfloat16,
        *,
        intermediate_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Return a high-precision tensor with the dequantized values."""

        values = cls.dequantize_values(tensor, dtype=intermediate_dtype)

        if tensor.scale_factors_are_in_blackwell_layout:
            scales = from_blocked(
                tensor.scale_factors,
                (
                    tensor.padded_shape[0],
                    tensor.padded_shape[1] // tensor.dtype.block_size,
                ),
            )
        else:
            scales = tensor.scale_factors

        if tensor.dtype.scale_type == ScaleType.nv_if:
            scales = torch.where(
                scales.view(torch.uint8) >= 128,  # noqa: PLR2004
                (scales.view(torch.uint8) - 128).view(torch.float8_e4m3fn),
                scales,
            )

        if tensor.dtype in {
            DataType.if3,
            DataType.if3_bs8,
            DataType.if4,
            DataType.if4_bs8,
            DataType.nvint3,
            DataType.nvint3_bs8,
            DataType.nvint4,
            DataType.nvint4_bs8,
            DataType.nvint6,
            DataType.if6_e2m3,
            DataType.if6_e3m2,
        }:
            scales = scales.reshape(
                tensor.padded_shape[0],
                tensor.padded_shape[1] // tensor.dtype.block_size,
            )

        result = values * scales.to(intermediate_dtype).repeat_interleave(
            tensor.dtype.block_size,
            -1,
        )

        if (
            tensor.dtype.scale_type in {ScaleType.nv, ScaleType.nv_if}
            and tensor.amax is not None
        ):
            result = (
                result.to(torch.float32)
                * tensor.amax
                / (
                    tensor.dtype.quantized_value_type.get_maximum_value(
                        tensor.scale_rule,
                    )
                    * tensor.dtype.scale_type.get_maximum_value(tensor.scale_rule)
                    * tensor.round_style.adjustment_factor
                )
            ).to(dtype)

        if result.shape != tensor.original_shape:
            result = result[: tensor.original_shape[0], : tensor.original_shape[1]]

        return result.to(dtype)

    @classmethod
    @abstractmethod
    def dequantize_values(
        cls,
        tensor: QuantizedTensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Get a high-precision representation of the raw values of a quantized tensor.
        Note that this returns raw values only, and not the fully dequantized tensor.
        """

        if tensor.dtype == DataType.if4:
            values = unpack_packed_if4(
                tensor.values,
                tensor.scale_factors.reshape(
                    tensor.padded_shape[0],
                    tensor.padded_shape[1] // tensor.dtype.block_size,
                ),
                dtype,
            )
        elif tensor.dtype == DataType.nvint4:
            values = unpack_packed_int4(tensor.values)
        else:
            values = unpack_packed_fp4(tensor.values)

        return values.to(dtype)

    @classmethod
    @abstractmethod
    def quantize(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        """
        Quantize a tensor to FP4 using the backend.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            config (QuantizationConfig): The quantization configuration.

        Returns:
            The quantized tensor.

        """

        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @classmethod
    def pseudo_quantize(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> torch.Tensor:
        """
        Perform a fused quantize-dequantize round-trip, returning a high-precision
        tensor without allocating intermediate quantized tensors.

        The default implementation falls back to quantize followed by dequantize.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            config (QuantizationConfig): The quantization configuration.

        Returns:
            A tensor of the same dtype as the input with fake-quantized values.

        """

        return cls.dequantize(cls.quantize(x, config), dtype=x.dtype)
