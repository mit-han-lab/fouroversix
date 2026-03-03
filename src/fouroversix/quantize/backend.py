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

        return tensor.dtype not in {DataType.nvfp6_e2m3, DataType.nvfp6_e3m2}

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

        return config.scale_rule in config.dtype.supported_scale_rules

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

        if tensor.dtype == DataType.if4:
            scales = torch.where(
                scales.view(torch.uint8) >= 128,  # noqa: PLR2004
                (scales.view(torch.uint8) - 128).view(
                    torch.float8_e4m3fn,
                ),
                scales,
            )

        if tensor.dtype in {DataType.if4, DataType.nvint4}:
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

        if tensor.values_are_packed:
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
        else:
            values = tensor.values

        return values.to(dtype)

    @classmethod
    @abstractmethod
    def quantize_to_fp4(
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
