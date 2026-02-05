from abc import ABC, abstractmethod

import torch
from fouroversix.quantize import QuantizedTensor


class MatmulBackendBase(ABC):
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool: ...

    @classmethod
    @abstractmethod
    def is_supported(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: torch.dtype,
    ) -> bool:
        if not cls.is_available():
            return False

        if input.dtype != other.dtype:
            msg = "Both inputs must have the same dtype"
            raise ValueError(msg)

        if input.original_shape[1] != other.original_shape[1]:
            msg = (
                "The first input must be in row-major layout, the second input must be"
                "in column-major layout, and both inputs must have the same inner "
                "dimension"
            )
            raise ValueError(msg)

        return True

    @classmethod
    @abstractmethod
    def fp4_matmul(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: torch.dtype,
    ) -> torch.Tensor: ...
