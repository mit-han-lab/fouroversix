import torch
from fouroversix.quantize import QuantizedTensor

from .backend import MatmulBackendBase


class PyTorchMatmulBackend(MatmulBackendBase):
    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def fp4_matmul(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        out_shape = (input.original_shape[0], other.original_shape[0])

        out = torch.matmul(
            input.dequantize(dtype=torch.float32),
            other.dequantize(dtype=torch.float32).T,
        ).to(out_dtype)

        if out.shape != out_shape:
            out = out[: out_shape[0], : out_shape[1]]

        return out
