import torch
from fouroversix.matmul.backend import MatmulBackendBase
from fouroversix.quantize import QuantizedTensor
from fouroversix.utils import DataType


class TritonMatmulBackend(MatmulBackendBase):
    """
    The Triton matrix multiplication backend. Uses Triton kernels to perform fast
    FP4 matrix multiplication. Requires a Blackwell GPU.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the Triton backend is available on the current machine."""
        # TODO(jack): Investigate slow speeds and re-enable later
        return False

    @classmethod
    def is_supported(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
    ) -> bool:
        """
        Return True if the Triton backend supports the given inputs and output data
        type.
        """

        if not super().is_supported(input, other, out_dtype=out_dtype):
            return False

        return (
            input.dtype == other.dtype
            and input.dtype
            in {
                DataType.nvfp4,
                DataType.nvfp6_e2m3,
                DataType.nvfp6_e3m2,
                DataType.nvint4,
                DataType.if4,
                DataType.if6_e2m3,
                DataType.if6_e3m2,
            }
            and input.device.type == "cuda"
        )

    @classmethod
    def quantized_matmul(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
        major_compute_capability: int | None = None,
    ) -> torch.Tensor:
        """
        Perform a matrix multiplication (`a @ b.T`) between two quantized tensors using
        the Triton backend.
        """

        from fouroversix.kernels.triton import matmul

        return matmul(
            input,
            other,
            out_dtype=out_dtype,
            major_compute_capability=major_compute_capability,
        )
