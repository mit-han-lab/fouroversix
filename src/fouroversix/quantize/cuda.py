import torch
from fouroversix.utils import SM_100, SM_110, SM_120, DataType

from .backend import QuantizeBackendBase
from .config import QuantizationConfig
from .quantized_tensor import QuantizedTensor


class CUDAQuantizeBackend(QuantizeBackendBase):
    """
    The CUDA quantization backend. Supports basic quantization options (no 2D block
    scaling, no stochastic rounding, no random Hadamard transform). As a result, it can
    be used for inference, but not training. Requires a Blackwell GPU.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the CUDA backend is available on the current machine."""

        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[
            0
        ] not in [SM_100, SM_110, SM_120]:
            return False

        try:
            import fouroversix._C  # noqa: F401
        except ModuleNotFoundError:
            return False

        return True

    @classmethod
    def is_supported(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        """
        Return True if the CUDA backend supports the given input and quantization
        configuration.
        """

        if not super().is_supported(x, config):
            return False

        return (
            x.device.type == "cuda"
            and config.dtype == DataType.nvfp4
            and not config.transpose
        )

    @classmethod
    def quantize_to_fp4(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        """
        Quantize a tensor to FP4 using the CUDA backend.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            config (QuantizationConfig): The quantization configuration.

        Returns:
            The quantized tensor.

        """

        msg = "The CUDA backend is currently disabled and will be updated soon"
        raise NotImplementedError(msg)
