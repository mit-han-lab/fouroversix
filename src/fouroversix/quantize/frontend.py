import torch
from fouroversix.utils import QuantizeBackend

from .config import QuantizationConfig
from .cuda import CUDAQuantizeBackend
from .pytorch import PyTorchQuantizeBackend
from .quantized_tensor import QuantizedTensor
from .transformer_engine import TransformerEngineQuantizeBackend
from .triton import TritonQuantizeBackend

AVAILABLE_BACKENDS = {
    QuantizeBackend.cuda: CUDAQuantizeBackend,
    QuantizeBackend.transformer_engine: TransformerEngineQuantizeBackend,
    QuantizeBackend.triton: TritonQuantizeBackend,
    QuantizeBackend.pytorch: PyTorchQuantizeBackend,
}


def quantize_to_fp4(
    x: torch.Tensor,
    config: QuantizationConfig | None = None,
) -> QuantizedTensor:
    """
    Quantize a tensor to FP4.

    ## Sample Code

    ### With Four Over Six

    ```
    x = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    x_quantized = quantize_to_fp4(x)
    ```

    ### Without Four Over Six

    ```
    x = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    x_quantized = quantize_to_fp4(x, scale_rule=ScaleRule.static_6)
    ```

    ### With Stochastic Rounding

    ```
    x = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    x_quantized = quantize_to_fp4(x, round_style=RoundStyle.stochastic)
    ```

    ### With the Random Hadamard Transform

    ```
    from fouroversix.quantize import get_rht_matrix

    x = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    had = get_rht_matrix()
    x_quantized = quantize_to_fp4(x, had=had)
    ```

    ## Backends

    We provide three different implementations of FP4 quantization:

    - **CUDA**: A fast implementation written in CUDA which currently does not support
        the operations required for training (2D block scaling, stochastic rounding,
        random Hadamard transform). Requires a Blackwell GPU.
    - **Triton**: A slightly slower implementation written in Triton which supports all
        operations needed for training. Requires a Blackwell GPU.
    - **PyTorch**: A slow implementation written in PyTorch which supports all
        operations and can be run on any GPU.

    If `quantize_to_fp4` is called with `backend=None`, a backend will be selected
    automatically based on the following rules:

    - If there is no GPU available, or if the available GPU is not a Blackwell GPU,
        select PyTorch.
    - If any quantization options are set other than `scale_rule`, select Triton.
        - However, if the available GPU is SM120 (i.e. RTX 5090, RTX 6000) and
            `round_style` is set to `RoundStyle.stochastic`, select PyTorch as
            stochastic rounding does not have hardware support on SM120 GPUs.
    - Otherwise, select CUDA.

    ## Parameters

    Args:
        x (torch.Tensor): The input tensor to quantize.
        backend (QuantizeBackend): The backend to use for quantization, either
            `QuantizeBackend.cuda`, `QuantizeBackend.triton`, or
            `QuantizeBackend.pytorch`. If no backend is provided, one will be selected
            automatically based on the available GPU and the options provided. See above
            for more details.
        scale_rule (AdaptiveBlockScalingRule): The scaling rule to use during
            quantization. See (Adaptive Block Scaling)[/adaptive_block_scaling] for more
            details.
        block_scale_2d (bool): If True, scale factors will be computed across 16x16
            chunks of the input rather than 1x16 chunks. This is useful to apply to the
            weight matrix during training, so that W and W.T will be equivalent after
            quantization.
        had (torch.Tensor): A high-precision Hadamard matrix to apply to the input prior
            to quantization.
        fp4_format (FP4Format): The FP4 format to quantize to, either `FP4Format.mxfp4`
            or `FP4Format.nvfp4`.
        round_style (RoundStyle): The rounding style to apply during quantization,
            either `RoundStyle.nearest` for round-to-nearest quantization, or
            `RoundStyle.stochastic` for stochastic rounding.
        transpose (bool): If True, the output will be a quantized version of the
            transposed input. This may be helpful for certain operations during training
            as `fp4_matmul` requires that both tensors are provided in row-major format.

    Returns:
        A quantized FP4Tensor, which contains the packed E2M1 values, the FP8 scale
        factors, and the tensor-wide FP32 scale factor.

    """

    if config is None:
        config = QuantizationConfig()

    selected_backend = config.backend

    if selected_backend is None:
        for backend in [
            QuantizeBackend.cuda,
            QuantizeBackend.triton,
            QuantizeBackend.pytorch,
        ]:
            if AVAILABLE_BACKENDS[backend] is not None and AVAILABLE_BACKENDS[
                backend
            ].is_supported(x, config):
                selected_backend = backend
                break
        else:
            msg = "No backend found that supports the given parameters"
            raise ValueError(msg)

    elif not AVAILABLE_BACKENDS[selected_backend].is_supported(x, config):
        msg = f"Backend {selected_backend} does not support the given parameters"
        raise ValueError(msg)

    return AVAILABLE_BACKENDS[selected_backend].quantize_to_fp4(x, config)
