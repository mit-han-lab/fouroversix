import dataclasses

import torch
from fouroversix.utils import QuantizeBackend

from .config import QuantizationConfig
from .cuda import CUDAQuantizeBackend
from .pytorch import PyTorchQuantizeBackend
from .quantized_tensor import QuantizedTensor
from .transformer_engine import TransformerEngineQuantizeBackend
from .triton import TritonQuantizeBackend
from .utils import get_hadamard_matrix

AVAILABLE_BACKENDS = {
    QuantizeBackend.cuda: CUDAQuantizeBackend,
    QuantizeBackend.transformer_engine: TransformerEngineQuantizeBackend,
    QuantizeBackend.triton: TritonQuantizeBackend,
    QuantizeBackend.pytorch: PyTorchQuantizeBackend,
}

_HADAMARD_DIM = 16


def _pseudo_quantize_with_precondition(
    x: torch.Tensor,
    config: QuantizationConfig,
    selected_backend: QuantizeBackend,
) -> torch.Tensor:
    """
    Kitchen-sink preprocessing for 2D pseudo-quantization.

    ``pseudo_quantize(..., transpose=True)`` expects input W [out, in] and returns
    a fake-quantized tensor in W.T layout [in, out].  We therefore operate on
    W.T directly and call the backend with ``transpose=False``:

    1. W_T = x.T  — work in [in, out]
    2. RMS precondition W_T (row-wise and col-wise)
    3. Apply fixed Hadamard to both dims
    4. Pseudo-quantize with ``transpose=False, block_scale_2d=True``
       → result ≈ W_T_had  [in, out]
    5. Undo Hadamard
    6. Undo RMS preconditioning
    7. Return  [in, out]  (W.T layout — same as what ``transpose=True`` returns)

    The caller then uses ``result.T`` in the matmul, giving ≈ W.
    """
    # Work with W.T because pseudo_quantize(transpose=True) returns W.T layout.
    x_t = x.T.contiguous()
    rows, cols = x_t.shape
    x_f = x_t.float()

    # RMS diagonal preconditioning on W.T
    row_rms = x_f.pow(2).mean(dim=1).sqrt().clamp(min=1e-8)
    col_rms = x_f.pow(2).mean(dim=0).sqrt().clamp(min=1e-8)
    x_precond = x_f / row_rms.unsqueeze(1) / col_rms.unsqueeze(0)

    # Fixed Hadamard rotation on both dims (only when divisible by 16)
    apply_had = (rows % _HADAMARD_DIM == 0) and (cols % _HADAMARD_DIM == 0)
    if apply_had:
        had = get_hadamard_matrix(_HADAMARD_DIM, device=x.device)
        x_precond = (
            had @ x_precond.reshape(rows // _HADAMARD_DIM, _HADAMARD_DIM, cols)
        ).reshape(rows, cols)
        x_precond = (
            x_precond.reshape(rows, cols // _HADAMARD_DIM, _HADAMARD_DIM) @ had.T
        ).reshape(rows, cols)

    # Pseudo-quantize W.T_had without the transpose flag (already transposed).
    inner_config = dataclasses.replace(config, precondition_2d=False, transpose=False)
    result = AVAILABLE_BACKENDS[selected_backend].pseudo_quantize(
        x_precond.to(x.dtype), inner_config
    ).float()

    # Undo Hadamard
    if apply_had:
        result = (
            result.reshape(rows, cols // _HADAMARD_DIM, _HADAMARD_DIM) @ had
        ).reshape(rows, cols)
        result = (
            had.T @ result.reshape(rows // _HADAMARD_DIM, _HADAMARD_DIM, cols)
        ).reshape(rows, cols)

    # Undo RMS preconditioning and return in W.T layout
    result = result * row_rms.unsqueeze(1) * col_rms.unsqueeze(0)
    return result.to(x.dtype)


def quantize(
    x: torch.Tensor,
    config: QuantizationConfig | None = None,
) -> QuantizedTensor | torch.Tensor:
    """
    Quantize a tensor to FP4.

    ## Sample Code

    ### With Four Over Six

    ```python
    x = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    x_quantized = quantize(x)
    ```

    ### Without Four Over Six

    ```python
    x = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(scale_rule="static_6")
    x_quantized = quantize(x, config)
    ```

    ### With Stochastic Rounding

    ```python
    x = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(round_style="stochastic")
    x_quantized = quantize(x, config)
    ```

    ### With the Random Hadamard Transform

    ```python
    from fouroversix.quantize import get_rht_matrix

    x = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(rht=True)
    x_quantized = quantize(x, config)
    ```

    ## Backends

    We provide three different implementations of FP4 quantization:

    - **CUDA**: A fast implementation written in CUDA which currently only supports
        basic quantization options (no 2D block scaling, no stochastic rounding, no
        random Hadamard transform). Can be used for inference, but not training.
        Requires a Blackwell GPU.
    - **Triton**: A slightly slower implementation written in Triton which supports all
        operations needed for training. Also requires a Blackwell GPU.
    - **PyTorch**: A slow implementation written in PyTorch which supports all
        operations and can be run on any GPU.

    If `quantize` is called with `backend=None`, a backend will be selected
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
        config (QuantizationConfig): The quantization configuration to use. If no
            configuration is provided, a default configuration will be used (NVFP4,
            1D block scaling, round-to-nearest, and 4/6 with the MSE selection rule).

    Returns:
        The quantized tensor.

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
            ].can_quantize(x, config):
                selected_backend = backend
                break
        else:
            msg = "No backend found that supports the given parameters"
            raise ValueError(msg)

    elif not AVAILABLE_BACKENDS[selected_backend].can_quantize(x, config):
        msg = f"Backend {selected_backend} does not support the given parameters"
        raise ValueError(msg)

    # Kitchen-sink preprocessing: apply when pseudo-quantizing a transposable 2D weight.
    # Reduces within-tile magnitude heterogeneity → better floating-point approximation.
    if (
        config.pseudo_quantize
        and config.block_scale_2d
        and config.transpose
        and config.precondition_2d
        and x.ndim == 2  # noqa: PLR2004
    ):
        return _pseudo_quantize_with_precondition(x, config, selected_backend)

    if config.pseudo_quantize:
        return AVAILABLE_BACKENDS[selected_backend].pseudo_quantize(x, config)

    return AVAILABLE_BACKENDS[selected_backend].quantize(x, config)


def dequantize(
    tensor: QuantizedTensor,
    dtype: torch.dtype = torch.bfloat16,
    *,
    backend: QuantizeBackend | None = None,
    intermediate_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    selected_backend = backend

    if selected_backend is None:
        for backend_candidate in [
            QuantizeBackend.cuda,
            QuantizeBackend.triton,
            QuantizeBackend.pytorch,
        ]:
            if AVAILABLE_BACKENDS[backend_candidate] is not None and AVAILABLE_BACKENDS[
                backend_candidate
            ].can_dequantize(tensor):
                selected_backend = backend_candidate
                break

    elif not AVAILABLE_BACKENDS[selected_backend].can_dequantize(tensor):
        msg = f"Backend {selected_backend} does not support the given parameters"
        raise ValueError(msg)

    return AVAILABLE_BACKENDS[selected_backend].dequantize(
        tensor,
        dtype,
        intermediate_dtype=intermediate_dtype,
    )
