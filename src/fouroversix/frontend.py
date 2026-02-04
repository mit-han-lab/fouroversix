from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .backend import MatmulBackend, QuantizeBackend
from .utils import AdaptiveBlockScalingRule, FP4Format, RoundStyle

if TYPE_CHECKING:
    from .quantize import FP4Tensor


def fp4_matmul(
    input: torch.Tensor | FP4Tensor,
    other: torch.Tensor | FP4Tensor,
    *,
    backend: MatmulBackend | None = None,
    input_quantize_kwargs: dict[str, Any] | None = None,
    other_quantize_kwargs: dict[str, Any] | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Perform a matrix multiplication (`a @ b.T`) with two FP4-quantized tensors provided
    in row-major layout.

    ## Sample Code

    Each tensor may be provided in either high or low precision. If provided in high
    precision, tensors will be quantized to FP4 prior to the matrix multiplication, and
    quantization may be configured with the `input_quantize_kwargs` and
    `other_quantize_kwargs` parameters. For example, the following two code samples are
    equivalent:

    ### With High-Precision Inputs

    ```
    a = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    b = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    out = fp4_matmul(a, b)
    ```

    ### With Low-Precision Inputs

    ```
    a = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    b = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")

    a_quantized = quantize_to_fp4(a)
    b_quantized = quantize_to_fp4(b)

    out = fp4_matmul(a_quantized, b_quantized)
    ```

    ## Backends

    We provide two different implementations of FP4 matrix multiplication:

    - **CUTLASS**: Uses CUTLASS kernels to perform fast FP4 matrix multiplication.
        Requires a Blackwell GPU.
    - **PyTorch**: A slow implementation which dequantizes FP4 tensors, and then
        performs a high-precision matrix multiplication.

    Note that our CUTLASS kernels accumulate in FP32, so it should be roughly
    equivalent to simulations done with the PyTorch backend.

    ## Parameters

    Args:
        input (torch.Tensor | FP4Tensor): The first tensor to be multiplied.
        other (torch.Tensor | FP4Tensor): The second tensor to be multiplied.
        backend (MatmulBackend): The backend to use for the matrix multiplication,
            either `MatmulBackend.cutlass` or `MatmulBackend.pytorch`. If no backend is
            provided, CUTLASS will be used if the machine has a Blackwell GPU, and
            PyTorch will be used otherwise.
        input_quantize_kwargs (dict): If `a` is provided in high precision, these
            parameters will be passed to the `quantize_to_fp4` call done prior to the
            matrix multiplication.
        other_quantize_kwargs (dict): If `other` is provided in high precision, these
            parameters will be passed to the `quantize_to_fp4` call done prior to the
            matrix multiplication.
        out_dtype (DataType): The data type of the output tensor, either
            `DataType.bfloat16` or `DataType.float16`.

    Returns:
        The output tensor.

    """

    if input_quantize_kwargs is None:
        input_quantize_kwargs = {}

    if other_quantize_kwargs is None:
        other_quantize_kwargs = {}

    if isinstance(input, torch.Tensor):
        input = quantize_to_fp4(input, **(input_quantize_kwargs or {}))

    if isinstance(other, torch.Tensor):
        other = quantize_to_fp4(other, **(other_quantize_kwargs or {}))

    if backend is None:
        backend = MatmulBackend.auto_select(input, other, out_dtype=out_dtype)
    elif not backend.is_available():
        msg = f"Backend {backend} is not available"
        raise ValueError(msg)

    return backend.fp4_matmul(input, other, out_dtype=out_dtype)


def quantize_to_fp4(
    x: torch.Tensor,
    *,
    backend: QuantizeBackend | None = None,
    scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    block_scale_2d: bool = False,
    had: torch.Tensor | None = None,
    fp4_format: FP4Format = FP4Format.nvfp4,
    round_style: RoundStyle = RoundStyle.nearest,
    transpose: bool = False,
) -> FP4Tensor:
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
    x_quantized = quantize_to_fp4(x, scale_rule=AdaptiveBlockScalingRule.always_6)
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

    kwargs = {
        "scale_rule": scale_rule,
        "block_scale_2d": block_scale_2d,
        "had": had,
        "fp4_format": fp4_format,
        "round_style": round_style,
        "transpose": transpose,
    }

    if backend is None:
        backend = QuantizeBackend.auto_select(x, **kwargs)
    elif not backend.is_supported(x, **kwargs):
        msg = f"Backend {backend} does not support the given parameters"
        raise ValueError(msg)

    return backend.quantize_to_fp4(x, **kwargs)
