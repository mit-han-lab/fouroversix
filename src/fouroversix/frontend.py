from __future__ import annotations

from typing import Any

import torch

from .backend import MatmulBackend, QuantizeBackend
from .utils import AdaptiveBlockScalingRule, FP4Format, RoundStyle


def fp4_matmul(
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    *,
    backend: MatmulBackend | None = None,
    a_e2m1: torch.Tensor | None = None,
    a_sf: torch.Tensor | None = None,
    a_amax: torch.Tensor | None = None,
    b_e2m1: torch.Tensor | None = None,
    b_sf: torch.Tensor | None = None,
    b_amax: torch.Tensor | None = None,
    a_quantize_kwargs: dict[str, Any] | None = None,
    b_quantize_kwargs: dict[str, Any] | None = None,
    fp4_format: FP4Format = FP4Format.nvfp4,
    scale_rule: AdaptiveBlockScalingRule = AdaptiveBlockScalingRule.mse,
    out_dtype: torch.dtype = torch.bfloat16,
    out_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Perform a matrix multiplication (`a @ b.T`) with two FP4-quantized tensors provided
    in row-major layout.

    ## Sample Code

    Each tensor may be provided in either high or low precision. If provided in high
    precision, tensors will be quantized to FP4 prior to the matrix multiplication, and
    quantization may be configured with the `a_quantization_kwargs` and
    `b_quantization_kwargs` parameters. For example, the following two code samples are
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

    a_e2m1, a_sf, a_normconst = quantize_to_fp4(a)
    b_e2m1, b_sf, b_normconst = quantize_to_fp4(b)

    out = fp4_matmul(
        a_e2m1=a_e2m1,
        a_sf=a_sf,
        a_normconst=a_normconst,
        b_e2m1=b_e2m1,
        b_sf=b_sf,
        b_normconst=b_normconst
    )
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
        a (torch.Tensor): The high-precision input tensor A.
        b (torch.Tensor): The high-precision input tensor B.
        backend (MatmulBackend): The backend to use for the matrix multiplication,
            either `MatmulBackend.cutlass` or `MatmulBackend.pytorch`. If no backend is
            provided, CUTLASS will be used if the machine has a Blackwell GPU, and
            PyTorch will be used otherwise.
        a_e2m1 (torch.Tensor): The values of the first input tensor in packed E2M1
            format (2 values per byte).
        a_sf (torch.Tensor): The scale factors of the first input tensor.
        a_amax (torch.Tensor): The absolute maximum value of the first input tensor.
        b_e2m1 (torch.Tensor): The values of the second input tensor in packed E2M1
            format (2 values per byte).
        b_sf (torch.Tensor): The scale factors of the second input tensor.
        b_amax (torch.Tensor): The absolute maximum value of the second input tensor.
        a_quantize_kwargs (dict): If `a` is provided in high precision, these parameters
            will be passed to the `quantize_to_fp4` call done prior to the matrix
            multiplication.
        b_quantize_kwargs (dict): If `b` is provided in high precision, these parameters
            will be passed to the `quantize_to_fp4` call done prior to the matrix
            multiplication.
        fp4_format (FP4Format): The FP4 format of the input tensors, either
            `FP4Format.nvfp4` or `FP4Format.mxfp4`.
        scale_rule (AdaptiveBlockScalingRule): The scaling rule that was used during
            quantization of the input tensors.
        out_dtype (DataType): The data type of the output tensor, either
            `DataType.bfloat16` or `DataType.float16`.
        out_shape (tuple[int, int] | None): The shape of the output tensor. This is
            helpful when the input tensors have shapes that are not multiples of 64,
            but which were padded to multiples of 64 during quantization.

    Returns:
        The output tensor.

    """

    if a is None and (a_e2m1 is None or a_sf is None):
        msg = "If a is None, a_e2m1 and a_sf must be provided"
        raise ValueError(msg)

    if b is None and (b_e2m1 is None or b_sf is None):
        msg = "If b is None, b_e2m1 and b_sf must be provided"
        raise ValueError(msg)

    if a_quantize_kwargs is None:
        a_quantize_kwargs = {}

    if b_quantize_kwargs is None:
        b_quantize_kwargs = {}

    if a_e2m1 is None or a_sf is None:
        a_e2m1, a_sf, a_amax = quantize_to_fp4(a, **a_quantize_kwargs)

    if b_e2m1 is None or b_sf is None:
        b_e2m1, b_sf, b_amax = quantize_to_fp4(b, **b_quantize_kwargs)

    kwargs = {
        "fp4_format": fp4_format,
        "scale_rule": scale_rule,
        "out_dtype": out_dtype,
        "out_shape": out_shape,
    }

    if backend is None:
        backend = MatmulBackend.auto_select()
    elif not backend.is_available():
        msg = f"Backend {backend} is not available"
        raise ValueError(msg)

    return backend.fp4_matmul(
        a_e2m1,
        a_sf,
        a_amax,
        b_e2m1,
        b_sf,
        b_amax,
        **kwargs,
    )


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Quantize a tensor to FP4.

    ## Sample Code

    ### With Four Over Six

    ```
    a = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    a_e2m1, a_sf, a_normconst = quantize_to_fp4(a)
    ```

    ### Without Four Over Six

    ```
    a = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    a_e2m1, a_sf, a_normconst = quantize_to_fp4(
        a,
        scale_rule=AdaptiveBlockScalingRule.always_6,
    )
    ```

    ### With Stochastic Rounding

    ```
    a = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    a_e2m1, a_sf, a_normconst = quantize_to_fp4(a, round_style=RoundStyle.stochastic)
    ```

    ### With the Random Hadamard Transform

    ```
    from fouroversix.quantize import get_rht_matrix

    a = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    had = get_rht_matrix()
    a_e2m1, a_sf, a_normconst = quantize_to_fp4(a, had=had)
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
        The packed E2M1 values.
        The FP8 scale factors.
        The tensor-wide FP32 scale factor.

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
