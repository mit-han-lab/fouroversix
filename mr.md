# Add `TransposableFourOverSixLinear` with Triton/CUDA kernel support

## Summary

This PR adds a new `TransposableFourOverSixLinear` module that eliminates the need to re-quantize the weight matrix in the backward pass when computing dgrad (`dX = dY @ W`).

The current `FourOverSixLinear` calls `quantize(weight, transpose=True)` on every backward iteration, which runs the full quantization pipeline (scale computation, fake-quantize, pack, blocked layout conversion) on the transposed weight. When `block_scale_2d=True`, this re-quantization produces FP4 codes that are **identical** to simply rearranging the nibbles of the already-quantized weight — because 16x16 tile scales are invariant to transposition.

`TransposableFourOverSixLinear` exploits this property: it quantizes the weight once in the forward pass, then obtains the transposed `QuantizedTensor` via a cheap nibble shuffle + scale grid transpose in the backward pass. No FP4 codes are recomputed. The operation is mathematically lossless.

The nibble transpose is accelerated with **Triton and CUDA kernels**, with automatic backend selection (Triton > CUDA > PyTorch fallback).

## Motivation

We are [GlamLabs](https://github.com/glamlabs) — we train LoRAs for image generation models on Blackwell GPUs with NVFP4 quantization. For our workload we use the straightforward `block_scale_2d=True` path without random Hadamard transforms or Quartet II — just plain 2D-block-scaled FP4. This is the simplest and most practical configuration for LoRA finetuning, where you want fast training without complex quantization overhead.

In this setup, we noticed the backward pass re-quantizes W on every iteration just to get the transposed packed layout for dgrad. This is entirely redundant when 2D block scaling is used, because:

1. Data is grouped into 16×16 tiles, each sharing a single FP8 scale factor
2. The scale is the same whether the tile is read row-major or column-major
3. Therefore `quantize(W, block_scale_2d=True)` transposed == `quantize(W, block_scale_2d=True, transpose=True)`

The nibble transpose is O(n) with no floating-point arithmetic — just integer bit manipulation and a memory transpose. The full quantize path involves floating-point scale computation, fake-quantization with rounding, BF16-to-FP4 encoding, and blocked layout conversion.

This work grew out of [NVFP4-transpose](https://github.com/awarebayes/NVFP4-transpose), a standalone library where we explored and benchmarked different strategies for runtime transposition of NVFP4 matrices (approximate nibble shuffle, exact dequant-requant, 2D tile scaling, and joint rounding). The 2D tile scaling approach proved to be the clear winner — lossless transpose with zero additional error — and integrates naturally into fouroversix's existing `block_scale_2d` support.

## What changed

### New files

**`src/fouroversix/quantize/transpose.py`** — Core transpose utility with backend dispatch

- `transpose_quantized_tensor(qt, backend=TransposeBackend.auto_select)`:
  - Auto-selects the fastest available backend: Triton > CUDA > PyTorch
  - Unpacks nibbles, transposes the 2D code matrix, re-packs into byte layout
  - Transposes the 2D block-scale tile grid (always PyTorch — the grid is small)
  - Returns a new `QuantizedTensor` with swapped shapes and correctly laid-out scale factors
- `TransposeBackend` enum: `auto_select`, `triton`, `cuda`, `pytorch`

**`src/fouroversix/kernels/triton/transpose.py`** — Triton `@triton.jit` kernel

- Fused unpack-transpose-repack in a single kernel pass
- Algebraic decomposition: splits source rows into even/odd, extracts low/high nibbles, transposes four `(BLOCK_M//2 × BLOCK_N//2)` sub-matrices via `tl.trans`, and repacks directly
- No intermediate full-size tensor allocated
- Tile size 16×16 matches the 2D block-scale boundary

**`src/fouroversix/kernels/triton/ops_transpose.py`** — Triton host launcher

- `transpose_packed_fp4(values, rows, cols)` — allocates output, configures grid, launches kernel

**`src/fouroversix/csrc/transpose_fp4.cu`** — CUDA kernel + torch binding

- Register-only transpose: each thread loads from source position `(k, j)` and writes to destination position `(j, k)` — the transpose is implicit in this coordinate swap
- No shared memory needed; each thread has exactly the four nibble codes it needs
- 64 threads per block (8×8), one block per 16×16 tile
- Registered via `TORCH_LIBRARY_IMPL(fouroversix, CUDA, m)`

**`src/fouroversix/model/modules/transposable_linear.py`** — New linear module

- `TransposableFourOverSixLinearFunction` (autograd Function):
  - **Forward**: identical to `FourOverSixLinearFunction` — quantize weight, `quantized_matmul(X, W_q)`
  - **Backward (dgrad)**: quantize weight once (row-major), then `transpose_quantized_tensor()` for W^T — no `transpose=True` in the quantize config
  - **Backward (wgrad)**: identical to the original (operates on activations/gradients, not weight transpose)
  - Supports `disable_dgrad_quantization` modes for compatibility

- `TransposableFourOverSixLinear(nn.Linear)`:
  - Enforces `weight_scale_2d=True` at construction (raises `ValueError` otherwise)
  - Same buffer registration and `get_quantized_parameters` API as `FourOverSixLinear` — drop-in replacement when 2D scaling is already enabled
  - Supports both `keep_master_weights=True` (training) and `False` (inference/PTQ)

**`tests/test_transposable_linear.py`** — 27 tests

**`tests/test_cuda_transpose_standalone.py`** — Standalone CUDA kernel test via JIT compilation

### Modified files

- `src/fouroversix/csrc/bindings.cpp` — Added `transpose_packed_fp4` op schema
- `src/fouroversix/kernels/triton/__init__.py` — Exports `transpose_packed_fp4`
- `src/fouroversix/quantize/__init__.py` — Exports `transpose_quantized_tensor`, `TransposeBackend`
- `src/fouroversix/model/modules/__init__.py` — Exports `TransposableFourOverSixLinear`
- `src/fouroversix/model/__init__.py` — Exports `TransposableFourOverSixLinear`

## How it works

### Current backward (dgrad)

```python
# FourOverSixLinearFunction.backward
dgrad_weight_config = ctx.config.get_weight_config(transpose=True)
#   -> QuantizationConfig(block_scale_2d=True, transpose=True)
#   -> quantize() runs: x.T, reshape into 16x16 tiles, compute scales,
#      fake-quantize, encode BF16->FP4, pack nibbles, to_blocked(scales)
grad_input = quantized_matmul(dY, W, other_config=dgrad_weight_config)
```

### New backward (dgrad)

```python
# TransposableFourOverSixLinearFunction.backward
weight_q = quantize(weight, ctx.config.get_weight_config())  # no transpose flag
weight_q_t = transpose_quantized_tensor(weight_q)
#   -> Triton/CUDA kernel: unpack nibbles (bit ops), transpose, repack (bit ops)
#   -> PyTorch: from_blocked(scales), transpose tile grid, to_blocked(scales)
#   -> new QuantizedTensor with swapped shapes
grad_input = quantized_matmul(dY, weight_q_t)
```

### Kernel design

Both kernels use the same algebraic insight: the full `BLOCK_M × BLOCK_N` code transpose decomposes into four `(BLOCK_M/2 × BLOCK_N/2)` sub-matrix transposes by separating even/odd source rows and low/high nibbles.

**Triton**: Uses `tl.trans()` on four sub-matrices in registers, strided loads for even/odd rows, strided stores for even/odd destination rows.

**CUDA**: Even simpler — each thread `(k, j)` loads one packed byte from source even-row and one from odd-row, extracts four nibbles, repacks them, and writes to destination at `(j, k)`. The transpose is implicit in the coordinate swap; no shared memory needed.

## Tests

All 27 tests pass:

| Test | What it verifies |
|------|-----------------|
| `test_transpose_matches_quantize_with_transpose_flag` (16 parametrizations: 4 shapes × 2 scale rules × {pytorch, triton}) | Transposed tensor matches `quantize(W, transpose=True)` — bitwise identical |
| `test_transpose_roundtrip_lossless` (4 parametrizations: 2 shapes × {pytorch, triton}) | Transposing twice recovers the original tensor exactly |
| `test_all_backends_bitwise_identical` (3 parametrizations) | All available backends produce bitwise-identical packed output |
| `test_transposable_linear_requires_2d_scales` | Construction raises `ValueError` when `weight_scale_2d=False` |
| `test_transposable_linear_forward_matches_original` | Forward output is bitwise identical to `FourOverSixLinear` |
| `test_transposable_linear_backward_runs` | Backward completes, gradients have correct shapes |
| `test_transposable_linear_backward_matches_original` | Input gradients match within tolerance; weight gradients are bitwise identical |
| `test_cuda_transpose_standalone.py` | CUDA kernel correctness and roundtrip via JIT compilation (tested on RTX 5090) |

```
tests/test_transposable_linear.py     27 passed in 4.35s
tests/test_cuda_transpose_standalone  All CUDA tests passed (RTX 5090, CUDA 12.8)
```

## Usage

```python
from fouroversix.model import TransposableFourOverSixLinear
from fouroversix.model.config import ModuleQuantizationConfig

config = ModuleQuantizationConfig(
    weight_scale_2d=True,        # required
    keep_master_weights=True,    # for training
)

base_linear = nn.Linear(4096, 4096).cuda().bfloat16()
layer = TransposableFourOverSixLinear(base_linear, config)

x = torch.randn(1, 128, 4096, dtype=torch.bfloat16, device="cuda")
out = layer(x)       # forward: quantized_matmul(X, W_q)
out.sum().backward()  # backward: nibble transpose for dgrad, no re-quantization
```

### Controlling the backend

```python
from fouroversix.quantize.transpose import TransposeBackend, transpose_quantized_tensor

# Auto-select (default): Triton > CUDA > PyTorch
qt_t = transpose_quantized_tensor(qt)

# Force a specific backend
qt_t = transpose_quantized_tensor(qt, backend=TransposeBackend.triton)
qt_t = transpose_quantized_tensor(qt, backend=TransposeBackend.cuda)
qt_t = transpose_quantized_tensor(qt, backend=TransposeBackend.pytorch)
```

## Scope and limitations

This targets the plain `block_scale_2d=True` training path — the most common configuration for LoRA finetuning on Blackwell. We intentionally don't touch:

- **Quartet II / Random Hadamard Transform** — these have their own transpose handling and are a separate optimization axis
- **`pseudo_quantize` mode** — same forward-path simplification as the existing module
- **1D block scales** — not transpose-invariant by design; this approach requires 2D

The CUDA kernel requires building the C++ extension (`SKIP_CUDA_BUILD=1` disables it). The Triton kernel works on any CUDA GPU without a build step.

## Related work

- [NVFP4-transpose](https://github.com/awarebayes/NVFP4-transpose) — our standalone library with benchmarks comparing all transpose strategies (1D approx/exact, 2D tiles, joint rounding). The 2D tile approach used here was developed and validated there first.
