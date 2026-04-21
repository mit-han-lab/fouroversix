"""Standalone test for the CUDA transpose_packed_fp4 kernel via JIT compilation."""

import os
import tempfile

import torch

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int HALF_M = TILE_M / 2;
constexpr int HALF_N = TILE_N / 2;

__global__ void transpose_packed_fp4_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int rows, int cols,
    int src_stride, int dst_stride
) {
    int tile_m = blockIdx.x;
    int tile_n = blockIdx.y;
    int k = threadIdx.x;
    int j = threadIdx.y;

    int src_even_row = tile_m * TILE_M + 2 * k;
    int src_odd_row  = src_even_row + 1;
    int src_pcol     = tile_n * HALF_N + j;

    uint8_t packed_even = src[src_even_row * src_stride + src_pcol];
    uint8_t packed_odd  = src[src_odd_row  * src_stride + src_pcol];

    uint8_t low_even  = packed_even & 0xF;
    uint8_t high_even = (packed_even >> 4) & 0xF;
    uint8_t low_odd   = packed_odd  & 0xF;
    uint8_t high_odd  = (packed_odd  >> 4) & 0xF;

    uint8_t result_even = (low_odd  << 4) | low_even;
    uint8_t result_odd  = (high_odd << 4) | high_even;

    int dst_even_row = tile_n * TILE_N + 2 * j;
    int dst_odd_row  = dst_even_row + 1;
    int dst_pcol     = tile_m * HALF_M + k;

    dst[dst_even_row * dst_stride + dst_pcol] = result_even;
    dst[dst_odd_row  * dst_stride + dst_pcol] = result_odd;
}


torch::Tensor transpose_packed_fp4(torch::Tensor values, int64_t rows, int64_t cols) {
    TORCH_CHECK(values.is_cuda(), "values must be on CUDA");
    TORCH_CHECK(values.dtype() == torch::kUInt8, "values must be uint8");
    const at::cuda::CUDAGuard device_guard(values.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto dst = torch::empty({cols, rows / 2}, torch::dtype(torch::kUInt8).device(values.device()));
    dim3 grid(rows / TILE_M, cols / TILE_N);
    dim3 block(HALF_M, HALF_N);
    transpose_packed_fp4_kernel<<<grid, block, 0, stream>>>(
        values.data_ptr<uint8_t>(), dst.data_ptr<uint8_t>(),
        (int)rows, (int)cols, (int)values.stride(0), (int)dst.stride(0));
    return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_packed_fp4", &transpose_packed_fp4,
          "Transpose packed FP4 nibbles (CUDA)");
}
"""


def _build_cuda_module():
    """JIT-compile the standalone CUDA kernel, return the loaded module."""
    from torch.utils.cpp_extension import load

    tmpdir = tempfile.mkdtemp(prefix="fox_transpose_test_")
    src_path = os.path.join(tmpdir, "transpose_fp4_test.cu")
    with open(src_path, "w") as f:
        f.write(CUDA_SRC)
    return load(
        name="transpose_fp4_test",
        sources=[src_path],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def _unpack(x):
    low = x & 0xF
    high = (x >> 4) & 0xF
    return torch.stack([low, high], dim=-1).reshape(x.shape[0], x.shape[1] * 2)


def _pack(codes):
    r, c = codes.shape
    codes = codes.reshape(r, c // 2, 2)
    return (codes[:, :, 1] << 4) | codes[:, :, 0]


def _pytorch_transpose(values, rows, cols):
    codes = _unpack(values)[:rows, :cols]
    return _pack(codes.T.contiguous())


def test_cuda_transpose_correctness():
    cuda_mod = _build_cuda_module()

    for rows, cols in [(16, 16), (32, 64), (128, 128), (256, 512), (512, 256)]:
        torch.manual_seed(42)
        packed = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, device="cuda")

        ref = _pytorch_transpose(packed, rows, cols)
        out = cuda_mod.transpose_packed_fp4(packed, rows, cols)

        assert ref.shape == out.shape, f"{rows}x{cols}: shape {ref.shape} vs {out.shape}"
        assert torch.equal(ref, out), (
            f"{rows}x{cols}: mismatch at {(ref != out).nonzero(as_tuple=False).tolist()[:5]}"
        )
        print(f"  {rows}x{cols} OK")


def test_cuda_transpose_roundtrip():
    cuda_mod = _build_cuda_module()

    for rows, cols in [(128, 128), (256, 512)]:
        torch.manual_seed(99)
        packed = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, device="cuda")

        t1 = cuda_mod.transpose_packed_fp4(packed, rows, cols)
        t2 = cuda_mod.transpose_packed_fp4(t1, cols, rows)
        assert torch.equal(packed, t2), f"{rows}x{cols}: roundtrip mismatch"
        print(f"  {rows}x{cols} roundtrip OK")


if __name__ == "__main__":
    print("Building CUDA kernel...")
    print("test_cuda_transpose_correctness:")
    test_cuda_transpose_correctness()
    print("test_cuda_transpose_roundtrip:")
    test_cuda_transpose_roundtrip()
    print("All CUDA tests passed!")
