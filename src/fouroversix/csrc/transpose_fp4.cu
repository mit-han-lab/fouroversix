/**
 * CUDA kernel for fused unpack-transpose-repack of packed FP4 nibbles.
 *
 * Each thread loads one packed byte from even and odd source rows, unpacks
 * the four nibble codes, repacks them for the transposed layout, and writes
 * two destination bytes.  The transpose is implicit: thread (k, j) reads
 * source position (k, j) and writes destination position (j, k).
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace fouroversix {

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int HALF_M = TILE_M / 2;
constexpr int HALF_N = TILE_N / 2;

/**
 * Each block handles one TILE_M x TILE_N tile of unpacked codes.
 * Block dimensions: (HALF_M, HALF_N) = (8, 8) = 64 threads.
 *
 * Thread (k, j) loads src packed bytes at even/odd source rows (2k, 2k+1)
 * and packed column j, producing four nibble codes.  It then repacks them
 * for the transposed destination at even/odd rows (2j, 2j+1) and packed
 * column k.
 */
__global__ void transpose_packed_fp4_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int rows, int cols,
    int src_stride, int dst_stride
) {
    int tile_m = blockIdx.x;
    int tile_n = blockIdx.y;

    int k = threadIdx.x;  /* source row-pair index within HALF_M */
    int j = threadIdx.y;  /* source packed-col index within HALF_N */

    int src_even_row = tile_m * TILE_M + 2 * k;
    int src_odd_row  = src_even_row + 1;
    int src_pcol     = tile_n * HALF_N + j;

    uint8_t packed_even = src[src_even_row * src_stride + src_pcol];
    uint8_t packed_odd  = src[src_odd_row  * src_stride + src_pcol];

    uint8_t low_even  = packed_even & 0xF;         /* src_codes[2k,   2j]   */
    uint8_t high_even = (packed_even >> 4) & 0xF;   /* src_codes[2k,   2j+1] */
    uint8_t low_odd   = packed_odd  & 0xF;          /* src_codes[2k+1, 2j]   */
    uint8_t high_odd  = (packed_odd  >> 4) & 0xF;   /* src_codes[2k+1, 2j+1] */

    /* Repack for the transposed layout:
     * dst_packed[2j,   k] = (src_codes[2k+1, 2j]   << 4) | src_codes[2k, 2j]
     * dst_packed[2j+1, k] = (src_codes[2k+1, 2j+1] << 4) | src_codes[2k, 2j+1]
     */
    uint8_t result_even = (low_odd  << 4) | low_even;
    uint8_t result_odd  = (high_odd << 4) | high_even;

    int dst_even_row = tile_n * TILE_N + 2 * j;
    int dst_odd_row  = dst_even_row + 1;
    int dst_pcol     = tile_m * HALF_M + k;

    dst[dst_even_row * dst_stride + dst_pcol] = result_even;
    dst[dst_odd_row  * dst_stride + dst_pcol] = result_odd;
}


torch::Tensor transpose_packed_fp4_cuda(
    const torch::Tensor& values,
    int64_t rows,
    int64_t cols
) {
    TORCH_CHECK(values.is_cuda(), "values must be on CUDA");
    TORCH_CHECK(values.dtype() == torch::kUInt8, "values must be uint8");
    TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
    TORCH_CHECK(rows % TILE_M == 0, "rows must be a multiple of ", TILE_M);
    TORCH_CHECK(cols % TILE_N == 0, "cols must be a multiple of ", TILE_N);

    const at::cuda::CUDAGuard device_guard(values.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto dst = torch::empty({cols, rows / 2}, torch::dtype(torch::kUInt8).device(values.device()));

    dim3 grid(rows / TILE_M, cols / TILE_N);
    dim3 block(HALF_M, HALF_N);

    transpose_packed_fp4_kernel<<<grid, block, 0, stream>>>(
        values.data_ptr<uint8_t>(),
        dst.data_ptr<uint8_t>(),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(values.stride(0)),
        static_cast<int>(dst.stride(0))
    );

    return dst;
}


TORCH_LIBRARY_IMPL(fouroversix, CUDA, m) {
    m.impl("transpose_packed_fp4", &transpose_packed_fp4_cuda);
}

}  // namespace fouroversix
