/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h
 * Copyright (c) 2025, FourOverSix Team.
 ******************************************************************************/

#pragma once

// #include "philox_unpack.cuh" // For at::cuda::philox::unpack

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "kernel_traits.h"
#include "utils.h"
// #include "softmax.h"
// #include "mask.h"
// #include "dropout.h"
// #include "rotary.h"

namespace fouroversix
{

    using namespace cute;

    template <typename Kernel_traits, bool Is_nvfp4, bool Is_rht, bool Is_transpose, bool Is_rtn, int kSelectionRule, typename Params>
    inline __device__ void compute_fp4_quant_prologue_block(const Params &params, const int m_block, const int n_block)
    {
        using Element = typename Kernel_traits::Element;
        using ElementScaleFactor = typename Kernel_traits::ElementScaleFactor;
        using index_t = typename Kernel_traits::index_t;

        // Shared memory
        extern __shared__ char smem[];

        // Constants
        constexpr AdaptiveBlockScalingRuleType kAdaptiveBlockScalingRuleType = static_cast<AdaptiveBlockScalingRuleType>(kSelectionRule);
        constexpr bool Is_4o6 = kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::L1_NORM_4o6 || kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::MSE_4o6 || kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::ABS_MAX_4o6;
        constexpr int kBlockM = Kernel_traits::kBlockM;
        constexpr int kBlockN = Kernel_traits::kBlockN;
        constexpr int kNWarps = Kernel_traits::kNWarps;
        constexpr int kGroupN = Kernel_traits::kGroupN;
        constexpr int kNumGroupsInRow = Kernel_traits::kNumGroupsInRow;
        constexpr float E4M3_MAX_VALUE = Kernel_traits::E4M3_MAX_VALUE;
        constexpr float E2M1_MAX_VALUE = Kernel_traits::E2M1_MAX_VALUE;
        constexpr float TS_SCALE = Is_4o6 ? (384 * 4) : (E4M3_MAX_VALUE * (kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::ALL_4 ? 4 : E2M1_MAX_VALUE));

        const int tidx = threadIdx.x;
        const int num_groups = kNumGroupsInRow * kBlockM;

        // Pointers
        float *ts_ptr = reinterpret_cast<float *>(params.ts_ptr);

        // -------------------------------------------------------------------------
        // Tensor Definitions
        // -------------------------------------------------------------------------

        // Input X (Global Memory)
        Tensor mX = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.x_ptr)),
                                make_shape(params.M, params.N),
                                make_stride(params.x_row_stride, _1{}));
        Tensor gX = local_tile(mX(_, _), Shape<Int<kBlockM>, Int<kBlockN>>{},
                               make_coord(m_block, n_block));

        // Scale Factor Temp SFT (Global Memory)
        Tensor mSFT = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(params.x_sft_ptr)),
                                  make_shape(params.M, params.N_rounded / kGroupN),
                                  make_stride(params.x_sft_row_stride, _1{}));
        Tensor gSFT = local_tile(mSFT(_, _), Shape<Int<kBlockM>, Int<kBlockN / kGroupN>>{},
                                 make_coord(m_block, n_block));

        // Shared Memory Tensors
        Tensor sX = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem)),
                                typename Kernel_traits::SmemLayoutX{});

        // SFT in Shared Memory (placed after X)
        Tensor sSFT = make_tensor(make_smem_ptr(reinterpret_cast<float *>(reinterpret_cast<char *>(sX.data().get()) 
                                  + sizeof(Element) * size(sX))),
                                  typename Kernel_traits::SmemLayoutSFT{});

        // -------------------------------------------------------------------------
        // Data Loading (X -> Shared)
        // -------------------------------------------------------------------------

        typename Kernel_traits::GmemTiledCopyX gmem_tiled_copy_X;
        auto gmem_thr_copy_X = gmem_tiled_copy_X.get_thread_slice(tidx);

        Tensor tXgX = gmem_thr_copy_X.partition_S(gX);
        Tensor tXsX = gmem_thr_copy_X.partition_D(sX);

        // Construct predicates for bounds checking
        Tensor cX = make_identity_tensor(make_shape(size<0>(sX), size<1>(sX)));
        Tensor tXcX = gmem_thr_copy_X.partition_S(cX);
        Tensor tXpX = make_tensor<bool>(make_shape(size<2>(tXcX)));

        for (int i = 0; i < size(tXpX); ++i)
        {
            tXpX(i) = get<1>(tXcX(0, 0, i)) < params.N - n_block * kBlockN;
        }

        __syncthreads();

        // Async copy from Global to Shared
        fouroversix::copy<false, false, true /*Clear_OOB_MN*/, true /*Clear_OOB_K*/>(
            gmem_tiled_copy_X, tXgX, tXsX, tXcX, tXpX, params.M - m_block * kBlockM);

        cute::cp_async_fence();
        fouroversix::cp_async_wait<0>();
        __syncthreads();

        // -------------------------------------------------------------------------
        // Scale Factor Computation
        // -------------------------------------------------------------------------

        float thr_max_scale_factor = static_cast<float>(0.0f);
        for (int group_idx = tidx; group_idx < num_groups; group_idx += blockDim.x)
        {
            const int group_row_idx = group_idx / kNumGroupsInRow;
            const int group_col_idx = group_idx % kNumGroupsInRow;

            float scale_factor = static_cast<float>(0.0f);
#pragma unroll
            for (int i = 0; i < kGroupN; ++i)
            {
                float val = abs(static_cast<float>(sX(group_row_idx, group_col_idx * kGroupN + i)));
                if (val > scale_factor)
                {
                    scale_factor = val;
                }
            }

            if (scale_factor > thr_max_scale_factor)
            {
                thr_max_scale_factor = scale_factor;
            }

            sSFT(group_row_idx, group_col_idx) = scale_factor;
        }

        // -------------------------------------------------------------------------
        // Normalization Constant Reduction (Block-wide Max)
        // -------------------------------------------------------------------------

        // Warp-level reduce
        MaxOp<float> max_op;
        float max_val = static_cast<float>(thr_max_scale_factor);
        max_val = Allreduce<32>::run(max_val, max_op);
        thr_max_scale_factor = max_val;

        // Block-level reduce via Shared Memory
        float *sRed = reinterpret_cast<float *>(smem); // Reuse smem
        if (tidx % 32 == 0)
        {
            sRed[tidx / 32] = thr_max_scale_factor;
        }
        __syncthreads();

        if (tidx == 0)
        {
            float block_max = static_cast<float>(0.0f);
#pragma unroll
            for (int i = 0; i < kNWarps; ++i)
            {
                float t = sRed[i];
                if (t > block_max)
                {
                    block_max = t;
                }
            }
            float block_ts = block_max / TS_SCALE;
            atomicMaxFloat(ts_ptr, block_ts);
        }

        // -------------------------------------------------------------------------
        // Write Back SFT (Shared -> Global)
        // -------------------------------------------------------------------------

        using VecType = uint4;
        constexpr int kVecSize = sizeof(VecType) / sizeof(float);

        for (int r_idx = tidx; r_idx < kBlockM; r_idx += blockDim.x)
        {
#pragma unroll
            for (int i = 0; i < int(kBlockN / kGroupN); i += kVecSize)
            {
                *reinterpret_cast<VecType *>(&gSFT(r_idx, i)) = *reinterpret_cast<VecType *>(&sSFT(r_idx, i));
            }
        }
    }

    template <typename Kernel_traits, bool Is_nvfp4, bool Is_rht, bool Is_transpose, bool Is_rtn, int kSelectionRule, typename Params>
    inline __device__ void compute_fp4_quant_prologue(const Params &params)
    {
        // TODO: Implement the fp4 quant kernel
        const int m_block = blockIdx.x;
        // The block index for the batch.
        const int n_block = blockIdx.y;

        fouroversix::compute_fp4_quant_prologue_block<Kernel_traits, Is_nvfp4, Is_rht, Is_transpose, Is_rtn, kSelectionRule>(params, m_block, n_block);
    }

    template <typename Kernel_traits, bool Is_nvfp4, bool Is_rht, bool Is_transpose, bool Is_rtn, int kSelectionRule, typename Params>
    inline __device__ void compute_fp4_quant_block(const Params &params, const int m_block, const int n_block)
    {
        using Element = typename Kernel_traits::Element;
        using ElementScaleFactor = typename Kernel_traits::ElementScaleFactor;
        using ElementXe2m1Packed = typename Kernel_traits::ElementXe2m1Packed;
        using index_t = typename Kernel_traits::index_t;

        // Shared memory
        extern __shared__ char smem[];

        // Constants
        constexpr AdaptiveBlockScalingRuleType kAdaptiveBlockScalingRuleType = static_cast<AdaptiveBlockScalingRuleType>(kSelectionRule);
        constexpr bool Is_4o6 = kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::L1_NORM_4o6 || kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::MSE_4o6 || kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::ABS_MAX_4o6;
        constexpr int kBlockM = Kernel_traits::kBlockM;
        constexpr int kBlockN = Kernel_traits::kBlockN;
        constexpr int kBlockMSF = Kernel_traits::kBlockMSF;
        constexpr int kBlockNSF = Kernel_traits::kBlockNSF;
        constexpr int kNWarps = Kernel_traits::kNWarps;
        constexpr int kGroupN = Kernel_traits::kGroupN;
        constexpr int kNumGroupsInRow = Kernel_traits::kNumGroupsInRow;
        constexpr float E2M1_MAX_VALUE = Kernel_traits::E2M1_MAX_VALUE;
        constexpr float E4M3_MAX_VALUE = Kernel_traits::E4M3_MAX_VALUE;
        constexpr float E4M3_MIN_POSITIVE_NORMAL = Kernel_traits::E4M3_MIN_POSITIVE_NORMAL;
        constexpr int TS_SCALE = Is_4o6 ? (384 * 4) : (E4M3_MAX_VALUE * E2M1_MAX_VALUE);

        constexpr int kSmemBlockInRow = int(kNumGroupsInRow / 4);
        constexpr int kSmemBlockInCol = int(kBlockM / 128);

        const int tidx = threadIdx.x;
        const int num_groups = kNumGroupsInRow * kBlockM;

        // Pointers
        const float ts = *reinterpret_cast<float *>(params.ts_ptr);
        const float sf_scale_6 = max(ts * E2M1_MAX_VALUE, 1e-12f);
        const float sf_scale_4 = max(ts * 4, 1e-12f);

        // -------------------------------------------------------------------------
        // Tensor Definitions
        // -------------------------------------------------------------------------

        // Input X (Global Memory)
        Tensor mX = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.x_ptr)),
                                make_shape(params.M, params.N),
                                make_stride(params.x_row_stride, _1{}));
        Tensor gX = local_tile(mX(_, _), Shape<Int<kBlockM>, Int<kBlockN>>{},
                               make_coord(m_block, n_block));

        Tensor mXe2m1 = make_tensor(make_gmem_ptr(reinterpret_cast<uint8_t *>(params.x_e2m1_ptr)),
                                    make_shape(params.M, params.N_rounded / 2),
                                    make_stride(params.x_e2m1_row_stride, _1{}));
        Tensor gXe2m1 = local_tile(mXe2m1(_, _), Shape<Int<kBlockM>, Int<kBlockN / 2>>{},
                                   make_coord(m_block, n_block));

        // Scale Factor Temp SFT (Global Memory)
        Tensor mSFT = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(params.x_sft_ptr)),
                                  make_shape(params.M, params.N_rounded / kGroupN),
                                  make_stride(params.x_sft_row_stride, _1{}));
        Tensor gSFT = local_tile(mSFT(_, _), Shape<Int<kBlockM>, Int<kBlockN / kGroupN>>{},
                                 make_coord(m_block, n_block));

        Tensor gSF = make_tensor(make_gmem_ptr(reinterpret_cast<ElementScaleFactor *>(params.x_sf_ptr)),
                                 make_shape(params.M_sf, params.N_sf),
                                 make_stride(params.x_sf_row_stride, _1{}));
        // Tensor gSF = local_tile(mSF(_, _), Shape<Int<1>, Int<16>>{},
        //                          make_coord(m_block, n_block));

        // Shared Memory Tensors
        Tensor sX = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem)),
                                typename Kernel_traits::SmemLayoutX{});

        // SFT in Shared Memory (placed after X)
        Tensor sSFT = make_tensor(make_smem_ptr(reinterpret_cast<float *>(reinterpret_cast<char *>(sX.data().get()) 
                                  + sizeof(Element) * size(sX))),
                                  typename Kernel_traits::SmemLayoutSFT{});

        Tensor sXe2m1 = make_tensor(make_smem_ptr(reinterpret_cast<uint8_t *>(reinterpret_cast<char *>(sSFT.data().get()) 
                                  + sizeof(float) * size(sSFT))),
                                    Shape<Int<kBlockM>, Int<kBlockN / 2>>{},
                                    Stride<Int<kBlockN / 2>, _1>{});

        Tensor sSF = make_tensor(make_smem_ptr(reinterpret_cast<ElementScaleFactor *>(reinterpret_cast<char *>(sXe2m1.data().get()) + sizeof(uint8_t) * size(sXe2m1))),
                                 typename Kernel_traits::SmemLayoutSF{});

        // -------------------------------------------------------------------------
        // Data Loading (X -> Shared)
        // -------------------------------------------------------------------------

        typename Kernel_traits::GmemTiledCopyX gmem_tiled_copy_X;
        auto gmem_thr_copy_X = gmem_tiled_copy_X.get_thread_slice(tidx);

        Tensor tXgX = gmem_thr_copy_X.partition_S(gX);
        Tensor tXsX = gmem_thr_copy_X.partition_D(sX);

        // Construct predicates for bounds checking
        Tensor cX = make_identity_tensor(make_shape(size<0>(sX), size<1>(sX)));
        Tensor tXcX = gmem_thr_copy_X.partition_S(cX);
        Tensor tXpX = make_tensor<bool>(make_shape(size<2>(tXcX)));

        for (int i = 0; i < size(tXpX); ++i)
        {
            tXpX(i) = get<1>(tXcX(0, 0, i)) < params.N - n_block * kBlockN;
        }

        __syncthreads();

        // Async copy from Global to Shared
        fouroversix::copy<false, false, true /*Clear_OOB_MN*/, true /*Clear_OOB_K*/>(
            gmem_tiled_copy_X, tXgX, tXsX, tXcX, tXpX, params.M - m_block * kBlockM);

        cute::cp_async_fence();

        // -------------------------------------------------------------------------
        // Data Loading (SFT -> Shared)
        // -------------------------------------------------------------------------

        using VecTypeSFT = uint4;
        constexpr int kVecSizeSFT = sizeof(VecTypeSFT) / sizeof(float);

        for (int r_idx = tidx; r_idx < kBlockM; r_idx += blockDim.x)
        {
#pragma unroll
            for (int i = 0; i < int(kBlockN / kGroupN); i += kVecSizeSFT)
            {
                *reinterpret_cast<VecTypeSFT *>(&sSFT(r_idx, i)) = *reinterpret_cast<VecTypeSFT *>(&gSFT(r_idx, i));
            }
        }

        fouroversix::cp_async_wait<0>();
        __syncthreads();

        // -------------------------------------------------------------------------
        // Quantization
        // -------------------------------------------------------------------------

        for (int group_idx = tidx; group_idx < num_groups; group_idx += blockDim.x)
        {
            const int group_row_idx = group_idx / kNumGroupsInRow;
            const int group_col_idx = group_idx % kNumGroupsInRow;

            const float group_max = sSFT(group_row_idx, group_col_idx);

            const Tensor sGX = make_tensor(make_smem_ptr(sX.data() + group_idx * kGroupN),
                                           Shape<Int<1>, Int<kGroupN>>{},
                                           Stride<Int<kGroupN>, _1>{});

            using OutputType = cutlass::Array<cutlass::float_e2m1_t, 8>;
            OutputType res[int(kGroupN / 8)];
            float sf;
            if constexpr (Is_4o6)
            {
                float sf_[2] = {
                    clamp(
                        group_max / sf_scale_4,
                        E4M3_MIN_POSITIVE_NORMAL, E4M3_MAX_VALUE),
                    clamp(
                        group_max / sf_scale_6,
                        E4M3_MIN_POSITIVE_NORMAL, E4M3_MAX_VALUE)};

                sf_[0] = static_cast<float>(static_cast<ElementScaleFactor>(sf_[0]));
                sf_[1] = static_cast<float>(static_cast<ElementScaleFactor>(sf_[1]));

                sf = fp4_convertion<Is_nvfp4, true, Is_rtn, kAdaptiveBlockScalingRuleType>(sGX, ts, sf_, res);
                // if (cute::thread0()) {
                //     printf("in fp4_quant_block, 4o6, sf = %f, res[0] = %lx\n", sf, reinterpret_cast<uint64_t&>(res[0]));
                // }
            }
            else
            {
                // static_assert(kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::ALL_6 || kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::ALL_4, "kAdaptiveBlockScalingRuleType must be AdaptiveBlockScalingRuleType::ALL_6 or AdaptiveBlockScalingRuleType::ALL_4");
                float sf_val = 0.0f;
                if constexpr (kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::ALL_6)
                {
                    sf_val = clamp(
                        group_max / sf_scale_6,
                        E4M3_MIN_POSITIVE_NORMAL, E4M3_MAX_VALUE);
                }
                else if constexpr (kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::ALL_4)
                {
                    sf_val = clamp(
                        group_max / sf_scale_4,
                        E4M3_MIN_POSITIVE_NORMAL, E4M3_MAX_VALUE);
                }
                else
                {
                    printf("in fp4_quant_block, kAdaptiveBlockScalingRuleType = %d, not supported\n", kAdaptiveBlockScalingRuleType);
                    assert(false);
                }

                // Add by JXGuo: convert the float to ElementScaleFactor and convert back for better accuracy.
                sf_val = static_cast<float>(static_cast<ElementScaleFactor>(sf_val));

                sf = fp4_convertion<Is_nvfp4, false, Is_rtn, kAdaptiveBlockScalingRuleType>(sGX, ts, &sf_val, res);
                // if (cute::thread0()) {
                //     printf("in fp4_quant_block, not 4o6, sf = %f, res[0] = %lx\n", sf, reinterpret_cast<uint64_t&>(res[0]));
                // }
            }
            // printf("in fp4_quant_block, group_idx = %d, group_row_idx = %d, group_col_idx = %d, res[0] = %lx\n", group_idx, group_row_idx, group_col_idx, reinterpret_cast<uint64_t&>(res[0]));
            // if (group_row_idx == 30 || group_row_idx == 31 || group_row_idx == 32) {
            //     printf("in fp4_quant_block, group_row_idx = %d, group_col_idx = %d, res[0] = %lx\n", group_row_idx, group_col_idx, reinterpret_cast<uint64_t&>(res[0]));
            // }
            for (int i = 0; i < int(kGroupN / 8); i++)
            {
                *reinterpret_cast<OutputType *>(&sXe2m1(group_row_idx, group_col_idx * (kGroupN / 2) + i * 4)) = res[i];
            }

            // *reinterpret_cast<ElementXe2m1Packed*>(&sXe2m1(group_row_idx, group_col_idx * kGroupN / 2)) = *reinterpret_cast<ElementXe2m1Packed*>(res);
            const int row_in_block = group_row_idx % 128;
            const int col_in_block = group_col_idx % 4;
            const int block_row_idx = int(group_row_idx / 128);
            const int block_col_idx = int(group_col_idx / 4);
            // const int block_in_row = kSmemBlockInRow;
            const int row_sf_layout_idx = 32 * (block_row_idx * kSmemBlockInRow + block_col_idx) + row_in_block % 32;
            const int col_sf_layout_idx = int(row_in_block / 32) * 4 + col_in_block;
            // const int row_in_block = group_row_idx % 128;
            // const int col_in_block = group_col_idx % 4;
            // const int row_sf_layout_idx = int(group_row_idx / 128) * 32 + row_in_block % 32;
            // const int col_sf_layout_idx = int(row_in_block / 32) * 4 + col_in_block;
            sSF(row_sf_layout_idx, col_sf_layout_idx) = static_cast<ElementScaleFactor>(sf);
        }

        constexpr int kVecSizeX = sizeof(ElementXe2m1Packed) / sizeof(uint8_t);

        // if (cute::thread0()) {
        //     printf("in fp4_quant_block, sXe2m1(32, 0) = %lx\n", reinterpret_cast<uint64_t&>(sXe2m1(32, 0)));
        //     print_tensor(sXe2m1);
        //     printf("########################################################\n");
        // }

        __syncthreads();

        for (int r_idx = tidx; r_idx < kBlockM; r_idx += blockDim.x)
        {
// printf("in fp4_quant_block, r_idx = %d, kBlockM = %d, blockDim.x = %d, sXe2m1(r_idx, 0) = %lx\n", r_idx, kBlockM, blockDim.x, reinterpret_cast<uint64_t&>(sXe2m1(r_idx, 0)));
#pragma unroll
            for (int i = 0; i < int(kBlockN / 2); i += kVecSizeX)
            {
                // if (cute::thread0()) {
                //     printf("in fp4_quant_block, r_idx = %d, i = %d, sXe2m1(r_idx, i) = %lx\n", r_idx, i, reinterpret_cast<uint64_t&>(sXe2m1(r_idx, i)));
                // }
                *reinterpret_cast<ElementXe2m1Packed *>(&gXe2m1(r_idx, i)) = *reinterpret_cast<ElementXe2m1Packed *>(&sXe2m1(r_idx, i));
            }
        }

        // const int row_in_block = group_row_idx % 128;
        // const int col_in_block = group_col_idx % 4;
        // const int block_row_idx = int(group_row_idx / 128);
        // const int block_col_idx = int(group_col_idx / 4);
        // const int block_in_row = int(kNumGroupsInRow / 4);
        // const int row_sf_layout_idx = 32 * (block_row_idx * block_in_row + block_col_idx) + row_in_block_idx;
        // const int col_sf_layout_idx = int(row_in_block / 32) * 4 + col_in_block;

        using VecTypeSF = uint4;
        constexpr int kVecSizeSF = sizeof(VecTypeSF) / sizeof(ElementScaleFactor);

        // const int global_block_row_idx_base = int(kBlockM / 128) * m_block;
        // const int global_block_in_row = int(params.N_rounded / (kGroupN * 4));
        // // const index_t global_sf_row_idx_base = index_t(32) * global_block_in_row * global_block_row_idx_base;
        // for (int r_idx = tidx; r_idx < kBlockMSF; r_idx += blockDim.x) {
        //     const int local_block_row_idx = int(r_idx / 32);
        //     const index_t global_sf_row_idx_base = index_t(32) * (global_block_row_idx_base + local_block_row_idx) * global_block_in_row;

        //     #pragma unroll
        //     for (int i = 0; i < int(kBlockNSF); i += kVecSizeSF) {
        //         const int local_block_col_idx = int(i / 16);
        //         const index_t global_sf_row_idx = global_sf_row_idx_base + index_t(32) * local_block_col_idx;
        //         const index_t global_sf_col_idx = index_t(16) * local_block_col_idx;
        //         *reinterpret_cast<VecTypeSF*>(&gSF(global_sf_row_idx, global_sf_col_idx)) = *reinterpret_cast<VecTypeSF*>(&sSF(r_idx, i));
        //     }
        // }

        const int global_blk_row_stride = int(params.N_rounded / (kGroupN * 4));
        const int global_blk_col_stride = 1;
        const int global_blk_idx_base = (m_block * kSmemBlockInCol) * global_blk_row_stride + (n_block * kSmemBlockInRow) * global_blk_col_stride;

        static_assert(kVecSizeSF == kBlockNSF, "kVecSizeSF must be equal to kBlockNSF");
        for (int r_idx = tidx; r_idx < kBlockMSF; r_idx += blockDim.x)
        {
            const int local_block_idx = int(r_idx / 32);
            const int local_row_idx = r_idx % 32;
            const int local_block_row_idx = int(local_block_idx / kSmemBlockInRow);
            const int local_block_col_idx = int(local_block_idx % kSmemBlockInRow);
            const int global_blk_idx = global_blk_idx_base + local_block_row_idx * global_blk_row_stride + local_block_col_idx * global_blk_col_stride;
            const index_t global_row_idx = index_t(32) * global_blk_idx + local_row_idx;
            *reinterpret_cast<VecTypeSF *>(&gSF(global_row_idx, 0)) = *reinterpret_cast<VecTypeSF *>(&sSF(r_idx, 0));
        }
    }

    template <typename Kernel_traits, bool Is_nvfp4, bool Is_rht, bool Is_transpose, bool Is_rtn, int kSelectionRule, typename Params>
    inline __device__ void compute_fp4_quant(const Params &params)
    {
        // TODO: Implement the fp4 quant kernel
        const int m_block = blockIdx.x;
        // The block index for the batch.
        const int n_block = blockIdx.y;

        fouroversix::compute_fp4_quant_block<Kernel_traits, Is_nvfp4, Is_rht, Is_transpose, Is_rtn, kSelectionRule>(params, m_block, n_block);
    }

} // namespace fouroversix
