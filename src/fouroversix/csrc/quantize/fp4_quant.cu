#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState

#include <cutlass/numeric_types.h>

#include "hardware_info.h"
#include "fp4_quant.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace fouroversix
{
    void set_params_fp4_quant(
        FP4_quant_params &params,
        /*-------------- tensors ---------------*/
        const at::Tensor x,
        at::Tensor x_e2m1,
        at::Tensor x_sf,
        at::Tensor x_sft,
        at::Tensor ts,
        const int M,
        const int N,
        const int M_rounded,
        const int N_rounded,
        const int M_sf,
        const int N_sf,
        const bool is_nvfp4,
        const bool is_rtn,
        const bool is_rht,
        // const bool is_4o6,
        const bool is_transpose,
        const int selection_rule)
    {

        // Reset the parameters
        params = {};

        params.is_bf16 = x.dtype() == torch::kBFloat16;

        /**************** Pointers & strides ****************/
        params.x_ptr = x.data_ptr();
        params.x_e2m1_ptr = x_e2m1.data_ptr();
        params.x_sf_ptr = x_sf.data_ptr();
        params.x_sft_ptr = x_sft.data_ptr();
        params.ts_ptr = ts.data_ptr();

        // Element-based strides (not bytes)
        params.x_row_stride = x.stride(0);
        params.x_col_stride = x.stride(1);
        params.x_e2m1_row_stride = x_e2m1.stride(0);
        params.x_e2m1_col_stride = x_e2m1.stride(1);
        params.x_sf_row_stride = x_sf.stride(0);
        params.x_sf_col_stride = x_sf.stride(1);
        params.x_sft_row_stride = x_sft.stride(0);
        params.x_sft_col_stride = x_sft.stride(1);

        // Set the dimensions
        params.M = M;
        params.N = N;
        params.M_rounded = M_rounded;
        params.N_rounded = N_rounded;
        params.M_sf = M_sf;
        params.N_sf = N_sf;
        // Set FP4-specific parameters
        params.is_nvfp4 = is_nvfp4;
        params.is_rtn = is_rtn;
        params.is_rht = is_rht;
        // params.is_4o6 = is_4o6;
        params.is_transpose = is_transpose;
        params.selection_rule = selection_rule;
    }

    void run_fp4_quant(FP4_quant_params &params, cudaStream_t stream)
    {
        FP16_SWITCH(!params.is_bf16, [&]
                    { BOOL_SWITCH(params.is_nvfp4, Is_nvfp4, [&]
                                  { BOOL_SWITCH(params.is_rht, Is_rht, [&]
                                                { BOOL_SWITCH(params.is_transpose, Is_transpose, [&]
                                                              { run_fp4_quant_<fp16_type, Is_nvfp4, Is_rht, Is_transpose>(params, stream); }); }); }); });
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor> quantize_to_fp4(
        const at::Tensor &x,
        const bool is_nvfp4,
        const bool is_rtn,
        const bool is_rht,
        // const bool       is_4o6,
        const bool is_transpose,
        const int64_t selection_rule)
    {

        /*******
         * selection_rule:
         * 0: all_6
         * 1: all_4
         * 2: 4o6_l1_norm
         * 3: 4o6_mse
         * 4: 4o6_abs_max
         */
        TORCH_CHECK(selection_rule >= 0 && selection_rule <= 4, "Invalid selection_rule: " + std::to_string(selection_rule));
        // const int is_4o6 = selection_rule == 2 || selection_rule == 3;

        /**********************
         * 1. Sanity checks   *
         *********************/
        at::cuda::CUDAGuard device_guard{x.device()};

        // Hardware capability
        {
            auto [cc_major, _] = get_compute_capability(get_current_device());
            TORCH_CHECK(cc_major >= 10, "FP4Quant only supports Blackwell GPUs or newer.");
        }

        // Dtype / device checks
        auto dtype = x.dtype();
        TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16,
                    "FP4Quant only supports fp16 and bf16 data types");
        // TORCH_CHECK(km.dtype() == dtype, "q and km must have the same dtype");
        CHECK_DEVICE(x);
        // Layout / contiguity checks
        TORCH_CHECK(x.stride(-1) == 1, "x must be contiguous on the last dim");

        /**********************
         * 2. Dimension logic *
         *********************/

        const int M = is_transpose ? x.size(1) : x.size(0);
        const int N = is_transpose ? x.size(0) : x.size(1);

        const int n_round = is_nvfp4 ? 64 : 128;
        TORCH_CHECK(N % n_round == 0, "N must be multiple of " + std::to_string(n_round));

        /**********************
         * 4. Derived sizes   *
         *********************/
        auto ceil_div = [](int x, int m)
        { return (x + m - 1) / m; };
        auto round_up = [](int x, int m)
        { return (x + m - 1) / m * m; };

        const int M_rounded = round_up(M, 128);
        const int N_rounded = round_up(N, n_round);

        // const int max_seqlen_km       = ceil_div(max_seqlen_k, moba_chunk_size);
        // const int head_size_rounded   = round_up(head_size, head_size <= 128 ? 32 : 64);
        // const int seqlen_q_rounded    = round_up(max_seqlen_q, 128);
        // const int seqlen_km_rounded   = round_up(max_seqlen_km, 128);
        // const int moba_topk_rounded   = round_up(moba_topk, 16);

        /**********************
         * 3. Output buffers  *
         *********************/
        at::Tensor x_e2m1 = torch::zeros({M_rounded, int(N_rounded / 2)}, x.options().dtype(torch::kUInt8));
        at::Tensor x_sf, x_sft;
        int M_sf, N_sf;
        if (is_nvfp4)
        {
            M_sf = int(M_rounded / 128 * 32) * int(N_rounded / 64);
            N_sf = 16;
            // N_sf = int(N_rounded / 16 * 4);
            x_sf = torch::zeros({M_sf, N_sf}, x.options().dtype(torch::kFloat8_e4m3fn));
            x_sft = torch::zeros({M_rounded, int(N_rounded / 16)}, x.options().dtype(torch::kFloat32));
        }
        else
        {
            M_sf = int(M_rounded / 128 * 32) * int(N_rounded / 128);
            N_sf = 16;
            x_sf = torch::zeros({M_sf, N_sf}, x.options().dtype(torch::kUInt8));
            x_sft = torch::zeros({M_rounded, int(N_rounded / 32)}, x.options().dtype(torch::kFloat32));
        }
        at::Tensor ts = torch::zeros({1}, x.options().dtype(torch::kFloat32));

        /**********************
         * 5. Param struct    *
         *********************/
        FP4_quant_params params;
        // const at::Tensor x,
        // const at::Tensor x_e2m1,
        // const at::Tensor x_sf,
        // const at::Tensor ts,
        // cosnt int M,
        // const int N,
        // const bool is_nvfp4,
        // const bool is_rtn,
        // const bool is_4o6,
        // const bool is_transpose
        set_params_fp4_quant(
            params,
            /*-------------- tensors ---------------*/
            x, x_e2m1, x_sf, x_sft, ts, M, N, M_rounded, N_rounded, M_sf, N_sf,
            is_nvfp4, is_rtn, is_rht, is_transpose, selection_rule);

        /**********************
         * 6. Kernel launch   *
         *********************/

        if (M > 0)
        {
            run_fp4_quant(params, at::cuda::getCurrentCUDAStream().stream());
        }
        else
        {
            ts.fill_(0);
        }

        return std::make_tuple(x_e2m1, x_sf.flatten(), ts);
    }

    TORCH_LIBRARY_IMPL(fouroversix, CUDA, m)
    {
        m.impl("quantize_to_fp4", &quantize_to_fp4);
    }
}