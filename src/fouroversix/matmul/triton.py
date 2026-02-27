import torch
from fouroversix.matmul.backend import MatmulBackendBase
from fouroversix.quantize import QuantizedTensor, from_blocked
from fouroversix.utils import DataType, device_supports_cvt_rn_e2m1x2


class TritonMatmulBackend(MatmulBackendBase):
    """
    The Triton matrix multiplication backend. Uses Triton kernels to perform fast
    FP4 matrix multiplication. Requires a Blackwell GPU.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the Triton backend is available on the current machine."""
        return True

    @classmethod
    def is_supported(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
    ) -> bool:
        """
        Return True if the Triton backend supports the given inputs and output data
        type.
        """

        if not super().is_supported(input, other, out_dtype=out_dtype):
            return False

        return (
            input.dtype in {DataType.if4, DataType.nvfp4}
            and input.device.type == "cuda"
        )

    @classmethod
    def fp4_matmul(
        cls,
        input: QuantizedTensor,
        other: QuantizedTensor,
        *,
        out_dtype: DataType,
        use_blackwell_cvt_rn_instructions: bool | None = None,
    ) -> torch.Tensor:
        """
        Perform a matrix multiplication (`a @ b.T`) between two quantized tensors using
        the Triton backend.
        """

        import triton
        import triton.language as tl

        from .kernel import matmul_kernel

        m = input.original_shape[0]
        n = other.original_shape[0]
        k = input.original_shape[1]

        input_sf_shape = (
            input.padded_shape[0],
            input.padded_shape[1] // input.dtype.block_size,
        )
        other_sf_shape = (
            other.padded_shape[0],
            other.padded_shape[1] // other.dtype.block_size,
        )

        if input.scale_factors_are_in_blackwell_layout:
            input_scale_factors = from_blocked(
                input.scale_factors,
                input_sf_shape,
            )
        else:
            input_scale_factors = input.scale_factors.reshape(input_sf_shape)

        if other.scale_factors_are_in_blackwell_layout:
            other_scale_factors = from_blocked(
                other.scale_factors,
                other_sf_shape,
            )
        else:
            other_scale_factors = other.scale_factors.reshape(other_sf_shape)

        output = torch.empty(
            (m, n),
            device=input.values.device,
            dtype=out_dtype.torch_dtype,
        )

        grid = lambda meta: (  # noqa: E731
            triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

        matmul_kernel[grid](
            input.values,
            input_scale_factors,
            input.amax,
            other.values,
            other_scale_factors,
            other.amax,
            output,
            m,
            n,
            k,
            input.values.stride(0),
            input_scale_factors.stride(0),
            input.values.stride(1),
            input_scale_factors.stride(1),
            other.values.stride(0),
            other_scale_factors.stride(0),
            other.values.stride(1),
            other_scale_factors.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=6,
            A_E2M1_MAX_VALUE=input.dtype.quantized_value_type.get_maximum_value(
                input.scale_rule,
            ),
            B_E2M1_MAX_VALUE=other.dtype.quantized_value_type.get_maximum_value(
                other.scale_rule,
            ),
            A_E4M3_MAX_VALUE=input.dtype.scale_type.get_maximum_value(input.scale_rule),
            B_E4M3_MAX_VALUE=other.dtype.scale_type.get_maximum_value(other.scale_rule),
            DTYPE=input.dtype.value,
            INTERMEDIATE_DTYPE=tl.float16,
            OUT_DTYPE=tl.bfloat16,
            USE_BLACKWELL_CVT_RN_INSTRUCTIONS=(
                device_supports_cvt_rn_e2m1x2()
                if use_blackwell_cvt_rn_instructions is None
                else use_blackwell_cvt_rn_instructions
            ),
        )

        return output
