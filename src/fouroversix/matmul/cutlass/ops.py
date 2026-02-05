import torch

import fouroversix._C  # noqa: F401


def gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fouroversix.gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt.default(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
    )


@torch.library.register_fake("fouroversix::gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt")
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,  # noqa: ARG001
    b_sf: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty(m, n, dtype=torch.bfloat16, device=a.device)


def gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt_sm120(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fouroversix.gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt_sm120.default(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
    )


@torch.library.register_fake(
    "fouroversix::gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt_sm120",
)
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,  # noqa: ARG001
    b_sf: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty(m, n, dtype=torch.bfloat16, device=a.device)


def gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fouroversix.gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt.default(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
    )


@torch.library.register_fake("fouroversix::gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt")
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,  # noqa: ARG001
    b_sf: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty(m, n, dtype=torch.bfloat16, device=a.device)


def gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fouroversix.gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120.default(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
    )


@torch.library.register_fake(
    "fouroversix::gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120",
)
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,  # noqa: ARG001
    b_sf: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty(m, n, dtype=torch.bfloat16, device=a.device)


def gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fouroversix.gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt.default(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
    )


@torch.library.register_fake("fouroversix::gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt")
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,  # noqa: ARG001
    b_sf: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty(m, n, dtype=torch.float16, device=a.device)


def gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.fouroversix.gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120.default(
        a,
        b,
        a_sf,
        b_sf,
        alpha,
    )


@torch.library.register_fake(
    "fouroversix::gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120",
)
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,  # noqa: ARG001
    b_sf: torch.Tensor,  # noqa: ARG001
    alpha: torch.Tensor,  # noqa: ARG001
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty(m, n, dtype=torch.float16, device=a.device)


def quantize_to_fp4(
    x: torch.Tensor,
    is_nvfp4: bool,  # noqa: FBT001
    is_rtn: bool,  # noqa: FBT001
    is_rht: bool,  # noqa: FBT001
    is_transpose: bool,  # noqa: FBT001
    selection_rule: int,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    return torch.ops.fouroversix.quantize_to_fp4.default(
        x,
        is_nvfp4,
        is_rtn,
        is_rht,
        is_transpose,
        selection_rule,
    )


@torch.library.register_fake("fouroversix::quantize_to_fp4")
def _(
    x: torch.Tensor,
    is_nvfp4: bool,  # noqa: FBT001
    is_rtn: bool,  # noqa: ARG001, FBT001
    is_rht: bool,  # noqa: ARG001, FBT001
    is_transpose: bool,  # noqa: ARG001, FBT001
    selection_rule: int,  # noqa: ARG001
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    return (
        torch.empty(x.shape[0], x.shape[1] // 2, dtype=torch.uint8, device=x.device),
        torch.empty(
            x.shape[0] * x.shape[1] // (16 if is_nvfp4 else 32),
            dtype=torch.float8_e4m3fn if is_nvfp4 else torch.float8_e8m0fnu,
            device=x.device,
        ),
        torch.empty(1, dtype=torch.float32, device=x.device),
    )
