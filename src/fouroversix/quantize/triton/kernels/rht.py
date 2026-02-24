import triton
import triton.language as tl


@triton.jit
def rht_kernel(
    x_desc,
    h_desc,
    y_desc,
    # Meta-parameters
    # TODO(jack): Update RHT kernel to support unpadded dimensions
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TRANSPOSE: tl.constexpr,
) -> None:
    HAD_BLOCK_SIZE: tl.constexpr = h_desc.block_shape[0]

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load H [B, B]
    h_block = h_desc.load([0, 0])

    m_block_offset = pid_m * BLOCK_SIZE_M
    n_block_offset = pid_n * BLOCK_SIZE_N

    if not TRANSPOSE:
        x_block = x_desc.load([m_block_offset, n_block_offset])
    else:
        x_block = x_desc.load([n_block_offset, m_block_offset]).T

    y_block = tl.dot(
        x_block.reshape(
            BLOCK_SIZE_M * BLOCK_SIZE_N // HAD_BLOCK_SIZE,
            HAD_BLOCK_SIZE,
        ).to(tl.bfloat16),
        h_block,
    ).reshape(BLOCK_SIZE_M, BLOCK_SIZE_N)

    y_desc.store([m_block_offset, n_block_offset], y_block)
