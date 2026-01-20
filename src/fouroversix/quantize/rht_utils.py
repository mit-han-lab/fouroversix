import functools
import math

import torch

"""
Credit: TransformerEngine
https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/nvfp4_tensor.py
"""


def get_no_random_sign_vector(device: int) -> torch.Tensor:
    """Non-random sign vector for Hadamard transform."""
    return torch.tensor([1], dtype=torch.float32, device=device)


def get_wgrad_sign_vector(device: int) -> torch.Tensor:
    """
    Hard-coded random signs for Hadamard transform.

    https://xkcd.com/221/

    """
    return torch.tensor(
        [1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1],
        dtype=torch.float32,
        device=device,
    )


def get_hadamard_matrix(hadamard_dimension: int, device: int) -> torch.Tensor:
    """Construct a 16x16 Hadamard matrix."""
    assert hadamard_dimension == 16, "Only hadamard dimension 16 is supported."  # noqa: S101, PLR2004
    hadamard_scale = 1 / math.sqrt(hadamard_dimension)
    return (
        torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
                [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
                [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
                [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
                [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
                [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
                [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
                [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
                [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        * hadamard_scale
    )


@functools.cache
def get_rht_matrix(
    *,
    with_random_sign_mask: bool = True,
    device: str | int = "cuda",
) -> torch.Tensor:
    """Construct matrix used in random Hadamard transform."""
    hadamard_dimension = 16
    if with_random_sign_mask:
        signs = get_wgrad_sign_vector(device=device)
    else:
        signs = get_no_random_sign_vector(device=device)
    sign_matrix = signs * torch.eye(
        hadamard_dimension,
        dtype=torch.float32,
        device=device,
    )
    rht_matrix = sign_matrix @ get_hadamard_matrix(hadamard_dimension, device=device)
    return rht_matrix.to(dtype=torch.bfloat16)
