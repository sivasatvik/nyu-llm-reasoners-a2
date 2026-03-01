from __future__ import annotations

from typing import Type

import torch

import student.benchmark as benchmark
from student.flash_pytorch import FlashAttention2AutogradFunctionPytorch
from student.flash_triton import FlashAttention2AutogradFunctionTriton
from student.flashattention import FlashAttentionPytorch, FlashAttention



def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    return FlashAttentionPytorch


def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and (optionally) backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    return FlashAttention


