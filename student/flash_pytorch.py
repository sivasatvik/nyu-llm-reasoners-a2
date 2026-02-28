from __future__ import annotations

import torch
from einops import einsum

from student.flash_backward import flash_attention_backward


def flash_attention2_forward_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
    q_tile_size: int = 16,
    k_tile_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q_tile_size < 16 or k_tile_size < 16:
        raise ValueError("Tile sizes must be at least 16.")

    batch_size, n_queries, d = Q.shape
    n_keys = K.shape[-2]
    scale = d ** -0.5

    O = torch.zeros_like(Q)
    L = torch.empty((batch_size, n_queries), device=Q.device, dtype=torch.float32)

    for q_start in range(0, n_queries, q_tile_size):
        q_end = q_start + q_tile_size

        q_i = Q[:, q_start:q_end, :]

        m_i = torch.full((batch_size, q_tile_size), -float("inf"), device=Q.device, dtype=torch.float32)
        l_i = torch.zeros((batch_size, q_tile_size), device=Q.device, dtype=torch.float32)
        o_i = torch.zeros((batch_size, q_tile_size, d), device=Q.device, dtype=torch.float32)

        for k_start in range(0, n_keys, k_tile_size):
            k_end = k_start + k_tile_size

            k_j = K[:, k_start:k_end, :]
            v_j = V[:, k_start:k_end, :]

            s_ij = einsum(q_i, k_j, "b q d, b k d -> b q k") * scale
            s_ij = s_ij.to(torch.float32)

            if is_causal:
                q_idx = torch.arange(q_start, q_end, device=Q.device)
                k_idx = torch.arange(k_start, k_end, device=Q.device)
                causal_mask = q_idx[:, None] >= k_idx[None, :]
                s_ij = torch.where(causal_mask[None, :, :], s_ij, torch.full_like(s_ij, -1e6))

            m_ij = torch.max(s_ij, dim=-1).values
            p_ij = torch.exp(s_ij - m_ij[..., None])
            l_ij = torch.sum(p_ij, dim=-1)

            m_new = torch.maximum(m_i, m_ij)
            alpha = torch.exp(m_i - m_new)
            beta = torch.exp(m_ij - m_new)
            l_new = alpha * l_i + beta * l_ij

            pv = einsum(p_ij.to(v_j.dtype), v_j, "b q k, b k d -> b q d").to(torch.float32)
            o_i = (alpha * l_i / l_new)[..., None] * o_i + (beta / l_new)[..., None] * pv

            m_i = m_new
            l_i = l_new

        O[:, q_start:q_end, :] = o_i.to(Q.dtype)
        L[:, q_start:q_end] = m_i + torch.log(l_i)

    return O, L


class FlashAttention2AutogradFunctionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        O, L = flash_attention2_forward_pytorch(Q, K, V, is_causal=is_causal)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_attention_backward(Q, K, V, O, dO, L, bool(ctx.is_causal))
        return dQ, dK, dV, None
