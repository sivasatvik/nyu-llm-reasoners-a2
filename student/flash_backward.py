from __future__ import annotations

import torch
from einops import einsum


def _flash_backward_impl(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = Q.shape[-1]
    scale = d ** -0.5

    S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale
    if is_causal:
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        q_idx = torch.arange(n_queries, device=Q.device)
        k_idx = torch.arange(n_keys, device=Q.device)
        causal_mask = q_idx[:, None] >= k_idx[None, :]
        S = torch.where(causal_mask[None, :, :], S, torch.full_like(S, -1e6))

    P = torch.exp(S - L[..., None])

    dV = einsum(P, dO, "... q k, ... q d -> ... k d")
    dP = einsum(dO, V, "... q d, ... k d -> ... q k")

    D = torch.sum(dO * O, dim=-1)
    dS = P * (dP - D[..., None])

    dQ = einsum(dS, K, "... q k, ... k d -> ... q d") * scale
    dK = einsum(dS, Q, "... q k, ... q d -> ... k d") * scale

    return dQ, dK, dV


_COMPILED_BACKWARD = None


def flash_attention_backward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    dO: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _COMPILED_BACKWARD

    if hasattr(torch, "compile"):
        if _COMPILED_BACKWARD is None:
            try:
                _COMPILED_BACKWARD = torch.compile(_flash_backward_impl)
            except Exception:
                _COMPILED_BACKWARD = _flash_backward_impl
        return _COMPILED_BACKWARD(Q, K, V, O, dO, L, is_causal)

    return _flash_backward_impl(Q, K, V, O, dO, L, is_causal)
