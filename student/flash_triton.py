# pyright: reportMissingImports=false, reportInvalidTypeForm=false
from __future__ import annotations

import torch

from student.flash_backward import flash_attention_backward
triton = None
tl = None
_flash_fwd_kernel = None


def _ensure_triton_loaded() -> None:
    global triton, tl
    if triton is not None and tl is not None:
        return
    try:
        import triton as _triton
        import triton.language as _tl
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Triton is required for FlashAttention2AutogradFunctionTriton. "
            "Install it in the same environment used by pytest, e.g. `uv pip install triton`."
        ) from exc

    triton = _triton
    tl = _tl


def _get_flash_fwd_kernel():
    global _flash_fwd_kernel
    if _flash_fwd_kernel is not None:
        return _flash_fwd_kernel

    _ensure_triton_loaded()

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        L_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        N_QUERIES,
        N_KEYS,
        scale,
        is_causal: tl.constexpr,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        q_start = query_tile_index * Q_TILE_SIZE

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr)

        m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        for k_start in range(0, N_KEYS, K_TILE_SIZE):
            K_block_ptr = tl.make_block_ptr(
                K_ptr + batch_index * stride_kb,
                shape=(N_KEYS, D),
                strides=(stride_kk, stride_kd),
                offsets=(k_start, 0),
                block_shape=(K_TILE_SIZE, D),
                order=(1, 0),
            )
            V_block_ptr = tl.make_block_ptr(
                V_ptr + batch_index * stride_vb,
                shape=(N_KEYS, D),
                strides=(stride_vk, stride_vd),
                offsets=(k_start, 0),
                block_shape=(K_TILE_SIZE, D),
                order=(1, 0),
            )

            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)

            s_ij = tl.dot(q, tl.trans(k)) * scale

            if is_causal:
                q_idx = q_start + tl.arange(0, Q_TILE_SIZE)[:, None]
                k_idx = k_start + tl.arange(0, K_TILE_SIZE)[None, :]
                causal_mask = q_idx >= k_idx
                s_ij = tl.where(causal_mask, s_ij, -1e6)

            m_ij = tl.max(s_ij, axis=1)
            p_ij = tl.exp(s_ij - m_ij[:, None])
            l_ij = tl.sum(p_ij, axis=1)

            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            l_new = alpha * l_i + beta * l_ij

            p_cast = p_ij.to(v.dtype)
            pv = tl.dot(p_cast, v, acc=tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32))

            o_i = (alpha * l_i / l_new)[:, None] * o_i + (beta / l_new)[:, None] * pv

            m_i = m_new
            l_i = l_new

        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty))

        l_ptrs = L_ptr + batch_index * stride_lb + q_start * stride_lq + tl.arange(0, Q_TILE_SIZE) * stride_lq
        tl.store(l_ptrs, m_i + tl.log(l_i))

    _flash_fwd_kernel = flash_fwd_kernel
    return _flash_fwd_kernel


class FlashAttention2AutogradFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        _ensure_triton_loaded()

        if not Q.is_cuda:
            raise ValueError("FlashAttention2AutogradFunctionTriton expects CUDA tensors.")

        batch_size, n_queries, d = Q.shape
        n_keys = K.shape[-2]

        q_tile_size = 16
        k_tile_size = 16

        O = torch.empty_like(Q)
        L = torch.empty((batch_size, n_queries), device=Q.device, dtype=torch.float32)
        flash_fwd_kernel = _get_flash_fwd_kernel()

        grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            n_queries,
            n_keys,
            d ** -0.5,
            is_causal=bool(is_causal),
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_attention_backward(Q, K, V, O, dO, L, bool(ctx.is_causal))
        return dQ, dK, dV, None
