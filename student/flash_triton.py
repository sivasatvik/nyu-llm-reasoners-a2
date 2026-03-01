from __future__ import annotations

import torch, triton, os
import triton.language as tl
import math
from einops import rearrange, einsum


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal : tl.constexpr
    ):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
    Q_ptr + batch_index * stride_qb,
    shape=(N_QUERIES, D),
    strides=(stride_qq, stride_qd),
    offsets=(query_tile_index * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
    O_ptr + batch_index * stride_ob,
    shape=(N_QUERIES, D),
    strides=(stride_oq, stride_od),
    offsets=(query_tile_index * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
    L_ptr + batch_index * stride_lb,
    shape=(N_QUERIES,),
    strides=(stride_lq,),
    offsets=(query_tile_index * Q_TILE_SIZE,),
    block_shape=(Q_TILE_SIZE,),
    order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
    K_ptr + batch_index * stride_kb,
    shape=(N_KEYS, D),
    strides=(stride_kk, stride_kd),
    offsets=(0, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
    V_ptr + batch_index * stride_vb,
    shape=(N_KEYS, D),
    strides=(stride_vk, stride_vd),
    offsets=(0, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(1, 0),
    )

    m_i = tl.full((Q_TILE_SIZE,), value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    Q_i = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    idx_q_base = tl.arange(0, Q_TILE_SIZE)
    idx_k_base = tl.arange(0, K_TILE_SIZE)
    idx_q = idx_q_base + Q_TILE_SIZE*query_tile_index
    mask_scale = -1e6

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0,1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0,1), padding_option="zero")
        S = tl.dot(Q_i,tl.trans(K_j))*scale
        if is_causal:
            idx_k = idx_k_base + j*K_TILE_SIZE
            mask = mask_scale*(idx_k[None,:] > idx_q[:,None])
            S += mask
        m_i_new = tl.maximum(m_i, tl.max(S, axis=-1))
        tildeP = tl.exp(S - m_i_new[:, None])
        exp_scale = tl.exp(m_i - m_i_new)
        l_i = exp_scale*l_i + tl.sum(tildeP, axis=-1)
        O_i = O_i*exp_scale[:, None]
        O_i += tl.dot(tildeP.to(V_j.dtype), V_j)
        m_i = m_i_new
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    l_i = m_i + tl.log(l_i)
    tl.store(O_block_ptr, O_i, boundary_check=(0,1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))


@torch.compile
def flash_attention_backward(Q, K, V, L, O, dO, sqrt_d, is_causal):
    D = (O * dO).sum(dim=-1, keepdim=True)
    S = einsum(Q, K, "... i d, ... j d -> ... i j")/sqrt_d
    if is_causal:
        i = torch.arange(S.shape[-2], device=S.device)
        mask = i[None, :] > i[:, None]
        S = S.masked_fill(mask, float("-inf"))
    P = torch.exp(S - L.unsqueeze(-1))
    dV = einsum(P, dO, "... i j, ... i d -> ... j d")
    dP = einsum(dO, V, "... i d, ... j d -> ... i j")
    dS = P * (dP - D)
    dQ = einsum(dS, K, "... a b, ... b c-> ... a c")/sqrt_d
    dK = einsum(dS, Q, "... a b, ... a d -> ... b d")/sqrt_d

    return dQ, dK, dV, None


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q : torch.Tensor ,K : torch.Tensor,V : torch.Tensor, is_causal : tl.constexpr=False):
        B_q = B_k = 16
        # B_k = 1
        T_q = triton.cdiv(Q.shape[-2], B_q)
        d = Q.shape[-1]
        device = "cuda"
        # O_i = torch.empty(*Q.shape[:-2],B_q, d, device= device)
        O = torch.empty(Q.shape, device=Q.device)
        L = torch.empty(Q.shape[:-1], device=Q.device)
        batch_size = Q.shape[0]
        N_QUERIES = Q.shape[-2]
        N_KEYS = K.shape[-2]
        scale = 1/math.sqrt(d)
        Q_TILE_SIZE = B_q
        K_TILE_SIZE = B_k
        D = d
        stride_qb = Q.stride(0)
        stride_qq = Q.stride(1)
        stride_qd = Q.stride(2)
        stride_kb = K.stride(0)
        stride_kk = K.stride(1)
        stride_kd = K.stride(2)
        stride_vb = V.stride(0)
        stride_vk = V.stride(1)
        stride_vd = V.stride(2)
        stride_ob = O.stride(0)
        stride_oq = O.stride(1)
        stride_od = O.stride(2)
        stride_lb = L.stride(0)
        stride_lq = L.stride(1)
        flash_fwd_kernel[(T_q, batch_size)](
            Q, K, V,
            O, L,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal
            )
        ctx.save_for_backward(Q,K,V,L,O)
        ctx.is_causal = is_causal
        ctx.sqrt_d = math.sqrt(d)
        return O
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L, O = ctx.saved_tensors
        return flash_attention_backward(Q, K, V, L, O, dO, ctx.sqrt_d, ctx.is_causal)