from __future__ import annotations

import torch
import math
from einops import einsum


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q : torch.Tensor, K : torch.Tensor, V : torch.Tensor, is_causal=False):
        B_q = B_k = 16
        T_q = math.ceil(Q.shape[-2] // B_q)
        T_k = math.ceil(K.shape[-2] // B_k)
        d = Q.shape[-1]
        device = "cpu"
        O_i = torch.empty(*Q.shape[:-2],B_q, d, device= device)
        O = torch.empty(Q.shape, device=device)
        L = torch.empty(Q.shape[:-1], device=device)
        l_i = torch.empty(*Q.shape[:-2], B_q, device = device)
        m_i = torch.empty(*Q.shape[:-2], B_q, device = device)
        sqrt_d = math.sqrt(d)
        for i in range(T_q):
            offset_i = i*B_q
            Q_i = Q[:,offset_i:offset_i + B_q, :]
            O_i.zero_()
            l_i.zero_()
            m_i.fill_(float('-inf'))
            for j in range(T_k):
                offset_j = j*B_k
                K_j = K[:,offset_j:offset_j + B_k, :]
                V_j = V[:,offset_j:offset_j + B_k, :]
                S = einsum(Q_i, K_j, "... i d, ... j d -> ... i j")/sqrt_d
                m_i_new = torch.max(m_i, torch.max(S, dim = -1)[0])
                tildeP = torch.exp(S - m_i_new.unsqueeze(-1))
                l_i = torch.exp(m_i - m_i_new)*l_i + torch.sum(tildeP, dim=-1)
                O_i = einsum(torch.exp(m_i - m_i_new), O_i, "... a, ... a b-> ... a b") 
                O_i += einsum(tildeP, V_j, "... a b, ... b d -> ... a d")
                m_i = m_i_new
            O_i = einsum(l_i.reciprocal(), O_i, "... a, ... a b -> ... a b")
            O[:, offset_i:offset_i + B_q,:] = O_i
            L[:, offset_i:offset_i + B_q] = m_i + torch.log(l_i)
        ctx.save_for_backward(Q,K,V,L,O)
        ctx.is_causal = is_causal
        ctx.B_q = B_q
        ctx.B_k = B_k
        ctx.sqrt_d = sqrt_d
        ctx.T_q = T_q
        ctx.T_k = T_k
        ctx.d = d
        return O

    def backward(ctx, dO):
        Q,K,V,L,O = ctx.saved_tensors
        D = (O * dO).sum(dim=-1, keepdim=True)
        S = einsum(Q, K, "... i d, ... j d -> ... i j")/ctx.sqrt_d
        P = torch.exp(S - L.unsqueeze(-1))
        dV = einsum(P, dO, "... i j, ... i d -> ... j d")
        dP = einsum(dO, V, "... i d, ... j d -> ... i j")
        dS = P * (dP - D)
        dQ = einsum(dS, K, "... a b, ... b c-> ... a c")/ctx.sqrt_d
        dK = einsum(dS, Q, "... a b, ... a d -> ... b d")/ctx.sqrt_d

        return dQ, dK, dV, None
