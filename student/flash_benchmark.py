# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
import torch
from einops import einsum

from student.flash_triton import FlashAttention

try:
    import triton
except ModuleNotFoundError:
    triton = None


def regular_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    d = q.shape[-1]
    s = einsum(q, k, "... q d, ... k d -> ... q k") * (d ** -0.5)
    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        q_idx = torch.arange(n_queries, device=q.device)
        k_idx = torch.arange(n_keys, device=q.device)
        mask = q_idx[:, None] >= k_idx[None, :]
        s = torch.where(mask[None, :, :], s, torch.full_like(s, -1e6))
    p = torch.softmax(s, dim=-1)
    return einsum(p, v, "... q k, ... k d -> ... q d")


@dataclass
class BenchResult:
    impl: str
    seq_len: int
    d: int
    dtype: str
    forward_ms: float
    backward_ms: float
    end_to_end_ms: float


def _bench_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, impl: str, is_causal: bool) -> tuple[float, float, float]:
    if triton is None:
        raise RuntimeError("Triton is required for this benchmark script because it uses triton.testing.do_bench")

    do = torch.randn_like(q)

    if impl == "triton":
        def fwd():
            FlashAttention.apply(q, k, v, is_causal)

        def bwd():
            out = FlashAttention.apply(q, k, v, is_causal)
            out.backward(do, retain_graph=False)

        def e2e():
            out = FlashAttention.apply(q, k, v, is_causal)
            out.backward(do, retain_graph=False)
    else:
        def fwd():
            regular_attention(q, k, v, is_causal)

        def bwd():
            out = regular_attention(q, k, v, is_causal)
            out.backward(do, retain_graph=False)

        def e2e():
            out = regular_attention(q, k, v, is_causal)
            out.backward(do, retain_graph=False)

    fwd_ms = triton.testing.do_bench(lambda: fwd(), quantiles=[0.5])[0]

    def _bwd_wrapper():
        q.grad = None
        k.grad = None
        v.grad = None
        bwd()

    bwd_ms = triton.testing.do_bench(lambda: _bwd_wrapper(), quantiles=[0.5])[0]

    def _e2e_wrapper():
        q.grad = None
        k.grad = None
        v.grad = None
        e2e()

    e2e_ms = triton.testing.do_bench(lambda: _e2e_wrapper(), quantiles=[0.5])[0]
    return fwd_ms, bwd_ms, e2e_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Triton FlashAttention2 vs regular PyTorch attention")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--csv-out", type=str, default="flash_benchmark_results.csv")
    parser.add_argument("--max-seq-len", type=int, default=65536)
    parser.add_argument("--max-d", type=int, default=128)
    args = parser.parse_args()

    if args.device != "cuda":
        raise ValueError("This benchmark is intended for a single GPU run; use --device cuda")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    seq_lens = [2 ** p for p in range(7, 17) if 2 ** p <= args.max_seq_len]
    d_values = [2 ** p for p in range(4, 8) if 2 ** p <= args.max_d]
    dtypes = [torch.bfloat16, torch.float32]

    rows: list[BenchResult] = []

    for seq_len in seq_lens:
        for d in d_values:
            for dtype in dtypes:
                for impl in ("pytorch", "triton"):
                    try:
                        q = torch.randn(1, seq_len, d, device="cuda", dtype=dtype, requires_grad=True)
                        k = torch.randn(1, seq_len, d, device="cuda", dtype=dtype, requires_grad=True)
                        v = torch.randn(1, seq_len, d, device="cuda", dtype=dtype, requires_grad=True)

                        fwd_ms, bwd_ms, e2e_ms = _bench_impl(q, k, v, impl=impl, is_causal=True)
                        rows.append(
                            BenchResult(
                                impl=impl,
                                seq_len=seq_len,
                                d=d,
                                dtype=str(dtype).replace("torch.", ""),
                                forward_ms=fwd_ms,
                                backward_ms=bwd_ms,
                                end_to_end_ms=e2e_ms,
                            )
                        )
                        print(f"done: impl={impl} seq={seq_len} d={d} dtype={dtype}")
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"skip(oom): impl={impl} seq={seq_len} d={d} dtype={dtype}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    print(df)
    df.to_csv(args.csv_out, index=False)
    print(f"Saved to {args.csv_out}")


if __name__ == "__main__":
    main()
