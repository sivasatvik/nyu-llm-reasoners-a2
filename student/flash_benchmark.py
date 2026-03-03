# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

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
    gpu_name: str
    impl: str
    seq_len: int
    d: int
    dtype: str
    status: str
    forward_ms: float | None
    backward_ms: float | None
    end_to_end_ms: float | None


def _to_ms(bench_result) -> float:
    if isinstance(bench_result, (tuple, list)):
        return float(bench_result[0])
    return float(bench_result)


def _bench_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    impl: str,
    is_causal: bool,
    warmup_ms: int,
    rep_ms: int,
) -> tuple[float, float, float]:
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

    fwd_ms = _to_ms(triton.testing.do_bench(lambda: fwd(), warmup=warmup_ms, rep=rep_ms, quantiles=[0.5]))

    def _bwd_wrapper():
        q.grad = None
        k.grad = None
        v.grad = None
        bwd()

    bwd_ms = _to_ms(triton.testing.do_bench(lambda: _bwd_wrapper(), warmup=warmup_ms, rep=rep_ms, quantiles=[0.5]))

    def _e2e_wrapper():
        q.grad = None
        k.grad = None
        v.grad = None
        e2e()

    e2e_ms = _to_ms(triton.testing.do_bench(lambda: _e2e_wrapper(), warmup=warmup_ms, rep=rep_ms, quantiles=[0.5]))
    return fwd_ms, bwd_ms, e2e_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Triton FlashAttention2 vs regular PyTorch attention")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--csv-out", type=str, default="benchmark_results/flash_benchmark_results.csv")
    parser.add_argument("--markdown-out", type=str, default="benchmark_results/flash_benchmark_results.md")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=65536)
    parser.add_argument("--max-d", type=int, default=128)
    args = parser.parse_args()

    if args.device != "cuda":
        raise ValueError("This benchmark is intended for a single GPU run; use --device cuda")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"Using GPU: {gpu_name}")

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

                        fwd_ms, bwd_ms, e2e_ms = _bench_impl(
                            q,
                            k,
                            v,
                            impl=impl,
                            is_causal=True,
                            warmup_ms=args.warmup_ms,
                            rep_ms=args.rep_ms,
                        )
                        rows.append(
                            BenchResult(
                                gpu_name=gpu_name,
                                impl=impl,
                                seq_len=seq_len,
                                d=d,
                                dtype=str(dtype).replace("torch.", ""),
                                status="ok",
                                forward_ms=fwd_ms,
                                backward_ms=bwd_ms,
                                end_to_end_ms=e2e_ms,
                            )
                        )
                        print(f"done: impl={impl} seq={seq_len} d={d} dtype={dtype}")
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        rows.append(
                            BenchResult(
                                gpu_name=gpu_name,
                                impl=impl,
                                seq_len=seq_len,
                                d=d,
                                dtype=str(dtype).replace("torch.", ""),
                                status="oom",
                                forward_ms=None,
                                backward_ms=None,
                                end_to_end_ms=None,
                            )
                        )
                        print(f"skip(oom): impl={impl} seq={seq_len} d={d} dtype={dtype}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    print(df)
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out, index=False)
    print(f"Saved to {args.csv_out}")
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        try:
            table = df.to_markdown(index=False, floatfmt=".4f")
        except ModuleNotFoundError:
            table = df.to_string(index=False)
        Path(args.markdown_out).write_text(table + "\n", encoding="utf-8")
        print(f"Saved to {args.markdown_out}")


if __name__ == "__main__":
    main()
