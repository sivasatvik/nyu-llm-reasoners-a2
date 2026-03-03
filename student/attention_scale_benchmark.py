from __future__ import annotations

import argparse
import math
import statistics
import timeit
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from student.utils import annotated_scaled_dot_product_attention


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


@dataclass
class Row:
    d_model: int
    seq_len: int
    status: str
    forward_mean_ms: float | None
    forward_std_ms: float | None
    memory_after_forward_before_backward_warmup_mib: float | None
    backward_mean_ms: float | None
    backward_std_ms: float | None
    qkv_mib: float
    scores_mib: float
    probs_mib: float
    output_mib: float
    saved_backward_est_mib: float


def _mib_from_bytes(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def _estimate_memory_mib(batch_size: int, seq_len: int, d_model: int, bytes_per_elem: int) -> tuple[float, float, float, float, float]:
    qkv_bytes = 3 * batch_size * seq_len * d_model * bytes_per_elem
    scores_bytes = batch_size * seq_len * seq_len * bytes_per_elem
    probs_bytes = batch_size * seq_len * seq_len * bytes_per_elem
    output_bytes = batch_size * seq_len * d_model * bytes_per_elem

    # Rough estimate of tensors saved for backward in standard attention:
    # Q, K, V, probabilities, and output.
    saved_backward_bytes = qkv_bytes + probs_bytes + output_bytes

    return (
        _mib_from_bytes(qkv_bytes),
        _mib_from_bytes(scores_bytes),
        _mib_from_bytes(probs_bytes),
        _mib_from_bytes(output_bytes),
        _mib_from_bytes(saved_backward_bytes),
    )


def _to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except ModuleNotFoundError:
        return df.to_string(index=False)


def run(args: argparse.Namespace) -> pd.DataFrame:
    if args.device != "cuda":
        raise ValueError("This benchmark is designed for CUDA runs.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device(args.device)
    dtype = torch.float32
    bytes_per_elem = torch.finfo(dtype).bits // 8

    # Eagerly initialize CUDA primary context + cuBLAS handle to avoid lazy-init warnings.
    torch.cuda.set_device(device)
    torch.cuda.init()
    # _ctx_a = torch.randn(1, 1, device=device, dtype=dtype)
    # _ctx_b = torch.randn(1, 1, device=device, dtype=dtype)
    # _ = _ctx_a @ _ctx_b
    # torch.cuda.synchronize(device)

    d_models = _parse_int_list(args.d_models)
    seq_lens = _parse_int_list(args.seq_lens)

    rows: list[Row] = []

    for d_model in d_models:
        for seq_len in seq_lens:
            qkv_mib, scores_mib, probs_mib, output_mib, saved_backward_est_mib = _estimate_memory_mib(
                args.batch_size,
                seq_len,
                d_model,
                bytes_per_elem,
            )

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            try:
                Q = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                K = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                V = torch.randn(args.batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

                # Warmup forward
                for _ in range(args.warmup_steps):
                    _ = annotated_scaled_dot_product_attention(Q, K, V)
                    torch.cuda.synchronize(device)

                forward_times_ms: list[float] = []
                for _ in range(args.iters):
                    start = timeit.default_timer()
                    _ = annotated_scaled_dot_product_attention(Q, K, V)
                    torch.cuda.synchronize(device)
                    end = timeit.default_timer()
                    forward_times_ms.append((end - start) * 1000.0)

                torch.cuda.synchronize(device)
                memory_after_forward_before_backward_warmup_mib = torch.cuda.memory_allocated(device) / (1024**2)

                # Warmup backward
                for _ in range(args.warmup_steps):
                    if Q.grad is not None:
                        Q.grad = None
                        K.grad = None
                        V.grad = None
                    out = annotated_scaled_dot_product_attention(Q, K, V)
                    grad_out = torch.randn_like(out)
                    out.backward(grad_out)
                    torch.cuda.synchronize(device)

                backward_times_ms: list[float] = []
                for _ in range(args.iters):
                    if Q.grad is not None:
                        Q.grad = None
                        K.grad = None
                        V.grad = None

                    out = annotated_scaled_dot_product_attention(Q, K, V)
                    torch.cuda.synchronize(device)

                    grad_out = torch.randn_like(out)
                    start = timeit.default_timer()
                    out.backward(grad_out)
                    torch.cuda.synchronize(device)
                    end = timeit.default_timer()
                    backward_times_ms.append((end - start) * 1000.0)

                rows.append(
                    Row(
                        d_model=d_model,
                        seq_len=seq_len,
                        status="ok",
                        forward_mean_ms=statistics.mean(forward_times_ms),
                        forward_std_ms=statistics.pstdev(forward_times_ms) if len(forward_times_ms) > 1 else 0.0,
                        memory_after_forward_before_backward_warmup_mib=memory_after_forward_before_backward_warmup_mib,
                        backward_mean_ms=statistics.mean(backward_times_ms),
                        backward_std_ms=statistics.pstdev(backward_times_ms) if len(backward_times_ms) > 1 else 0.0,
                        qkv_mib=qkv_mib,
                        scores_mib=scores_mib,
                        probs_mib=probs_mib,
                        output_mib=output_mib,
                        saved_backward_est_mib=saved_backward_est_mib,
                    )
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                rows.append(
                    Row(
                        d_model=d_model,
                        seq_len=seq_len,
                        status="oom",
                        forward_mean_ms=None,
                        forward_std_ms=None,
                        memory_after_forward_before_backward_warmup_mib=None,
                        backward_mean_ms=None,
                        backward_std_ms=None,
                        qkv_mib=qkv_mib,
                        scores_mib=scores_mib,
                        probs_mib=probs_mib,
                        output_mib=output_mib,
                        saved_backward_est_mib=saved_backward_est_mib,
                    )
                )

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention scaling benchmark (single-head; batch fixed to 8 by default)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-models", type=str, default="16,32,64,128")
    parser.add_argument("--seq-lens", type=str, default="256,1024,4096,8192,16384")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--csv-out", type=str, default="benchmark_results/attention_scale_benchmark.csv")
    parser.add_argument("--markdown-out", type=str, default="benchmark_results/attention_scale_benchmark.md")
    args = parser.parse_args()

    df = run(args)

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    markdown = _to_markdown(df)
    md_path = Path(args.markdown_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown + "\n", encoding="utf-8")

    print("=== Attention Scale Benchmark Results ===")
    print(markdown)
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved Markdown to: {md_path}")


if __name__ == "__main__":
    main()
