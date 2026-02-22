from __future__ import annotations

import argparse
import importlib
import sys
import timeit
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


def _load_basics_transformer_lm() -> type[torch.nn.Module]:
	try:
		return importlib.import_module("a1_basics.model").BasicsTransformerLM
	except ModuleNotFoundError:
		repo_root = Path(__file__).resolve().parent.parent
		local_basics_src = repo_root / "a1-basics"
		if local_basics_src.exists():
			sys.path.insert(0, str(local_basics_src))
			return importlib.import_module("a1_basics.model").BasicsTransformerLM
		raise


BasicsTransformerLM = _load_basics_transformer_lm()

@dataclass(frozen=True)
class ModelSpec:
	d_model: int
	d_ff: int
	num_layers: int
	num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
	"small": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
	"medium": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
	"large": ModelSpec(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
	"xl": ModelSpec(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
	"2.7B": ModelSpec(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

DTYPE_CHOICES = {
	"float32": torch.float32,
	"float16": torch.float16,
	"bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Benchmark Transformer forward/backward runtime.")

	parser.add_argument("--model-size", choices=list(MODEL_SPECS), default="small")
	parser.add_argument("--d-model", type=int, default=None)
	parser.add_argument("--d-ff", type=int, default=None)
	parser.add_argument("--num-layers", type=int, default=None)
	parser.add_argument("--num-heads", type=int, default=None)

	parser.add_argument("--vocab-size", type=int, default=10000)
	parser.add_argument("--context-length", type=int, default=256)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--rope-theta", type=float, default=10_000.0)

	parser.add_argument("--mode", choices=["forward", "forward-backward"], default="forward-backward")
	parser.add_argument("--warmup-steps", type=int, default=5)
	parser.add_argument("--benchmark-steps", type=int, default=10)

	parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
	parser.add_argument("--dtype", choices=list(DTYPE_CHOICES), default="bfloat16")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--markdown-out", type=str, default=None)
	parser.add_argument("--latex-out", type=str, default=None)

	return parser.parse_args()


def _synchronize_if_cuda(device: torch.device) -> None:
	if device.type == "cuda":
		torch.cuda.synchronize(device)


def _resolve_model_spec(args: argparse.Namespace) -> ModelSpec:
	base = MODEL_SPECS[args.model_size]
	return ModelSpec(
		d_model=args.d_model if args.d_model is not None else base.d_model,
		d_ff=args.d_ff if args.d_ff is not None else base.d_ff,
		num_layers=args.num_layers if args.num_layers is not None else base.num_layers,
		num_heads=args.num_heads if args.num_heads is not None else base.num_heads,
	)


def _validate_args(args: argparse.Namespace, spec: ModelSpec, device: torch.device, dtype: torch.dtype) -> None:
	if spec.d_model % spec.num_heads != 0:
		raise ValueError(f"d_model ({spec.d_model}) must be divisible by num_heads ({spec.num_heads}).")
	if args.warmup_steps < 0:
		raise ValueError("warmup-steps must be >= 0")
	if args.benchmark_steps <= 0:
		raise ValueError("benchmark-steps must be > 0")
	if args.batch_size <= 0:
		raise ValueError("batch-size must be > 0")
	if args.context_length <= 0:
		raise ValueError("context-length must be > 0")
	if args.vocab_size <= 0:
		raise ValueError("vocab-size must be > 0")
	if device.type == "cpu" and dtype == torch.float16:
		raise ValueError("float16 on CPU is not supported for this benchmark. Use float32 or bfloat16.")


def _build_model(args: argparse.Namespace, spec: ModelSpec, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
	model = BasicsTransformerLM(
		vocab_size=args.vocab_size,
		context_length=args.context_length,
		d_model=spec.d_model,
		num_layers=spec.num_layers,
		num_heads=spec.num_heads,
		d_ff=spec.d_ff,
		rope_theta=args.rope_theta,
	)
	model.to(device=device, dtype=dtype)
	return model


def _make_random_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
	x = torch.randint(
		low=0,
		high=args.vocab_size,
		size=(args.batch_size, args.context_length),
		device=device,
		dtype=torch.long,
	)
	y = torch.randint(
		low=0,
		high=args.vocab_size,
		size=(args.batch_size, args.context_length),
		device=device,
		dtype=torch.long,
	)
	return x, y


def _single_step(
	model: torch.nn.Module,
	x: torch.Tensor,
	y: torch.Tensor,
	mode: str,
	device: torch.device,
) -> None:
	if mode == "forward":
		with torch.no_grad():
			_ = model(x)
		_synchronize_if_cuda(device)
		return

	logits = model(x)
	loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
	loss.backward()
	_synchronize_if_cuda(device)


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, float | int | str], list[float]]:
	if args.device == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA requested but no CUDA device is available.")

	device = torch.device(args.device)
	dtype = DTYPE_CHOICES[args.dtype]
	spec = _resolve_model_spec(args)
	_validate_args(args, spec, device, dtype)

	torch.manual_seed(args.seed)
	if device.type == "cuda":
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.reset_peak_memory_stats(device)

	model = _build_model(args, spec, device, dtype)
	x, y = _make_random_batch(args, device)

	model.train(args.mode == "forward-backward")

	for _ in range(args.warmup_steps):
		if args.mode == "forward-backward":
			model.zero_grad(set_to_none=True)
		_single_step(model, x, y, args.mode, device)

	step_times: list[float] = []
	for _ in range(args.benchmark_steps):
		if args.mode == "forward-backward":
			model.zero_grad(set_to_none=True)
		start = timeit.default_timer()
		_single_step(model, x, y, args.mode, device)
		end = timeit.default_timer()
		step_times.append(end - start)

	total_seconds = sum(step_times)
	mean_seconds = total_seconds / len(step_times)
	
	if len(step_times) > 1:
		variance = sum((t - mean_seconds) ** 2 for t in step_times) / (len(step_times) - 1)
		std_seconds = variance ** 0.5
	else:
		std_seconds = 0.0
	
	tokens_per_step = args.batch_size * args.context_length
	tokens_per_second = tokens_per_step / mean_seconds

	max_mem_mib = 0.0
	if device.type == "cuda":
		max_mem_mib = torch.cuda.max_memory_allocated(device) / (1024**2)

	results = {
		"device": str(device),
		"dtype": args.dtype,
		"mode": args.mode,
		"model_size": args.model_size,
		"d_model": spec.d_model,
		"d_ff": spec.d_ff,
		"num_layers": spec.num_layers,
		"num_heads": spec.num_heads,
		"batch_size": args.batch_size,
		"context_length": args.context_length,
		"warmup_steps": args.warmup_steps,
		"benchmark_steps": args.benchmark_steps,
		"mean_step_ms": mean_seconds * 1000.0,
		"std_step_ms": std_seconds * 1000.0,
		"total_time_s": total_seconds,
		"tokens_per_second": tokens_per_second,
		"max_memory_mib": max_mem_mib,
	}
	return results, [t * 1000.0 for t in step_times]


def _build_observations_table(step_times_ms: list[float]) -> pd.DataFrame:
	if len(step_times_ms) == 0:
		raise ValueError("No benchmark measurements were collected.")

	mean_step_ms = sum(step_times_ms) / len(step_times_ms)
	if len(step_times_ms) > 1:
		variance = sum((t - mean_step_ms) ** 2 for t in step_times_ms) / (len(step_times_ms) - 1)
		std_step_ms = variance ** 0.5
	else:
		std_step_ms = 0.0

	row: dict[str, float] = {}
	for idx, value in enumerate(step_times_ms, start=1):
		row[f"measurement_{idx}_ms"] = value

	row["mean_step_ms"] = mean_step_ms
	row["std_step_ms"] = std_step_ms
	return pd.DataFrame([row])


def _emit_observations_tables(observations_df: pd.DataFrame, markdown_out: str | None, latex_out: str | None) -> None:
	markdown_table = observations_df.to_markdown(index=False, floatfmt=".4f")
	latex_table = observations_df.to_latex(index=False, float_format="%.4f")

	print("\n=== Observations (Markdown) ===")
	print(markdown_table)

	print("\n=== Observations (LaTeX) ===")
	print(latex_table)

	if markdown_out:
		Path(markdown_out).write_text(markdown_table + "\n", encoding="utf-8")
	if latex_out:
		Path(latex_out).write_text(latex_table + "\n", encoding="utf-8")


def _print_results(results: dict[str, float | int | str]) -> None:
	print("=== Benchmark Results ===")
	for key, value in results.items():
		if isinstance(value, float):
			print(f"{key:>18}: {value:.4f}")
		else:
			print(f"{key:>18}: {value}")


def main() -> None:
	args = parse_args()
	results, step_times_ms = run_benchmark(args)
	_print_results(results)
	observations_df = _build_observations_table(step_times_ms)
	_emit_observations_tables(observations_df, args.markdown_out, args.latex_out)


if __name__ == "__main__":
	main()

