import torch.cuda.nvtx as nvtx
from jaxtyping import Float, Bool, Int
from torch import Tensor
import torch
import math
from einops import rearrange, einsum
import importlib
import sys
from pathlib import Path


def _load_nn_utils():
	try:
		return importlib.import_module("a1_basics.nn_utils")
	except ModuleNotFoundError:
		repo_root = Path(__file__).resolve().parent.parent
		local_basics_src = repo_root / "a1-basics"
		if local_basics_src.exists():
			sys.path.insert(0, str(local_basics_src))
			return importlib.import_module("a1_basics.nn_utils")
		raise


nn_utils = _load_nn_utils()

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
	"""Scaled dot-product attention.

	This function implements Eq. 1 of the Transformer paper.

	Args
	    Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """
	d_k = K.shape[-1]
	with nvtx.range("computing attention scores"):
		attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

	if mask is not None:
		attention_scores = torch.where(mask, attention_scores, float("-inf"))

	with nvtx.range("computing softmax"):
		attention_weights = nn_utils.softmax(attention_scores, dim=-1)  # Softmax over the key dimension

	with nvtx.range("final matmul"):
		X = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

	return X