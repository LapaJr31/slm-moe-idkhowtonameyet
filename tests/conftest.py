import sys
from pathlib import Path

import pytest
import torch

# make `src/` importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_slm.model import MoESLM  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tiny config: exercises every code path (GQA, MoE, tied embeddings) in <1s
TINY = dict(
    vocab_size=256,
    d_model=64,
    n_layers=2,
    n_q_heads=4,
    n_kv_heads=2,
    num_experts=4,
    d_ffn=48,
    top_k=2,
    max_seq_len=256,
)


def tiny_model(**overrides) -> MoESLM:
    kwargs = {**TINY, **overrides}
    m = MoESLM(vocab_size=kwargs.pop("vocab_size"), **kwargs)
    return m.to(DEVICE)


def random_batch(batch_size=2, seqlen=32, vocab=256):
    ids = torch.randint(1, vocab, (batch_size, seqlen), device=DEVICE)
    return {"input_ids": ids}


@pytest.fixture(scope="session")
def device():
    return DEVICE
