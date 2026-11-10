import os

from pathlib import Path

from transformers import AutoTokenizer

from .model import MoESLM
from .dataset import Dataset_Mixed
from .tokenizer import BPETokenizer, train_bpe_on_mix
from .train import preTrain

VOCAB_SIZE = 50000
SEQLEN = preTrain.SEQLEN


def main():
    mix = Dataset_Mixed(
        testing=True,
        total_tokens_b=1.0,
        tokenizer="HuggingFaceTB/SmolLM2-1.7B",
    )

    out = "tokenizers/smol-data-1b-50k"
    if os.path.exists(os.path.join(out, "tokenizer.json")):
        print(f"tokenizer already exists at {out}, loading it")
    else:
        bpe = train_bpe_on_mix(
            mix,
            vocab_size=VOCAB_SIZE,
            tokens_b=1.0,
            cache_name="smol-data-1b",
        )
        bpe.save_pretrained(out)

    ds = Path("dataset/")
    if ds.exists():
        print("Dataset exists")
    else:
        tok = AutoTokenizer.from_pretrained(out)
        ds = mix.tokenize(mix.dataset, tok)
        print(f"tokenized dataset ready: {len(ds):,} x {SEQLEN}-token rows")

    latest_ckpt = Path("checkpoints/latest.pt")
    if latest_ckpt.exists():
        print(f"resuming from checkpoint {latest_ckpt}")

    model = MoESLM(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        n_layers=6,
        n_q_heads=8,
        n_kv_heads=2,
        num_experts=4,
        d_ffn=1024,
        top_k=2,
    )
    trainer = preTrain(ds, model, run_type="dry_run")
    if latest_ckpt.exists():
        trainer.load_checkpoint(latest_ckpt)
    trainer.train()
