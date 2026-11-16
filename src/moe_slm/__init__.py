import os

from pathlib import Path

from transformers import AutoTokenizer
from datasets import load_from_disk

from .model import MoESLM
from .dataset import Dataset_Mixed
from .tokenizer import BPETokenizer, train_bpe_on_mix
from .train import preTrain

VOCAB_SIZE = 32768
SEQLEN = preTrain.SEQLEN
SUBSAMPLE_TOKENS = False


def main():
    out = "tokenizers/smol-data-1b-50k"
    ds_path = Path("dataset/")
    have_tokenizer = os.path.exists(os.path.join(out, "tokenizer.json"))
    have_dataset = ds_path.exists()

    if have_tokenizer and have_dataset:
        print(f"tokenizer already exists at {out}, loading it")
        print(f"loading dataset from {ds_path}")
        ds = load_from_disk(ds_path)
        if SUBSAMPLE_TOKENS:
            n_rows = int(SUBSAMPLE_TOKENS / SEQLEN)
            ds = ds.select(range(n_rows))
            print(f"subsampled to first {n_rows:,} rows (~{SUBSAMPLE_TOKENS/1e6:.0f}M tokens)")
    else:
        mix = Dataset_Mixed(
            testing=True,
            total_tokens_b=10.0,
            tokenizer="HuggingFaceTB/SmolLM2-1.7B",
            num_workers=5,
        )

        if have_tokenizer:
            print(f"tokenizer already exists at {out}, loading it")
            tok = AutoTokenizer.from_pretrained(out)
        else:
            bpe = train_bpe_on_mix(
                mix,
                vocab_size=VOCAB_SIZE,
                tokens_b=1.0,
                cache_name="smol-data-1b",
            )
            bpe.save_pretrained(out)
            tok = AutoTokenizer.from_pretrained(out)

        if have_dataset:
            print(f"loading dataset from {ds_path}")
            ds = load_from_disk(ds_path)
            if SUBSAMPLE_TOKENS:
                n_rows = int(SUBSAMPLE_TOKENS / SEQLEN)
                ds = ds.select(range(n_rows))
                print(f"subsampled to first {n_rows:,} rows (~{SUBSAMPLE_TOKENS/1e6:.0f}M tokens)")
        else:
            ds = mix.tokenize(mix.dataset, tok)
            print(f"tokenized dataset ready: {len(ds):,} x {SEQLEN}-token rows")

    latest_ckpt = Path("checkpoints/latest.pt")
    if latest_ckpt.exists():
        print(f"resuming from checkpoint {latest_ckpt}")

    model = MoESLM(
        vocab_size=VOCAB_SIZE,
        d_model=768,
        n_layers=12,
        n_q_heads=12,
        n_kv_heads=4,
        num_experts=8,
        d_ffn=960,
        top_k=2,
        attn_implementation="flex",
    )
    trainer = preTrain(ds, model, run_type="benchmarking_CE")
    if latest_ckpt.exists():
        trainer.load_checkpoint(latest_ckpt)
    trainer.train()
