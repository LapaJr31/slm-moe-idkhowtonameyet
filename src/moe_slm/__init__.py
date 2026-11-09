import os

from transformers import AutoTokenizer

from .model import MoESLM
from .dataset import Dataset_Mixed
from .tokenizer import BPETokenizer, train_bpe_on_mix

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
            vocab_size=50000,
            tokens_b=1.0,
            cache_name="smol-data-1b",
        )
        bpe.save_pretrained(out)

    tok = AutoTokenizer.from_pretrained(out)
    ds = mix.tokenize(mix.dataset, tok)
    print(f"tokenized dataset ready: {len(ds):,} x {1024}-token rows")
