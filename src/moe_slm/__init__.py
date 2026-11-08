from .model import MoESLM
from .dataset import Dataset_Mixed
from .tokenizer import BPETokenizer, train_bpe_on_mix

def main():
    mix = Dataset_Mixed(testing=True, total_tokens_b=1.0)

    tok = train_bpe_on_mix(
        mix,
        vocab_size=50000,
        tokens_b=1.0,
        cache_name="smol-data-1b"
    )

    tok.save_pretrained("tokenizers/smol-data-1b-50k")
