import itertools
import hashlib
import os

from datasets import load_dataset, interleave_datasets, Dataset
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer
import pyarrow.parquet as pq
from .tokenizer import BPETokenizer
from pathlib import Path


def _parquet_text_stream(name: str, text_column: str = "text", batch_rows: int = 1024):
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{name}/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found in dataset {name}")
    for f in files:
        with fs.open(f) as fh:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(batch_size=batch_rows, columns=[text_column]):
                yield from batch.column(text_column).to_pylist()


def _interleave_streams(sources, probabilities, seed=42):
    import random

    rng = random.Random(seed)
    iters = [iter(s) for s in sources]
    while iters:
        it = rng.choices(iters, weights=probabilities, k=1)[0]
        try:
            yield next(it)
        except StopIteration:
            i = iters.index(it)
            iters.pop(i)
            probabilities.pop(i)


def load_stream(name: str, text_column: str = "text", buffer_size: int = 10000):
    def shuffled():
        import random

        rng = random.Random(42)
        buf = []
        for text in _parquet_text_stream(name, text_column):
            if len(buf) < buffer_size:
                buf.append(text)
            else:
                i = rng.randrange(buffer_size)
                buf[i], text = text, buf[i]
                yield {"text": text}
        for text in buf:
            yield {"text": text}

    return shuffled()


def _make_token_counter(tokenizer, text_stream, text_column, bpe_vocab_size, bpe_train_docs):
    if isinstance(tokenizer, BPETokenizer):
        bpe = tokenizer

        def count(texts):
            return [len(bpe.tokenize(t)) for t in texts]

        return count

    if tokenizer == "bpe":
        sample = list(itertools.islice(text_stream, bpe_train_docs))
        bpe = BPETokenizer(sample, vocab_size=bpe_vocab_size)

        def count(texts):
            return [len(bpe.tokenize(t)) for t in texts]

        return count

    hf = AutoTokenizer.from_pretrained(tokenizer)

    def count(texts):
        return [len(ids) for ids in hf(texts, add_special_tokens=False)["input_ids"]]

    return count


def _token_count_budget(
    name: str,
    tokens: int,
    tokenizer,
    text_column: str = "text",
    batch_size: int = 512,
    bpe_vocab_size: int = 50000,
    bpe_train_docs: int = 5000,
):
    def gen():
        text_stream = _parquet_text_stream(name, text_column)
        count = _make_token_counter(
            tokenizer, text_stream, text_column, bpe_vocab_size, bpe_train_docs
        )
        counted = 0
        done = False
        while not done:
            texts = list(itertools.islice(text_stream, batch_size))
            if not texts:
                break
            lengths = count(texts)
            for text, n in zip(texts, lengths):
                yield {"text": text}
                counted += n
                if counted >= tokens:
                    done = True
                    break

    return gen


def tokenize_to_disk(
    mix,
    tokenizer,
    path,
    seq_len=1024,
    text_column="text",
    batch_size=512,
    add_eos=True,
):
    if os.path.exists(path):
        print(f"tokenized dataset already exists at {path}, loading it")
        return Dataset.load_from_disk(path)

    eos_id = tokenizer.eos_token_id if add_eos else None

    def gen():
        buf = []
        it = iter(mix)
        while True:
            batch = list(itertools.islice(it, batch_size))
            if not batch:
                break
            texts = [ex[text_column] if isinstance(ex, dict) else ex for ex in batch]
            enc = tokenizer(texts, add_special_tokens=False)["input_ids"]
            for ids in enc:
                if eos_id is not None:
                    buf.extend(ids)
                    buf.append(eos_id)
                else:
                    buf.extend(ids)
                while len(buf) >= seq_len:
                    yield {"input_ids": buf[:seq_len]}
                    buf = buf[seq_len:]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ds = Dataset.from_generator(gen)
    ds.save_to_disk(path)
    return ds


def _tokenizer_cache_key(tokenizer, bpe_vocab_size, bpe_train_docs):
    if isinstance(tokenizer, BPETokenizer):
        digest = hashlib.blake2b(
            "\n".join(tokenizer.vocab).encode(), digest_size=16
        ).hexdigest()
        return f"bpe-inst|{tokenizer.vocab_size}|{digest}"
    if tokenizer == "bpe":
        return f"bpe|{bpe_vocab_size}|{bpe_train_docs}"
    return f"hf|{tokenizer}"


def load_exact(
    name: str,
    tokens: int,
    tokenizer,
    text_column: str = "text",
    seed: int = 42,
    bpe_vocab_size: int = 50000,
    bpe_train_docs: int = 5000,
):
    cache_key = _tokenizer_cache_key(tokenizer, bpe_vocab_size, bpe_train_docs)
    cache_dir = os.path.join(
        ".cache", "moe_slm_test_mixes",
        hashlib.blake2b(f"{name}|{tokens}|{cache_key}".encode(), digest_size=16).hexdigest(),
    )

    if os.path.exists(cache_dir):
        ds = Dataset.load_from_disk(cache_dir)
    else:
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        ds = Dataset.from_generator(
            _token_count_budget(
                name, tokens, tokenizer, text_column,
                bpe_vocab_size=bpe_vocab_size, bpe_train_docs=bpe_train_docs,
            )
        )
        ds.save_to_disk(cache_dir)
    return ds.shuffle(seed=seed)


class Dataset_Mixed:
    def __init__(
        self,
        datasets: list[str] = [
            "HuggingFaceFW/dclm_100BT",
            "HuggingFaceFW/finepdfs_100BT",
            "HuggingFaceFW/finepdfs_edu_100BT",
            "HuggingFaceFW/fineweb_100BT",
            "HuggingFaceFW/fineweb_edu_100BT",
        ],
        split: str = "train",
        seed=42,
        testing=False,
        total_tokens_b=1.0,
        probabilities=[0.35, 0.25, 0.20, 0.1, 0.1],
        tokenizer=None,
        text_column: str = "text",
    ):
        self.datasets = datasets
        self.testing = testing
        self.seed = seed

        if testing:
            self.tokenizer = tokenizer if tokenizer is not None else "bpe"
            total_tokens = int(total_tokens_b * 1e9)
            if len(datasets) > 1:
                assert len(probabilities) == len(datasets)
                budgets = [int(total_tokens * p) for p in probabilities]
                sources = [
                    load_exact(x, budget, self.tokenizer, text_column, seed)
                    for x, budget in zip(datasets, budgets)
                ]
                mix = interleave_datasets(
                    sources,
                    probabilities=probabilities,
                )
            else:
                mix = load_exact(
                    datasets[0], total_tokens, self.tokenizer, text_column, seed
                )
        else:
            if len(datasets) > 1:
                assert len(probabilities) == len(datasets)
                sources = [load_stream(x, text_column) for x in datasets]
                mix = _interleave_streams(sources, list(probabilities), seed)
            else:
                mix = load_stream(datasets[0], text_column)

        self.dataset = mix

    def tokenize(self, dataset, tokenizer, path = Path(__file__).resolve().parents[2] / "dataset"):
        dataset = self.dataset
        tokenizer = self.tokenizer

        ds = tokenize_to_disk(dataset, tokenizer, path=path)

    def __iter__(self):
        return iter(self.dataset)
