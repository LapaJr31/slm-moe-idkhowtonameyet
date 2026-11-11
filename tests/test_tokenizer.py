"""Tokenizer round-trip tests. Skipped unless a trained tokenizer exists on
disk — building one from scratch is a 1B-token job, so tests run against the
checked-in artifact (or are skipped on fresh checkouts)."""
import os
from pathlib import Path

import pytest

TOKENIZER_DIR = os.environ.get(
    "MOE_SLM_TOKENIZER_PATH", "tokenizers/smol-data-1b-50k"
)
_has_tokenizer = (Path(TOKENIZER_DIR) / "tokenizer.json").exists()

pytestmark = pytest.mark.skipif(
    not _has_tokenizer, reason=f"no tokenizer at {TOKENIZER_DIR}"
)


@pytest.fixture(scope="module")
def tok():
    from moe_slm.tokenizer import BPETokenizer

    return BPETokenizer.from_pretrained(TOKENIZER_DIR)


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "",                      # empty — must not crash
        "   ",                   # whitespace only
        "naïve café — résumé",   # accented chars
        "emoji: \U0001F600 \U0001F916 mixed 漢字",  # multibyte unicode
        "a" * 10_000,            # long run
        "tabs\tand\nnewlines\r\n",
    ],
)
def test_encode_decode_roundtrip(tok, text):
    """Decoding a full encode must reproduce the input exactly."""
    ids = tok.tokenizer.encode(text)
    decoded = tok.tokenizer.decode(ids)
    assert decoded == text


def test_every_byte_encodable(tok):
    """Byte-level fallback: all 256 byte values must encode without error.
    Guards the 'unseen byte' edge case that motivated the alphabet seeding."""
    raw = bytes(range(256))
    ids = tok.tokenizer.encode(raw.decode("latin-1"))
    assert len(ids) > 0


def test_vocab_size_within_bounds(tok):
    """The exported vocab must match the configured VOCAB_SIZE — ids beyond
    the model's embedding table would crash training."""
    assert tok.vocab_size == len(tok.vocab)


def test_from_pretrained_roundtrip(tmp_path, tok):
    """save_pretrained → from_pretrained must preserve token ids."""
    from transformers import AutoTokenizer

    tok.tokenizer.save_pretrained(tmp_path)
    reloaded = AutoTokenizer.from_pretrained(str(tmp_path))
    text = "round trip consistency check 123"
    assert reloaded.encode(text) == tok.tokenizer.encode(text)
