import pytest
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from moe_slm import dataset as dataset_module


@pytest.fixture(autouse=True)
def isolated_huggingface_cache(monkeypatch, tmp_path):
    import datasets.config

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(tmp_path / "hf_cache"))


def test_budgeted_sources_load_concurrently(monkeypatch):
    from threading import Barrier

    barrier = Barrier(2)
    calls = []

    def load_source(name, tokens, tokenizer, text_column, seed):
        calls.append((name, tokens, tokenizer, text_column, seed))
        barrier.wait(timeout=10)
        return Dataset.from_dict({"text": [name]})

    def combine(sources, **kwargs):
        assert [source[0]["text"] for source in sources] == ["first", "second"]
        assert kwargs["probabilities"] == [0.25, 0.75]
        return sources[0]

    monkeypatch.setattr(dataset_module, "load_exact", load_source)
    monkeypatch.setattr(dataset_module, "interleave_datasets", combine)
    dataset_module.Dataset_Mixed(
        datasets=["first", "second"], probabilities=[0.25, 0.75],
        testing=True, total_tokens_b=0.0000001, tokenizer="local",
        num_workers=2,
    )
    assert sorted(calls) == [
        ("first", 25, "local", "text", 42),
        ("second", 75, "local", "text", 42),
    ]


def test_worker_count_must_be_positive():
    with pytest.raises(ValueError, match="num_workers"):
        dataset_module.Dataset_Mixed(num_workers=0)


def test_progress_setting_restored_on_failure():
    from datasets.utils.tqdm import are_progress_bars_disabled

    before = are_progress_bars_disabled()
    with pytest.raises(RuntimeError):
        with dataset_module._quiet_dataset_progress():
            assert are_progress_bars_disabled()
            raise RuntimeError("source failed")
    assert are_progress_bars_disabled() == before


def test_count_long_documents_without_truncation_or_warning(monkeypatch, caplog):
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", model_max_length=2
    )
    monkeypatch.setattr(dataset_module.AutoTokenizer, "from_pretrained", lambda _: tokenizer)
    count = dataset_module._make_token_counter("local", iter([]), "text", 100, 10)
    assert count(["hello " * 12]) == [12]
    assert "sequence length" not in caplog.text


def test_parallel_sources_build_and_reuse_cache(monkeypatch, tmp_path):
    def texts(name, text_column="text", batch_rows=1024):
        for _ in range(10):
            yield f"{name} document"

    def counter(*args):
        return lambda batch: [2] * len(batch)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dataset_module, "_parquet_text_stream", texts)
    monkeypatch.setattr(dataset_module, "_make_token_counter", counter)
    kwargs = dict(
        datasets=["first", "second"], probabilities=[0.5, 0.5],
        testing=True, total_tokens_b=0.000000008, tokenizer="local",
        num_workers=2,
    )
    first = dataset_module.Dataset_Mixed(**kwargs)
    assert sorted(row["text"] for row in first) == [
        "first document", "first document", "second document", "second document"
    ]
    assert list(dataset_module.Dataset_Mixed(**kwargs)) == list(first)


@pytest.mark.parametrize("names", [["first"], ["first", "second"]])
def test_stream_can_restart_and_tokenize_to_disk(monkeypatch, tmp_path, names):
    def local_text_stream(name, text_column="text", batch_rows=1024):
        yield "hello world"
        yield "world hello"

    monkeypatch.setattr(dataset_module, "_parquet_text_stream", local_text_stream)
    mix = dataset_module.Dataset_Mixed(
        datasets=names, probabilities=[1 / len(names)] * len(names)
    )
    expected = list(mix)
    assert len(expected) == 2 * len(names)
    assert list(mix) == expected
    # Partially consuming a stream (as BPE training does) must not lose rows.
    next(iter(mix))
    assert list(mix) == expected

    backend = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1, "world": 2, "<eos>": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>"
    )
    path = tmp_path / "dataset"
    result = dataset_module.tokenize_to_disk(
        mix.dataset, tokenizer, path, seq_len=3, batch_size=1
    )
    assert result["input_ids"] == [[1, 2, 3], [2, 1, 3]] * len(names)
    assert Dataset.load_from_disk(path)[:] == result[:]
