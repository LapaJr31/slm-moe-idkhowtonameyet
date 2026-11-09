import hashlib
import os
import pickle
from collections import defaultdict

from datasets import load_dataset
from tokenizers import pre_tokenizers
from transformers import AutoTokenizer


class BPETokenizer:
    def __init__(
        self,
        dataset=None,
        vocab_size=50000,
        tokenizer="gpt2",
        wordfreq=None,
        text_column=None,
        special_tokens=("<pad>", "<eos>"),
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.wordfreq = wordfreq if wordfreq is not None else defaultdict(int)
        self.alphabet = []
        self.dataset = dataset
        self.text_column = text_column
        self.vocab_size = vocab_size
        self.special_tokens = list(special_tokens)
        self.merges: dict[tuple[str, str], str] = {}

        if wordfreq is None:
            if dataset is None:
                raise ValueError("provide either `dataset` or a prebuilt `wordfreq`")
            self._build_wordfreq()

        self.vocab, self.splits = self.compute_vocab()
        self.train_merges()
        self._merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    def to_huggingface(self):
        from tokenizers import Tokenizer, models, decoders
        from transformers import PreTrainedTokenizerFast

        vocab = {tok: i for i, tok in enumerate(self.vocab)}

        # byte chars unseen during training get ids after the merges so every
        # possible byte still encodes; normally a no-op because compute_vocab
        # seeds the full byte alphabet
        next_id = len(vocab)
        for ch in pre_tokenizers.ByteLevel.alphabet():
            if ch not in vocab:
                vocab[ch] = next_id
                next_id += 1

        merges = [(a, b) for a, b in self.merges.keys()]
        bpe = models.BPE(vocab=vocab, merges=merges)

        tk = Tokenizer(bpe)
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tk.decoder = decoders.ByteLevel()

        return PreTrainedTokenizerFast(
            tokenizer_object=tk,
            pad_token="<pad>" if "<pad>" in self.special_tokens else None,
            eos_token="<eos>" if "<eos>" in self.special_tokens else None,
        )

    def save_pretrained(self, path):
        self.to_huggingface().save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path, vocab_size=None):

        import json

        hf = AutoTokenizer.from_pretrained(path)
        backend = hf.backend_tokenizer
        vocab = backend.get_vocab()

        with open(os.path.join(path, "tokenizer.json"), encoding="utf-8") as f:
            data = json.load(f)
        if data["model"]["type"] != "BPE":
            raise ValueError(f"{path} is not a BPE tokenizer")
        merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1))
                  for m in data["model"]["merges"]]

        id_to_tok = {i: t for t, i in vocab.items()}
        vocab_list = [id_to_tok[i] for i in range(len(id_to_tok))]
        tok = cls.__new__(cls)
        tok.tokenizer = hf
        tok.vocab = vocab_list
        tok.merges = {pair: pair[0] + pair[1] for pair in merges}
        tok.wordfreq = defaultdict(int)
        tok.vocab_size = vocab_size if vocab_size is not None else len(vocab_list)
        tok.alphabet = [t for t in vocab_list if len(t) == 1]
        tok.special_tokens = [t for t in vocab_list if t in ("<pad>", "<eos>", "<bos>", "<unk>")]
        tok.splits = None
        tok.dataset = None
        tok.text_column = None
        tok._merge_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        return tok

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "vocab": self.vocab,
            "merges": self.merges,
            "wordfreq": self.wordfreq,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "tokenizer": self.tokenizer.name_or_path,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        tok = cls.__new__(cls)
        tok.tokenizer = AutoTokenizer.from_pretrained(state["tokenizer"])
        tok.vocab = state["vocab"]
        tok.merges = state["merges"]
        tok.wordfreq = state["wordfreq"]
        tok.vocab_size = state["vocab_size"]
        tok.special_tokens = state.get("special_tokens", [])
        tok.splits = None
        tok._merge_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        return tok

    def _pretokenize(self, text):
        return [
            word
            for word, _ in self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        ]

    def _build_wordfreq(self):
        for ex in self.dataset:
            text = ex[self.text_column] if isinstance(ex, dict) else ex
            for word in self._pretokenize(text):
                self.wordfreq[word] += 1

    def compute_vocab(self):

        observed = set(self.alphabet)
        for word in self.wordfreq.keys():
            observed.update(word)
        # seed the full byte-level alphabet so every possible byte has a token
        # even when unseen in the corpus; keeps the exported vocab exactly
        # `vocab_size` tokens so ids stay within the model's embedding bounds
        self.alphabet = sorted(observed | set(pre_tokenizers.ByteLevel.alphabet()))

        vocab = self.special_tokens.copy() + self.alphabet.copy()
        splits = {word: [c for c in word] for word in self.wordfreq.keys()}

        return vocab, splits

    def train_merges(self):
        import heapq

        self.pair_counts = defaultdict(int)
        pair_to_words: dict[tuple[str, str], set] = defaultdict(set)

        for word, freq in self.wordfreq.items():
            split = self.splits[word]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                self.pair_counts[pair] += freq
                pair_to_words[pair].add(word)

        heap = [(-freq, pair) for pair, freq in self.pair_counts.items() if freq > 0]
        heapq.heapify(heap)

        n_target = self.vocab_size - len(self.vocab)
        while len(self.vocab) < self.vocab_size:
            while heap:
                neg_freq, pair = heapq.heappop(heap)
                if self.pair_counts.get(pair, 0) == -neg_freq and -neg_freq > 0:
                    break
            else:
                break
            a, b = pair
            merged = a + b
            self.merges[pair] = merged
            self.vocab.append(merged)
            if len(self.merges) % 10000 == 0:
                print(f"merges: {len(self.merges)}/{n_target}", flush=True)

            changed_pairs = set()
            for word in pair_to_words.pop(pair, ()):
                freq = self.wordfreq[word]
                old_split = self.splits[word]

                for i in range(len(old_split) - 1):
                    old = (old_split[i], old_split[i + 1])
                    self.pair_counts[old] -= freq
                    changed_pairs.add(old)
                    words_for_old = pair_to_words.get(old)
                    if words_for_old is not None:
                        words_for_old.discard(word)

                new_split = []
                i = 0
                while i < len(old_split):
                    if i < len(old_split) - 1 and old_split[i] == a and old_split[i + 1] == b:
                        new_split.append(merged)
                        i += 2
                    else:
                        new_split.append(old_split[i])
                        i += 1
                self.splits[word] = new_split

                for i in range(len(new_split) - 1):
                    new = (new_split[i], new_split[i + 1])
                    self.pair_counts[new] += freq
                    changed_pairs.add(new)
                    pair_to_words[new].add(word)

            for p in changed_pairs:
                freq = self.pair_counts.get(p, 0)
                if freq > 0:
                    heapq.heappush(heap, (-freq, p))

        if len(self.merges) < n_target:
            print(
                f"pair counts exhausted at {len(self.merges)} merges (wanted {n_target})",
                flush=True,
            )

    def tokenize(self, text):
        pre_tokenize_result = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        ranks = self._merge_ranks

        tokens = []
        for word in pre_tokenized_text:
            split = list(word)
            while len(split) > 1:
                best = None
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    rank = ranks.get(pair)
                    if rank is not None and (best is None or rank < best[0]):
                        best = (rank, pair)
                if best is None:
                    break
                a, b = best[1]
                merged = a + b
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                        new_split.append(merged)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                split = new_split
            tokens.extend(split)

        return tokens


def train_bpe_on_mix(
    mix,
    vocab_size=50000,
    tokens_b=1.0,
    text_column="text",
    batch_size=512,
    ref_tokenizer="gpt2",
    cache_name=None,
):

    ref = AutoTokenizer.from_pretrained(ref_tokenizer)
    pretok = ref.backend_tokenizer.pre_tokenizer

    budget = int(tokens_b * 1e9)

    if cache_name:
        key = hashlib.blake2b(
            f"{cache_name}|{budget}|{ref_tokenizer}".encode(), digest_size=16
        ).hexdigest()
        cache_dir = os.path.join(".cache", "moe_slm_bpe_wordfreq", key)
        cache_file = os.path.join(cache_dir, "wordfreq.pkl")
        if os.path.exists(cache_file):
            print(f"loading wordfreq cache {cache_file}", flush=True)
            with open(cache_file, "rb") as f:
                wordfreq = pickle.load(f)
            return BPETokenizer(vocab_size=vocab_size, wordfreq=wordfreq)
        os.makedirs(cache_dir, exist_ok=True)

    wordfreq = defaultdict(int)
    counted = 0
    batch = []
    n_batches = 0

    def flush(texts):
        nonlocal counted
        lengths = [len(ids) for ids in ref(texts, add_special_tokens=False)["input_ids"]]
        for text, n in zip(texts, lengths):
            for word, _ in pretok.pre_tokenize_str(text):
                wordfreq[word] += 1
            counted += n

    for ex in mix:
        text = ex[text_column] if isinstance(ex, dict) else ex
        batch.append(text)
        if len(batch) >= batch_size:
            flush(batch)
            n_batches += 1
            batch = []
            if n_batches % 100 == 0:
                print(
                    f"wordfreq: {counted/1e6:.0f}M/{budget/1e6:.0f}M tokens, "
                    f"{len(wordfreq):,} unique words",
                    flush=True,
                )
            if counted >= budget:
                break
    if batch and counted < budget:
        flush(batch)

    if cache_name:
        with open(cache_file, "wb") as f:
            pickle.dump(dict(wordfreq), f)

    return BPETokenizer(vocab_size=vocab_size, wordfreq=wordfreq)
