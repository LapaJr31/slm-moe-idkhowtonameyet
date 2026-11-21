import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, dynamic_module_utils

from moe_slm import export_hf
from moe_slm.model import MoESLM


@pytest.mark.parametrize("tokenizer_subfolder", ["", "tokenizers/skynet-32768"])
def test_export_compiled_checkpoint_with_tied_weights(tmp_path, monkeypatch, tokenizer_subfolder):
    def build_model(*, attn_implementation):
        return MoESLM(
            vocab_size=8, d_model=16, n_layers=1, n_q_heads=2, n_kv_heads=1,
            num_experts=2, d_ffn=24, top_k=1, max_seq_len=16,
            attn_implementation=attn_implementation,
        )

    monkeypatch.setattr(export_hf, "build_model", build_model)
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "modules"))

    original = build_model(attn_implementation="sdpa").eval()
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": {f"_orig_mod.{name}": value for name, value in original.state_dict().items()}}, checkpoint)
    tokenizer_source = tmp_path / "tokenizer-source"
    tokenizer_dir = tokenizer_source / tokenizer_subfolder
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<eos>": 0, "hello": 1})), eos_token="<eos>",
    )
    tokenizer.save_pretrained(tokenizer_dir)

    output = tmp_path / "export"
    export_hf.export_model(checkpoint, str(tokenizer_source), output, tokenizer_subfolder=tokenizer_subfolder)
    model = AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(output, config=model.config)

    assert tokenizer.eos_token_id == model.config.eos_token_id == 0
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    input_ids = torch.tensor([[1, 1, 0]])
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids).logits, original(input_ids)[0])
        torch.testing.assert_close(
            model(input_ids, labels=input_ids).loss,
            model(input_ids, labels=input_ids, loss_only=True).loss,
        )
        torch.testing.assert_close(
            model(input_ids, labels=input_ids, loss_only=True, num_items_in_batch=4).loss,
            model(input_ids, labels=input_ids).loss / 2,
        )
