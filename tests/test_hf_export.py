import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, dynamic_module_utils

from moe_slm import export_hf
from moe_slm.model import MoESLM
from sft_post_training import sft


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

    monkeypatch.setattr(sft.mlflow, "set_tag", lambda *args: None)
    monkeypatch.setattr(sft.mlflow, "log_params", lambda *args: None)
    pipeline = sft.SFT_Post_train(str(output), str(tmp_path / "sft"), trust_remote_code=True)
    attention_mask = torch.tensor([[1, 1, 0]])
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    inputs = dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels, loss_only=True)
    original_experts = model.model.blocks[0].moe
    loaded_experts = pipeline.model.model.blocks[0].moe
    weights_before = (original_experts.w13, original_experts.w2)
    weights_after = (loaded_experts.w13, loaded_experts.w2)
    for before, after in zip(weights_before, weights_after, strict=True):
        assert after.transpose(-1, -2).is_contiguous()
        assert (after.dtype, after.device, after.requires_grad) == (before.dtype, before.device, before.requires_grad)
        torch.testing.assert_close(after, before, rtol=0, atol=0)

    expected = model(**inputs).loss
    pipeline.model.train()
    pipeline.model.gradient_checkpointing_enable()
    compiled = torch.compile(pipeline.model, backend="aot_eager", fullgraph=True)
    actual = compiled(**inputs).loss
    torch.testing.assert_close(actual, expected)
    gradients_before = torch.autograd.grad(expected, weights_before)
    gradients_after = torch.autograd.grad(actual, weights_after)
    for before, after in zip(gradients_before, gradients_after, strict=True):
        torch.testing.assert_close(after, before)

    sft._restore_expert_weight_layout(pipeline.model)
    assert loaded_experts.w13 is weights_after[0]
    assert loaded_experts.w2 is weights_after[1]
