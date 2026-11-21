import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from . import build_model
from .modeling_moe_slm import MoESLMConfig, MoESLMForCausalLM


def export_model(
    checkpoint_path: Path,
    tokenizer_source: str,
    output_dir: Path,
    *,
    tokenizer_subfolder: str = "",
) -> None:
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Export requires an empty output directory: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, subfolder=tokenizer_subfolder)
    with torch.device("meta"):
        native = build_model(attn_implementation="sdpa").config
    if max(tokenizer.get_vocab().values()) >= native["vocab_size"]:
        raise ValueError("The tokenizer has token IDs outside the checkpoint's embedding table")

    MoESLMConfig.register_for_auto_class()
    MoESLMForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    config = MoESLMConfig(
        vocab_size=native["vocab_size"],
        hidden_size=native["d_model"],
        num_hidden_layers=native["n_layers"],
        num_attention_heads=native["n_q_heads"],
        num_key_value_heads=native["n_kv_heads"],
        num_experts=native["num_experts"],
        intermediate_size=native["d_ffn"],
        num_experts_per_tok=native["top_k"],
        max_position_embeddings=native["max_seq_len"],
        tie_word_embeddings=native["tie_embeddings"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = MoESLMForCausalLM(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weights = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(weights, dict) or not all(isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError("Expected a MoESLM state dict or a training checkpoint with a 'model' state dict")
    model.model.load_state_dict({name.removeprefix("_orig_mod."): value for name, value in weights.items()})
    del checkpoint, weights

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Export a native MoESLM checkpoint for Transformers.")
    parser.add_argument("--checkpoint", required=True, help="Local .pt path or Hub filename")
    parser.add_argument("--checkpoint-repo", help="Hub repo containing --checkpoint")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer directory or Hub repo ID")
    parser.add_argument("--tokenizer-subfolder", default="", help="Tokenizer path within the directory or Hub repo")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    if args.checkpoint_repo:
        checkpoint_path = hf_hub_download(repo_id=args.checkpoint_repo, filename=checkpoint_path)
    export_model(
        Path(checkpoint_path), args.tokenizer, args.output_dir, tokenizer_subfolder=args.tokenizer_subfolder,
    )


if __name__ == "__main__":
    main()
