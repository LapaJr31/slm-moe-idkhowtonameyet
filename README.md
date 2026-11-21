# moe_slm

[Background checkpoint offloading](docs/checkpoint-offloading.md)

## Export MoESLM for Transformers and SFT

Export a checkpoint produced by this repo with its original tokenizer. Pass a
local tokenizer directory or its Hugging Face repo ID.

```sh
uv run --env-file .env moe_slm_export \
  --checkpoint checkpoints/best.pt \
  --tokenizer lapkaaaa/skynet_0.1_not_killing_anybody_yet \
  --tokenizer-subfolder tokenizers/skynet-32768 \
  --output-dir checkpoints/skynet-hf
```

For a checkpoint stored on the Hub, add `--checkpoint-repo OWNER/REPO` and
give `--checkpoint` the exact filename in that repo. The export contains
`config.json`, safetensors weights, the tokenizer, and the custom model code.
You can upload that directory to the Hub and pass its repo ID to SFT.

```sh
uv run --env-file .env --cache-dir /tmp/uv-cache --project gpu_envs/fa2 \
  sft_post_training --model checkpoints/skynet-hf --trust-remote-code
```

Use `--trust-remote-code` only for a custom export you trust, including exports
you created locally. The MoESLM wrapper uses PyTorch SDPA. `--project` selects
the Python environment.
