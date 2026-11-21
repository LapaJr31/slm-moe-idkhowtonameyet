import torch
from transformers import GenerationMixin, PreTrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .model import BatchedSwiGLUMoE, MoESLM, RotaryPositionalEmbeddings


class MoESLMConfig(PreTrainedConfig):
    model_type = "moe_slm"

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        num_experts=8,
        intermediate_size=960,
        num_experts_per_tok=2,
        max_position_embeddings=8192,
        tie_word_embeddings=True,
        use_cache=False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache


class MoESLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MoESLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _tied_weights_keys = {"model.lm_head.weight": "model.tok_emb.weight"}
    accepts_loss_kwargs = True

    def __init__(self, config):
        super().__init__(config)
        self.model = MoESLM(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            n_layers=config.num_hidden_layers,
            n_q_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            num_experts=config.num_experts,
            d_ffn=config.intermediate_size,
            top_k=config.num_experts_per_tok,
            max_seq_len=config.max_position_embeddings,
            tie_embeddings=config.tie_word_embeddings,
            attn_implementation="sdpa",
        )
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, RotaryPositionalEmbeddings):
            positions = torch.arange(0, module.dim, 2, device=module.inv_freq.device, dtype=torch.float32)
            module.inv_freq.copy_(1.0 / (module.base ** (positions / module.dim)))
            module._build_cache(self.config.max_position_embeddings)
        elif isinstance(module, BatchedSwiGLUMoE):
            module.drop_rate.zero_()
            module.active_expert_assignments.zero_()

    def get_input_embeddings(self):
        return self.model.tok_emb

    def set_input_embeddings(self, embeddings):
        self.model.tok_emb = embeddings

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, embeddings):
        self.model.lm_head = embeddings

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        loss_only=False,
        num_items_in_batch=None,
    ):
        if use_cache:
            raise ValueError("The Transformers MoESLM wrapper currently requires use_cache=False")

        if labels is not None and loss_only:
            loss, _, _ = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            if num_items_in_batch is not None:
                token_count = (labels[:, 1:] != -100).sum()
                batch_token_count = torch.as_tensor(num_items_in_batch, device=loss.device).clamp_min(1)
                loss = loss * token_count / batch_token_count
            output = CausalLMOutput(loss=loss)
        else:
            logits, _, _ = self.model(input_ids, attention_mask=attention_mask)
            loss = None if labels is None else self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size,
                num_items_in_batch=num_items_in_batch,
            )
            output = CausalLMOutput(loss=loss, logits=logits)
        if return_dict is None:
            return_dict = self.config.return_dict
        return output if return_dict else output.to_tuple()
