import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.utils.checkpoint import checkpoint


flex_attention = torch.compile(flex_attention)  # ty:ignore[invalid-assignment]
_MASK_CACHE_MAX = 64
_MASK_CACHE: dict = {}


def _mask_cache_put(key, value):
    if len(_MASK_CACHE) >= _MASK_CACHE_MAX:
        _MASK_CACHE.pop(next(iter(_MASK_CACHE)))
    _MASK_CACHE[key] = value
    return value


def _mask_fingerprint(attention_mask: torch.Tensor):
    m = attention_mask.detach().to(torch.bool)
    digest = hashlib.blake2b(m.cpu().numpy().tobytes(), digest_size=16).hexdigest()
    return tuple(m.shape), str(m.device), digest


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def get_causal_block_mask(seq_len: int, device):
    key = ("flex_causal", seq_len, str(device))
    mask = _MASK_CACHE.get(key)
    if mask is None:
        mask = _mask_cache_put(
            key,
            create_block_mask(
                causal_mask,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
            ),
        )
    return mask


def get_padded_block_mask(attention_mask: torch.Tensor):
    key = ("flex_padded", *_mask_fingerprint(attention_mask))
    mask = _MASK_CACHE.get(key)
    if mask is None:
        B, T = attention_mask.shape
        am = attention_mask.detach()

        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (am[b, kv_idx] != 0)

        mask = _mask_cache_put(
            key,
            create_block_mask(
                mask_mod,
                B=B,
                H=None,
                Q_LEN=T,
                KV_LEN=T,
                device=am.device,
            ),
        )
    return mask


def get_sdpa_additive_mask(attention_mask: torch.Tensor, dtype: torch.dtype):
    key = ("sdpa_additive", *_mask_fingerprint(attention_mask), str(dtype))
    mask = _MASK_CACHE.get(key)
    if mask is None:
        B, T = attention_mask.shape
        am = attention_mask.detach().to(torch.bool)
        allow = torch.ones(T, T, dtype=torch.bool, device=am.device).tril()
        allow = allow & am[:, None, None, :]

        fully_masked = am.cumsum(-1) == 0                       # [B, T]
        diag = torch.eye(T, dtype=torch.bool, device=am.device)[None, None]
        allow = allow | (diag & fully_masked[:, None, :, None])  # [B, 1, T, T]
        mask = _mask_cache_put(
            key,
            torch.zeros(B, 1, T, T, dtype=dtype, device=am.device).masked_fill_(
                ~allow, torch.finfo(dtype).min
            ),
        )
    return mask


def make_padded_causal_block_mask(attention_mask: torch.Tensor):
    if bool(attention_mask.all()):
        return get_causal_block_mask(attention_mask.shape[-1], attention_mask.device)
    return get_padded_block_mask(attention_mask)


class RotaryPositionalEmbeddings(nn.Module):
    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, dim: int, base: int = 10_000, max_seq_len: int = 8192):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._capacity = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, capacity: int) -> None:
        t = torch.arange(capacity, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = angles.cos()
        self.sin_cached = angles.sin()
        self._capacity = capacity

    def _ensure_capacity(self, seq_len: int) -> None:
        if seq_len > self._capacity:
            self._build_cache(max(seq_len, 2 * self._capacity))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, position_ids=None):
        B, _, T, D = q.shape
        assert D == self.dim
        assert k.shape[0] == B and k.shape[-2:] == (T, D)

        self._ensure_capacity(T)

        if position_ids is None:
            cos = self.cos_cached[:T].view(1, 1, T, D)
            sin = self.sin_cached[:T].view(1, 1, T, D)
        else:
            pos = position_ids.long()
            cos = self.cos_cached[pos]
            sin = self.sin_cached[pos]
            if pos.ndim == 1:
                cos = cos.view(1, 1, T, D)
                sin = sin.view(1, 1, T, D)
            else:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)

        q = q * cos.to(q.dtype) + self._rotate_half(q) * sin.to(q.dtype)
        k = k * cos.to(k.dtype) + self._rotate_half(k) * sin.to(k.dtype)
        return q, k

#todo
class KVcache(nn.Module):
    def __init__(self, d_size, n_layers, n_kv_heads, max_seq_len, dtype, device, max_batch_size):
        pass

    def update(self, layer_idx, k_new, v_new):
        pass

    def get(self, layer_idx):
        pass

    def reset(self):
        pass

class GQAAttention(nn.Module):
    def __init__(self, d_model=1024, n_q_heads=16, n_kv_heads=4,
                 max_seq_len=8192, attn_implementation="flex"):
        super().__init__()
        assert d_model % n_q_heads == 0
        assert n_q_heads % n_kv_heads == 0
        assert attn_implementation in ("flex", "sdpa")

        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.attn_implementation = attn_implementation
        self.enable_gqa = n_q_heads != n_kv_heads

        self.q_proj = nn.Linear(d_model, n_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim, max_seq_len=max_seq_len
        )


    def forward(self, x, block_mask, attention_mask, position_ids):

        B, T, _ = x.shape

        q = self.q_proj(x).view(
            B, T, self.n_q_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            B, T, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            B, T, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        q, k = self.rope(q, k, position_ids)

        if self.attn_implementation == "flex":
            y = flex_attention(
                q, k, v,
                block_mask=block_mask,
                enable_gqa=self.enable_gqa,
            )
        elif attention_mask is None:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=self.enable_gqa
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=get_sdpa_additive_mask(attention_mask, q.dtype),
                enable_gqa=self.enable_gqa,
            )

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)

# --- MoE as a custom op ---------------------------------------------------------
# The MoE routing (topk/argsort/bincount/grouped_mm) doesn't trace cleanly by
# dynamo, so the whole computation is exposed to torch.compile as ONE opaque
# op — avoiding the graph-break-in-loop that kills whole-model compile.
#
# Autograd: register_autograd with a recompute-based backward. The backward is
# a PLAIN python function (not an op impl), so torch.enable_grad() works inside
# it: it re-runs the deterministic impl and differentiates through it.
# (Grouped_mm has autograd; verified empirically.) NOTE: custom_op *impls*
# force grad-mode off and enable_grad cannot override — that's why the
# recompute must live in the backward function, never inside an op impl.
#
# The backward formula runs eager during compile (AOTAutograd treats custom-op
# backward formulas as opaque calls) — acceptable: it's a small fraction of
# step time and correctness is unaffected.
#
# Router jitter stays OUTSIDE the op (in BatchedSwiGLUMoE.forward): noise is
# generated by dynamo/checkpoint-managed RNG, keeping the op deterministic.

@torch.library.custom_op("moe_slm::batched_swiglu", mutates_args=())
def batched_swiglu(x: torch.Tensor, logits: torch.Tensor, w13: torch.Tensor,
                   w2: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _moe_impl(x, logits, w13, w2, top_k)


def _moe_impl(x, logits, w13, w2, top_k: int):
    scores = logits.float().softmax(-1)
    topk_scores, topk_ids = scores.topk(top_k, -1)

    flat_ids = topk_ids.reshape(-1)
    flat_scores = topk_scores.reshape(-1)
    order = flat_ids.argsort()
    expert_sorted = flat_ids[order]
    token_idx = order // top_k
    x_sorted = x[token_idx]

    counts = torch.bincount(expert_sorted, minlength=w13.shape[0])
    offs = counts.cumsum(0, dtype=torch.int32)

    # gemm 1
    h13 = torch._grouped_mm(x_sorted, w13.transpose(-1, -2), offs=offs)
    hidden_dim = w13.shape[1] // 2
    gate, up = h13.split(hidden_dim, dim=-1)
    h2 = F.silu(gate) * up

    # gemm 2
    h3 = torch._grouped_mm(h2, w2.transpose(-1, -2), offs=offs)

    P = scores.mean(dim=0)
    f = counts.detach().float() / flat_ids.numel()
    aux = w13.shape[0] * (f * P).sum()
    z = w13.shape[0] * (scores ** 2).sum(-1).mean()

    out = torch.zeros_like(x)
    out.index_add_(0, token_idx, h3 * flat_scores[order, None])
    return out, aux, z


@batched_swiglu.register_fake
def _(x, logits, w13, w2, top_k):
    # output shapes are data-independent: out mirrors x, aux/z are scalars
    return (
        torch.empty_like(x),
        x.new_empty((), dtype=torch.float32),
        x.new_empty((), dtype=torch.float32),
    )


def _moe_setup_context(ctx, inputs, output):
    x, logits, w13, w2, top_k = inputs
    ctx.save_for_backward(x, logits, w13, w2)
    ctx.top_k = top_k


def _moe_backward(ctx, grad_out, grad_aux, grad_z):
    """Recompute-based backward: re-run the deterministic impl under grad mode
    and differentiate through it. Plain function — enable_grad works here
    (unlike inside custom-op impls)."""
    x, logits, w13, w2 = ctx.saved_tensors
    top_k = ctx.top_k
    with torch.enable_grad():
        x_c = x.detach().requires_grad_(True)
        lg_c = logits.detach().requires_grad_(True)
        w13_c = w13.detach().requires_grad_(True)
        w2_c = w2.detach().requires_grad_(True)
        out_c, aux_c, z_c = _moe_impl(x_c, lg_c, w13_c, w2_c, top_k)
        gx, gl, g13, g2 = torch.autograd.grad(
            (out_c, aux_c, z_c),
            (x_c, lg_c, w13_c, w2_c),
            grad_outputs=(grad_out, grad_aux, grad_z),
        )
    # one grad per op input: x, logits, w13, w2, top_k(int → None)
    return gx, gl, g13, g2, None


batched_swiglu.register_autograd(_moe_backward, setup_context=_moe_setup_context)


class BatchedSwiGLUMoE(nn.Module):
    def __init__(self, num_experts: int, d_ffn: int, d_model: int = 1024, top_k: int = 2):
        super().__init__()
        # 16-byte stride alignment requirement
        hidden_dim = -(-2 * d_ffn // 3 // 8) * 8
        self.top_k, self.num_experts, self.hidden_dim = top_k, num_experts, hidden_dim
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * hidden_dim, d_model))
        self.w2  = nn.Parameter(torch.empty(num_experts, d_model, hidden_dim))
        self.jitter_std = 0.1
        for p in (self.w13, self.w2):
            nn.init.kaiming_uniform_(p, a=5**0.5)

    def forward(self, x):
        logits = self.router(x)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.jitter_std
        out, aux, z = torch.ops.moe_slm.batched_swiglu(
            x, logits, self.w13, self.w2, self.top_k
        )  # ty: ignore[invalid-assignment]
        return out, aux, z


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight.to(dtype) * x.to(dtype)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int,
                 num_experts: int, d_ffn: int, top_k: int,
                 max_seq_len: int = 8192, attn_implementation: str = "flex"):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GQAAttention(d_model, n_q_heads, n_kv_heads,
                                 max_seq_len, attn_implementation)
        self.moe_norm = RMSNorm(d_model)
        self.moe = BatchedSwiGLUMoE(num_experts, d_ffn, d_model, top_k)

    def forward(self, x, block_mask, attention_mask, position_ids):
        x = x + self.attn(self.attn_norm(x), block_mask, attention_mask, position_ids)

        B, T, D = x.shape
        moe_out, aux, z_loss = self.moe(self.moe_norm(x).view(B * T, D))
        return x + moe_out.view(B, T, D), aux, z_loss


class MoESLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 1024,
        n_layers: int = 12,
        n_q_heads: int = 16,
        n_kv_heads: int = 4,
        num_experts: int = 8,
        d_ffn: int = 4224,
        top_k: int = 2,
        tie_embeddings: bool = True,
        max_seq_len: int = 8192,
        attn_implementation: str = "flex",
    ):
        super().__init__()
        assert d_model % n_q_heads == 0

        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_q_heads": n_q_heads,
            "n_kv_heads": n_kv_heads,
            "num_experts": num_experts,
            "d_ffn": d_ffn,
            "top_k": top_k,
            "tie_embeddings": tie_embeddings,
            "max_seq_len": max_seq_len,
            "attn_implementation": attn_implementation,
        }
        self.attn_implementation = attn_implementation

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_q_heads, n_kv_heads,
                             num_experts, d_ffn, top_k,
                             max_seq_len, attn_implementation)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None and bool(attention_mask.all()):
            attention_mask = None

        flex_path = self.attn_implementation == "flex"

        if attention_mask is None:

            position_ids = None
            block_mask = (
                get_causal_block_mask(input_ids.shape[-1], input_ids.device)
                if flex_path
                else None
            )
            padding_mask = None
        else:

            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            block_mask = get_padded_block_mask(attention_mask) if flex_path else None
            padding_mask = attention_mask

        x = self.tok_emb(input_ids)
        aux_total = x.new_zeros(())
        z_total = x.new_zeros(())

        use_ckpt = self.training and self.gradient_checkpointing and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                x, block_aux, block_z = checkpoint(
                    block, x, block_mask, padding_mask, position_ids,
                    use_reentrant=False,
                )
            else:
                x, block_aux, block_z = block(x, block_mask, padding_mask, position_ids)
            aux_total += block_aux
            z_total +=  block_z
        x = self.norm(x)
        return self.lm_head(x), aux_total, z_total

def compile_model(model: MoESLM, mode: str | None = "default", strategy: str = "whole") -> MoESLM:
    # either whole model or per block of a model
    if strategy == "whole":
        return torch.compile(model, dynamic=True, mode=mode)
    for block in model.blocks:
        block.compile(dynamic=True, mode=mode)
    return model
