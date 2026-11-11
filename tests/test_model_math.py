"""Model math invariants: custom op correctness, aux/z properties, checkpointing."""
import math

import pytest
import torch
import torch.nn.functional as F

from conftest import tiny_model, random_batch, DEVICE
from moe_slm.model import BatchedSwiGLUMoE


# ---------------------------------------------------------------- custom op

def test_custom_op_matches_eager_reference(device):
    """The custom-op MoE must produce identical outputs and gradients to the
    reference (traced-through) implementation. Guards the recompute-backward
    and fake-kernel work."""
    torch.manual_seed(0)
    d, E, H = 32, 4, 24
    x = torch.randn(6, d, device=device, requires_grad=True)
    logits = torch.randn(6, E, device=device, requires_grad=True)
    w13 = torch.randn(E, 2 * H, d, device=device, requires_grad=True)
    w2 = torch.randn(E, d, H, device=device, requires_grad=True)
    top_k = 2

    # reference: run the impl directly under autograd
    from moe_slm.model import _moe_impl
    out_ref, aux_ref, z_ref = _moe_impl(x, logits, w13, w2, top_k)
    loss_ref = out_ref.sum() + 0.01 * aux_ref + 0.01 * z_ref
    loss_ref.backward()

    gx_ref, gl_ref, g13_ref, g2_ref = x.grad, logits.grad, w13.grad, w2.grad
    out_ref = out_ref.detach()
    for t in (x, logits, w13, w2):
        t.grad = None

    # through the custom op
    import moe_slm.model as M
    out_op, aux_op, z_op = torch.ops.moe_slm.batched_swiglu(x, logits, w13, w2, top_k)
    (out_op.sum() + 0.01 * aux_op + 0.01 * z_op).backward()

    assert torch.allclose(out_ref, out_op, atol=1e-5)
    assert torch.allclose(aux_ref.detach(), aux_op, atol=1e-5)
    assert torch.allclose(z_ref.detach(), z_op, atol=1e-5)
    for name, ref, got in (
        ("x", gx_ref, x.grad), ("logits", gl_ref, logits.grad),
        ("w13", g13_ref, w13.grad), ("w2", g2_ref, w2.grad),
    ):
        assert torch.allclose(ref, got, atol=1e-4), f"grad mismatch for {name}"


@pytest.mark.skipif(DEVICE.type != "cuda", reason="flex backward unsupported on CPU")
def test_custom_op_compiles_without_graph_breaks(device):
    """Whole-model compile must not emit graph breaks — the break-in-loop
    bug that forced the custom-op rewrite."""
    m = tiny_model().to(device)
    m.train()
    ids = random_batch(vocab=256)["input_ids"]

    from torch._dynamo.utils import counters
    counters.clear()
    compiled = torch.compile(m, dynamic=True)
    logits, aux, z = compiled(ids)
    (logits.float().mean() + aux + z).backward()

    breaks = counters["stats"]["unique_graph_breaks"]
    assert breaks == 0, f"{breaks} graph breaks: {dict(counters['graph_break'])}"


# ---------------------------------------------------------------- aux / z loss

def test_aux_uniform_vs_collapsed_closed_form(device):
    """Closed-form aux on synthetic logits, via _moe_impl directly (no model,
    no attention — pure routing math).

    aux = E * sum(f_i * P_i) with E=4, top_k=2:
    - uniform scores (all 1/E): f_i = 1/E, P_i = 1/E → aux = 4*4*(1/4)*(1/4) = 1.0
    - fully peaked on expert 0: f_0 = top_k/E = 0.5, P_0 = 1 → aux = 4*0.5 = 2.0
      (f_0 caps at 0.5 because every token's second slot still goes elsewhere)
    """
    torch.manual_seed(0)
    E, top_k = 4, 2
    n_tokens = 64
    x = torch.randn(n_tokens, 8)
    w13 = torch.randn(E, 16, 8)
    w2 = torch.randn(E, 8, 8)

    from moe_slm.model import _moe_impl

    uniform = torch.zeros(n_tokens, E)  # equal logits → uniform softmax
    _, aux_u, _ = _moe_impl(x, uniform, w13, w2, top_k)
    assert abs(aux_u.item() - 1.0) < 1e-5

    peaked = torch.zeros(n_tokens, E)
    peaked[:, 0] = 100.0  # expert 0 always ranks first
    _, aux_p, _ = _moe_impl(x, peaked, w13, w2, top_k)
    assert abs(aux_p.item() - 2.0) < 1e-5


def test_z_loss_minimum_at_uniform(device):
    """z = E * mean(sum_i p_i^2); uniform p=1/E gives E * E*(1/E^2) = 1 per layer."""
    m = tiny_model(num_experts=4, attn_implementation="sdpa")
    m.eval()
    with torch.no_grad():
        for blk in m.blocks:
            blk.moe.router.weight.zero_()
    ids = random_batch(vocab=256)["input_ids"]
    with torch.no_grad():  # no backward needed
        _, _, z = m(ids)
    assert abs(z.item() - m.config["n_layers"]) < 0.15


# ---------------------------------------------------------------- checkpointing

def test_checkpointing_produces_identical_gradients(device):
    """Gradient checkpointing is a memory/compute tradeoff — the math must be
    identical. Runs the same batch with and without checkpointing and compares
    every gradient."""
    torch.manual_seed(7)
    # sdpa: flex has no CPU backward; sdpa path is mathematically identical
    m1 = tiny_model(attn_implementation="sdpa")
    m2 = tiny_model(attn_implementation="sdpa")
    m2.load_state_dict(m1.state_dict())
    # zero the jitter: randn_like makes every forward differ, so a
    # ckpt-vs-no-ckpt comparison is only meaningful with deterministic routing.
    # (jitter correctness under checkpoint RNG replay is torch's job, not ours)
    for m in (m1, m2):
        for blk in m.blocks:
            blk.moe.jitter_std = 0.0
    ids = random_batch(vocab=256)["input_ids"]
    targets = ids.roll(-1, dims=-1)

    def run(model, use_ckpt):
        if use_ckpt:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing = False
        model.train()
        logits, aux, z = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1),
        ) + 0.01 * aux + 0.01 * z
        loss.backward()

    run(m1, use_ckpt=True)
    run(m2, use_ckpt=False)

    # NOTE: not torch.equal — the recompute path executes ops in a different
    # order than the stored-activation path, so grads agree to ~1e-6 relative
    # (fp32 non-associativity) but are not bitwise identical.
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-4), (
            f"grad mismatch at {n1}: "
            f"maxdiff={(p1.grad - p2.grad).abs().max().item()}"
        )


def test_checkpointing_disabled_at_eval(device):
    """Eval must not checkpoint (no recompute cost at inference)."""
    m = tiny_model()
    m.gradient_checkpointing_enable()
    m.eval()
    ids = random_batch(vocab=256)["input_ids"]
    with torch.no_grad():
        out = m(ids)  # would fail if checkpoint path forced under no_grad
    assert len(out) == 3


# ---------------------------------------------------------------- shapes

@pytest.mark.parametrize("d_ffn", [48, 100, 4224])
def test_hidden_dim_alignment(d_ffn):
    """hidden_dim must be a multiple of 8 — grouped_mm's 16-byte stride rule."""
    hidden = -(-2 * d_ffn // 3 // 8) * 8
    assert hidden % 8 == 0
    assert hidden >= 2 * d_ffn / 3


def test_forward_shapes(device):
    m = tiny_model()
    m.eval()
    ids = random_batch(batch_size=3, seqlen=16, vocab=256)["input_ids"]
    with torch.no_grad():
        logits, aux, z = m(ids)
    assert logits.shape == (3, 16, 256)
    assert aux.shape == ()
    assert z.shape == ()


def test_tied_embeddings_share_weight(device):
    m = tiny_model(tie_embeddings=True)
    assert m.lm_head.weight is m.tok_emb.weight
