"""Model math invariants: grouped MoE correctness, aux/z properties, checkpointing."""
import pytest
import torch
import torch.nn.functional as F

from conftest import tiny_model, random_batch
from moe_slm.model import BatchedSwiGLUMoE


def reference_moe(moe, x):
    """Plain per-expert matmuls, independent of grouped dispatch and scatter."""
    scores = moe.router(x).float().softmax(-1)
    weights, ids = scores.topk(moe.top_k, dim=-1)
    output = torch.zeros_like(x)
    for expert in range(moe.num_experts):
        gate, up = F.linear(x, moe.w13[expert]).chunk(2, dim=-1)
        value = F.linear(F.silu(gate) * up, moe.w2[expert])
        weight = (weights * (ids == expert)).sum(-1, keepdim=True)
        output = output + value * weight
    fractions = F.one_hot(ids, moe.num_experts).float().mean(dim=(0, 1))
    aux = moe.num_experts * (fractions * scores.mean(0)).sum()
    z = moe.num_experts * scores.square().sum(-1).mean()
    return output, aux, z


@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("collapsed", [False, True])
def test_moe_matches_per_expert_reference(device, top_k, collapsed):
    """Check dispatch, unnormalized routing weights, and all gradients.

    Collapsed routing leaves experts unused when top_k < num_experts.
    """
    torch.manual_seed(0)
    moe = BatchedSwiGLUMoE(4, 48, d_model=32, top_k=top_k).to(device).eval()
    x = torch.randn(6, 32, device=device)
    if collapsed:
        x = x.abs() + 1
        with torch.no_grad():
            moe.router.weight.copy_(
                torch.arange(4, device=device)[:, None].expand(4, 32) * 0.1
            )
    x.requires_grad_()
    inputs = (x, *moe.parameters())
    expected = reference_moe(moe, x)
    actual = moe(x)
    probe = torch.randn_like(x)
    def gradients(outputs):
        out, aux, z = outputs
        return torch.autograd.grad((out * probe).sum() + 0.03 * aux + 0.07 * z, inputs)
    expected_grads = gradients(expected)
    actual_grads = gradients(actual)
    for ref, got in zip(expected, actual):
        torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-4)
    for ref, got in zip(expected_grads, actual_grads):
        torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-4)


def test_compile_matches_eager_with_moe_graph_breaks(device):
    """Compilation may break around eager MoE, but must preserve training math."""
    torch.manual_seed(4)
    m = tiny_model(attn_implementation="sdpa").to(device).eval()
    ids = random_batch(vocab=256)["input_ids"]
    expected = m(ids)
    sum(t.float().square().mean() for t in expected).backward()
    expected_grads = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    compiled = torch.compile(m, dynamic=True)
    actual = compiled(ids)
    sum(t.float().square().mean() for t in actual).backward()
    for ref, got in zip(expected, actual):
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)
    for name, param in m.named_parameters():
        torch.testing.assert_close(param.grad, expected_grads[name], atol=1e-5, rtol=1e-4)


def test_aux_uniform_vs_collapsed_closed_form(device):
    """Uniform routing has aux=1; fully peaked top-2 routing has aux=E/2."""
    moe = BatchedSwiGLUMoE(4, 24, d_model=8, top_k=2).to(device).eval()
    x = torch.ones(64, 8, device=device)
    with torch.no_grad():
        moe.router.weight.zero_()
        _, aux, z = moe(x)
        torch.testing.assert_close(aux, aux.new_tensor(1.0))
        torch.testing.assert_close(z, z.new_tensor(1.0))
        moe.router.weight[0].fill_(100.0)
        _, aux, z = moe(x)
        torch.testing.assert_close(aux, aux.new_tensor(2.0))
        torch.testing.assert_close(z, z.new_tensor(4.0))


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
    ids = random_batch(vocab=256)["input_ids"]
    targets = ids.roll(-1, dims=-1)

    def run(model, use_ckpt):
        if use_ckpt:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing = False
        model.train()
        # Replay routing jitter for both runs; checkpointing must replay it too.
        torch.manual_seed(123)
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
    hidden = BatchedSwiGLUMoE(2, d_ffn, d_model=8).hidden_dim
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


@pytest.mark.parametrize("attention", ["sdpa", "flex"])
def test_future_tokens_do_not_change_prefix_logits(attention):
    """Detect causal-mask leaks using different suffixes with a shared prefix."""
    torch.manual_seed(13)
    model = tiny_model(attn_implementation=attention).eval()
    original = random_batch(seqlen=16)["input_ids"]
    changed = original.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 256
    with torch.no_grad():
        before = model(original)[0]
        after = model(changed)[0]
    torch.testing.assert_close(before[:, :8], after[:, :8], atol=1e-6, rtol=1e-5)
