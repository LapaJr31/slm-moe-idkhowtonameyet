"""Loss-function tests: chunked CE must be numerically identical to plain CE."""
import pytest
import torch
import torch.nn.functional as F

from moe_slm.train import chunked_cross_entropy
from moe_slm.model import chunked_linear_cross_entropy
from conftest import tiny_model, random_batch


@pytest.mark.parametrize("chunk", [1, 7, 100])
@pytest.mark.parametrize("ignored", ["some", "all"])
@pytest.mark.parametrize("bf16", [False, True])
def test_chunked_projection_loss_and_gradients(device, chunk, ignored, bf16):
    torch.manual_seed(12)
    hidden = torch.randn(2, 5, 16, device=device, requires_grad=True)
    weight = torch.randn(23, 16, device=device, requires_grad=True)
    targets = torch.randint(0, 23, (2, 5), device=device)
    targets[:, 2] = -100
    if ignored == "all":
        targets.fill_(-100)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        logits = F.linear(hidden, weight)
    expected = chunked_cross_entropy(logits, targets, chunk=chunk)
    expected_grads = torch.autograd.grad(expected, (hidden, weight))
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        actual = chunked_linear_cross_entropy(hidden, weight, targets, chunk=chunk)
    actual_grads = torch.autograd.grad(actual, (hidden, weight))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    for got, ref in zip(actual_grads, expected_grads):
        # BF16 chunked matmuls accumulate weight gradients in a different order.
        torch.testing.assert_close(got, ref, atol=5e-3 if bf16 else 2e-6, rtol=1e-2 if bf16 else 1e-5)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        evaluated = chunked_linear_cross_entropy(hidden, weight, targets, chunk=chunk)
    torch.testing.assert_close(evaluated, expected.detach(), atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("mask_kind", [None, "all_tokens", "padded"])
def test_model_loss_path_matches_logits_and_all_gradients(compiled, tied, mask_kind):
    torch.manual_seed(13)
    model = tiny_model(attn_implementation="sdpa", tie_embeddings=tied)
    ids = random_batch(batch_size=2, seqlen=5)["input_ids"]
    attention_mask = None if mask_kind is None else torch.ones_like(ids)
    labels = ids
    if mask_kind == "padded":
        attention_mask[0, :2] = 0
        attention_mask[1, -1] = 0
        labels = ids.masked_fill(attention_mask == 0, -100)
    logits, aux, z = model(ids, attention_mask=attention_mask)
    expected = chunked_cross_entropy(logits, labels) + 0.01 * (aux + z)
    expected.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    # AOTAutograd exercises compiled forward/backward without GPU codegen.
    run = torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=False) if compiled else model
    lm, aux, z = run(ids, attention_mask=attention_mask, labels=labels, ce_chunk=3)
    actual = lm + 0.01 * (aux + z)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, grads[name], atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("chunk", [1, 7, 512, 4096, 10**9])
@pytest.mark.parametrize("vocab", [13, 50_000])
def test_chunked_ce_matches_plain_ce(chunk, vocab, device):
    """Any chunk size must give the same loss as the monolithic computation.
    Includes the degenerate chunk=1 (per-row CE) and chunk > n (single slice)."""
    torch.manual_seed(0)
    logits = torch.randn(4, 16, vocab, device=device)
    targets = torch.randint(0, vocab, (4, 16), device=device)

    plain = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, vocab),
        targets[:, 1:].reshape(-1),
    )
    chunked = chunked_cross_entropy(logits, targets, chunk=chunk)

    assert torch.allclose(plain, chunked, atol=1e-6), (
        f"chunk={chunk}, vocab={vocab}: {plain.item()} vs {chunked.item()}"
    )


def test_chunked_ce_gradients_match(device):
    """Gradient equality matters more than the forward value — a chunking bug
    that only corrupts backward would pass a forward-only test."""
    torch.manual_seed(1)
    logits_a = torch.randn(2, 32, 97, device=device, requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 97, (2, 32), device=device)

    F.cross_entropy(
        logits_a[:, :-1].float().reshape(-1, 97),
        targets[:, 1:].reshape(-1),
    ).backward()
    chunked_cross_entropy(logits_b, targets, chunk=100).backward()

    assert torch.allclose(logits_a.grad, logits_b.grad, atol=1e-5)


def test_chunked_ce_handles_ignore_index(device):
    """-100 targets are excluded from the mean (used only if padding is ever
    reintroduced)."""
    logits = torch.randn(1, 8, 10, device=device)
    targets = torch.full((1, 8), -100, device=device)
    targets[:, 3:] = torch.randint(0, 10, (1, 5), device=device)

    plain = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, 10),
        targets[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    chunked = chunked_cross_entropy(logits, targets, chunk=3)
    assert torch.allclose(plain, chunked, atol=1e-6)


@pytest.mark.parametrize("chunk", [1, 7, 100])
def test_all_ignored_targets_have_zero_loss_and_gradients(device, chunk):
    """An empty supervised batch must not introduce NaNs or update parameters."""
    logits = torch.randn(2, 8, 13, device=device, requires_grad=True)
    targets = torch.full((2, 8), -100, device=device)
    loss = chunked_cross_entropy(logits, targets, chunk=chunk)
    torch.testing.assert_close(loss, loss.new_zeros(()))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
