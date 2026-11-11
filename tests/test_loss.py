"""Loss-function tests: chunked CE must be numerically identical to plain CE."""
import pytest
import torch
import torch.nn.functional as F

from moe_slm.train import chunked_cross_entropy


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
