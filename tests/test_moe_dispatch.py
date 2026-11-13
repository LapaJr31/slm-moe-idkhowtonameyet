import copy

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters

from moe_slm.model import BatchedSwiGLUMoE, MoESLM
from moe_slm.train import ExpertMuon, preTrain


def small_model():
    return MoESLM(32, d_model=16, n_layers=1, n_q_heads=2, n_kv_heads=1,
                  num_experts=4, d_ffn=24, max_seq_len=32, attn_implementation="sdpa")


def test_balanced_and_collapsed():
    moe = BatchedSwiGLUMoE(4, 24, d_model=4, top_k=1)
    with torch.no_grad():
        moe.router.weight.copy_(20 * torch.eye(4))
    _, aux, z = moe(torch.eye(4).repeat(8, 1))
    torch.testing.assert_close(aux, torch.tensor(1.0))
    assert z.ndim == 0 and torch.isfinite(z)
    out, aux, z = moe(torch.eye(4)[0].repeat(32, 1))
    assert out.shape == (32, 4)
    torch.testing.assert_close(aux, torch.tensor(4.0))
    assert moe.drop_rate > 0
    (out.square().mean() + aux + z).backward()
    assert all(torch.isfinite(p.grad).all() for p in moe.parameters())


@pytest.mark.slow
def test_dynamic_fullgraph_no_recompile():
    torch._dynamo.reset()
    counters.clear()
    model = small_model()
    compiled = torch.compile(model, dynamic=True, mode="default", fullgraph=True)
    for batch in (2, 3):
        logits, aux, z = compiled(torch.randint(32, (batch, 8)))
        (logits.square().mean() + .01 * aux + .01 * z).backward()
        model.zero_grad(set_to_none=True)
        assert counters["stats"]["unique_graphs"] == 1
        assert not counters["graph_break"]


def test_muon_split_updates_and_roundtrips():
    model = small_model()
    muon = preTrain.get_muon_params(model)
    adamw = preTrain.get_adamw_params(model, muon)
    ids = {id(p) for p in muon}
    assert not ids.intersection(id(p) for p in adamw)
    assert len(muon) + len(adamw) == len(list(model.parameters()))
    moe = model.blocks[0].moe
    assert id(moe.w13) in ids and id(moe.w2) in ids
    assert id(model.tok_emb.weight) not in ids
    assert id(model.lm_head.weight) not in ids
    assert id(moe.router.weight) not in ids
    optimizer = ExpertMuon(muon, lr=.01)
    before = moe.w13.detach().clone()
    for p in muon:
        p.grad = torch.randn_like(p)
    optimizer.step()
    assert not torch.equal(before, moe.w13)
    state = copy.deepcopy(optimizer.state_dict())
    optimizer.load_state_dict(state)
    optimizer.zero_grad()
    assert all(p.grad is None for p in muon)


@pytest.mark.slow
def test_random_dataset_50_steps():
    torch.manual_seed(17)
    model = small_model()
    muon = preTrain.get_muon_params(model)
    optimizers = [ExpertMuon(muon, lr=.01),
                  torch.optim.AdamW(preTrain.get_adamw_params(model, muon), lr=.01)]
    data = torch.randint(32, (8, 8))
    losses = []
    for _ in range(50):
        for optimizer in optimizers:
            optimizer.zero_grad()
        logits, aux, z = model(data)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 32), data[:, 1:].reshape(-1))
        loss = loss + preTrain.AUX_COEF * aux + preTrain.Z_COEF * z
        assert torch.isfinite(loss)
        losses.append(loss.item())
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
    assert sum(losses[-5:]) < sum(losses[:5])
    print(f"50-step random dataset loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
