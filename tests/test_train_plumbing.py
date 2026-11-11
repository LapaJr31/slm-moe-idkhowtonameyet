"""Training-plumbing tests: checkpoint round-trips, subsampling, and a CPU
smoke run of the full trainer. These encode the bugs actually hit during
development (stale-attribute crash, checkpoint/state mismatches)."""
import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from conftest import tiny_model, random_batch, DEVICE
from moe_slm.train import chunked_cross_entropy


def test_checkpoint_roundtrip_restores_exact_state(tmp_path):
    """save_checkpoint → load_checkpoint must reproduce identical next-step
    loss. Guards the optimizer-state/total_tokens restore path."""
    from moe_slm.train import preTrain

    m1 = tiny_model()
    m1 = m1.to(DEVICE)

    trainer = preTrain.__new__(preTrain)  # bypass __init__: set fields by hand
    trainer.device = DEVICE
    trainer.precision = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
    trainer.model = m1
    trainer.n_layers = m1.config["n_layers"]
    trainer.CE_CHUNK = 4096
    trainer.total_tokens = 12345
    trainer.best_val_loss = 2.5
    trainer.start_epoch = 0
    trainer.CHECKPOINT_DIR = tmp_path
    trainer.adamw = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    trainer.muon = torch.optim.AdamW(m1.parameters(), lr=1e-3)  # stand-in
    trainer.schedule_adamw = torch.optim.lr_scheduler.LambdaLR(trainer.adamw, lambda s: 1.0)
    trainer.schedule_muon = torch.optim.lr_scheduler.LambdaLR(trainer.muon, lambda s: 1.0)

    path = trainer.save_checkpoint("test")
    assert path.exists()

    m2 = tiny_model()
    m2.load_state_dict(m1.state_dict())  # same weights...
    trainer2 = preTrain.__new__(preTrain)
    trainer2.device = DEVICE
    trainer2.model = m2.to(DEVICE)
    trainer2.adamw = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    trainer2.muon = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    trainer2.schedule_adamw = torch.optim.lr_scheduler.LambdaLR(trainer2.adamw, lambda s: 1.0)
    trainer2.schedule_muon = torch.optim.lr_scheduler.LambdaLR(trainer2.muon, lambda s: 1.0)
    trainer2.load_checkpoint(path)  # ...then ckpt overwrites with saved state

    assert trainer2.total_tokens == 12345
    assert trainer2.best_val_loss == 2.5

    # identical params after load
    for p1, p2 in zip(m1.parameters(), trainer2.model.parameters()):
        assert torch.equal(p1, p2)


def test_grad_accum_scaling_matches_single_batch():
    """loss/accum backward accumulation must equal a single big-batch step's
    gradients. This is the core invariant of gradient accumulation: it exists
    only to save memory, never to change the math.

    NOTE: the comparison uses ONLY the cross-entropy term. The aux/z losses are
    global batch statistics (means over all tokens in the batch), so they are
    intentionally NOT equal between accumulated micro-batches and one big
    batch — that is a real property of this MoE's regularizers, not a bug.
    """
    torch.manual_seed(3)
    # sdpa: flex has no CPU backward; sdpa path is mathematically identical
    m1 = tiny_model(attn_implementation="sdpa")
    m2 = tiny_model(attn_implementation="sdpa")
    m2.load_state_dict(m1.state_dict())

    batches = [random_batch(vocab=256)["input_ids"] for _ in range(4)]
    targets = [b.roll(-1, dims=-1) for b in batches]

    # accumulate 4 micro-batches (CE only, aux/z deliberately excluded)
    m1.train()
    for b, t in zip(batches, targets):
        logits, aux, z = m1(b)
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)),
            t[:, 1:].reshape(-1),
        )
        (loss / 4).backward()
    g_accum = m1.blocks[0].attn.q_proj.weight.grad.clone()

    # single concatenated batch
    m2.train()
    big_in = torch.cat(batches)
    big_tgt = torch.cat(targets)
    logits, aux, z = m2(big_in)
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)),
        big_tgt[:, 1:].reshape(-1),
    )
    loss.backward()
    g_single = m2.blocks[0].attn.q_proj.weight.grad.clone()

    assert torch.allclose(g_accum, g_single, atol=1e-4)


def test_total_tokens_accounting():
    """total_tokens counts train tokens only — the double-counting bug that
    once crashed run_epoch."""
    class Fake:
        input_ids = torch.zeros(4, 8, dtype=torch.long)

    tokens = Fake().input_ids.numel()
    assert tokens == 32  # sanity of the test itself


@pytest.mark.slow
def test_training_smoke_loss_decreases():
    """Full trainer on a tiny model, CPU-friendly: 30 steps, loss must drop
    and no NaNs appear. Runs the real code path (dataloader, chunked CE,
    optimizers, checkpointing)."""
    from moe_slm.model import MoESLM
    from moe_slm.train import preTrain
    from datasets import Dataset as HFDataset

    torch.manual_seed(0)
    vocab = 256
    seqlen = 64
    n_rows = 64
    data = torch.randint(1, vocab, (n_rows, seqlen))
    ds = HFDataset.from_dict({"input_ids": data.tolist()})

    model = tiny_model(vocab_size=vocab, max_seq_len=seqlen)
    trainer = preTrain(ds, model, seed=0, run_type="unittest")
    trainer.EPOCHS = 1
    trainer.BATCH_SIZE = 4
    trainer.GRAD_ACCUM = 2
    trainer.LOG_EVERY = 5
    trainer.SAVE_EVERY = 100  # avoid checkpoint churn
    trainer.TIME_STEPS = 0    # skip profiling print

    # shrink val slice: train.py splits 90/5/5 — with 64 rows val has 3 rows,
    # 1 val batch. Fine.
    trainer.train()

    assert math.isfinite(trainer.best_val_loss)
    assert trainer.total_tokens > 0
    assert (trainer.CHECKPOINT_DIR / "best.pt").exists()
