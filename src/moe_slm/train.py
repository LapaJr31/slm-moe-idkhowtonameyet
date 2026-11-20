import itertools
import math
import os
import time
import warnings

import mlflow
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler

from datasets import load_from_disk, Dataset
from mlflow.data.huggingface_dataset import from_huggingface


from pathlib import Path
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
)
from .model import MoESLM, compile_model, get_causal_block_mask


def _fmt_hms(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def chunked_cross_entropy(logits, targets, chunk=4096, ignore_index=-100):
    B, T, V = logits.shape
    flat_logits = logits[:, :-1].reshape(-1, V)
    flat_targets = targets[:, 1:].reshape(-1)
    n = flat_logits.shape[0]

    total = flat_logits.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk):
        li = flat_logits[i:i + chunk].float()
        ti = flat_targets[i:i + chunk]
        total = total + F.cross_entropy(li, ti, ignore_index=ignore_index, reduction="sum")
    n_valid = (flat_targets != ignore_index).sum().clamp(min=1)
    return total / n_valid


class ExpertMuon(optim.Muon):
    def __init__(self, params, **kwargs):
        self.sources = list(params)
        self.slices = [
            [torch.nn.Parameter(part) for part in p.unbind(0)] if p.ndim == 3 else [p]
            for p in self.sources
        ]
        super().__init__([part for parts in self.slices for part in parts], **kwargs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for source, parts in zip(self.sources, self.slices):
            if source.ndim == 3:
                for i, part in enumerate(parts):
                    part.grad = None if source.grad is None else source.grad[i]
        result = super().step()
        return loss if closure is not None else result

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for source in self.sources:
            if source.ndim == 3 and source.grad is not None:
                if set_to_none:
                    source.grad = None
                else:
                    source.grad.zero_()


class EpochSampler(Sampler):
    """Recreate an epoch's permutation and start at a consumed sample offset."""
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.offset = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        return itertools.islice(indices, self.offset, None)

    def __len__(self):
        return max(0, len(self.dataset) - self.offset)


class preTrain:
    EPOCHS = 1
    BATCH_SIZE = 128        # micro-batch
    GRAD_ACCUM = 1        # effective batch BATCH_SIZE * GRAD_ACCUM
    TIME_STEPS = 50        # how many optimizer steps to profile timing for
    AUX_COEF = 0.01       # load-balancing loss weight
    Z_COEF = 0.01         # router confidence penalty weight
    LR_MUON = 0.02
    LR_ADAMW = 3e-4
    SEQLEN = 2048
    CE_CHUNK = 8192
    SAVE_EVERY = 1000
    LOG_EVERY = 50
    CHECKPOINT_DIR = Path("checkpoints")

    def __init__(self, dataset_tokenized: "Path | Dataset", model : MoESLM, seed : int = 99, run_type: str = "dry_run"):
        torch.manual_seed(seed)

        mlflow.set_experiment("moe-slm-pretrain")
        mlflow.start_run(run_name=f"pretrain-{run_type}")
        mlflow.set_tag("run_type", run_type)
        mlflow.log_params({
            "epochs": self.EPOCHS,
            "batch_size": self.BATCH_SIZE,
            "grad_accum": self.GRAD_ACCUM,
            "ce_chunk": self.CE_CHUNK,
            "seqlen": self.SEQLEN,
            "lr_muon": self.LR_MUON,
            "lr_adamw": self.LR_ADAMW,
            "seed": seed,
            "aux_coef": self.AUX_COEF,
            "z_coef": self.Z_COEF,
            **{f"model/{k}": v for k, v in model.config.items()},
            "model/param_count": sum(p.numel() for p in model.parameters()),
        })

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = torch.bfloat16

        model = model.to(self.device)
        # model.gradient_checkpointing_enable()

        if model.attn_implementation == "flex":
            get_causal_block_mask(self.SEQLEN, next(model.parameters()).device)

        print(f"Compiling model")
        model = compile_model(model, mode="max-autotune", strategy="whole")
        if self.device == "cuda":
            torch.cuda.synchronize()
        print(f"Finished compilation")

        if isinstance(dataset_tokenized, Dataset):
            dataset = dataset_tokenized
        else:
            dataset = load_from_disk(dataset_tokenized)
        dataset = dataset.with_format("torch")
        self.data_fingerprint = dataset._fingerprint
        self.seed = seed
        n = len(dataset)
        train_slice = dataset.select(range(int(n * 0.90)))
        val_slice = dataset.select(range(int(n * 0.90), int(n * 0.95)))
        test_slice = dataset.select(range(int(n * 0.95), n))

        source = (
            str(Path(dataset_tokenized).resolve())
            if not isinstance(dataset_tokenized, Dataset)
            else None
        )

        for split, context in [
            (train_slice, "training"),
            (val_slice, "validation"),
            (test_slice, "testing"),
        ]:
            tracked = from_huggingface(
                split,
                source=source,
                name=f"moe-slm-{context}",
            )
            mlflow.log_input(tracked, context=context)


        mlflow.log_dict(
            {
                "sources": ["lapkaaaa/smol-10B"],
                "revision": "v1",
                "tokenizer": "tokenizer-32768",
            },
            "data/dataset_data.json",
        )

        muon_params = self.get_muon_params(model)
        adamw_params = self.get_adamw_params(model, muon_params)

        muon_ids = {id(p) for p in muon_params}
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(f"Optimizer {'Muon' if id(p) in muon_ids else 'AdamW'}: {name} {tuple(p.shape)}")

        self.muon = ExpertMuon(
            muon_params,
            lr=self.LR_MUON,
            momentum=0.95,
            weight_decay=0.01,
        )
        self.adamw = optim.AdamW(
            adamw_params,
            lr=self.LR_ADAMW,
            weight_decay=0.01,
            fused=self.device == "cuda",
        )

        self.train_dl = self.create_dataloader(
            train_slice,
            batch=self.BATCH_SIZE,
            num_workers=4,
            shuffle=True,
            seed=seed,
        )
        self.val_dl = self.create_dataloader(
            val_slice,
            batch=self.BATCH_SIZE,
            num_workers=4,
            shuffle=False,
        )
        self.test_dl = self.create_dataloader(
            test_slice,
            batch=self.BATCH_SIZE,   # 512 would OOM at 1B scale: logits alone are ~B*2047*50k
            num_workers=4,
            shuffle=False,
        )

        total_steps = math.ceil(len(self.train_dl) / self.GRAD_ACCUM) * self.EPOCHS
        self.schedule_adamw = self.make_wsd_scheduler(
            optimizer= self.adamw,
            total_steps=total_steps,
        )
        self.schedule_muon = self.make_wsd_scheduler(
            optimizer= self.muon,
            total_steps=total_steps,
        )

        self.model = model
        self.n_layers = model.config["n_layers"]

        self.optim_steps_train = len(self.train_dl)
        self.optim_steps_val = len(self.val_dl)

        self.tokens_seen = 0
        self.total_tokens = 0

        self.timing = {"bwd": 0.0, "opt": 0.0}
        self.start_epoch = 0
        self.epoch_batches = 0
        self.best_val_loss = float("inf")
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_wsd_scheduler(
        optimizer,
        total_steps : int,
        warmup_ratio=0.03,
        decay_ratio=0.20,
        min_lr_ratio=0.10,
    ):
        warmup = max(1, int(total_steps * warmup_ratio))
        decay = max(1, int(total_steps * decay_ratio))
        decay_start = total_steps - decay

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup

            if step < decay_start:
                return 1.0

            progress = min(1.0, (step - decay_start) / decay)

            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0+ math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def get_muon_params(model):
        return [
            p for name, p in model.named_parameters()
            if (
                p.requires_grad
                and (p.ndim == 2 or (p.ndim == 3 and name.rsplit(".", 1)[-1] in ("w13", "w2")))
                and not any(
                    excluded in name.lower()
                    for excluded in (
                        "embed",
                        "tok_emb",
                        "lm_head",
                        "router",
                    )
                )
            )
        ]

    @staticmethod
    def get_adamw_params(model, muon_params):
        muon_ids = {id(p) for p in muon_params}
        return [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in muon_ids
        ]

    @staticmethod
    def create_dataloader(dataset_slice, batch:int, num_workers:int, shuffle:bool, seed:int = 99):
        return DataLoader(
            dataset=dataset_slice,
            batch_size=batch,
            num_workers=num_workers,
            pin_memory=True,
            sampler=EpochSampler(dataset_slice, seed) if shuffle else None,
            generator=torch.Generator().manual_seed(seed),
            drop_last=True,
        )

    def load_checkpoint(self, path: Path, *, benchmark: bool = False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if benchmark:
            # Benchmark updates are disposable and can use a different batch size.
            self.model.load_state_dict(ckpt["model"])
            self.adamw.load_state_dict(ckpt["optimizer_adamw"])
            self.muon.load_state_dict(ckpt["optimizer_muon"])
            mlflow.log_params({"benchmark_checkpoint": str(path)})
            return

        progress = ckpt.get("data_progress")
        if progress is not None and hasattr(self, "data_fingerprint"):
            expected = (self.data_fingerprint, self.seed, self.BATCH_SIZE, self.GRAD_ACCUM)
            saved = (progress["dataset"], progress["seed"], progress["batch_size"], progress["grad_accum"])
            if saved != expected:
                raise ValueError("Cannot resume: dataset, seed, batch size, or gradient accumulation differs from the checkpoint")
            batches = progress["batches"]
            if not 0 <= batches <= len(self.train_dl) or (batches != len(self.train_dl) and batches % self.GRAD_ACCUM):
                raise ValueError("Cannot resume: checkpoint batch position is invalid")
            if progress.get("epochs") != self.EPOCHS:
                warnings.warn("Epoch count changed; the restored scheduler will follow the new training budget.")
        elif progress is None:
            warnings.warn("Legacy checkpoint has no data position or shuffle state; resuming restarts the saved epoch and may repeat data.")

        self.model.load_state_dict(ckpt["model"])
        self.adamw.load_state_dict(ckpt["optimizer_adamw"])
        self.muon.load_state_dict(ckpt["optimizer_muon"])
        self.schedule_adamw.load_state_dict(ckpt["scheduler_adamw"])
        self.schedule_muon.load_state_dict(ckpt["scheduler_muon"])

        self.total_tokens = ckpt["tokens_seen"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.start_epoch = ckpt.get("start_epoch", 0)
        self.epoch_batches = progress["batches"] if progress is not None else 0
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if self.device == "cuda" and ckpt.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng_state"]])

        mlflow.log_params({"resumed_from": str(path)})

    def save_checkpoint(self, name: str):
        path = self.CHECKPOINT_DIR / f"{name}.pt"
        state = {
            "model": self.model.state_dict(),
            "optimizer_adamw": self.adamw.state_dict(),
            "optimizer_muon": self.muon.state_dict(),
            "scheduler_adamw": self.schedule_adamw.state_dict(),
            "scheduler_muon": self.schedule_muon.state_dict(),
            "tokens_seen": self.total_tokens,
            "best_val_loss": self.best_val_loss,
            "start_epoch": self.start_epoch,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if self.device == "cuda" else None,
        }
        if hasattr(self, "data_fingerprint"):
            state["data_progress"] = {
                "dataset": self.data_fingerprint, "seed": self.seed,
                "batch_size": self.BATCH_SIZE, "grad_accum": self.GRAD_ACCUM,
                "epochs": self.EPOCHS, "batches": self.epoch_batches,
            }
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)
        offloader = getattr(self, "offloader", None)
        if offloader is not None:
            try:
                offloader.enqueue(path)
            except (OSError, RuntimeError) as exc:
                print(f"[offload] could not queue {path.name}: {exc}; local checkpoint remains available", flush=True)
        return path

    def run_epoch(
        self,
        loader,
        train=True,
        benchmark_steps=0,
        benchmark_warmup=5,
    ):
        self.model.train(train)

        if len(loader) == 0:
            raise ValueError("DataLoader is empty")
        if benchmark_steps and (not train or not 0 <= benchmark_warmup < benchmark_steps):
            raise ValueError("Benchmark requires training steps after warmup")

        total_loss = torch.zeros((), device=self.device)
        total_aux_loss = torch.zeros_like(total_loss)
        total_lm_loss = torch.zeros_like(total_loss)
        total_z_loss = torch.zeros_like(total_loss)

        total_tokens = 0

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[green]{task.fields[tok_s]:,.0f} tok/s"),
            TextColumn("[yellow]Loss: {task.fields[loss]:.4f}"),
            TextColumn("[yellow]LM: {task.fields[lm_loss]:.4f}"),
            TextColumn("[yellow]Aux: {task.fields[aux_loss]:.4f}"),
            TextColumn("[yellow]Grad: {task.fields[grad_norm]:.3f}"),
            TextColumn("[yellow]Mem: {task.fields[gpu_mem_gb]:.2f}GB"),
            TextColumn("[cyan]{task.fields[elapsed]}[/] | [magenta]eta {task.fields[eta]}[/]"),
            expand=False,
            refresh_per_second=2,
        ) as progress:

            task = progress.add_task(
                "Train" if train else "Val",
                total=len(loader),
                tok_s=0.0,
                loss=0.0,
                lm_loss=0.0,
                aux_loss=0.0,
                grad_norm=0.0,
                gpu_mem_gb=0.0,
                elapsed="0:00",
                eta="-",
            )

            start = time.perf_counter()

            accum = self.GRAD_ACCUM if train else 1
            n_steps = math.ceil(len(loader) / accum)
            if benchmark_steps:
                if n_steps < benchmark_steps:
                    raise ValueError("Not enough batches for the benchmark")
                n_steps = benchmark_steps
            benchmark_tokens = 0
            progress.update(task, total=n_steps)

            it = iter(loader)
            for step in range(n_steps):
                if benchmark_steps and step == benchmark_warmup:
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    benchmark_start = time.perf_counter()
                micros = list(itertools.islice(it, accum))
                if not micros:
                    break

                should_log = not benchmark_steps and (
                    (step + 1) % self.LOG_EVERY == 0
                    or step + 1 == n_steps
                )

                if train and not benchmark_steps and step + 1 == self.TIME_STEPS:
                    t = self.timing
                    total = t["bwd"] + t["opt"]
                    print(
                        f"\n[timing over steps 5-{self.TIME_STEPS}, wall-clock incl. sync] "
                        f"fwd+bwd+loss: {t['bwd']:.2f}s "
                        f"({t['bwd']/total*100:.1f}%) | "
                        f"optimizer: {t['opt']:.2f}s ({t['opt']/total*100:.1f}%)"
                    )
                if train:
                    self.adamw.zero_grad(set_to_none=accum == 1)
                    self.muon.zero_grad(set_to_none=accum == 1)
                    if accum > 1:
                        for p in self.model.parameters():
                            if p.requires_grad and p.grad is None:
                                p.grad = torch.zeros_like(p)

                step_loss = torch.zeros((), device=self.device)
                step_aux = torch.zeros_like(step_loss)
                step_lm = torch.zeros_like(step_loss)
                step_z = torch.zeros_like(step_loss)
                step_tok = 0

                # skip the first few steps: they absorb torch.compile recompiles
                profile = train and not benchmark_steps and 5 <= step < self.TIME_STEPS
                for batch in micros:
                    batch = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in batch.items()
                    }

                    t0 = time.perf_counter()
                    with torch.set_grad_enabled(train):
                        with torch.autocast("cuda", dtype=self.precision):
                            lm_loss, aux_loss, z_loss = self.model(
                                input_ids=batch["input_ids"],
                                labels=batch["input_ids"],
                                ce_chunk=self.CE_CHUNK,
                            )

                        loss = (
                            lm_loss
                            + self.AUX_COEF * aux_loss.float()
                            + self.Z_COEF * z_loss.float()
                        )
                        if train:
                            (loss / len(micros)).backward()
                    if profile and self.device == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    if profile:
                        self.timing["bwd"] += t1 - t0

                    if train:
                        self.total_tokens += batch["input_ids"].numel()
                    step_tok += batch["input_ids"].numel()
                    step_loss += loss.detach()
                    step_aux += aux_loss.detach()
                    step_lm += lm_loss.detach()
                    step_z += z_loss.detach()

                loss, lm_loss, aux_loss, z_loss = (
                    step_loss / len(micros), step_lm / len(micros),
                    step_aux / len(micros), step_z / len(micros),
                )
                total_tokens += step_tok
                if benchmark_steps and step >= benchmark_warmup:
                    benchmark_tokens += step_tok
                total_loss += loss
                total_aux_loss += aux_loss
                total_lm_loss += lm_loss
                total_z_loss += z_loss

                grad_norm = None
                if train:
                    t2 = time.perf_counter()
                    if should_log:
                        grad_norm = torch.nn.utils.get_total_norm([
                            p.grad for p in self.model.parameters() if p.grad is not None
                        ])

                    self.adamw.step()
                    self.muon.step()

                    self.schedule_adamw.step()
                    self.schedule_muon.step()

                    if not benchmark_steps:
                        self.epoch_batches += len(micros)

                    if profile and self.device == "cuda":
                        torch.cuda.synchronize()
                    if profile:
                        self.timing["opt"] += time.perf_counter() - t2

                    if not benchmark_steps and ((self.epoch_batches // accum) % self.SAVE_EVERY == 0 or step + 1 == n_steps):
                        self.save_checkpoint("latest")

                elapsed = time.perf_counter() - start
                eta = elapsed / (step + 1) * (n_steps - step - 1)
                progress.update(
                    task,
                    completed=step + 1,
                    elapsed=_fmt_hms(elapsed),
                    eta=_fmt_hms(eta) if eta > 0 else "0:00",
                )

                if should_log:
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    tok_s = total_tokens / elapsed
                    gpu_mem_gb = (
                        torch.cuda.memory_allocated() / 1024**3
                        if self.device == "cuda" else 0.0
                    )
                    peak_mem_gb = (
                        torch.cuda.max_memory_allocated() / 1024**3
                        if self.device == "cuda" else 0.0
                    )

                    progress.update(
                        task,
                        tok_s=tok_s,
                        loss=loss.item(),
                        lm_loss=lm_loss.item(),
                        aux_loss=aux_loss.item() / self.n_layers,
                        grad_norm=(
                            grad_norm.item()
                            if grad_norm is not None
                            else 0.0
                        ),
                        adamw_lr=self.adamw.param_groups[0]["lr"],
                        muon_lr=self.muon.param_groups[0]["lr"],
                        gpu_mem_gb=gpu_mem_gb,
                        peak_mem_gb=peak_mem_gb,
                    )

                    if train:
                        mlflow.log_metrics({
                            "train/loss": loss.item(),
                            "train/lm_loss": lm_loss.item(),
                            "train/aux_loss": aux_loss.item() / self.n_layers,
                            "train/z_loss": z_loss.item(),
                            "train/grad_norm": (
                                grad_norm.item()
                                if grad_norm is not None
                                else 0.0
                            ),
                            "train/lr_adamw": self.adamw.param_groups[0]["lr"],
                            "train/lr_muon": self.muon.param_groups[0]["lr"],
                            "train/tok_s": tok_s,
                            "train/tokens_seen": self.total_tokens,
                        }, step=self.total_tokens)
                    else:
                        mlflow.log_metrics({
                            "val/running_loss": (total_loss / (step + 1)).item(),
                            "val/running_lm_loss": (total_lm_loss / (step + 1)).item(),
                        }, step=self.total_tokens + total_tokens)

        if benchmark_steps:
            if self.device == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - benchmark_start
            peak = torch.cuda.max_memory_allocated() / 1024**3 if self.device == "cuda" else 0.0
            print(
                f"[benchmark] batch={self.BATCH_SIZE} accum={accum} "
                f"ce_chunk={self.CE_CHUNK} timed_steps={n_steps - benchmark_warmup} "
                f"tokens={benchmark_tokens:,} seconds={seconds:.3f} "
                f"tok/s={benchmark_tokens / seconds:,.0f} peak_allocated={peak:.2f} GiB"
            )

        return (
            (total_loss / (step + 1)).item(),
            (total_aux_loss / (step + 1)).item(),
            (total_lm_loss / (step + 1)).item(),
            (total_z_loss / (step + 1)).item(),
            total_tokens,
        )

    def run_test(self):
        best_path = self.CHECKPOINT_DIR / "best.pt"
        if best_path.exists():
                ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["model"])
                print(f"\n[test] loaded best checkpoint ({best_path})")

        self.model.eval()
        loader = self.test_dl
        if len(loader) == 0:
            print("[test] test split is empty, skipping")
            return

        total_lm = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=self.precision):
                    lm_loss, aux_loss, z_loss = self.model(
                        input_ids=batch["input_ids"],
                        labels=batch["input_ids"],
                        ce_chunk=self.CE_CHUNK,
                    )
                    total_lm += lm_loss.item()
                    n_batches += 1

        avg_lm = total_lm / n_batches
        ppl = math.exp(avg_lm)
        print(
            f"\n{'=' * 50}\nTEST (held-out, best checkpoint)\n"
            f"{'-' * 40}\n"
            f"LM Loss:        {avg_lm:.4f}\n"
            f"Perplexity:     {ppl:.2f}\n"
            f"Batches:        {n_batches}\n"
            f"{'=' * 50}"
        )
        mlflow.log_metrics({
            "test/lm_loss": avg_lm,
            "test/perplexity": ppl,
        })

    def train(self):
        train_vals = {"Train": True, "Val": False}
        print(f"\n{'-' * 50}\nStarting training")
        ran_epoch = False
        for epoch in range(self.start_epoch, self.EPOCHS):
            ran_epoch = True
            self.start_epoch = epoch
            sampler = self.train_dl.sampler
            sampler.epoch = epoch
            sampler.offset = self.epoch_batches * self.BATCH_SIZE
            if self.epoch_batches:
                print(f"Resuming epoch {epoch + 1} after {self.epoch_batches} training batches")
            print(f"Epoch {epoch + 1}/{self.EPOCHS}")

            for key, flag in train_vals.items():
                loader = self.train_dl if flag else self.val_dl

                if flag and len(loader) == 0 and self.epoch_batches:
                    print("Training for this epoch is complete; running validation")
                    continue

                if self.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                start = time.perf_counter()
                (epoch_loss, epoch_aux_loss, epoch_lm_loss, epoch_z_loss, total_tokens) = self.run_epoch(
                    loader=loader,
                    train=flag,
                )

                if self.device == "cuda":
                    torch.cuda.synchronize()

                epoch_time = time.perf_counter() - start
                time_per_step = epoch_time / len(loader)
                tokens_per_second = total_tokens / epoch_time
                perplexity = math.exp(epoch_lm_loss)
                peak_mem_gb = (torch.cuda.max_memory_allocated() / 1024**3 if self.device == "cuda" else 0.0)

                print(
                    f"\nEpoch {epoch + 1} | {key}\n"
                    f"{'-' * 40}\n"
                    f"Total Loss:     {epoch_loss:.4f}\n"
                    f"LM Loss:        {epoch_lm_loss:.4f}\n"
                    f"Aux Loss:       {epoch_aux_loss / self.n_layers:.4f}\n"
                    f"Z Loss:         {epoch_z_loss:.4f}\n"
                    f"Perplexity:     {perplexity:.2f}\n"
                    f"Avg Throughput: {tokens_per_second:,.0f} tok/s\n"
                    f"Time per Batch: {time_per_step:.3f}s\n"
                    f"Peak GPU Mem:   {peak_mem_gb:.2f} GB\n"
                    f"Tokens Seen:    {self.total_tokens / 1e9:.3f}B\n"
                )

                prefix = "train" if flag else "val"
                mlflow.log_metrics({
                    f"{prefix}/loss": epoch_loss,
                    f"{prefix}/lm_loss": epoch_lm_loss,
                    f"{prefix}/aux_loss": epoch_aux_loss / self.n_layers,
                    f"{prefix}/z_loss": epoch_z_loss,
                    f"{prefix}/perplexity": perplexity,
                    f"{prefix}/tok_s": tokens_per_second,
                    f"{prefix}/peak_mem_gb": peak_mem_gb,
                    "tokens_seen": self.total_tokens,
                }, step=epoch)

                if not flag:
                    self.start_epoch = epoch + 1
                    self.epoch_batches = 0
                    if epoch_lm_loss < self.best_val_loss:
                        self.best_val_loss = epoch_lm_loss
                        best_path = self.save_checkpoint("best")
                        mlflow.log_artifact(best_path, artifact_path="checkpoints")
                        mlflow.log_metrics({"best_val_loss": self.best_val_loss}, step=epoch)
                        print(f"! New best val loss, saved {best_path}")
                    self.save_checkpoint("latest")

        if ran_epoch:
            self.run_test()

        mlflow.end_run()

#todo
