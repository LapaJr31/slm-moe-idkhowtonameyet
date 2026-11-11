import itertools
import math
import time

import mlflow
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint_sequential

from datasets import load_from_disk, Dataset


from pathlib import Path
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
)
from .model import MoESLM, compile_model


def _fmt_hms(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class preTrain:
    EPOCHS = 1
    BATCH_SIZE = 8        # micro-batch
    GRAD_ACCUM = 16        # effective batch 64
    AUX_COEF = 0.05       # load-balancing loss weight
    LR_MUON = 0.02
    LR_ADAMW = 3e-4
    SEQLEN = 2048
    SAVE_EVERY = 100
    LOG_EVERY = 50
    CHECKPOINT_DIR = Path("checkpoints")

    def __init__(self, dataset_tokenized : Path, model : MoESLM, seed : int = 99, run_type: str = "dry_run"):
        torch.manual_seed(seed)

        mlflow.set_experiment("moe-slm-pretrain")
        mlflow.start_run(run_name=f"pretrain-{run_type}")
        mlflow.set_tag("run_type", run_type)
        mlflow.log_params({
            "epochs": self.EPOCHS,
            "batch_size": self.BATCH_SIZE,
            "seqlen": self.SEQLEN,
            "lr_muon": self.LR_MUON,
            "lr_adamw": self.LR_ADAMW,
            "seed": seed,
            "aux_coef": self.AUX_COEF,
            **{f"model/{k}": v for k, v in model.config.items()},
            "model/param_count": sum(p.numel() for p in model.parameters()),
        })

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = torch.bfloat16
        self.criterion = F.cross_entropy

        model = model.to(self.device)
        model = compile_model(model = model,  mode = "default")

        if isinstance(dataset_tokenized, Dataset):
            dataset = dataset_tokenized
        else:
            dataset = load_from_disk(dataset_tokenized)
        dataset = dataset.with_format("torch")
        n = len(dataset)
        train_slice = dataset.select(range(int(n * 0.90)))
        val_slice = dataset.select(range(int(n * 0.90), int(n * 0.95)))
        test_slice = dataset.select(range(int(n * 0.95), n))

        muon_params = self.get_muon_params(model)
        adamw_params = self.get_adamw_params(model, muon_params)


        self.muon = optim.Muon(
            muon_params,
            lr=self.LR_MUON,
            momentum=0.95,
            weight_decay=0.01,
        )
        self.adamw = optim.AdamW(
            adamw_params,
            lr=self.LR_ADAMW,
            weight_decay=0.01,
        )

        self.train_dl = self.create_dataloader(
            train_slice,
            batch=self.BATCH_SIZE,
            num_workers=4,
            shuffle=True,
        )
        self.val_dl = self.create_dataloader(
            val_slice,
            batch=self.BATCH_SIZE,
            num_workers=4,
            shuffle=False,
        )
        self.test_dl = self.create_dataloader(
            test_slice,
            batch=512,
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

        self.optim_steps_train = len(self.train_dl)
        self.optim_steps_val = len(self.val_dl)

        self.tokens_seen = 0
        self.total_tokens = 0
        self.start_epoch = 0
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
                and p.ndim == 2
                and not any(
                    excluded in name.lower()
                    for excluded in (
                        "embed",
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
    def create_dataloader(dataset_slice, batch:int, num_workers:int, shuffle:bool):
        return DataLoader(
            dataset=dataset_slice,
            batch_size=batch,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=True,
        )

    def load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model"])
        self.adamw.load_state_dict(ckpt["optimizer_adamw"])
        self.muon.load_state_dict(ckpt["optimizer_muon"])
        self.schedule_adamw.load_state_dict(ckpt["scheduler_adamw"])
        self.schedule_muon.load_state_dict(ckpt["scheduler_muon"])

        self.total_tokens = ckpt["tokens_seen"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.start_epoch = ckpt.get("start_epoch", 0)

        mlflow.log_params({"resumed_from": str(path)})

    def save_checkpoint(self, name: str):
        path = self.CHECKPOINT_DIR / f"{name}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer_adamw": self.adamw.state_dict(),
            "optimizer_muon": self.muon.state_dict(),
            "scheduler_adamw": self.schedule_adamw.state_dict(),
            "scheduler_muon": self.schedule_muon.state_dict(),
            "tokens_seen": self.total_tokens,
            "best_val_loss": self.best_val_loss,
            "start_epoch": self.start_epoch,
        }, path)
        return path

    def run_epoch(
        self,
        loader,
        train=True,
    ):
        self.model.train(train)

        if len(loader) == 0:
            raise ValueError("DataLoader is empty")

        total_loss = torch.zeros((), device=self.device)
        total_aux_loss = torch.zeros_like(total_loss)
        total_lm_loss = torch.zeros_like(total_loss)

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
            TextColumn("[cyan]{task.fields[elapsed]}[/]:[magenta]{task.fields[eta]}[/]"),
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
            progress.update(task, total=n_steps)

            it = iter(loader)
            for step in range(n_steps):
                micros = list(itertools.islice(it, accum))
                if not micros:
                    break

                should_log = (
                    (step + 1) % self.LOG_EVERY == 0
                    or step + 1 == n_steps
                )
                if train:
                    self.adamw.zero_grad(set_to_none=True)
                    self.muon.zero_grad(set_to_none=True)

                step_loss = torch.zeros((), device=self.device)
                step_aux = torch.zeros_like(step_loss)
                step_lm = torch.zeros_like(step_loss)
                step_tok = 0

                for batch in micros:
                    batch = {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in batch.items()
                    }

                    with torch.set_grad_enabled(train):
                        with torch.autocast("cuda", dtype=self.precision):
                            logits, aux_loss = self.model(
                                input_ids=batch["input_ids"]
                            )

                        lm_loss = self.criterion(
                            logits[:, :-1].float().reshape(-1, logits.size(-1)),
                            batch["input_ids"][:, 1:].reshape(-1),
                        )

                        loss = lm_loss + self.AUX_COEF * aux_loss.float()
                        if train:
                            (loss / len(micros)).backward()

                    if train:
                        self.total_tokens += batch["input_ids"].numel()
                    step_tok += batch["input_ids"].numel()
                    step_loss += loss.detach()
                    step_aux += aux_loss.detach()
                    step_lm += lm_loss.detach()

                loss, lm_loss, aux_loss = (
                    step_loss / len(micros), step_lm / len(micros),
                    step_aux / len(micros),
                )
                total_tokens += step_tok
                total_loss += loss
                total_aux_loss += aux_loss
                total_lm_loss += lm_loss

                grad_norm = None
                if train:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=float("inf"),
                    )

                    self.adamw.step()
                    self.muon.step()

                    self.schedule_adamw.step()
                    self.schedule_muon.step()

                    if (step + 1) % self.SAVE_EVERY == 0 or step + 1 == n_steps:
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
                        aux_loss=aux_loss.item(),
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
                            "train/aux_loss": aux_loss.item(),
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

        return (
            (total_loss / (step + 1)).item(),
            (total_aux_loss / (step + 1)).item(),
            (total_lm_loss / (step + 1)).item(),
            total_tokens,
        )

    def train(self):
        train_vals = {"Train": True, "Val": False}
        print(f"\n{'-' * 50}\nStarting training")
        for epoch in range(self.start_epoch, self.EPOCHS):
            print(f"Epoch {epoch + 1}/{self.EPOCHS}")

            for key, flag in train_vals.items():
                loader = self.train_dl if flag else self.val_dl

                if self.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                start = time.perf_counter()
                (epoch_loss, epoch_aux_loss, epoch_lm_loss, total_tokens) = self.run_epoch(
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
                    f"Aux Loss:       {epoch_aux_loss:.4f}\n"
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
                    f"{prefix}/aux_loss": epoch_aux_loss,
                    f"{prefix}/perplexity": perplexity,
                    f"{prefix}/tok_s": tokens_per_second,
                    f"{prefix}/peak_mem_gb": peak_mem_gb,
                    "tokens_seen": self.total_tokens,
                }, step=epoch)

                if not flag:
                    if epoch_lm_loss < self.best_val_loss:
                        self.best_val_loss = epoch_lm_loss
                        best_path = self.save_checkpoint("best")
                        mlflow.log_artifact(best_path, artifact_path="checkpoints")
                        mlflow.log_metrics({"best_val_loss": self.best_val_loss}, step=epoch)
                        print(f"! New best val loss, saved {best_path}")
                    else:
                        self.save_checkpoint("latest")

        mlflow.end_run()

#todo
# test
