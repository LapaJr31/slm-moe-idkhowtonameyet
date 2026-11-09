import math

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint_sequential

from datasets import load_from_disk

from pathlib import Path

from .model import MoESLM, compile_model

class preTrain:
    EPOCHS = 10
    SAVE_EVERY = 100
    BATCH_SIZE = 512
    LR_MUON = 0.02
    LR_ADAMW = 3e-4
    SEQLEN = 2048

    def __init__(self, dataset_tokenized : Path, model : MoESLM, seed : int = 99):
        torch.manual_seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = torch.bfloat16
        self.criterion = F.cross_entropy

        model = model.to(self.device)
        model = compile_model(model = model,  mode = "default")

        dataset = load_from_disk(dataset_tokenized).with_format("torch")
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

        total_steps = len(self.train_dl) * self.EPOCHS
        schedule_adamw = self.make_wsd_scheduler(
            optimizer= self.adamw,
            total_steps=total_steps,
        )
        schedule_muon = self.make_wsd_scheduler(
            optimizer= self.muon,
            total_steps=total_steps,
        )

        self.model = model

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
            pin_memory=True,
            shuffle=shuffle,
            drop_last=True,
        )


    def run_epoch(
        self,
        model,
        loader,
        optimizer_adamw, scheduler_adamw,
        optimizer_muon, scheduler_muon,
        criterion,
        train = True,
    ):
        model.train(train)
        total_loss = 0.0

        for batch in loader:
            batch = {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

            if train:
                optimizer_adamw.zero_grad(set_to_none=True)
                optimizer_muon.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, aux_loss = model(input_ids=batch["input_ids"])

                lm_loss = criterion(
                    logits[:, :-1].float().reshape(-1, logits.size(-1)),
                    batch["input_ids"][:, 1:].reshape(-1),
                )
                loss = lm_loss + 0.01 * aux_loss.float()

                if train:
                    loss.backward()
                    optimizer_adamw.step()
                    optimizer_muon.step()
                    scheduler_adamw.step()
                    scheduler_muon.step()

            total_loss += loss.detach().item()

        if len(loader) == 0:
            raise ValueError("DataLoader is empty")

        return total_loss / len(loader)

    def train(self):
        pass

#todo
# unpack logits and axu
# train empty
# model checkpointing
# mlflow
# val and test
