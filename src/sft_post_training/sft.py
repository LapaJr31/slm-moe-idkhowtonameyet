import os

import mlflow
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


class MoESLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return super().compute_loss(
            model, {**inputs, "loss_only": True}, return_outputs, num_items_in_batch,
        )


class SFT_Post_train:
    EPOCHS = 2
    MICRO_BATCH = 16
    EFFECTIVE_BATCH = 32
    LR = 5e-5
    SEQLEN = 4096
    LOG_EVERY = 50
    SAVE_EVERY = 1000

    def __init__(
        self,
        model_id: str,
        output_dir: str,
        seed: int = 42,
        run_type: str = "dry_run",
        *,
        trust_remote_code: bool = False,
    ):

        self.output_dir = output_dir
        self.seed = seed

        mlflow.set_tag("run_type", run_type)
        mlflow.log_params({
            "epochs" : self.EPOCHS,
            "lr" : self.LR,
            "micro_batch" : self.MICRO_BATCH,
            "effective_batch" : self.EFFECTIVE_BATCH,
            "seqlen" : self.SEQLEN
        })

        torch.manual_seed(self.seed)

        has_cuda = torch.cuda.is_available()
        use_bf16 = has_cuda and torch.cuda.is_bf16_supported()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=trust_remote_code, attn_implementation="sdpa",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, config=self.model.config, trust_remote_code=trust_remote_code, padding_side="right",
        )
        if self.tokenizer.eos_token is None:
            raise ValueError("The SFT tokenizer requires an EOS token")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        role_tokens = ["<|user|>", "<|assistant|>", "<|system|>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": role_tokens}, replace_extra_special_tokens=False)
        required_vocab_size = max(self.tokenizer.get_vocab().values()) + 1
        if required_vocab_size > self.model.config.vocab_size:
            self.model.resize_token_embeddings(required_vocab_size)

        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|' + message['role'] + '|>\\n' }}"
            "{% if message['role'] == 'assistant' %}"
            "{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}"
            "{% else %}"
            "{{ message['content'] + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
        )

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        assert self.EFFECTIVE_BATCH % (self.MICRO_BATCH * world_size) == 0

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.EPOCHS,

            per_device_train_batch_size=self.MICRO_BATCH,
            gradient_accumulation_steps=self.EFFECTIVE_BATCH// (self.MICRO_BATCH * world_size),
            per_device_eval_batch_size=self.EFFECTIVE_BATCH,

            learning_rate=self.LR,
            lr_scheduler_type="cosine",
            optim="adamw_torch_fused" if has_cuda else "adamw_torch",
            warmup_steps=0.03,
            weight_decay=0.01,

            bf16=use_bf16,
            fp16=has_cuda and not use_bf16,
            tf32=has_cuda and torch.cuda.get_device_capability()[0] >= 8,
            gradient_checkpointing=True,

            eval_strategy="epoch",

            remove_unused_columns=True,
            label_names=["labels"],

            logging_steps=self.LOG_EVERY,
            save_strategy="steps",
            save_steps=self.SAVE_EVERY,
            save_total_limit=4,

            dataloader_num_workers=4,
            dataloader_pin_memory=has_cuda,
            dataloader_persistent_workers=True,
            dataloader_prefetch_factor=2,

            prediction_loss_only=True,

            seed=seed,

            report_to=["mlflow"],

            torch_compile=has_cuda,
            torch_compile_mode="max-autotune" if has_cuda else None,
        )

    def tokenize_batch(self, batch):
        encoded = self.tokenizer.apply_chat_template(
            batch["messages"],
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=self.SEQLEN,
        )
        encoded["labels"] = [
            [token if is_assistant else -100 for token, is_assistant in zip(ids, mask, strict=True)]
            for ids, mask in zip(encoded["input_ids"], encoded.pop("assistant_masks"), strict=True)
        ]
        return encoded

    def init_dataset(self, dataset_id, subset=None, split="train"):
        mlflow.log_dict(
            {
                "source": [f"{dataset_id}"],
                "subset": [f"{subset}"],
                "split": [f"{split}"],
            },
            "data/dataset_data.json"
        )
        ds = load_dataset(dataset_id, name=subset, split=split)
        ds = ds.map(self.tokenize_batch, batched=True, remove_columns=ds.column_names)
        ds = ds.filter(lambda example: any(label != -100 for label in example["labels"][1:]))
        if len(ds) < 2:
            raise ValueError("SFT requires at least two examples with assistant tokens after truncation")
        splits = ds.train_test_split(test_size=0.02, seed=self.seed)
        return splits["train"], splits["test"]

    def train(self, train_dataset, eval_dataset):
        trainer = MoESLMTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset= eval_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer),
        )

        trainer.train()
        trainer.save_model(self.output_dir)
