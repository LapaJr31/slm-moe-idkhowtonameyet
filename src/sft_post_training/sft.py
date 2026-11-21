from pandas.core.algorithms import mode
import unsloth
import torch
import mlflow
import os

from enum import StrEnum
from pathlib import Path

from datasets import load_dataset
from unsloth import FastLanguageModel
# from accelerate import PartialState

from trl import (
    SFTConfig,
    SFTTrainer
)

# def log_rank0(*args, **kwargs):
#     if PartialState().local_process_index == 0:
#         print(*args, **kwargs)


class ATTN(StrEnum):
    FA2 = "flash_attention_2"
    FA3 = "flash_attention_3"
    FA4 = "flash_attention_4"
    SDPA = "sdpa"

class SFT_Post_train:
    EPOCHS = 2
    MICRO_BATCH = 16
    EFFECTIVE_BATCH = 32
    LR = 5e-5
    SEQLEN = 4096
    LOG_EVERY = 50
    SAVE_EVERY = 1000
    CHECKPOINT_DIR = Path("checkpoints")

    def __init__(
        self,
        model_id : str,
        output_dir : str,
        attn_imp: str = "sdpa",
        seed: int = 42,
        run_type: str = "dry_run"
    ):

        if not isinstance(attn_imp, ATTN):
            raise ValueError(
                """
                Invalid attention implementation. Only allowed values are:
                - "flash_attention_2"
                - "flash_attention_3"
                - "flash_attention_4"
                - "sdpa"
                """
            )
        self.output_dir = output_dir
        self.seed = seed

        mlflow.set_experiment("skynet-sft-posttrain")
        mlflow.start_run(run_name=f"posttrain-{run_type}")
        mlflow.set_tag("run_type", run_type)
        mlflow.log_params({
            "epochs" : self.EPOCHS,
            "lr" : self.LR,
            "micro_batch" : self.MICRO_BATCH,
            "effective_batch" : self.EFFECTIVE_BATCH,
            "seqlen" : self.SEQLEN
        })

        torch.manual_seed(self.seed)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=self.SEQLEN,
            load_in_4bit=False,
            full_finetuning=True,
            dtype=compute_dtype,
            use_gradient_checkpointing="unsloth",
            device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
            attn_implementation=attn_imp,
        )
        self.model = model
        self.model.print_trainable_parameters()

        self.tokenizer = tokenizer

        role_tokens = ["<|user|>", "<|assistant|>", "<|system|>"]
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": role_tokens}
        )
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|' + message['role'] + '|>\\n' }}"
            "{% if message['role'] == 'assistant' %}"
            "{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}"
            "{% else %}"
            "{{ message['content'] + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        assert self.EFFECTIVE_BATCH % (self.MICRO_BATCH * world_size) == 0

        self.training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=2,

            per_device_train_batch_size=self.MICRO_BATCH,
            gradient_accumulation_steps=self.EFFECTIVE_BATCH// (self.MICRO_BATCH * world_size),
            per_device_eval_batch_size=self.EFFECTIVE_BATCH,

            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            optim="adamw_torch_fused",
            warmup_steps=0.03,
            weight_decay=0.01,

            bf16=True,
            fp16=False,
            tf32=True,

            max_length=self.SEQLEN,
            packing=False,

            completion_only_loss=False,
            assistant_only_loss=True,

            eval_strategy="no",

            remove_unused_columns=True,
            label_names=["labels"],

            logging_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=4,

            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            dataloader_prefetch_factor=2,

            padding_free=False,

            seed=seed,

            report_to=["mlflow"],

            torch_compile=True,
            torch_compile_mode="max-autotune"
        )

    def init_dataset(self, dataset_id, cache_dir, subset=None, split=None):
        mlflow.log_dict(
            {
                "source": [f"{dataset_id}"],
                "subset": [f"{subset}"],
                "split": [f"{split}"],
                "revision": "v1",
            },
            "data/dataset_data.json"
        )
        return load_dataset(dataset_id, name=subset, split=split)

    def train(self, dataset):
        trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        try:
            trainer.train()
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
        finally:
            mlflow.end_run()
