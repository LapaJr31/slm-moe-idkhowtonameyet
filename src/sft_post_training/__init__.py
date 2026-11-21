from .sft import SFT_Post_train

from pathlib import Path
from contextlib import nullcontext
from accelerate import PartialState

import mlflow

MODEL_ID = "lapkaaaa/skynet_0.1_not_killing_anybody_yet"
DATASET_ID = "HuggingFaceTB/smoltalk2"
SUBSET = "SFT"
SPLIT = "tulu_3_sft_personas_instruction_following_no_think"
ATTENTION = "flash_attention_2"


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "nli-adapter"

RUN_TYPE = "dry_run"

def main():
    state = PartialState()
    mlflow_context = nullcontext()

    if state.is_main_process:
        mlflow.set_experiment("skynet-sft-posttrain")
        mlflow.start_run(run_name=f"posttrain-{RUN_TYPE}")

    with mlflow_context:
        trainer = SFT_Post_train(
            model_id=MODEL_ID,
            output_dir=str(OUTPUT_DIR),
            attn_imp=ATTENTION,
            seed=SEED,
            run_type=RUN_TYPE,
        )

        train, val = trainer.init_dataset(
            dataset_id = DATASET_ID,
            subset=SUBSET,
            split=SPLIT,
        )

        trainer.train(
            train_dataset=train,
            eval_dataset=val,
        )

if __name__ == "__main__":
    main()
