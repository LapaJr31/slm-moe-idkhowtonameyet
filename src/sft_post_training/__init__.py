import argparse
from contextlib import nullcontext
from pathlib import Path

import mlflow
from accelerate import PartialState

from .sft import SFT_Post_train

DATASET_ID = "HuggingFaceTB/smoltalk2"
SUBSET = "SFT"
SPLIT = "tulu_3_sft_personas_instruction_following_no_think"


SEED = 42
OUTPUT_DIR = Path("checkpoints/skynet-sft")

RUN_TYPE = "dry_run"

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers export of MoESLM.")
    parser.add_argument("--model", required=True, help="Export directory or Hub repo ID")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--trust-remote-code", action="store_true", help="Load the custom code in a trusted export")
    args = parser.parse_args()

    state = PartialState()
    mlflow_context = nullcontext()

    if state.is_main_process:
        mlflow.set_experiment("skynet-sft-posttrain")
        mlflow_context = mlflow.start_run(run_name=f"posttrain-{RUN_TYPE}")

    with mlflow_context:
        trainer = SFT_Post_train(
            model_id=args.model,
            output_dir=str(args.output_dir),
            seed=SEED,
            run_type=RUN_TYPE,
            trust_remote_code=args.trust_remote_code,
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
