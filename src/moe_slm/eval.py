import argparse
import math
import pickle
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from . import SEQLEN, TOKENIZER_PATH, VOCAB_SIZE, build_model
from .model import MoESLM

DEFAULT_PROMPTS = (
    "Once upon a time",
    "The main reason for this is",
    "To make a cup of coffee,",
)
DEFAULT_CHECKPOINT = Path("checkpoints/best.pt")
TOP_K = 50
TEMPERATURE = 0.8
ARITHMETIC_EXAMPLE = "Problem: (2 + 3) * 4\nWork: 2 + 3 = 5, then 5 * 4 = 20\nAnswer: 20\n\n"
ARITHMETIC_PROBLEMS = (
    *((f"({a} + {b}) * {c}", (a + b) * c)
      for a, b, c in ((3, 5, 2), (6, 4, 3), (8, 7, 4), (5, 9, 6))),
    *((f"{a} * {b} - {c}", a * b - c)
      for a, b, c in ((3, 4, 5), (7, 5, 8), (9, 6, 4), (3, 4, 19))),
)


def load_model(path: Path) -> MoESLM:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    weights = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(weights, dict):
        raise ValueError("expected a training checkpoint containing a model state dict")

    model = build_model(attn_implementation="sdpa").to("cpu")
    model.load_state_dict({key.removeprefix("_orig_mod."): value for key, value in weights.items()})
    for block in model.blocks:
        block.moe.inference_no_drop = True
    return model.eval()


def configure_routing(
    model: MoESLM,
    top_p: float,
    min_k: int | None,
    max_k: int | None,
    *,
    preserve_top_k_mass: bool = False,
) -> None:
    for block in model.blocks:
        moe = block.moe
        lower = moe.top_k if min_k is None else min_k
        upper = moe.num_experts if max_k is None else max_k
        if not 1 <= lower <= upper <= moe.num_experts:
            raise ValueError("expected 1 <= --moe-min-k <= --moe-max-k <= num_experts")
        moe.top_p, moe.min_k, moe.max_k = top_p, lower, upper
        moe.preserve_top_k_mass = preserve_top_k_mass


@torch.inference_mode()
def reference_cross_entropy(
    model: MoESLM, prompt_ids: torch.Tensor, reference_ids: torch.Tensor
) -> float:
    ids = torch.cat((prompt_ids, reference_ids), dim=1)
    labels = ids.clone()
    labels[:, :prompt_ids.size(1)] = -100
    loss, _, _ = model(ids, labels=labels)
    return loss.item()


def sampling_weights(
    logits: torch.Tensor,
    temperature: float,
    top_p: float | None,
    min_p: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if top_p is not None:
        values, indices = logits.sort(descending=True)
        weights = (values / temperature).softmax(dim=0)
        if top_p < 1:
            weights[1:] = weights[1:].masked_fill(weights.cumsum(dim=0)[:-1] >= top_p, 0)
        return weights, indices
    if min_p is not None:
        weights = (logits / temperature).softmax(dim=0)
        weights = weights.masked_fill(weights < min_p * weights.max(), 0)
        return weights, torch.arange(logits.numel(), device=logits.device)

    values, indices = logits.topk(min(TOP_K, logits.numel()))
    return (values / temperature).softmax(dim=0), indices


@torch.inference_mode()
def sample_tokens(
    model: MoESLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    generator: torch.Generator | None,
    *,
    temperature: float = TEMPERATURE,
    top_p: float | None = None,
    min_p: float | None = None,
) -> tuple[list[int], float]:
    kv_cache = model.make_kv_cache(prompt_ids.size(1) + max_new_tokens, prompt_ids.size(0))
    input_ids = prompt_ids
    generated = []
    assignments = 0
    routed_tokens = 0
    for _ in range(max_new_tokens):
        logits, _, _ = model(input_ids, kv_cache=kv_cache)
        assignments += sum(block.moe.active_expert_assignments.item() for block in model.blocks)
        routed_tokens += input_ids.numel()
        weights, indices = sampling_weights(logits[0, -1], temperature, top_p, min_p)
        token = indices[torch.multinomial(weights, num_samples=1, generator=generator)]
        token_id = token.item()
        if token_id == eos_token_id:
            break
        generated.append(token_id)
        input_ids = token.view(1, 1)
    active_experts = assignments / (routed_tokens * len(model.blocks))
    return generated, active_experts


def run_arithmetic(model: MoESLM, tokenizer, args: argparse.Namespace) -> None:
    seed = 42 if args.seed is None else args.seed
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first_correct = verified_correct = total_tokens = 0
    for expression, expected in ARITHMETIC_PROBLEMS:
        prompt = f"{ARITHMETIC_EXAMPLE}Problem: {expression}\nAnswer:"
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        if ids.size(1) + args.max_new_tokens > SEQLEN:
            raise ValueError(f"arithmetic prompt and requested continuation exceed {SEQLEN} tokens")

        answers = []
        for _ in range(4):
            tokens, _ = sample_tokens(
                model, ids, args.max_new_tokens, tokenizer.eos_token_id, generator,
                temperature=args.temperature, top_p=args.top_p, min_p=args.min_p,
            )
            total_tokens += len(tokens)
            continuation = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            first_line = continuation.split("\n", 1)[0].strip()
            answer = int(first_line) if re.fullmatch(r"[+-]?[0-9]+", first_line) else None
            answers.append(answer)

        first = answers[0]
        verified = next((answer for answer in answers if answer == expected), None)
        first_correct += first == expected
        verified_correct += verified is not None
        print(f"{expression} | First: {first} | Calculator-verified: {verified}")

    total = len(ARITHMETIC_PROBLEMS)
    print(f"First-sample accuracy: {first_correct}/{total} ({first_correct / total:.1%})")
    print(f"Calculator-assisted success within four samples: {verified_correct}/{total} ({verified_correct / total:.1%})")
    print(f"Continuation tokens: {total_tokens}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text and measure expert routing on CPU.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=Path(TOKENIZER_PATH))
    parser.add_argument("--prompt", action="append", help="Prompt to continue. Repeat for more samples.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, help="Optional seed for reproducible samples.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--top-p", type=float, help="Use vocabulary nucleus sampling.")
    sampling.add_argument("--min-p", type=float, help="Use vocabulary min-p sampling.")
    parser.add_argument("--arithmetic", action="store_true", help="Run the calculator-assisted arithmetic benchmark.")
    parser.add_argument(
        "--moe-top-p", type=float, action="append", default=[],
        help="Enable dynamic expert top-p routing. Repeat to compare thresholds with fixed top-k.",
    )
    parser.add_argument("--moe-min-k", type=int)
    parser.add_argument("--moe-max-k", type=int)
    parser.add_argument(
        "--moe-mass-preserving-top-p", type=float,
        help="Add an expert top-p configuration that preserves the trained top-k gate mass.",
    )
    parser.add_argument(
        "--reference",
        help="Report cross-entropy and perplexity for this continuation after a single --prompt.",
    )
    args = parser.parse_args()

    if not 1 <= args.max_new_tokens < SEQLEN:
        parser.error(f"--max-new-tokens must be between 1 and {SEQLEN - 1}")
    if args.seed is not None and not -(2**63) <= args.seed < 2**64:
        parser.error("--seed must be between -2**63 and 2**64 - 1")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    for name, value in (("--top-p", args.top_p), ("--min-p", args.min_p)):
        if value is not None and not 0 < value <= 1:
            parser.error(f"{name} must be in (0, 1]")
    if any(not 0 < top_p <= 1 for top_p in args.moe_top_p):
        parser.error("--moe-top-p must be in (0, 1]")
    mass_preserving_top_p = args.moe_mass_preserving_top_p
    if args.arithmetic and (
        args.prompt or args.reference is not None or args.moe_top_p
        or mass_preserving_top_p is not None or args.moe_min_k is not None or args.moe_max_k is not None
    ):
        parser.error("--arithmetic uses built-in prompts and fixed top-k expert routing")
    if mass_preserving_top_p is not None and not 0 < mass_preserving_top_p <= 1:
        parser.error("--moe-mass-preserving-top-p must be in (0, 1]")
    if not args.moe_top_p and mass_preserving_top_p is None and (
        args.moe_min_k is not None or args.moe_max_k is not None
    ):
        parser.error("--moe-min-k and --moe-max-k require an expert top-p threshold")
    if args.reference is not None and (len(args.prompt or []) != 1 or not args.reference.strip()):
        parser.error("--reference requires nonempty text and exactly one --prompt")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint file not found: {args.checkpoint}")
    if not args.tokenizer.is_dir():
        parser.error(f"tokenizer directory not found: {args.tokenizer}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(f"could not load tokenizer {args.tokenizer}: {exc}")
    if len(tokenizer) != VOCAB_SIZE:
        parser.error(f"tokenizer vocabulary has {len(tokenizer)} entries, expected {VOCAB_SIZE}")

    prompts = []
    prompt_texts = () if args.arithmetic else (args.prompt or DEFAULT_PROMPTS)
    for prompt in prompt_texts:
        if not prompt.strip():
            parser.error("--prompt must contain text")
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        if ids.numel() == 0:
            parser.error("--prompt must encode to at least one token")
        if ids.size(1) + args.max_new_tokens > SEQLEN:
            parser.error(f"prompt and requested continuation exceed {SEQLEN} tokens")
        prompts.append((prompt, ids))

    reference_ids = None
    if args.reference is not None:
        reference_ids = tokenizer(args.reference, add_special_tokens=False, return_tensors="pt").input_ids
        if reference_ids.numel() == 0:
            parser.error("--reference must encode to at least one token")
        if prompts[0][1].size(1) + reference_ids.size(1) > SEQLEN:
            parser.error(f"prompt and reference exceed {SEQLEN} tokens")

    try:
        model = load_model(args.checkpoint)
    except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        parser.error(f"could not load checkpoint {args.checkpoint}: {exc}")

    if args.arithmetic:
        try:
            run_arithmetic(model, tokenizer, args)
        except ValueError as exc:
            parser.error(str(exc))
        return

    configurations = [(top_p, False) for top_p in args.moe_top_p]
    if mass_preserving_top_p is not None:
        configurations.append((mass_preserving_top_p, True))
    if len(configurations) != 1:
        configurations.insert(0, (None, False))

    moe = model.blocks[0].moe
    for top_p, preserve_mass in configurations:
        if top_p is None:
            routing = f"top-k={moe.top_k}"
        else:
            try:
                configure_routing(
                    model, top_p, args.moe_min_k, args.moe_max_k,
                    preserve_top_k_mass=preserve_mass,
                )
            except ValueError as exc:
                parser.error(str(exc))
            routing = f"top-p={top_p:g}, k={moe.min_k}–{moe.max_k}"
            if preserve_mass:
                routing += ", mass-preserving"

        generator = None
        if args.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
        for prompt, ids in prompts:
            tokens, active_experts = sample_tokens(
                model, ids, args.max_new_tokens, tokenizer.eos_token_id, generator,
                temperature=args.temperature, top_p=args.top_p, min_p=args.min_p,
            )
            continuation = tokenizer.decode(tokens, skip_special_tokens=True)
            lines = [
                f"Expert routing: {routing}",
                f"Prompt: {prompt}",
                f"Continuation: {continuation}",
                f"Active experts/token/layer during generation: {active_experts:.4f}",
            ]
            if reference_ids is not None:
                loss = reference_cross_entropy(model, ids, reference_ids)
                lines.append(f"Reference cross-entropy: {loss:.6f}")
                lines.append(f"Reference perplexity: {math.exp(loss):.2f}")
            print("\n".join(lines), end="\n\n", flush=True)


if __name__ == "__main__":
    main()
