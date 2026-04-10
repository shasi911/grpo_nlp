"""
Expert Iteration training script for Qwen/Qwen2.5-Math-1.5B on the MATH dataset.

Algorithm per EI step:
  1. Sample Db questions from train
  2. Generate G rollouts per question with vLLM
  3. Keep rollouts with correct answers
  4. SFT on kept rollouts for sft_epochs epochs
  5. Evaluate validation accuracy and log response entropy

Usage examples:
  python scripts/run_expert_iteration.py --db 1024 --rollouts 4 --sft-epochs 1
  python scripts/run_expert_iteration.py --db 2048 --rollouts 8 --sft-epochs 3
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #

def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_question(ex: dict) -> str:
    return ex.get("problem", ex.get("question", ex.get("prompt", "")))


def get_answer(ex: dict) -> str:
    return str(ex.get("answer", ex.get("solution", "")))


# --------------------------------------------------------------------------- #
# vLLM generation
# --------------------------------------------------------------------------- #

def generate_rollouts(
    model_path: str,
    prompts: list[str],
    rollouts_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    gpu_memory_utilization: float = 0.5,
) -> list[list[str]]:
    """Generate `rollouts_per_prompt` responses per prompt. Returns list[list[str]]."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, trust_remote_code=True,
              gpu_memory_utilization=gpu_memory_utilization)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_new_tokens,
        n=rollouts_per_prompt,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling_params)
    del llm

    import torch
    torch.cuda.empty_cache()

    # outputs[i].outputs is a list of length rollouts_per_prompt
    return [[o.text for o in out.outputs] for out in outputs]


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #

def filter_correct_rollouts(
    prompts: list[str],
    rollouts: list[list[str]],
    ground_truths: list[str],
) -> tuple[list[str], list[str]]:
    """Return (prompt_list, response_list) for correct rollouts only."""
    from alignment.drgrpo_grader import r1_zero_reward_fn

    kept_prompts, kept_responses = [], []
    total, correct = 0, 0
    for prompt, responses, gt in zip(prompts, rollouts, ground_truths):
        for resp in responses:
            total += 1
            result = r1_zero_reward_fn(resp, gt)
            if result["answer_reward"] == 1.0:
                kept_prompts.append(prompt)
                kept_responses.append(resp)
                correct += 1

    logger.info("Correct rollouts: %d / %d (%.1f%%)", correct, total,
                100 * correct / max(total, 1))
    return kept_prompts, kept_responses


# --------------------------------------------------------------------------- #
# SFT step
# --------------------------------------------------------------------------- #

def sft_step(
    model_path: str,
    prompts: list[str],
    responses: list[str],
    tokenizer,
    args,
) -> tuple[str, list[dict]]:
    """Fine-tune the model on (prompt, response) pairs. Returns new checkpoint path."""
    import torch
    from torch.optim import AdamW, SGD
    from transformers import AutoModelForCausalLM
    from alignment.tokenizer_prompt_and_output import tokenize_prompt_and_output
    from alignment.sft_microbatch_train_step import sft_microbatch_train_step
    from alignment.compute_entropy import compute_entropy

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = SGD(model.parameters(), lr=args.lr,
                    weight_decay=args.weight_decay, momentum=0.9)

    pairs = list(zip(prompts, responses))
    gradient_accumulation_steps = max(1, args.batch_size // args.micro_batch_size)
    step_metrics = []

    for epoch in range(args.sft_epochs):
        random.shuffle(pairs)
        optimizer.zero_grad()
        epoch_entropy = []

        for micro_step, (prompt, response) in enumerate(pairs):
            batch = tokenize_prompt_and_output(
                prompt_strs=[prompt],
                output_strs=[response],
                tokenizer=tokenizer,
            )
            input_ids = batch["input_ids"][:, :args.max_seq_len].to(args.device)
            labels = batch["labels"][:, :args.max_seq_len].to(args.device)
            response_mask = batch["response_mask"][:, :args.max_seq_len].to(args.device)

            logits = model(input_ids=input_ids).logits
            # compute entropy before freeing logits
            token_entropy = compute_entropy(logits)  # (1, seq_len)
            masked_entropy = (token_entropy * response_mask).sum() / response_mask.sum().clamp(min=1)
            epoch_entropy.append(masked_entropy.item())

            log_probs_all = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            del logits, log_probs_all

            loss, _ = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

            if (micro_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        avg_entropy = sum(epoch_entropy) / max(len(epoch_entropy), 1)
        step_metrics.append({"sft_epoch": epoch + 1, "avg_entropy": avg_entropy})
        logger.info("  SFT epoch %d | avg_entropy=%.4f", epoch + 1, avg_entropy)

    # save checkpoint
    ckpt_dir = Path(args.output_dir) / f"checkpoint"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    del model
    torch.cuda.empty_cache()

    return str(ckpt_dir), step_metrics


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def evaluate_accuracy(
    model_path: str,
    val_data: list[dict],
    prompt_template: str,
    num_val_examples: int,
    max_new_tokens: int,
    gpu_memory_utilization: float = 0.5,
) -> float:
    import torch
    from vllm import LLM, SamplingParams
    from alignment.drgrpo_grader import r1_zero_reward_fn

    val_data = val_data[:num_val_examples]
    prompts = [prompt_template.format(question=get_question(ex)) for ex in val_data]
    answers = [get_answer(ex) for ex in val_data]

    llm = LLM(model=model_path, trust_remote_code=True,
              gpu_memory_utilization=gpu_memory_utilization)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling_params)
    del llm
    torch.cuda.empty_cache()

    correct = 0
    for output, gold in zip(outputs, answers):
        gen = output.outputs[0].text
        result = r1_zero_reward_fn(gen, gold)
        correct += result["answer_reward"]

    return correct / len(val_data)


# --------------------------------------------------------------------------- #
# Main EI loop
# --------------------------------------------------------------------------- #

def run_expert_iteration(args):
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = load_jsonl(args.train_path)
    val_data = load_jsonl(args.val_path)

    with open(args.prompt_path) as f:
        prompt_template = f.read()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    metrics_file = open(metrics_path, "w")

    current_model_path = args.model_id

    for ei_step in range(1, args.n_ei_steps + 1):
        logger.info("=" * 60)
        logger.info("EI step %d / %d", ei_step, args.n_ei_steps)

        # 1. Sample Db questions
        sample = random.sample(train_data, min(args.db, len(train_data)))
        prompts = [prompt_template.format(question=get_question(ex)) for ex in sample]
        ground_truths = [get_answer(ex) for ex in sample]

        # 2. Generate G rollouts per question
        logger.info("Generating %d rollouts for %d questions ...", args.rollouts, len(sample))
        rollouts = generate_rollouts(
            model_path=current_model_path,
            prompts=prompts,
            rollouts_per_prompt=args.rollouts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        # 3. Filter correct rollouts
        kept_prompts, kept_responses = filter_correct_rollouts(
            prompts, rollouts, ground_truths
        )

        if len(kept_prompts) == 0:
            logger.warning("No correct rollouts found at EI step %d, skipping SFT.", ei_step)
            metrics_file.write(json.dumps({
                "ei_step": ei_step, "kept_rollouts": 0,
                "val_accuracy": None, "avg_entropy": None,
            }) + "\n")
            metrics_file.flush()
            continue

        logger.info("SFT on %d correct rollouts for %d epochs ...",
                    len(kept_prompts), args.sft_epochs)

        # 4. SFT step
        current_model_path, sft_metrics = sft_step(
            model_path=current_model_path,
            prompts=kept_prompts,
            responses=kept_responses,
            tokenizer=tokenizer,
            args=args,
        )

        avg_entropy = sft_metrics[-1]["avg_entropy"] if sft_metrics else None

        # 5. Validate
        logger.info("Evaluating on validation set ...")
        val_acc = evaluate_accuracy(
            model_path=current_model_path,
            val_data=val_data,
            prompt_template=prompt_template,
            num_val_examples=args.num_val_examples,
            max_new_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        logger.info("EI step %d | val_accuracy=%.4f | avg_entropy=%.4f",
                    ei_step, val_acc, avg_entropy or 0.0)

        record = {
            "ei_step": ei_step,
            "kept_rollouts": len(kept_prompts),
            "val_accuracy": val_acc,
            "avg_entropy": avg_entropy,
            "sft_metrics": sft_metrics,
        }
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    metrics_file.close()
    logger.info("Expert iteration complete. Metrics saved to %s", metrics_path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train-path",   default="data/MATH/train.jsonl")
    p.add_argument("--val-path",     default="data/MATH/validation.jsonl")
    p.add_argument("--prompt-path",  default="alignment/prompts/r1_zero.prompt")
    p.add_argument("--output-dir",   default="outputs/ei")
    # model
    p.add_argument("--model-id",     default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--device",       default="cuda")
    # EI hyperparams
    p.add_argument("--n-ei-steps",   type=int,   default=5)
    p.add_argument("--db",           type=int,   default=1024,
                   help="Number of questions to sample per EI step {512,1024,2048}")
    p.add_argument("--rollouts",     type=int,   default=4,
                   help="Number of rollouts (G) per question")
    p.add_argument("--temperature",  type=float, default=0.8,
                   help="Sampling temperature for rollout generation")
    p.add_argument("--sft-epochs",   type=int,   default=1,
                   help="SFT epochs per EI step")
    # training
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--max-seq-len",  type=int,   default=256)
    p.add_argument("--max-grad-norm",type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    # validation
    p.add_argument("--num-val-examples", type=int, default=200)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_expert_iteration(parse_args())
