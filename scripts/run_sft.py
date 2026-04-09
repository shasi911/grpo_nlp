"""
SFT training script for Qwen/Qwen2.5-Math-1.5B on the MATH dataset.

Usage examples:
  # Train on 512 examples, log to wandb
  python scripts/run_sft.py --num-examples 512 --lr 1e-5 --batch-size 8

  # Train on full dataset
  python scripts/run_sft.py --num-examples full --lr 2e-5 --batch-size 16

  # Train on filtered (correct-answer-only) dataset
  python scripts/run_sft.py --filtered --num-examples full
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from torch.optim import SGD

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Heavy deps (torch, wandb, vllm, transformers) are imported inside train()
# so that --help works without a GPU environment.

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_sft_data(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def filter_correct_examples(data: list[dict]) -> list[dict]:
    """Keep only examples where the response contains a correct answer."""
    from alignment.drgrpo_grader import r1_zero_reward_fn  # needs math deps on remote
    filtered = []
    for ex in data:
        response = ex.get("response", ex.get("output", ""))
        ground_truth = ex.get("answer", ex.get("solution", ""))
        result = r1_zero_reward_fn(response, str(ground_truth))
        if result["answer_reward"] == 1.0:
            filtered.append(ex)
    logger.info(
        "Filtered dataset: %d / %d examples kept (correct answers only)",
        len(filtered),
        len(data),
    )
    return filtered


def make_prompt_response_pairs(data: list[dict], prompt_template: str) -> list[tuple[str, str]]:
    """Convert raw data records into (prompt_str, response_str) pairs."""
    pairs = []
    for ex in data:
        question = ex.get("problem", ex.get("question", ex.get("prompt", "")))
        response = ex.get("response", ex.get("output", ex.get("solution", "")))
        prompt = prompt_template.format(question=question)
        pairs.append((prompt, response))
    return pairs


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def evaluate_accuracy(
    model_path: str,
    val_data: list[dict],
    prompt_template: str,
    max_new_tokens: int = 1024,
    num_val_examples: int | None = None,
) -> float:
    """Generate responses with vLLM and compute validation accuracy."""
    if num_val_examples is not None:
        val_data = val_data[:num_val_examples]

    prompts = [
        prompt_template.format(question=ex.get("problem", ex.get("question", "")))
        for ex in val_data
    ]
    answers = [str(ex.get("answer", ex.get("solution", ""))) for ex in val_data]

    from vllm import LLM, SamplingParams
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.5)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, sampling_params)

    from alignment.drgrpo_grader import r1_zero_reward_fn  # needs math deps on remote
    correct = 0
    for output, gold in zip(outputs, answers):
        gen = output.outputs[0].text
        result = r1_zero_reward_fn(gen, gold)
        correct += result["answer_reward"]

    acc = correct / len(val_data)
    # free vllm memory
    del llm
    torch.cuda.empty_cache()
    return acc


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train(args):
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from alignment.tokenizer_prompt_and_output import tokenize_prompt_and_output
    from alignment.sft_microbatch_train_step import sft_microbatch_train_step

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- load data ---
    train_data = load_sft_data(args.train_path)
    val_data = load_sft_data(args.val_path)

    if args.filtered:
        train_data = filter_correct_examples(train_data)
        run_name = f"sft_filtered_lr{args.lr}_bs{args.batch_size}"
    else:
        run_name = f"sft_n{args.num_examples}_lr{args.lr}_bs{args.batch_size}"

    # subsample if requested
    if args.num_examples != "full":
        n = int(args.num_examples)
        random.shuffle(train_data)
        train_data = train_data[:n]

    logger.info("Training on %d examples", len(train_data))

    # --- load prompt template ---
    with open(args.prompt_path) as f:
        prompt_template = f.read()

    pairs = make_prompt_response_pairs(train_data, prompt_template)

    # --- wandb ---
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )
    wandb.config.update({"num_train_examples": len(train_data)})

    # --- model + tokenizer ---
    logger.info("Loading model: %s", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)
    model.gradient_checkpointing_enable()
    model.train()

    from torch.optim import AdamW, SGD
    if args.use_sgd:
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        logger.info("Using SGD with momentum")
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        logger.info("Using fp32 AdamW")

    # --- training ---
    global_step = 0
    gradient_accumulation_steps = max(1, args.batch_size // args.micro_batch_size)

    for epoch in range(args.num_epochs):
        random.shuffle(pairs)
        optimizer.zero_grad()

        for micro_step, (prompt, response) in enumerate(pairs):
            # tokenize single example as a batch of 1
            batch = tokenize_prompt_and_output(
                prompt_strs=[prompt],
                output_strs=[response],
                tokenizer=tokenizer,
            )
            # truncate to max_seq_len to avoid OOM on long responses
            input_ids = batch["input_ids"][:, :args.max_seq_len].to(args.device)
            labels = batch["labels"][:, :args.max_seq_len].to(args.device)
            response_mask = batch["response_mask"][:, :args.max_seq_len].to(args.device)

            # get log-probs (free logits immediately to save memory)
            logits = model(input_ids=input_ids).logits
            log_probs_all = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            del logits, log_probs_all

            loss, _ = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

            wandb.log({"train/loss": loss.item(), "train/step": global_step})

            # optimizer step after accumulating gradients
            if (micro_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    logger.info("Step %d | loss %.4f", global_step, loss.item())

        # --- validation at end of each epoch ---
        logger.info("Running validation after epoch %d ...", epoch + 1)
        # save checkpoint for vllm; free training model first to avoid OOM
        ckpt_dir = Path(args.output_dir) / f"epoch_{epoch + 1}"
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        del model
        torch.cuda.empty_cache()

        val_acc = evaluate_accuracy(
            model_path=str(ckpt_dir),
            val_data=val_data,
            prompt_template=prompt_template,
            num_val_examples=args.num_val_examples,
        )
        logger.info("Epoch %d | val_accuracy = %.4f", epoch + 1, val_acc)
        wandb.log({"val/accuracy": val_acc, "val/epoch": epoch + 1, "train/step": global_step})

        # reload model for next epoch (vllm unloads it)
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(args.device)
        model.gradient_checkpointing_enable()
        model.train()
        if args.use_sgd:
            optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        else:
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    wandb.finish()
    logger.info("Training complete.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train-path",   default="data/sft.jsonl")
    p.add_argument("--val-path",     default="data/MATH/validation.jsonl")
    p.add_argument("--prompt-path",  default="alignment/prompts/r1_zero.prompt")
    p.add_argument("--output-dir",   default="outputs/sft")
    # model
    p.add_argument("--model-id",     default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--device",       default="cuda")
    # training
    p.add_argument("--num-examples", default="full",
                   help="Number of training examples: 128/256/512/1024/full")
    p.add_argument("--filtered",     action="store_true",
                   help="Use only correct-answer examples")
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size",   type=int,   default=16,
                   help="Effective batch size (gradient accumulation applied)")
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--max-seq-len",  type=int,   default=256,
                   help="Truncate sequences to this length to avoid OOM")
    p.add_argument("--num-epochs",   type=int,   default=3)
    p.add_argument("--max-grad-norm",type=float, default=1.0)
    p.add_argument("--num-val-examples", type=int, default=200,
                   help="Number of val examples to evaluate (None = all)")
    p.add_argument("--log-every",    type=int,   default=50)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--use-sgd",      action="store_true",
                   help="Use SGD instead of AdamW to reduce memory (no 2nd moment)")
    # wandb
    p.add_argument("--wandb-project", default="sft-math")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
