"""
GRPO training loop for Qwen/Qwen2.5-Math-1.5B on the MATH dataset.

Algorithm per step:
  1. Sample n_prompts questions from train
  2. Generate G rollouts per question with vLLM
  3. Compute rewards → group-normalize to get advantages
  4. Load HF model; compute old_log_probs (for grpo_clip) and train
  5. Save updated checkpoint; periodically evaluate

Usage:
  python scripts/run_grpo.py --n-steps 100 --rollouts 8 --n-prompts 32
  python scripts/run_grpo.py --loss-type grpo_clip --cliprange 0.2
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

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_question(ex):
    return ex.get("problem", ex.get("question", ex.get("prompt", "")))


def get_answer(ex):
    return str(ex.get("answer", ex.get("solution", "")))


# --------------------------------------------------------------------------- #
# vLLM generation
# --------------------------------------------------------------------------- #

def generate_rollouts(model_path, prompts, G, max_new_tokens, temperature, gpu_mem):
    """Generate G rollouts per prompt. Returns list[list[str]]."""
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=gpu_mem)
    params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_new_tokens,
        n=G,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, params)
    del llm
    torch.cuda.empty_cache()
    return [[o.text for o in out.outputs] for out in outputs]


# --------------------------------------------------------------------------- #
# Reward + advantage computation
# --------------------------------------------------------------------------- #

def compute_rewards_and_advantages(
    prompts, rollouts, ground_truths, G, advantage_eps, normalize_by_std,
):
    """Returns flat lists: all_prompts, all_rollouts, advantages, raw_rewards, reward_dicts."""
    from alignment.drgrpo_grader import r1_zero_reward_fn
    from alignment.compute_group_normalized_rewards import compute_group_normalized_rewards

    flat_prompts, flat_rollouts, flat_gts = [], [], []
    for prompt, responses, gt in zip(prompts, rollouts, ground_truths):
        for resp in responses:
            flat_prompts.append(prompt)
            flat_rollouts.append(resp)
            flat_gts.append(gt)

    normalized, raw_rewards, metadata = compute_group_normalized_rewards(
        reward_fn=r1_zero_reward_fn,
        rollout_responses=flat_rollouts,
        repeated_ground_truths=flat_gts,
        group_size=G,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
    )

    reward_dicts = [
        r1_zero_reward_fn(resp, gt)
        for resp, gt in zip(flat_rollouts, flat_gts)
    ]

    return flat_prompts, flat_rollouts, normalized, raw_rewards, reward_dicts, metadata


# --------------------------------------------------------------------------- #
# Training step
# --------------------------------------------------------------------------- #

def train_on_rollouts(
    model, tokenizer, flat_prompts, flat_rollouts, advantages, raw_rewards,
    old_log_probs_list, args, device,
):
    """Run one pass of gradient accumulation over all rollouts. Returns avg loss and entropy."""
    import torch
    from alignment.tokenizer_prompt_and_output import tokenize_prompt_and_output
    from alignment.get_response_log_probs import get_response_log_probs
    from alignment.grpo_microbatch_train_step import grpo_microbatch_train_step
    from alignment.compute_entropy import compute_entropy
    from alignment.masked_mean import masked_mean

    n = len(flat_prompts)
    gradient_accumulation_steps = max(1, args.batch_size // args.micro_batch_size)
    losses, entropies = [], []

    for i in range(n):
        batch = tokenize_prompt_and_output(
            prompt_strs=[flat_prompts[i]],
            output_strs=[flat_rollouts[i]],
            tokenizer=tokenizer,
        )
        input_ids = batch["input_ids"][:, :args.max_seq_len].to(device)
        labels = batch["labels"][:, :args.max_seq_len].to(device)
        response_mask = batch["response_mask"][:, :args.max_seq_len].to(device)

        # forward for policy log-probs + entropy
        logits = model(input_ids=input_ids).logits
        token_entropy = compute_entropy(logits)
        masked_ent = masked_mean(token_entropy, response_mask).item()
        entropies.append(masked_ent)

        import torch as _torch
        log_probs_all = _torch.log_softmax(logits, dim=-1)
        policy_log_probs = log_probs_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        del logits, log_probs_all

        # Skip if no response tokens survived truncation
        if response_mask.sum() == 0:
            continue

        adv = advantages[i].unsqueeze(0).unsqueeze(0).to(device)   # (1,1)
        raw = raw_rewards[i].unsqueeze(0).unsqueeze(0).to(device)  # (1,1)

        # Skip if advantage is NaN (happens when entire group reward is 0 and std=0)
        if adv.isnan().any():
            continue

        old_lp = None
        if old_log_probs_list is not None:
            old_lp = old_log_probs_list[i][:, :args.max_seq_len].to(device)

        loss, _ = grpo_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            loss_type=args.loss_type,
            raw_rewards=raw,
            advantages=adv,
            old_log_probs=old_lp,
            cliprange=args.cliprange,
        )
        if not loss.isnan():
            losses.append(loss.item())

        if (i + 1) % gradient_accumulation_steps == 0 or i == n - 1:
            import torch.nn as nn
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

    return sum(losses) / max(len(losses), 1), sum(entropies) / max(len(entropies), 1)


# --------------------------------------------------------------------------- #
# Compute old log-probs (for grpo_clip)
# --------------------------------------------------------------------------- #

def compute_old_log_probs(model, tokenizer, flat_prompts, flat_rollouts, max_seq_len, device):
    """Compute log-probs with no_grad for use as old_log_probs in grpo_clip."""
    import torch
    from alignment.tokenizer_prompt_and_output import tokenize_prompt_and_output

    old_lp_list = []
    with torch.no_grad():
        for prompt, response in zip(flat_prompts, flat_rollouts):
            batch = tokenize_prompt_and_output(
                prompt_strs=[prompt],
                output_strs=[response],
                tokenizer=tokenizer,
            )
            input_ids = batch["input_ids"][:, :max_seq_len].to(device)
            labels = batch["labels"][:, :max_seq_len].to(device)

            logits = model(input_ids=input_ids).logits
            log_probs_all = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            del logits, log_probs_all
            old_lp_list.append(log_probs.cpu())
    return old_lp_list


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def evaluate_accuracy(model_path, val_data, prompt_template, num_val, max_new_tokens, gpu_mem):
    import torch
    from vllm import LLM, SamplingParams
    from alignment.drgrpo_grader import r1_zero_reward_fn

    val_data = val_data[:num_val]
    prompts = [prompt_template.format(question=get_question(ex)) for ex in val_data]
    answers = [get_answer(ex) for ex in val_data]

    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=gpu_mem)
    params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_new_tokens,
        stop=["</answer>"], include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, params)
    del llm
    torch.cuda.empty_cache()

    correct = sum(
        r1_zero_reward_fn(out.outputs[0].text, gold)["answer_reward"]
        for out, gold in zip(outputs, answers)
    )
    return correct / len(val_data)


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def train(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from alignment.log_generations import log_generations

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = load_jsonl(args.train_path)
    val_data   = load_jsonl(args.val_path)

    with open(args.prompt_path) as f:
        prompt_template = f.read()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    gen_log_path = str(Path(args.output_dir) / "generations.jsonl")
    metrics_file = open(metrics_path, "w")

    # initial checkpoint = base model
    current_ckpt = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    model = None  # loaded lazily

    def load_model(path):
        nonlocal model
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(args.device)
        m.gradient_checkpointing_enable()
        m.train()
        return m

    def save_model(m, step):
        ckpt = Path(args.output_dir) / f"step_{step}"
        m.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        return str(ckpt)

    from torch.optim import SGD
    optimizer = None

    for step in range(1, args.n_steps + 1):
        logger.info("=" * 60)
        logger.info("GRPO step %d / %d", step, args.n_steps)

        # ---- Save model for vLLM (or use base model at step 1) ----
        if model is not None:
            current_ckpt = save_model(model, step - 1)
            del model
            torch.cuda.empty_cache()
            model = None

        # ---- Generate rollouts ----
        sample = random.sample(train_data, min(args.n_prompts, len(train_data)))
        prompts = [prompt_template.format(question=get_question(ex)) for ex in sample]
        ground_truths = [get_answer(ex) for ex in sample]

        logger.info("Generating %d rollouts for %d prompts ...", args.rollouts, len(prompts))
        rollouts = generate_rollouts(
            current_ckpt, prompts, args.rollouts,
            args.max_new_tokens, args.temperature, args.gpu_memory_utilization,
        )

        # ---- Rewards + advantages ----
        flat_prompts, flat_rollouts, advantages, raw_rewards, reward_dicts, reward_meta = \
            compute_rewards_and_advantages(
                prompts, rollouts, ground_truths,
                args.rollouts, args.advantage_eps, args.normalize_by_std,
            )

        mean_reward = raw_rewards.mean().item()
        logger.info("Mean reward: %.4f | %s", mean_reward, reward_meta)

        # Log sample generations
        log_generations(
            prompts=flat_prompts,
            generations=flat_rollouts,
            rewards=reward_dicts,
            step=step,
            log_file=gen_log_path,
            num_examples=args.log_generations,
        )

        # ---- Load model for training ----
        model = load_model(current_ckpt)
        optimizer = SGD(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay, momentum=0.9)
        optimizer.zero_grad()

        # ---- Compute old_log_probs if using grpo_clip ----
        old_log_probs_list = None
        if args.loss_type == "grpo_clip":
            logger.info("Computing old log-probs ...")
            old_log_probs_list = compute_old_log_probs(
                model, tokenizer, flat_prompts, flat_rollouts, args.max_seq_len, args.device
            )

        # ---- Train ----
        avg_loss, avg_entropy = train_on_rollouts(
            model, tokenizer, flat_prompts, flat_rollouts,
            advantages, raw_rewards, old_log_probs_list, args, args.device,
        )
        optimizer.step()
        optimizer.zero_grad()

        logger.info("Step %d | loss=%.4f | entropy=%.4f | mean_reward=%.4f",
                    step, avg_loss, avg_entropy, mean_reward)

        record = {
            "step": step,
            "train_loss": avg_loss,
            "avg_entropy": avg_entropy,
            "mean_reward": mean_reward,
            "frac_nonzero": reward_meta.get("frac_nonzero", None),
        }

        # ---- Periodic validation ----
        if step % args.val_every == 0 or step == args.n_steps:
            current_ckpt = save_model(model, step)
            del model
            torch.cuda.empty_cache()
            model = None

            logger.info("Evaluating on %d val examples ...", args.num_val_examples)
            val_acc = evaluate_accuracy(
                current_ckpt, val_data, prompt_template,
                args.num_val_examples, args.max_new_tokens, args.gpu_memory_utilization,
            )
            logger.info("Step %d | val_accuracy=%.4f", step, val_acc)
            record["val_accuracy"] = val_acc

            # Reload model to continue training
            if step < args.n_steps:
                model = load_model(current_ckpt)
                optimizer = SGD(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay, momentum=0.9)

        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    # Final save
    if model is not None:
        save_model(model, args.n_steps)

    metrics_file.close()
    logger.info("Training complete. Metrics: %s", metrics_path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train-path",   default="data/MATH/train.jsonl")
    p.add_argument("--val-path",     default="data/MATH/validation.jsonl")
    p.add_argument("--prompt-path",  default="alignment/prompts/r1_zero.prompt")
    p.add_argument("--output-dir",   default="outputs/grpo")
    # model
    p.add_argument("--model-id",     default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--device",       default="cuda")
    # GRPO hyperparams
    p.add_argument("--n-steps",      type=int,   default=100)
    p.add_argument("--n-prompts",    type=int,   default=32,
                   help="Questions sampled per step")
    p.add_argument("--rollouts",     type=int,   default=8,
                   help="Rollouts G per question")
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--advantage-eps",type=float, default=1e-6)
    p.add_argument("--normalize-by-std", action="store_true", default=True)
    p.add_argument("--loss-type",    default="reinforce_with_baseline",
                   choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])
    p.add_argument("--cliprange",    type=float, default=0.2)
    # training
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--max-seq-len",  type=int,   default=512)
    p.add_argument("--max-grad-norm",type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    # logging / eval
    p.add_argument("--val-every",    type=int,   default=10)
    p.add_argument("--num-val-examples", type=int, default=200)
    p.add_argument("--log-generations", type=int, default=4,
                   help="Number of rollouts to print per step")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
