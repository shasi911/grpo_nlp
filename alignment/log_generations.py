import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def log_generations(
    prompts: list[str],
    generations: list[str],
    rewards: list[dict],
    step: int,
    log_file: str | None = None,
    num_examples: int = 4,
) -> None:
    """Log model generations along with their rewards.

    Prints a sample of generations to the Python logger and optionally
    appends all examples to a JSONL file.

    Args:
        prompts: The input prompt strings fed to the model.
        generations: The model-generated response strings.
        rewards: List of reward dicts (keys: "reward", "format_reward",
                 "answer_reward") for each generation.
        step: Current training step, used for console logging.
        log_file: Optional path to a JSONL file. Each call appends one
                  record per generation.
        num_examples: Number of examples to print to the logger.
    """
    # --- console logging (sample) ---
    logger.info("=== Generations at step %d ===", step)
    for i, (prompt, gen, rew) in enumerate(zip(prompts, generations, rewards)):
        if i >= num_examples:
            break
        reward_val = rew.get("reward", rew) if isinstance(rew, dict) else rew
        format_reward = rew.get("format_reward", "n/a") if isinstance(rew, dict) else "n/a"
        answer_reward = rew.get("answer_reward", "n/a") if isinstance(rew, dict) else "n/a"
        logger.info(
            "[%d/%d] reward=%.3f  format=%.3f  answer=%.3f\n"
            "  PROMPT   : %s\n"
            "  RESPONSE : %s",
            i + 1,
            len(prompts),
            reward_val,
            format_reward,
            answer_reward,
            prompt[:200],
            gen[:400],
        )

    # --- file logging (all examples) ---
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            for prompt, gen, rew in zip(prompts, generations, rewards):
                record = {
                    "step": step,
                    "prompt": prompt,
                    "generation": gen,
                    "reward": rew.get("reward", rew) if isinstance(rew, dict) else rew,
                    "format_reward": rew.get("format_reward", None) if isinstance(rew, dict) else None,
                    "answer_reward": rew.get("answer_reward", None) if isinstance(rew, dict) else None,
                }
                f.write(json.dumps(record) + "\n")
