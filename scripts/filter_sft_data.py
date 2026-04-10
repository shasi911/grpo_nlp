

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def build_answer_lookup(math_train_path: str) -> dict[str, str]:
    """Build a dict mapping problem text → ground truth answer from MATH train."""
    lookup = {}
    with open(math_train_path) as f:
        for line in f:
            ex = json.loads(line.strip())
            problem = ex.get("problem", ex.get("question", "")).strip()
            answer  = str(ex.get("answer", ex.get("solution", ""))).strip()
            if problem:
                lookup[problem] = answer
    logger.info("Loaded %d ground truth answers from %s", len(lookup), math_train_path)
    return lookup


def main(input_path: str, math_train_path: str, output_path: str) -> None:
    from alignment.drgrpo_grader import r1_zero_reward_fn

    answer_lookup = build_answer_lookup(math_train_path)

    total, kept, no_gt = 0, 0, 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            ex = json.loads(line.strip())
            total += 1

            # Get the problem text to look up ground truth
            problem = ex.get("problem", ex.get("question", ""))
            if not problem:
                # Try to extract from prompt field
                prompt = ex.get("prompt", "")
                # The prompt contains "User: <problem>\nAssistant:" — extract it
                if "User:" in prompt:
                    problem = prompt.split("User:")[-1].split("Assistant:")[0].strip()

            ground_truth = answer_lookup.get(problem.strip(), "")

            # Fall back to inline ground truth if present
            if not ground_truth:
                ground_truth = str(ex.get("answer", ex.get("solution", "")))

            if not ground_truth:
                no_gt += 1
                continue

            response = ex.get("response", ex.get("output", ""))
            result = r1_zero_reward_fn(response, ground_truth)

            if result["answer_reward"] == 1.0:
                fout.write(json.dumps(ex) + "\n")
                kept += 1

    logger.info(
        "Filtered dataset: %d / %d examples kept (%.1f%%)",
        kept, total, 100 * kept / max(total, 1),
    )
    if no_gt:
        logger.warning("%d examples skipped (no ground truth found)", no_gt)
    logger.info("Written to: %s", output_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-path",       default="data/sft.jsonl")
    p.add_argument("--math-train-path",  default="data/MATH/train.jsonl")
    p.add_argument("--output-path",      default="data/sft_filtered.jsonl")
    args = p.parse_args()
    main(args.input_path, args.math_train_path, args.output_path)
