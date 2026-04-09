"""
Filter sft.jsonl to keep only examples that produce the correct answer,
and report the resulting dataset size.

Usage:
  python scripts/filter_sft_data.py \
      --input-path data/sft.jsonl \
      --output-path data/sft_filtered.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main(input_path: str, output_path: str) -> None:
    from alignment.drgrpo_grader import extract_boxed_answer  # needs math deps on remote

    total, kept = 0, 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            ex = json.loads(line.strip())
            total += 1

            response = ex.get("response", ex.get("output", ""))

            # Check 1: proper </think> <answer>...</answer> format
            if "</think> <answer>" not in response or "</answer>" not in response:
                continue

            # Check 2: <answer> section contains a parseable \boxed{} expression
            answer_section = response.split("<answer>")[-1].replace("</answer>", "")
            if "\\boxed" not in answer_section:
                continue
            extracted = extract_boxed_answer(answer_section)
            if extracted is None:
                continue

            fout.write(json.dumps(ex) + "\n")
            kept += 1

    logger.info(
        "Filtered dataset: %d / %d examples kept (%.1f%%)",
        kept, total, 100 * kept / max(total, 1),
    )
    logger.info("Written to: %s", output_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-path",  default="data/sft.jsonl")
    p.add_argument("--output-path", default="data/sft_filtered.jsonl")
    args = p.parse_args()
    main(args.input_path, args.output_path)
