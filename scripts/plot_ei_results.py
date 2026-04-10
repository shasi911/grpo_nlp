"""
Plot EI validation accuracy and entropy curves from metrics.jsonl files.

Usage:
  python scripts/plot_ei_results.py --output-dirs outputs/ei_G4_db1024_ep1 outputs/ei_G8_db1024_ep1
  python scripts/plot_ei_results.py  # auto-discovers all outputs/ei_* dirs
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(output_dir: str) -> list[dict]:
    path = Path(output_dir) / "metrics.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dirs", nargs="*", default=None,
                   help="Dirs to plot. Auto-discovers outputs/ei_* if not specified.")
    p.add_argument("--save-dir", default="outputs/plots")
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_dirs:
        dirs = args.output_dirs
    else:
        dirs = sorted(str(p) for p in Path("outputs").glob("ei_*") if (Path(p) / "metrics.jsonl").exists())

    if not dirs:
        print("No output dirs found.")
        return

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # --- Accuracy plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in dirs:
        try:
            records = load_metrics(d)
        except FileNotFoundError:
            print(f"Skipping {d}: metrics.jsonl not found")
            continue
        steps = [r["ei_step"] for r in records if r.get("val_accuracy") is not None]
        accs  = [r["val_accuracy"] for r in records if r.get("val_accuracy") is not None]
        label = Path(d).name
        ax.plot(steps, accs, marker="o", label=label)

    ax.set_xlabel("EI Step")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Expert Iteration: Validation Accuracy")
    ax.legend(fontsize=7)
    ax.grid(True)
    fig.tight_layout()
    acc_path = Path(args.save_dir) / "ei_val_accuracy.png"
    fig.savefig(acc_path)
    print(f"Saved: {acc_path}")

    # --- Entropy plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in dirs:
        try:
            records = load_metrics(d)
        except FileNotFoundError:
            continue
        steps   = [r["ei_step"] for r in records if r.get("avg_entropy") is not None]
        entropy = [r["avg_entropy"] for r in records if r.get("avg_entropy") is not None]
        if not steps:
            continue
        label = Path(d).name
        ax.plot(steps, entropy, marker="o", label=label)

    ax.set_xlabel("EI Step")
    ax.set_ylabel("Avg Token Entropy (response tokens)")
    ax.set_title("Expert Iteration: Response Entropy over Training")
    ax.legend(fontsize=7)
    ax.grid(True)
    fig.tight_layout()
    ent_path = Path(args.save_dir) / "ei_entropy.png"
    fig.savefig(ent_path)
    print(f"Saved: {ent_path}")


if __name__ == "__main__":
    main()
