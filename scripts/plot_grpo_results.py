

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_metrics(output_dir):
    records = []
    with open(Path(output_dir) / "metrics.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dirs", nargs="*", default=None)
    p.add_argument("--save-dir", default="outputs/plots")
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_dirs:
        dirs = args.output_dirs
    else:
        dirs = sorted(
            str(d) for d in Path("outputs").glob("grpo*")
            if (Path(d) / "metrics.jsonl").exists()
        )

    if not dirs:
        print("No output dirs found.")
        return

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for d in dirs:
        try:
            records = load_metrics(d)
        except FileNotFoundError:
            print(f"Skipping {d}: no metrics.jsonl")
            continue
        label = Path(d).name

        steps        = [r["step"] for r in records]
        rewards      = [r.get("mean_reward") for r in records]
        entropies    = [r.get("avg_entropy") for r in records]
        val_steps    = [r["step"] for r in records if "val_accuracy" in r]
        val_accs     = [r["val_accuracy"] for r in records if "val_accuracy" in r]

        axes[0].plot(steps, rewards, label=label, alpha=0.8)
        axes[1].plot(steps, entropies, label=label, alpha=0.8)
        if val_steps:
            axes[2].plot(val_steps, val_accs, marker="o", label=label)

    for ax, title, ylabel in zip(
        axes,
        ["Mean Rollout Reward", "Avg Token Entropy", "Validation Accuracy"],
        ["Reward", "Entropy (nats)", "Accuracy"],
    ):
        ax.set_xlabel("Training Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True)

    fig.tight_layout()
    out = Path(args.save_dir) / "grpo_training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
