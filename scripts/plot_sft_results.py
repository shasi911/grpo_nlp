

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
    p.add_argument("--output-dirs", nargs="*", default=None,
                   help="Dirs to plot. Auto-discovers outputs/sft_* if not specified.")
    p.add_argument("--save-dir", default="outputs/plots")
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_dirs:
        dirs = args.output_dirs
    else:
        dirs = sorted(
            str(d) for d in Path("outputs").glob("sft_*")
            if (Path(d) / "metrics.jsonl").exists()
        )

    if not dirs:
        print("No SFT output dirs found under outputs/sft_*")
        return

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for d in dirs:
        try:
            records = load_metrics(d)
        except FileNotFoundError:
            print(f"Skipping {d}: no metrics.jsonl")
            continue

        # Extract per-epoch val accuracy
        val_records = [r for r in records if "val_accuracy" in r]
        if not val_records:
            print(f"Skipping {d}: no val_accuracy entries")
            continue

        epochs = [r["epoch"] for r in val_records]
        accs   = [r["val_accuracy"] for r in val_records]
        label  = Path(d).name

        ax.plot(epochs, accs, marker="o", label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("SFT Validation Accuracy by Dataset Size")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()

    out = Path(args.save_dir) / "sft_val_accuracy.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
