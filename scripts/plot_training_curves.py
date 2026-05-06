"""Plot train/eval loss curves + LR schedule for a W&B training run.

Pulls history from the Weights & Biases API (re-uses the credentials in
~/_netrc), so the plot is reproducible by anyone who can read the project.
Saves a single PNG to outputs/training_curves_<run_name>.png.

Usage:
    python scripts/plot_training_curves.py                            # uses the canonical r=16 run
    python scripts/plot_training_curves.py --run-id <id> --label r=32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display backend on Windows
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
# Saved under docs/img/ rather than outputs/ so the PNG is tracked in git and
# renders inline in the README on GitHub.
OUT_DIR = REPO / "docs" / "img"

DEFAULT_ENTITY = "pedromussi-pedro-mussi"
DEFAULT_PROJECT = "paper-tldr"
DEFAULT_RUN_ID = "rc05szkq"  # canonical r=16 training run
DEFAULT_LABEL = "QLoRA 3B, rank=16"


def _first_existing_col(columns: list[str], *candidates: str) -> str | None:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=DEFAULT_ENTITY)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--run-id", default=DEFAULT_RUN_ID)
    ap.add_argument("--label", default=DEFAULT_LABEL)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import pandas as pd  # imported lazily so --help is fast

    import wandb

    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{args.run_id}")
    history = pd.DataFrame(run.scan_history())
    print(f"loaded {len(history)} history rows; columns: {list(history.columns)[:12]}...")

    step_col = _first_existing_col(list(history.columns), "_step", "global_step", "step")
    train_loss_col = _first_existing_col(list(history.columns), "train/loss", "loss")
    eval_loss_col = _first_existing_col(list(history.columns), "eval/loss", "eval_loss")
    lr_col = _first_existing_col(list(history.columns), "train/learning_rate", "learning_rate")
    if step_col is None:
        raise SystemExit(f"could not find a step column in {list(history.columns)}")
    if train_loss_col is None:
        raise SystemExit(f"could not find a train-loss column in {list(history.columns)}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    train = history[[step_col, train_loss_col]].dropna()
    axes[0].plot(train[step_col], train[train_loss_col], label="train", color="#1f77b4")
    if eval_loss_col is not None:
        ev = history[[step_col, eval_loss_col]].dropna()
        axes[0].plot(ev[step_col], ev[eval_loss_col], "o-", label="eval (val subset)", color="#d62728")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"Loss curves — {args.label}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if lr_col is not None:
        lr = history[[step_col, lr_col]].dropna()
        axes[1].plot(lr[step_col], lr[lr_col], color="#2ca02c")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("learning rate")
        axes[1].set_title("Cosine LR schedule (3% warmup)")
        axes[1].grid(alpha=0.3)
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    out = args.out or (OUT_DIR / f"training_curves_{args.run_id}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
