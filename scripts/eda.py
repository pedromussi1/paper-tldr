"""Exploratory data analysis for the SciTLDR splits.

Reports per-split sizes, token-length distributions for abstracts and summaries
(measured with the Llama 3.2 tokenizer so they match training-time numbers),
compression ratio, and a small set of example pairs. Saves one plot summarizing
the length distributions to outputs/eda/lengths.png.

Run:
    python scripts/eda.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display backend on Windows
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "processed"
OUT_DIR = REPO / "outputs" / "eda"

TOKENIZER_NAME = "meta-llama/Llama-3.2-3B-Instruct"
SAMPLE_K = 4


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _percentiles(arr: np.ndarray) -> dict:
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    splits: dict[str, list[dict]] = {
        s: _read_jsonl(DATA_DIR / f"{s}.jsonl") for s in ("train", "val", "test")
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    report: dict = {}

    for split, rows in splits.items():
        abs_lens = np.array([len(tok.encode(r["abstract"], add_special_tokens=False)) for r in rows])
        sum_lens = np.array([len(tok.encode(r["summary"], add_special_tokens=False)) for r in rows])
        compress = sum_lens.astype(float) / np.maximum(abs_lens, 1)

        report[split] = {
            "size": len(rows),
            "abstract_tokens": _percentiles(abs_lens),
            "summary_tokens": _percentiles(sum_lens),
            "compression_ratio": {
                "mean": float(compress.mean()),
                "median": float(np.median(compress)),
            },
        }

        axes[0].hist(abs_lens, bins=40, alpha=0.5, label=f"{split} (n={len(rows)})")
        axes[1].hist(sum_lens, bins=30, alpha=0.5, label=f"{split} (n={len(rows)})")

    axes[0].set_title("Abstract length (tokens)")
    axes[0].set_xlabel("tokens")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[1].set_title("Summary length (tokens)")
    axes[1].set_xlabel("tokens")
    axes[1].legend()
    fig.tight_layout()
    plot_path = OUT_DIR / "lengths.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    random.seed(7)
    samples = random.sample(splits["train"], SAMPLE_K)
    report["examples"] = [
        {
            "abstract": r["abstract"][:300] + ("…" if len(r["abstract"]) > 300 else ""),
            "summary": r["summary"],
        }
        for r in samples
    ]

    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    print(f"\nSaved plot: {plot_path}")
    print(f"Saved full report: {report_path}")


if __name__ == "__main__":
    main()
