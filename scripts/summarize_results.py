"""Aggregate baseline + fine-tuned eval metrics into a single markdown table.

Reads every `outputs/baselines/<name>__<split>.metrics.json`, joins it with the
mean output-word count from the matching `<name>__<split>.jsonl`, and prints a
markdown row in a stable, hand-curated order. Use this to refresh the README
results table without manual editing.

Usage:
    python scripts/summarize_results.py                    # val table, current canonical rows
    python scripts/summarize_results.py --split test       # test table
    python scripts/summarize_results.py --include-deprecated  # also show plain-language-prompt baselines
    python scripts/summarize_results.py --out outputs/results_val.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO / "outputs" / "baselines"

# Hand-curated display labels and row order. Anything not in this map is shown
# at the bottom in alphabetical order with its raw name as the label.
DISPLAY = {
    "extractive_first_sentence": "First-sentence extractive",
    "zero_shot_neutral__Llama-3.2-1B-Instruct": "Zero-shot Llama 3.2 1B Instruct",
    "zero_shot_neutral__Llama-3.2-3B-Instruct": "Zero-shot Llama 3.2 3B Instruct",
    "qlora_3b_r8": "QLoRA 3B, rank=8",
    "qlora_3b_r16": "QLoRA 3B, rank=16",
    "qlora_3b_r32": "QLoRA 3B, rank=32",
    "qlora_3b_r64": "QLoRA 3B, rank=64",
}
ORDER = list(DISPLAY)

# These are kept as ablation footnote rows but excluded from the canonical table
# unless --include-deprecated is passed.
DEPRECATED = {
    "zero_shot__Llama-3.2-1B-Instruct": "Zero-shot Llama 3.2 1B (plain-language prompt, deprecated)",
    "zero_shot__Llama-3.2-3B-Instruct": "Zero-shot Llama 3.2 3B (plain-language prompt, deprecated)",
}


def _mean_output_words(jsonl_path: Path) -> float | None:
    if not jsonl_path.exists():
        return None
    n, total = 0, 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pred = row.get("prediction")
            if pred is None:
                continue
            total += len(pred.split())
            n += 1
    return total / n if n else None


def _row(name: str, label: str, metrics: dict, words: float | None, *, bold: bool = False) -> str:
    def fmt(v: float | None, fmt_str: str) -> str:
        return format(v, fmt_str) if v is not None else "—"

    cells = [
        f"**{label}**" if bold else label,
        fmt(metrics.get("rouge1"), ".3f"),
        fmt(metrics.get("rouge2"), ".3f"),
        fmt(metrics.get("rougeL"), ".3f"),
        fmt(metrics.get("bertscore_f1"), ".3f"),
        fmt(words, ".1f") if words is not None else "—",
    ]
    if bold:
        cells = [f"**{c}**" if i > 0 and c != "—" else c for i, c in enumerate(cells)]
    return "| " + " | ".join(cells) + " |"


def _load_run(name: str, split: str) -> tuple[dict, float | None] | None:
    metrics_path = BASELINES_DIR / f"{name}__{split}.metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    words = _mean_output_words(BASELINES_DIR / f"{name}__{split}.jsonl")
    return metrics, words


def build_table(split: str, *, include_deprecated: bool) -> str:
    header = (
        "| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | mean output words |\n"
        "|---|---|---|---|---|---|"
    )
    lines = [header]

    # Find the best fine-tune by ROUGE-1 so we can bold it.
    best_qlora_name = None
    best_qlora_r1 = -1.0
    for name in ORDER:
        if not name.startswith("qlora_"):
            continue
        loaded = _load_run(name, split)
        if loaded is None:
            continue
        r1 = loaded[0].get("rouge1", -1.0)
        if r1 > best_qlora_r1:
            best_qlora_r1 = r1
            best_qlora_name = name

    seen: set[str] = set()
    for name in ORDER:
        loaded = _load_run(name, split)
        if loaded is None:
            continue
        metrics, words = loaded
        lines.append(_row(name, DISPLAY[name], metrics, words, bold=(name == best_qlora_name)))
        seen.add(name)

    if include_deprecated:
        for name, label in DEPRECATED.items():
            loaded = _load_run(name, split)
            if loaded is None:
                continue
            metrics, words = loaded
            lines.append(_row(name, label, metrics, words))

    # Anything we didn't account for explicitly — show alphabetically at the bottom.
    extras: list[tuple[str, dict, float | None]] = []
    for path in sorted(BASELINES_DIR.glob(f"*__{split}.metrics.json")):
        name = path.stem.replace(f"__{split}.metrics", "")
        if name in seen or name in DEPRECATED:
            continue
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        words = _mean_output_words(BASELINES_DIR / f"{name}__{split}.jsonl")
        extras.append((name, metrics, words))
    for name, metrics, words in extras:
        lines.append(_row(name, name, metrics, words))

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--include-deprecated", action="store_true",
                    help="Also include the plain-language-prompt zero-shot rows.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional path to write the table to (in addition to stdout).")
    args = ap.parse_args()

    table = build_table(args.split, include_deprecated=args.include_deprecated)
    print(f"### Results - `{args.split}` split\n")
    print(table)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
        print(f"\n[wrote {args.out}]")


if __name__ == "__main__":
    main()
