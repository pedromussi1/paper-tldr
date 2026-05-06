"""Run a baseline on a JSONL split, score it, and (optionally) log to W&B.

Usage:
    # extractive (no model load — runs in seconds, no auth needed)
    python scripts/run_baselines.py --baseline extractive --split val

    # zero-shot with a chat-tuned LLM (requires `huggingface-cli login` first)
    python scripts/run_baselines.py --baseline zero-shot --split val \
        --model meta-llama/Llama-3.2-3B-Instruct

Outputs:
    outputs/baselines/<name>__<split>.jsonl     (predictions per row)
    outputs/baselines/<name>__<split>.metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# allow `python scripts/run_baselines.py ...` from repo root without -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import (  # noqa: E402
    extractive_first_sentence,
    qlora_finetuned_llm,
    zero_shot_llm,
)
from src.eval import compute_metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "processed"
OUT_DIR = REPO / "outputs" / "baselines"


def _load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save_predictions(name: str, split: str, rows: list[dict], preds: list[str]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}__{split}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row, pred in zip(rows, preds, strict=True):
            f.write(json.dumps({**row, "prediction": pred}, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--baseline",
        choices=["extractive", "zero-shot", "qlora"],
        required=True,
        help="Which baseline to run.",
    )
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="HF model id (zero-shot/qlora base).")
    ap.add_argument("--adapter-path", default=None, help="Path to LoRA adapter (qlora mode).")
    ap.add_argument("--batch-size", type=int, default=4, help="Generation batch size (LLM modes only).")
    ap.add_argument("--max-new-tokens", type=int, default=96, help="Generation cap (LLM modes only).")
    ap.add_argument("--limit", type=int, default=0, help="Cap rows for a quick smoke run (0 = all).")
    ap.add_argument("--name", default=None, help="Override the run name (used in output filenames).")
    ap.add_argument("--no-wandb", action="store_true", help="Skip W&B logging.")
    args = ap.parse_args()

    rows = _load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    abstracts = [r["abstract"] for r in rows]
    refs = [r["summary"] for r in rows]
    print(f"loaded {len(rows)} rows from {args.split}")

    if args.baseline == "extractive":
        name = args.name or "extractive_first_sentence"
        t0 = time.perf_counter()
        preds = [extractive_first_sentence(a) for a in abstracts]
        elapsed = time.perf_counter() - t0
    elif args.baseline == "zero-shot":
        name = args.name or f"zero_shot__{args.model.split('/')[-1]}"
        t0 = time.perf_counter()
        preds = zero_shot_llm(
            abstracts,
            model_name=args.model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.perf_counter() - t0
    else:  # qlora
        if not args.adapter_path:
            raise SystemExit("qlora mode requires --adapter-path")
        adapter_name = Path(args.adapter_path).parent.name + "_" + Path(args.adapter_path).name
        name = args.name or f"qlora__{adapter_name}"
        t0 = time.perf_counter()
        preds = qlora_finetuned_llm(
            abstracts,
            base_model_name=args.model,
            adapter_path=args.adapter_path,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.perf_counter() - t0
    print(f"generated {len(preds)} predictions in {elapsed:.1f}s")

    pred_path = _save_predictions(name, args.split, rows, preds)
    print(f"wrote predictions: {pred_path}")

    print("computing metrics...")
    metrics = compute_metrics(preds, refs)
    metrics_row = metrics.as_row(name)
    metrics_path = pred_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics_row, indent=2))
    print(json.dumps(metrics_row, indent=2))

    if not args.no_wandb:
        try:
            import wandb

            wandb.init(project="paper-tldr", name=name, job_type="baseline", config=vars(args))
            wandb.log({k: v for k, v in metrics_row.items() if isinstance(v, (int, float))})
            wandb.finish()
            print("logged to W&B")
        except Exception as e:
            print(f"W&B logging failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
