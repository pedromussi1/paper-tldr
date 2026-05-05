"""Download and prepare the abstract → plain-language summary dataset.

Source: SciTLDR (Cachola et al., 2020) — extreme summarization of CS papers,
where each abstract has 1–5 reference TLDRs. We use the "Abstract" config
(input = abstract only, no full text) and take the first TLDR as the canonical
target, since that's how the original paper evaluates.

Outputs:
    data/processed/train.jsonl
    data/processed/val.jsonl
    data/processed/test.jsonl

Each line: {"id": str, "abstract": str, "summary": str, "source": str}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

SEED = 1337
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "processed"


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


def _normalize_scitldr(example: dict) -> dict | None:
    """SciTLDR row → unified schema. Returns None if the row is unusable."""
    abstract_tokens = example.get("source") or []
    targets = example.get("target") or []
    if not abstract_tokens or not targets:
        return None
    abstract = _clean(" ".join(abstract_tokens))
    summary = _clean(targets[0])
    if len(abstract) < 100 or len(summary) < 10:
        return None
    return {
        "id": _hash(abstract),
        "abstract": abstract,
        "summary": summary,
        "source": "scitldr",
    }


def _dedupe(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        out.append(row)
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(out_dir: Path = DEFAULT_OUT) -> dict[str, int]:
    random.seed(SEED)
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        ds = load_dataset("allenai/scitldr", "Abstract", split=split)
        rows = [_normalize_scitldr(r) for r in ds]
        rows = [r for r in rows if r is not None]
        rows = _dedupe(rows)
        random.shuffle(rows)
        out_name = "val.jsonl" if split == "validation" else f"{split}.jsonl"
        _write_jsonl(out_dir / out_name, rows)
        counts[split] = len(rows)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    args = ap.parse_args()
    counts = prepare(args.out)
    print(json.dumps({"split_sizes": counts, "out_dir": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
