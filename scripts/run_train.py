"""CLI for QLoRA fine-tuning. Loads a YAML config and optionally applies
key=value overrides for ablation runs.

Usage:
    # full training run with the default config
    python scripts/run_train.py --config configs/base.yaml

    # smoke test (5 steps)
    python scripts/run_train.py --config configs/base.yaml \
        --override num_epochs=0.05 logging_steps=1 eval_steps=5 save_steps=5 \
                   run_name=smoke output_dir=checkpoints/smoke

    # rank ablation
    python scripts/run_train.py --config configs/base.yaml \
        --override lora_rank=32 lora_alpha=64 \
                   run_name=qlora-3b-r32 output_dir=checkpoints/qlora-3b-r32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import TrainConfig, train  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _coerce_like(template, raw: str):
    """Cast a CLI string to the type of the existing config value.

    Using the value's runtime type rather than the dataclass annotation avoids
    issues with PEP 563 stringified annotations (`from __future__ import
    annotations`), where `field.type` would be the literal string `"int"`.
    """
    if isinstance(template, bool):
        return raw.lower() in {"1", "true", "yes", "y"}
    if isinstance(template, int):
        return int(raw)
    if isinstance(template, float):
        return float(raw)
    if isinstance(template, Path):
        return Path(raw)
    return raw


def _apply_overrides(cfg_dict: dict, overrides: list[str]) -> dict:
    for kv in overrides:
        if "=" not in kv:
            raise SystemExit(f"override must be key=value, got: {kv!r}")
        key, raw = kv.split("=", 1)
        if key not in cfg_dict:
            raise SystemExit(f"unknown config field: {key!r}")
        cfg_dict[key] = _coerce_like(cfg_dict[key], raw)
    return cfg_dict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "base.yaml")
    ap.add_argument("--override", nargs="*", default=[], help="key=value overrides applied on top of the YAML config.")
    args = ap.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg_dict: dict = yaml.safe_load(f)
    cfg_dict = _apply_overrides(cfg_dict, args.override)
    cfg_dict["processed_dir"] = REPO / cfg_dict["processed_dir"]
    cfg_dict["output_dir"] = REPO / cfg_dict["output_dir"]

    cfg = TrainConfig(**cfg_dict)
    print(f"resolved config: {cfg}")
    final_path = train(cfg)
    print(f"\nfinal adapter saved to: {final_path}")


if __name__ == "__main__":
    main()
