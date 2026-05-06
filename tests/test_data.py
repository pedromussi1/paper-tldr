"""Tests for src/data.py — model-agnostic helpers only.

The tokenizer-dependent helpers (tokenize_with_completion_mask, map_for_training,
_render_chat) are not unit-tested here because they need to download a chat-
template-aware tokenizer. They're exercised end-to-end by the training run.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data import CollateForCausalLM, load_split


def test_load_split_reads_jsonl(tmp_path: Path):
    rows = [
        {"id": "a1", "abstract": "Abstract one.", "summary": "Summary one.", "source": "test"},
        {"id": "a2", "abstract": "Abstract two.", "summary": "Summary two.", "source": "test"},
    ]
    (tmp_path / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    ds = load_split(tmp_path, "train")
    assert len(ds) == 2
    assert ds[0]["abstract"] == "Abstract one."
    assert ds[1]["summary"] == "Summary two."


class TestCollator:
    def test_pads_input_ids_to_longest(self):
        collator = CollateForCausalLM(pad_token_id=0)
        batch = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
        ]
        out = collator(batch)
        assert out["input_ids"].shape == (2, 3)
        assert out["input_ids"][0].tolist() == [1, 2, 3]
        assert out["input_ids"][1].tolist() == [4, 5, 0]
        assert out["attention_mask"][1].tolist() == [1, 1, 0]

    def test_pads_labels_with_label_pad_id(self):
        collator = CollateForCausalLM(pad_token_id=0, label_pad_token_id=-100)
        batch = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
            {"input_ids": [4], "attention_mask": [1], "labels": [4]},
        ]
        out = collator(batch)
        assert out["labels"][1].tolist() == [4, -100, -100]

    def test_preserves_aligned_rows(self):
        collator = CollateForCausalLM(pad_token_id=0)
        batch = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]},
            {"input_ids": [3, 4], "attention_mask": [1, 1], "labels": [3, 4]},
        ]
        out = collator(batch)
        assert torch.equal(out["input_ids"], torch.tensor([[1, 2], [3, 4]]))
        assert torch.equal(out["attention_mask"], torch.tensor([[1, 1], [1, 1]]))

    def test_returns_long_tensors(self):
        collator = CollateForCausalLM(pad_token_id=0)
        batch = [{"input_ids": [1], "attention_mask": [1], "labels": [1]}]
        out = collator(batch)
        for key in ("input_ids", "attention_mask", "labels"):
            assert out[key].dtype == torch.long, f"{key} should be int64"
