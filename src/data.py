"""Dataset loading and tokenization for QLoRA training.

Loads the JSONL produced by scripts/prepare_data.py and renders each example as
a chat-formatted conversation that Llama 3.2 Instruct expects, then tokenizes
with the model's tokenizer. Labels mask everything except the assistant's
summary span so the loss is computed only on the target tokens.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

SYSTEM_PROMPT = (
    "You are a science communicator. Given a paper abstract, write a plain-language "
    "TL;DR (one or two sentences) that preserves the core finding for a non-specialist. "
    "Output only the TL;DR sentences themselves — no preamble, no quotation marks, "
    "no markdown, no explanation."
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(processed_dir: Path, split: str) -> Dataset:
    """split is one of {'train', 'val', 'test'}."""
    rows = _read_jsonl(processed_dir / f"{split}.jsonl")
    return Dataset.from_list(rows)


def _render_chat(tokenizer: PreTrainedTokenizerBase, abstract: str, summary: str | None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Abstract:\n{abstract}\n\nWrite the TL;DR."},
    ]
    if summary is not None:
        messages.append({"role": "assistant", "content": summary})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=summary is None,
    )


@dataclass
class TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


def tokenize_with_completion_mask(
    tokenizer: PreTrainedTokenizerBase,
    abstract: str,
    summary: str,
    max_length: int = 1024,
) -> TokenizedExample:
    """Tokenize a (abstract, summary) pair, masking the prompt portion in labels."""
    prompt_text = _render_chat(tokenizer, abstract, summary=None)
    full_text = _render_chat(tokenizer, abstract, summary=summary)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100  # mask prompt tokens from the loss

    attention_mask = [1] * len(full_ids)
    return TokenizedExample(input_ids=full_ids, attention_mask=attention_mask, labels=labels)


def map_for_training(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 1024,
) -> Dataset:
    def _map(example: dict) -> dict:
        ex = tokenize_with_completion_mask(
            tokenizer, example["abstract"], example["summary"], max_length=max_length
        )
        return {"input_ids": ex.input_ids, "attention_mask": ex.attention_mask, "labels": ex.labels}

    return ds.map(_map, remove_columns=ds.column_names, desc="tokenizing")


@dataclass
class CollateForCausalLM:
    """Right-pads input_ids/labels/attention_mask to the longest sequence in the batch."""

    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), self.label_pad_token_id, dtype=torch.long)
        for i, ex in enumerate(batch):
            n = len(ex["input_ids"])
            input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
            attention_mask[i, :n] = torch.tensor(ex["attention_mask"], dtype=torch.long)
            labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
