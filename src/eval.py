"""Evaluation metrics for abstract → TL;DR generation.

Reports ROUGE-1, ROUGE-2, ROUGE-L (F-measure means) and BERTScore F1.
A held-out test JSONL is the reference; predictions are passed alongside as a
parallel list of strings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import evaluate


@dataclass
class GenerationMetrics:
    rouge1: float
    rouge2: float
    rougeL: float
    bertscore_f1: float
    n: int

    def as_row(self, name: str) -> dict:
        return {"name": name, **asdict(self)}


def compute_metrics(
    predictions: list[str],
    references: list[str],
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    bertscore_lang: str = "en",
) -> GenerationMetrics:
    if len(predictions) != len(references):
        raise ValueError(f"length mismatch: {len(predictions)} preds vs {len(references)} refs")

    rouge = evaluate.load("rouge")
    rouge_out = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
        use_aggregator=True,
    )

    bertscore = evaluate.load("bertscore")
    bertscore_out = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type=bertscore_model,
        lang=bertscore_lang,
    )
    bert_f1 = sum(bertscore_out["f1"]) / len(bertscore_out["f1"])

    return GenerationMetrics(
        rouge1=float(rouge_out["rouge1"]),
        rouge2=float(rouge_out["rouge2"]),
        rougeL=float(rouge_out["rougeL"]),
        bertscore_f1=float(bert_f1),
        n=len(predictions),
    )
