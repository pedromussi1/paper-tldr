# Paper TL;DR

> Fine-tuning Llama 3.2 3B Instruct with QLoRA to produce plain-language summaries of scientific paper abstracts.
>
> **Status:** Week 1 — data + baselines

## Problem

Scientific paper abstracts are dense and jargon-heavy. This project fine-tunes a small open-weight LLM to rewrite arXiv-style abstracts as plain-language TL;DRs that preserve the core finding while being readable at a layperson level. The result is a quantized GGUF model that runs locally and integrates with PaperPal as a summarization tool.

## Approach

| Component | Choice |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Adapter | QLoRA, 4-bit NF4 quantization, rank 16 / alpha 32 (default) |
| Dataset | SciTLDR (~5K pairs) + PubMed lay summaries (filtered + deduped) |
| Training | HuggingFace Trainer + PEFT + bitsandbytes, ~1–2 hr on RTX 4070 Ti |
| Evaluation | ROUGE-1/2/L + BERTScore + n=30 blind human Likert |
| Deployment | Merged adapter → GGUF (`q4_K_M`) → Ollama Modelfile |

## Baselines

1. **Extractive** — first sentence of the abstract
2. **Zero-shot Llama 3.2 3B Instruct** — same model, no fine-tune
3. **Zero-shot Llama 3.2 1B Instruct** — smaller-model lower bound

## Results

Reported on SciTLDR validation split (n=618). Test-set numbers will be reported in the final results section once the QLoRA model converges; we do not touch test during development.

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | mean output words | Human Likert (1–5) |
|---|---|---|---|---|---|---|
| First-sentence extractive | 0.259 | 0.093 | 0.205 | 0.637 | — | TBD |
| Zero-shot Llama 3.2 1B Instruct | 0.297 | 0.096 | 0.223 | 0.657 | 20.2 | TBD |
| Zero-shot Llama 3.2 3B Instruct | 0.274 | 0.072 | 0.201 | 0.657 | 31.3 | TBD |
| **QLoRA 3B (ours)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

*Reference summaries average 19.0 words.*

### Baseline observations

- **ROUGE is non-monotonic in model size on this benchmark.** The 3B Instruct scores *lower* than the 1B Instruct on every ROUGE variant despite being a stronger model. Manual inspection shows the 3B follows the system prompt's "plain-language" instruction faithfully ("Researchers have developed a new approach..."), while SciTLDR's reference summaries are technical paper-style TLDRs (often the authors' own language). The 3B's outputs are 60% longer than the 1B's (31.3 vs 20.2 words) and ~65% longer than the references (19.0 words) — the wordiness directly drives the ROUGE drop.
- **BERTScore is essentially tied** (0.6566 / 0.6566), confirming the semantic content is equivalent — the two models disagree on style, not substance.
- **Implication for fine-tuning:** the QLoRA training signal teaches the model to match SciTLDR's terse, technical style. This may conflict with the "plain-language" framing in the system prompt; we'll either drop that framing during fine-tuning (cleaner signal) or report both prompt variants as an ablation. Decision logged in the project notes.

## Ablations (planned)

- LoRA rank: 8 / 16 / 32 / 64 (alpha = 2 × rank)
- Dataset size: 25% / 50% / 100%
- Inference quantization: 4-bit / 8-bit / fp16

## Error analysis (planned)

Failure modes categorized from the human eval: hallucination, oversimplification, term-dropping, register mismatch, length blow-out.

## Limitations

TBD — discussed at end of project.

## Setup

Hardware tested: NVIDIA RTX 4070 Ti (12 GB VRAM), 64 GB RAM, Windows 11, CUDA 12.x

```powershell
# create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install PyTorch with CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# install everything else
pip install -r requirements.txt

# log into Hugging Face (needed for Llama 3.2 weights) and W&B
huggingface-cli login
wandb login
```

## Project layout

```
src/         training, eval, baselines (Python package)
scripts/     CLI entrypoints (prepare_data, run_train, run_eval)
configs/     training config YAMLs
notebooks/   EDA, error analysis
data/        DVC-tracked datasets (gitignored)
tests/       unit tests
```

## License

MIT
