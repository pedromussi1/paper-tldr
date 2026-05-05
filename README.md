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
| Zero-shot Llama 3.2 1B Instruct | 0.302 | 0.102 | 0.229 | 0.657 | 20.4 | TBD |
| Zero-shot Llama 3.2 3B Instruct | 0.337 | 0.131 | 0.258 | 0.674 | 30.2 | TBD |
| **QLoRA 3B (ours)** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

*Reference summaries average 19.0 words.*

### Prompt-sensitivity ablation

We ran the LLM baselines twice. The first prompt described the task as writing a *"plain-language TL;DR for a non-specialist."* The second dropped that framing in favor of a neutral *"write a TL;DR capturing the paper's core contribution."* The numbers above use the neutral prompt; the older numbers are kept here as an ablation.

| Prompt variant | Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | mean output words |
|---|---|---|---|---|---|---|
| `plain-language` | Llama 3.2 1B | 0.297 | 0.096 | 0.223 | 0.657 | 20.2 |
| `plain-language` | Llama 3.2 3B | 0.274 | 0.072 | 0.201 | 0.657 | 31.3 |
| `neutral` | Llama 3.2 1B | 0.302 | 0.102 | 0.229 | 0.657 | 20.4 |
| `neutral` | Llama 3.2 3B | 0.337 | 0.131 | 0.258 | 0.674 | 30.2 |

**Observations:**
- With the `plain-language` prompt, **the 3B scored *lower* than the 1B on every ROUGE variant** despite being a stronger model. Manual inspection showed the 3B was faithfully following the instruction ("Researchers have developed a new approach…") while SciTLDR references are terse, technical author-style TLDRs. The 1B partly ignored the instruction and stayed in technical vocabulary, accidentally matching the reference style.
- Switching to the neutral prompt closed and reversed the gap: ROUGE-1 jumped from 0.274 → 0.337 on the 3B (+0.063), while the 1B barely moved (+0.005). BERTScore also rose for the 3B (0.657 → 0.674) but stayed flat for the 1B.
- Mean output length only changed by ≤1 word in either case, so the ROUGE jump is driven by **vocabulary alignment**, not length. The 3B's plain-language paraphrases used different surface forms than the technical references; the neutral prompt let it match them.
- Takeaway for fine-tuning: training prompts that telegraph a *style* the dataset doesn't actually exhibit punish stronger models more than weaker ones. We'll use the neutral prompt for QLoRA so the training signal is clean.

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
