# Changelog

All notable changes to **paper-tldr** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this
project uses [Semantic Versioning](https://semver.org/). Each release is also
published on the [GitHub releases page](https://github.com/pedromussi1/paper-tldr/releases).

## [1.0.0] — 2026-05-13

Final training + evaluation milestone: QLoRA fine-tune shipped with rigorous
held-out results.

### Added
- Test-set evaluation on the held-out SciTLDR split (n=618) with ROUGE-1/2/L
  and BERTScore.
- Rank ablation (LoRA r=8 vs r=16); shipped **r=8** as the best size/quality
  trade-off.
- Verified result: the QLoRA fine-tune of Llama 3.2 3B beats the zero-shot 3B
  baseline on every reported metric.

## [0.2.0] — 2026-05-13

### Added
- Week 2 — QLoRA fine-tuning run: 4-bit NF4 + LoRA (rank 16, alpha 32) with
  completion-only label masking; beats the zero-shot 3B baseline on every metric.

## [0.1.0] — 2026-05-13

### Added
- Week 1 — data pipeline (SciTLDR prep + EDA), three baselines (first-sentence
  extractive, zero-shot 3B, zero-shot 1B), and the ROUGE/BERTScore eval harness.
- MIT license, pytest suite, GitHub Actions CI.

[1.0.0]: https://github.com/pedromussi1/paper-tldr/releases/tag/v1.0.0
[0.2.0]: https://github.com/pedromussi1/paper-tldr/releases/tag/v0.2.0
[0.1.0]: https://github.com/pedromussi1/paper-tldr/releases/tag/v0.1.0
