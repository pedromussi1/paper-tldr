"""Three baselines for abstract → TL;DR summarization.

1. ``extractive_first_sentence`` — return the first sentence of the abstract.
   Lower bound; often a surprisingly tough baseline on SciTLDR.
2. ``zero_shot_llm`` — prompt a chat-tuned LLM with the same SYSTEM_PROMPT we
   use for fine-tuning. Apples-to-apples comparison vs. the QLoRA-tuned model.

For ``zero_shot_llm`` we run in fp16 (no quantization) so the baseline isn't
artificially weakened. The fine-tuned model will be evaluated in both fp16 and
4-bit so we can also report the quantization-induced drop.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import SYSTEM_PROMPT

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

_HERES_PREAMBLE = re.compile(
    r"^(?:sure[!,.]?\s+)?"
    r"(?:here(?:'|’)?s|here is)\s+"
    r"(?:a|the|my)?\s*"
    r"(?:rewritten|plain[\s\-]?language|paraphrased|short|brief|simple)?\s*"
    r"tl[;\s]?dr\s*[:\-—]?\s*\n*",
    re.IGNORECASE,
)
# A short leading line that ends in ":" followed by a newline — model self-labels its own output
# (e.g. "TL;DR:", "Plain-language summary:", "Abstract of X:"). Capped at 100 chars to avoid
# eating real content.
_LABEL_LINE = re.compile(r"^[^\n]{1,100}:\s*\n+")


def strip_llm_preamble(text: str) -> str:
    """Strip common chat-LLM boilerplate so ROUGE/BERTScore see only the summary.

    Applied uniformly to every LLM output (zero-shot baselines and the QLoRA model)
    so the comparison remains symmetric.
    """
    text = text.strip()
    text = _HERES_PREAMBLE.sub("", text).strip()
    text = _LABEL_LINE.sub("", text).strip()
    # Strip surrounding straight or curly quotes.
    if len(text) >= 2 and text[0] in '"“‘' and text[-1] in '"”’':
        text = text[1:-1].strip()
    # Strip surrounding single asterisks (markdown italic), but leave **bold** alone.
    if text.startswith("*") and text.endswith("*") and not text.startswith("**"):
        text = text[1:-1].strip()
    return text


def extractive_first_sentence(abstract: str) -> str:
    """Return the first sentence of the abstract, or the whole thing if no boundary."""
    abstract = abstract.strip()
    parts = _SENT_BOUNDARY.split(abstract, maxsplit=1)
    return parts[0].strip()


def _build_prompt_messages(abstract: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Abstract:\n{abstract}\n\nWrite the TL;DR."},
    ]


def _prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for batched generation
    return tokenizer


def _generate_with_chat_template(
    model,
    tokenizer,
    abstracts: list[str],
    batch_size: int,
    max_new_tokens: int,
    device: str = "cuda",
) -> list[str]:
    """Shared generation core used by zero-shot baselines and the QLoRA model."""
    prompts = [
        tokenizer.apply_chat_template(
            _build_prompt_messages(a),
            tokenize=False,
            add_generation_prompt=True,
        )
        for a in abstracts
    ]

    completions: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for ids in out:
            gen_ids = ids[enc["input_ids"].shape[1] :]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            completions.append(strip_llm_preamble(text))
        del enc, out

    return completions


def zero_shot_llm(
    abstracts: Iterable[str],
    model_name: str,
    batch_size: int = 4,
    max_new_tokens: int = 96,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> list[str]:
    """Run zero-shot generation with the model's chat template.

    Returns a list of generated TLDRs, one per input abstract, in order.
    """
    tokenizer = _prepare_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    completions = _generate_with_chat_template(
        model, tokenizer, list(abstracts), batch_size, max_new_tokens, device
    )

    del model
    torch.cuda.empty_cache()
    return completions


def qlora_finetuned_llm(
    abstracts: Iterable[str],
    base_model_name: str,
    adapter_path: str,
    batch_size: int = 4,
    max_new_tokens: int = 96,
    dtype: torch.dtype = torch.bfloat16,  # match training compute dtype
    device: str = "cuda",
) -> list[str]:
    """Generate from a base model with a QLoRA adapter applied on top.

    The base model is loaded in fp16/bf16 (no 4-bit quantization here) so the
    comparison against the fp16 zero-shot baseline is apples-to-apples — only
    the LoRA adapter introduces the change. A separate quantized-inference
    eval can be run later to measure the deployment-time quantization drop.
    """
    from peft import PeftModel  # imported lazily to keep base import surface small

    tokenizer = _prepare_tokenizer(base_model_name)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    completions = _generate_with_chat_template(
        model, tokenizer, list(abstracts), batch_size, max_new_tokens, device
    )

    del model, base
    torch.cuda.empty_cache()
    return completions
