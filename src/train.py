"""QLoRA fine-tuning of Llama 3.2 3B Instruct on SciTLDR.

Uses HuggingFace Trainer + PEFT directly rather than TRL's SFTTrainer so the
chat-template rendering and completion-only label masking from `src.data`
(`map_for_training`, `CollateForCausalLM`) are the single source of truth shared
between training and inference. Less magic, fewer drift opportunities.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.data import CollateForCausalLM, load_split, map_for_training

os.environ.setdefault("WANDB_PROJECT", "paper-tldr")


@dataclass
class TrainConfig:
    # data + model
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    output_dir: Path = field(default_factory=lambda: Path("checkpoints/qlora-3b-r16"))
    run_name: str = "qlora-3b-r16"
    max_length: int = 512

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # optim
    num_epochs: float = 3.0
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # logging / eval
    logging_steps: int = 10
    eval_steps: int = 25
    save_steps: int = 25
    save_total_limit: int = 2
    val_subset_size: int = 64

    # misc
    seed: int = 1337


def _build_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _build_lora_config(cfg: TrainConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )


def train(cfg: TrainConfig) -> Path:
    cfg.output_dir = Path(cfg.output_dir)
    cfg.processed_dir = Path(cfg.processed_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=_build_quant_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, _build_lora_config(cfg))
    model.print_trainable_parameters()

    train_ds = load_split(cfg.processed_dir, "train")
    val_ds = load_split(cfg.processed_dir, "val")
    train_tok = map_for_training(train_ds, tokenizer, max_length=cfg.max_length)
    val_tok = map_for_training(val_ds, tokenizer, max_length=cfg.max_length)
    val_subset = val_tok.select(range(min(cfg.val_subset_size, len(val_tok))))

    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        run_name=cfg.run_name,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb"],
        seed=cfg.seed,
        label_names=["labels"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_subset,
        data_collator=CollateForCausalLM(pad_token_id=tokenizer.pad_token_id),
    )

    trainer.train()

    final_path = cfg.output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    return final_path
