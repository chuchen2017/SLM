"""Fine-tune SLM on geospatial reasoning."""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from functools import partial
from pathlib import Path

# Set before NumPy. A multi-GPU relaunch inherits this process, and the Intel
# MKL layer aborts once libgomp is loaded.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slm.runtime import prefer_env_libstdcxx

prefer_env_libstdcxx()

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler

from slm.build import build_slm, load_checkpoint, save_checkpoint
from slm.chat import resolve_thinking_tags
from slm.config import TrainConfig, load_config, to_dict
from slm.data import build_geo_datasets, collate_slm, load_benchmarks, system_prompt_for, task_decimals
from slm.evaluate import evaluate_benchmarks
from slm.lora import (
    build_optimizer_param_groups,
    freeze_llm_parameters,
    set_trainable_parameters,
    trainable_parameter_summary,
)
from slm.runtime import apply_runtime_config, parse_cuda_visible_devices, relaunch_distributed_training

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SLM.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def peek_geo_input_dim(config) -> int:
    if config.model.geo_input_dim:
        return int(config.model.geo_input_dim)
    path = Path(config.data.osm_dir) / config.data.city / config.data.train_embeddings
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2:
        raise ValueError(f"{path} has shape {array.shape}; expected (N, D).")
    return int(array.shape[1])


def resolve_geo_learning_rate(train_cfg: TrainConfig) -> float:
    if train_cfg.geo_learning_rate is not None:
        return train_cfg.geo_learning_rate
    return train_cfg.learning_rate * train_cfg.geo_learning_rate_multiplier


def optimizer_steps_per_epoch(num_batches: int, gradient_accumulation_steps: int) -> int:
    return max(1, math.ceil(num_batches / max(1, gradient_accumulation_steps)))


def resolve_warmup_steps(train_cfg: TrainConfig, num_training_steps: int) -> int:
    if train_cfg.warmup_steps is not None:
        return train_cfg.warmup_steps
    return int(num_training_steps * train_cfg.warmup_ratio)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.init_from_checkpoint:
        config.train.init_from_checkpoint = args.init_from_checkpoint
    if args.cuda_visible_devices:
        config.runtime.cuda_visible_devices = args.cuda_visible_devices

    apply_runtime_config(config.runtime)
    visible = parse_cuda_visible_devices(config.runtime.cuda_visible_devices)
    relaunch_distributed_training(visible, config.train.mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        mixed_precision=config.train.mixed_precision,
    )
    seed_everything(config.train.seed + accelerator.process_index)

    output_dir = Path(config.train.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(to_dict(config), handle, sort_keys=False)

    geo_input_dim = peek_geo_input_dim(config)
    if config.train.init_from_checkpoint:
        model = load_checkpoint(config.model, config.train.init_from_checkpoint, geo_input_dim=geo_input_dim)
    else:
        model = build_slm(config.model, config.lora, geo_input_dim=geo_input_dim)
    set_trainable_parameters(model, config.lora)

    thinking_tags = ("<think>", "</think>")
    if config.data.include_thinking:
        if config.data.thinking_open_tag and config.data.thinking_close_tag:
            thinking_tags = (config.data.thinking_open_tag, config.data.thinking_close_tag)
        else:
            thinking_tags = resolve_thinking_tags(model.tokenizer)
        accelerator.print(f"thinking tags: open={thinking_tags[0]!r} close={thinking_tags[1]!r}")

    train_dataset, _ = build_geo_datasets(config.data, thinking_tags=thinking_tags, build_test=False)
    example = train_dataset[0]
    if int(example["geo_feature"].shape[-1]) not in {0, geo_input_dim}:
        raise ValueError(
            f"Training features have width {example['geo_feature'].shape[-1]}, expected {geo_input_dim}."
        )
    accelerator.print(f"example task={example['task']} prompt={example['prompt']}")
    accelerator.print(f"example label={example['labels']}")

    collate = partial(collate_slm, tokenizer=model.tokenizer, system_prompt=system_prompt_for(config.data))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        drop_last=False,
    )

    geo_learning_rate = resolve_geo_learning_rate(config.train)
    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(
            model,
            learning_rate=config.train.learning_rate,
            geo_learning_rate=geo_learning_rate,
            weight_decay=config.train.weight_decay,
        )
    )
    if accelerator.is_main_process:
        trainable, total, percent = trainable_parameter_summary(model)
        accelerator.print(
            f"trainable={trainable:,} total={total:,} trainable_pct={percent:.4f} "
            f"llm_lr={config.train.learning_rate:.3g} geo_lr={geo_learning_rate:.3g}"
        )
    encoder_only_epochs = config.train.encoder_only_epochs
    if encoder_only_epochs:
        freeze_llm_parameters(model)
        accelerator.print(
            f"stage 1 freezes the language model for {encoder_only_epochs} epoch(s); "
            f"stage 2 trains it for {config.train.epochs} epoch(s)."
        )
    total_epochs = encoder_only_epochs + config.train.epochs
    num_training_steps = optimizer_steps_per_epoch(len(train_loader), config.train.gradient_accumulation_steps) * total_epochs
    num_warmup_steps = resolve_warmup_steps(config.train, num_training_steps)
    scheduler = get_scheduler(
        config.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    if accelerator.is_main_process:
        accelerator.print(f"steps={num_training_steps} warmup={num_warmup_steps}")

    benchmarks = []
    if accelerator.is_main_process and config.train.eval_every_epochs > 0:
        benchmarks = load_benchmarks(config.data, limit=config.train.eval_num_samples)

    best_score = float("-inf")
    for epoch in range(total_epochs):
        if encoder_only_epochs and epoch == encoder_only_epochs:
            set_trainable_parameters(accelerator.unwrap_model(model), config.lora)
            accelerator.print(f"encoder-only stage finished; joint training starts at epoch {epoch + 1}.")
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(
                    query_ids=batch["prompt_input_ids"],
                    geo_tensor=batch["geo_feature"],
                    geo_attention_mask=batch["geo_attention_mask"],
                    labels=batch["labels"],
                    query_mask=batch["prompt_attention_mask"],
                    label_mask=batch["labels_attention_mask"],
                )
                accelerator.backward(outputs.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(outputs.loss.detach().float())
        mean_loss = running_loss / max(1, len(train_loader))
        stage = "encoder_only" if epoch < encoder_only_epochs else "joint"
        accelerator.print(f"epoch={epoch + 1} stage={stage} mean_loss={mean_loss:.4f}")

        should_eval = bool(benchmarks) and (epoch + 1) % config.train.eval_every_epochs == 0
        if should_eval:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                result = evaluate_benchmarks(
                    unwrapped,
                    unwrapped.tokenizer,
                    benchmarks,
                    generation=config.generation,
                    system_prompt=system_prompt_for(config.data),
                    decimals_for_task=lambda task: task_decimals(config.data, task),
                    device=accelerator.device,
                )
                accelerator.print(f"epoch={epoch + 1} benchmark_score={result['score']:.4f}")
                if result["score"] > best_score:
                    best_score = result["score"]
                    save_checkpoint(unwrapped, output_dir / "best")
                    accelerator.print(f"saved best checkpoint at epoch {epoch + 1}")
            accelerator.wait_for_everyone()

        should_save = (epoch + 1) % config.train.save_every_epochs == 0 or epoch + 1 == total_epochs
        if should_save:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(accelerator.unwrap_model(model), output_dir / f"epoch_{epoch + 1}")
                accelerator.print(f"saved epoch_{epoch + 1}")


if __name__ == "__main__":
    main()
