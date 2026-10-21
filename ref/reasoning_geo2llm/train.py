from __future__ import annotations

import os

# Set before NumPy. A multi-GPU relaunch inherits this process environment, and
# MKL_THREADING_LAYER=INTEL crashes the Accelerate workers once libgomp is loaded.
os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse
import math
import random
from functools import partial
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import get_scheduler

from reasoning_geo2llm.build import build_geo2llm, load_checkpoint, save_checkpoint
from reasoning_geo2llm.config import TrainConfig, load_config, to_dict
from reasoning_geo2llm.data import (
    LocationDescriptionDataset,
    collate_geo_chat_template,
    grid_bounds_from_records,
    load_all_records,
    load_train_test_records,
    resolve_reasoning_config,
    resolve_system_prompt,
)
from reasoning_geo2llm.evaluate import evaluate_generation
from reasoning_geo2llm.lora import (
    build_optimizer_param_groups,
    freeze_llm_parameters,
    resolve_tuning_embedding_tokens,
    set_trainable_parameters,
    trainable_parameter_breakdown,
    trainable_parameter_summary,
)
from reasoning_geo2llm.runtime import (
    apply_runtime_config,
    parse_cuda_visible_devices,
    relaunch_distributed_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a LoRA Geo2LLM model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override train.output_dir.")
    parser.add_argument("--init-from-checkpoint", type=str, default=None, help="Override train.init_from_checkpoint.")
    parser.add_argument("--cuda-visible-devices", type=str, default=None, help="Override runtime.cuda_visible_devices.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_geo_learning_rate(train_cfg: TrainConfig) -> float:
    """LR for the randomly-initialized geo modules: absolute override, else a multiple of the base."""
    if train_cfg.geo_learning_rate is not None:
        return train_cfg.geo_learning_rate
    return train_cfg.learning_rate * train_cfg.geo_learning_rate_multiplier


def optimizer_steps_per_epoch(num_batches: int, gradient_accumulation_steps: int) -> int:
    return max(1, math.ceil(num_batches / max(1, gradient_accumulation_steps)))


def resolve_warmup_steps(train_cfg: TrainConfig, num_training_steps: int) -> int:
    """Warmup measured in optimizer steps: absolute override, else a fraction of the whole run."""
    if train_cfg.warmup_steps is not None:
        return train_cfg.warmup_steps
    return int(num_training_steps * train_cfg.warmup_ratio)


def resolve_encoder_only_epochs(train_cfg: TrainConfig, spatial_division_method: str | None) -> int:
    """Epochs of the stage-1 encoder-only alignment phase (LLM + LoRA frozen).

    Applies to every spatial method with a learned geo module (Fourier/GeoCLIP default, HYBRID,
    grid, SCALE, sphere). DIRECT/geohash put the location in the prompt text only, so there is no
    encoder to pre-align and the stage is skipped.
    """
    if train_cfg.encoder_only_epochs <= 0:
        return 0
    if spatial_division_method in ("DIRECT", "geohash"):
        return 0
    return train_cfg.encoder_only_epochs


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    if args.init_from_checkpoint:
        cfg.train.init_from_checkpoint = args.init_from_checkpoint
    if args.cuda_visible_devices:
        cfg.runtime.cuda_visible_devices = args.cuda_visible_devices

    apply_runtime_config(cfg.runtime)
    visible_devices = parse_cuda_visible_devices(cfg.runtime.cuda_visible_devices)
    # Expert LoRA on GPT-OSS does not fit as a full replica on one 48GB GPU.
    # Shard that model across the listed GPUs. Dense runs still use data parallel.
    shard_model = cfg.lora.moe_rank is not None and len(visible_devices) > 1
    if not shard_model:
        # One process per listed GPU. Must happen before any CUDA call so the
        # relaunched workers inherit the device mask and bind one rank each.
        relaunch_distributed_training(visible_devices, cfg.train.mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=cfg.train.mixed_precision,
    )
    seed_everything(cfg.train.seed + accelerator.process_index)

    if accelerator.is_main_process:
        output_dir = Path(cfg.train.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        import yaml

        with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(to_dict(cfg), f, sort_keys=False)

    train_df, eval_df = load_train_test_records(cfg.data)

    if cfg.train.init_from_checkpoint:
        model = load_checkpoint(cfg.model, cfg.train.init_from_checkpoint)
    else:
        model = build_geo2llm(
            cfg.model,
            cfg.lora,
            device_map="balanced" if shard_model else None,
        )
    set_trainable_parameters(model, cfg.lora)
    if shard_model:
        # Keep the non-LLM modules on the same device as the inputs. The LLM itself
        # stays split by the balanced device map.
        first_device = torch.device("cuda:0")
        for module in (
            model.geo_clip,
            model.geo_projector,
            model.embedding_layer,
            model.scale_geo_clip_log,
            model.scale_geo_clip_linear,
            model.scale_geo_projector_log,
            model.scale_geo_projector_linear,
        ):
            if module is not None:
                module.to(first_device)
        device_counts: dict[str, int] = {}
        for param in model.llm.parameters():
            key = str(param.device)
            device_counts[key] = device_counts.get(key, 0) + param.numel()
        accelerator.print(
            "MoE model sharded across "
            + ", ".join(f"{device}={count / 1e9:.2f}B params" for device, count in sorted(device_counts.items()))
        )

    if cfg.reasoning.enabled:
        # Auto-detect <think>-style tags from this model's own chat template (unless pinned
        # explicitly in the YAML config), so the same 'reasoning' config works across base models.
        cfg.reasoning = resolve_reasoning_config(cfg.reasoning, model.tokenizer)
        if accelerator.is_main_process:
            accelerator.print(
                "reasoning thinking tags resolved from tokenizer chat template: "
                f"open={cfg.reasoning.thinking_open_tag!r} close={cfg.reasoning.thinking_close_tag!r}"
            )
            import yaml

            with (Path(cfg.train.output_dir) / "resolved_config.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(to_dict(cfg), f, sort_keys=False)

    if cfg.train.compile_model:
        model = torch.compile(model)

    train_dataset = LocationDescriptionDataset(
        train_df,
        latitude_col=cfg.data.latitude_col,
        longitude_col=cfg.data.longitude_col,
        description_col=cfg.data.description_col,
        id_col=cfg.data.id_col,
        num_samples=cfg.data.max_train_samples,
        random_sample=True,
        spatial_division_method=cfg.model.spatial_division_method,
        grid_num_axis=cfg.model.grid_num_axis,
        geohash_length=cfg.model.geohash_length,
        reasoning=cfg.reasoning,
    )
    eval_dataset = LocationDescriptionDataset(
        eval_df,
        latitude_col=cfg.data.latitude_col,
        longitude_col=cfg.data.longitude_col,
        description_col=cfg.data.description_col,
        id_col=cfg.data.id_col,
        num_samples=cfg.data.max_eval_samples,
        random_sample=False,
        spatial_division_method=cfg.model.spatial_division_method,
        grid_num_axis=cfg.model.grid_num_axis,
        geohash_length=cfg.model.geohash_length,
        reasoning=cfg.reasoning,
    )
    _ = eval_dataset

    collate_fn = partial(
        collate_geo_chat_template,
        tokenizer=model.tokenizer,
        system_prompt=resolve_system_prompt(cfg.reasoning),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False,
    )

    geo_learning_rate = resolve_geo_learning_rate(cfg.train)
    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(
            model,
            learning_rate=cfg.train.learning_rate,
            geo_learning_rate=geo_learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    )

    # LLaVA-style stage 1: the optimizer above was built while the full joint-stage parameter set
    # was unfrozen, so it already holds the LLM/LoRA group. Freezing here merely stops gradients;
    # AdamW skips parameters whose .grad is None, and the stage-2 transition below re-opens them
    # with `set_trainable_parameters` (they then start stepping with fresh optimizer state).
    encoder_only_epochs = resolve_encoder_only_epochs(cfg.train, cfg.model.spatial_division_method)
    if cfg.train.encoder_only_epochs > 0 and encoder_only_epochs == 0:
        accelerator.print(
            "encoder_only_epochs ignored: spatial_division_method="
            f"{cfg.model.spatial_division_method!r} is text-only and has "
            "no geo encoder to pre-align."
        )
    if encoder_only_epochs:
        freeze_llm_parameters(model)
    total_epochs = encoder_only_epochs + cfg.train.epochs

    # Schedule length is counted in optimizer steps, not micro-batches, so gradient accumulation
    # does not silently stretch the warmup/decay over `gradient_accumulation_steps` times too long.
    num_training_steps = (
        optimizer_steps_per_epoch(len(train_loader), cfg.train.gradient_accumulation_steps)
        * total_epochs
    )
    num_warmup_steps = resolve_warmup_steps(cfg.train, num_training_steps)
    lr_scheduler = get_scheduler(
        cfg.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # `accelerator.prepare` wraps the scheduler so it only advances on steps where the optimizer
    # actually stepped, which is why `lr_scheduler.step()` can be called every micro-batch below.
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        trainable, total, pct = trainable_parameter_summary(unwrapped)
        accelerator.print(
            f"train_rows={len(train_df)} eval_rows={len(eval_df)} "
            f"trainable={trainable:,} total={total:,} trainable_pct={pct:.4f}"
        )
        breakdown = trainable_parameter_breakdown(unwrapped)
        breakdown_msg = " ".join(
            f"{name}={trainable_count:,}/{group_total:,}"
            for name, (trainable_count, group_total) in breakdown.items()
        )
        accelerator.print(f"trainable_breakdown {breakdown_msg}")
        accelerator.print(
            f"lora_enabled={cfg.lora.enabled} freeze_base_model={cfg.lora.freeze_base_model} "
            f"r={cfg.lora.r} alpha={cfg.lora.alpha} moe_rank={cfg.lora.moe_rank} "
            f"tuning_embedding={cfg.lora.tuning_embedding}"
            + (
                f" tokens={resolve_tuning_embedding_tokens(cfg.lora.tuning_embedding_tokens)}"
                if cfg.lora.tuning_embedding
                else ""
            )
        )
        accelerator.print(
            f"llm_lr={cfg.train.learning_rate:.3g} geo_lr={geo_learning_rate:.3g} "
            f"lr_scheduler={cfg.train.lr_scheduler_type} warmup_steps={num_warmup_steps} "
            f"total_optimizer_steps={num_training_steps}"
        )
        if encoder_only_epochs:
            accelerator.print(
                f"stage 1 (multimodal alignment): epochs 1-{encoder_only_epochs} train only the "
                f"geo encoder/projector with the LLM and LoRA frozen; stage 2 (joint): "
                f"epochs {encoder_only_epochs + 1}-{total_epochs} restore the configured "
                "trainable set."
            )
        if cfg.runtime.cuda_visible_devices:
            accelerator.print(
                f"CUDA_VISIBLE_DEVICES={cfg.runtime.cuda_visible_devices} "
                f"processes={accelerator.num_processes}"
            )
        if cfg.train.init_from_checkpoint:
            accelerator.print(f"Initialized model from checkpoint: {cfg.train.init_from_checkpoint}")
        accelerator.print(f"Saving checkpoints under: {cfg.train.output_dir}")
    print(
        f"process={accelerator.process_index}/{accelerator.num_processes} device={accelerator.device}",
        flush=True,
    )

    eval_grid_bounds = None
    if cfg.model.spatial_division_method == "grid":
        eval_grid_bounds = grid_bounds_from_records(
            load_all_records(cfg.data),
            cfg.data.latitude_col,
            cfg.data.longitude_col,
        )

    # Load once, up front: `evaluate_generation` also accepts a lazy `(model_name, device)`
    # tuple and constructs `SentenceTransformer` itself, but doing that here means every
    # periodic eval (every `eval_every_epochs`) reloads the model from scratch - wasteful, and
    # if the HF Hub metadata check is ever slow (rate limiting, network hiccups), it stalls that
    # epoch with no GPU/CPU activity to show for it.
    sentence_model = None
    # if accelerator.is_main_process and cfg.train.eval_every_epochs > 0:
    #     sentence_model = SentenceTransformer(cfg.train.eval_sentence_model, device=str(accelerator.device))
    sentence_model = (cfg.train.eval_sentence_model, str(accelerator.device))

    best_eval_f1 = float("-inf")
    global_step = 0
    for epoch in range(total_epochs):
        if encoder_only_epochs and epoch == encoder_only_epochs:
            # Stage-1 -> stage-2 transition: re-open the trainable set configured under `lora:`.
            # The optimizer already owns these parameters (built pre-freeze), so no rebuild needed.
            set_trainable_parameters(accelerator.unwrap_model(model), cfg.lora)
            if accelerator.is_main_process:
                trainable, total, pct = trainable_parameter_summary(accelerator.unwrap_model(model))
                accelerator.print(
                    f"encoder-only stage done after {encoder_only_epochs} epoch(s); unfroze the "
                    f"LLM-side parameters for joint tuning: trainable={trainable:,} ({pct:.4f}%)"
                )
        model.train()
        running_loss = 0.0
        torch.cuda.empty_cache()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(
                    query_ids=batch["prompt_input_ids"],
                    geo_tensor=batch["geo_feature"],
                    geo_attention_mask=batch["geo_attention_mask"],
                    labels=batch["labels"],
                    query_mask=batch["prompt_attention_mask"],
                    label_mask=batch["labels_attention_mask"],
                )
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.detach().float().item()
            global_step += 1

        mean_loss = running_loss / max(1, len(train_loader))
        current_lrs = "/".join(f"{group['lr']:.3g}" for group in optimizer.param_groups)
        stage = "encoder_only" if epoch < encoder_only_epochs else "joint"
        accelerator.print(
            f"epoch={epoch + 1} stage={stage} mean_loss={mean_loss:.4f} "
            f"global_step={global_step} lr={current_lrs}"
        )

        should_eval = cfg.train.eval_every_epochs > 0 and (epoch + 1) % cfg.train.eval_every_epochs == 0
        if should_eval:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                result = evaluate_generation(
                    unwrapped,
                    unwrapped.tokenizer,
                    eval_df,
                    sentence_model,
                    batch_size=cfg.generation.batch_size,
                    num_samples=cfg.train.eval_num_samples,
                    temperature=cfg.generation.temperature,
                    top_k=cfg.generation.top_k,
                    top_p=cfg.generation.top_p,
                    spatial_division_method=cfg.model.spatial_division_method,
                    grid_num_axis=cfg.model.grid_num_axis,
                    grid_bounds=eval_grid_bounds,
                    geohash_length=cfg.model.geohash_length,
                    reasoning=cfg.reasoning,
                )
                metrics = " ".join(f"{key}={value:.4f}" for key, value in result.items())
                eval_f1 = result["semantic_f1"]
                accelerator.print(f"epoch={epoch + 1} eval {metrics}")
                if 1: #eval_f1 > best_eval_f1
                    best_eval_f1 = eval_f1
                    #save_checkpoint(unwrapped, Path(cfg.train.output_dir) / f"epoch_{epoch + 1}")
                    save_checkpoint(unwrapped, Path(cfg.train.output_dir) / "best")
                    accelerator.print(
                        f"Saved new best checkpoint at epoch={epoch + 1} semantic_f1={eval_f1:.4f}"
                    )
                else:
                    accelerator.print(
                        f"Skipped checkpoint at epoch={epoch + 1}; "
                        f"semantic_f1={eval_f1:.4f} best={best_eval_f1:.4f}"
                    )
            accelerator.wait_for_everyone()
        elif cfg.train.eval_every_epochs <= 0 or not cfg.train.save_on_eval_improvement:
            should_save = (epoch + 1) % cfg.train.save_every_epochs == 0 or epoch + 1 == total_epochs
            if should_save:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    save_checkpoint(unwrapped, Path(cfg.train.output_dir) / f"epoch_{epoch + 1}")


        if cfg.train.save_every_epochs and epoch + 1 == total_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
            save_checkpoint(unwrapped, Path(cfg.train.output_dir) / f"epoch_{epoch + 1}")
            accelerator.print(
                f"Saved checkpoint at epoch={epoch + 1}"
            )


if __name__ == "__main__":
    main()
