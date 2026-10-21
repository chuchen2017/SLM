"""LoRA on the language model. The geo encoder and projector stay fully trainable."""

from __future__ import annotations

import logging

import torch

from slm.config import LoraConfigValues

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover - surfaced when LoRA is actually requested.
    LoraConfig = None
    TaskType = None
    get_peft_model = None

logger = logging.getLogger(__name__)


def discover_lora_targets(model: torch.nn.Module) -> list[str]:
    """Full module path of every linear layer except the output head.

    This is the notebook rule: ``model.layers.0.self_attn.q_proj`` rather than
    the shared suffix ``q_proj``. PEFT matches each path with ``endswith``, so
    each layer is adapted on its own.
    """
    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            names.append(name)
    return names


def maybe_apply_lora(model: torch.nn.Module, config: LoraConfigValues) -> torch.nn.Module:
    if not config.enabled:
        return model
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise RuntimeError("peft is not installed but LoRA is enabled.")
    target_modules = config.target_modules or discover_lora_targets(model)
    preview = ", ".join(target_modules[:4])
    logger.info("LoRA targets %d linear modules (%s)", len(target_modules), preview)
    return get_peft_model(
        model,
        LoraConfig(
            r=config.r,
            lora_alpha=config.alpha,
            target_modules=target_modules,
            lora_dropout=config.dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )


def set_trainable_parameters(model, lora_config: LoraConfigValues) -> None:
    if not lora_config.enabled:
        for parameter in model.llm.parameters():
            parameter.requires_grad = True
    elif lora_config.freeze_base_model:
        for name, parameter in model.llm.named_parameters():
            parameter.requires_grad = "lora_" in name
    else:
        for parameter in model.llm.parameters():
            parameter.requires_grad = True
    for parameter in model.geo_encoder.parameters():
        parameter.requires_grad = True
    for parameter in model.geo_projector.parameters():
        parameter.requires_grad = True


def freeze_llm_parameters(model) -> None:
    """Stage-1 alignment: train the geo modules against a frozen language model."""
    base = getattr(model, "_orig_mod", model)
    for parameter in base.llm.parameters():
        parameter.requires_grad = False


def build_optimizer_param_groups(
    model,
    *,
    learning_rate: float,
    geo_learning_rate: float,
    weight_decay: float,
) -> list[dict]:
    base = getattr(model, "_orig_mod", model)
    language_params: list[torch.nn.Parameter] = []
    geo_params: list[torch.nn.Parameter] = []
    for name, parameter in base.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("llm."):
            language_params.append(parameter)
        else:
            geo_params.append(parameter)
    groups: list[dict] = []
    if language_params:
        groups.append({"params": language_params, "lr": learning_rate, "weight_decay": weight_decay})
    if geo_params:
        groups.append({"params": geo_params, "lr": geo_learning_rate, "weight_decay": weight_decay})
    if not groups:
        raise ValueError("No trainable parameters. Check lora.enabled and freeze_base_model.")
    return groups


def trainable_parameter_summary(model: torch.nn.Module) -> tuple[int, int, float]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    percent = 100.0 * trainable / total if total else 0.0
    return trainable, total, percent
