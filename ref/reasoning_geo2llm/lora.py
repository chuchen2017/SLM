from __future__ import annotations

import logging

import torch

from reasoning_geo2llm.config import LoraConfigValues

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover - import error is surfaced when LoRA is requested.
    LoraConfig = None
    TaskType = None
    get_peft_model = None

logger = logging.getLogger(__name__)

COMMON_CAUSAL_LM_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

# Fused expert weights stored as 3D nn.Parameter, not nn.Linear. PEFT adapts these through
# `target_parameters`. Suffix match skips the sibling `*_bias` parameters and the router.
MOE_EXPERT_PARAMETER_SUFFIXES = (
    "experts.gate_up_proj",
    "experts.down_proj",
)

# Digit characters that spell latitude/longitude in DIRECT and HYBRID prompts, e.g. (34.062403, -118.372775).
# Geohash prompts use the Base32 alphabet instead; override `tuning_embedding_tokens` if needed.
DEFAULT_TUNING_EMBEDDING_TOKENS = [str(digit) for digit in range(10)]


def discover_lora_targets(model: torch.nn.Module) -> list[str]:
    leaf_names: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf == "lm_head":
            continue
        leaf_names.add(leaf)

    common = sorted(leaf_names.intersection(COMMON_CAUSAL_LM_TARGETS))
    return common if common else sorted(leaf_names)


def input_embedding_leaf_names(model: torch.nn.Module) -> list[str]:
    """Leaf names of `get_input_embeddings()`, e.g. `embed_tokens` or `wte`."""
    if not hasattr(model, "get_input_embeddings"):
        return []
    embed = model.get_input_embeddings()
    if embed is None:
        return []
    names: set[str] = set()
    for name, module in model.named_modules():
        if module is embed:
            names.add(name.split(".")[-1])
    return sorted(names)


def _word_embedding_parameters(llm: torch.nn.Module) -> list[torch.nn.Parameter]:
    if not hasattr(llm, "get_input_embeddings"):
        return []
    embed = llm.get_input_embeddings()
    if embed is None:
        return []
    return list(embed.parameters())


def resolve_tuning_embedding_tokens(tokens: list[str] | None) -> list[str]:
    if not tokens:
        return list(DEFAULT_TUNING_EMBEDDING_TOKENS)
    return [str(token) for token in tokens]


def resolve_tuning_embedding_token_ids(tokenizer, tokens: list[str] | None = None) -> list[int]:
    """Map token strings (default: digits 0-9) to tokenizer ids.

    Each piece must be a single vocab entry. `convert_tokens_to_ids` is preferred; `encode`
    is the fallback when the tokenizer only exposes the digit through encoding.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required to resolve tuning_embedding token ids.")
    unk = getattr(tokenizer, "unk_token_id", None)
    ids: set[int] = set()
    missing: list[str] = []
    for piece in resolve_tuning_embedding_tokens(tokens):
        chosen: int | None = None
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            tid = tokenizer.convert_tokens_to_ids(piece)
            if isinstance(tid, int) and tid != unk and tid >= 0:
                back = tokenizer.convert_ids_to_tokens(tid) if hasattr(tokenizer, "convert_ids_to_tokens") else piece
                if back == piece:
                    chosen = tid
        if chosen is None and hasattr(tokenizer, "encode"):
            encoded = tokenizer.encode(piece, add_special_tokens=False)
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if len(encoded) == 1 and encoded[0] != unk:
                chosen = int(encoded[0])
        if chosen is None:
            missing.append(piece)
        else:
            ids.add(int(chosen))
    if missing:
        raise ValueError(f"Could not map tuning_embedding_tokens to tokenizer ids: {missing}")
    return sorted(ids)


def discover_moe_expert_parameters(model: torch.nn.Module) -> list[str]:
    """Suffixes of 3D expert parameters present on `model`."""
    found: list[str] = []
    for suffix in MOE_EXPERT_PARAMETER_SUFFIXES:
        for name, param in model.named_parameters():
            if name.endswith(suffix) and getattr(param, "ndim", 0) == 3:
                found.append(suffix)
                break
    return found


def moe_expert_lora_kwargs(model: torch.nn.Module, config: LoraConfigValues) -> dict:
    """LoRA kwargs for the standard MoE rule, or {} when that rule does not apply.

    Attention projections keep `config.r` / `config.alpha`. Every expert feed-forward
    parameter is adapted at `config.moe_rank`, with alpha scaled by the same alpha/r
    ratio. The router is not a target, so it stays frozen.
    """
    if config.moe_rank is None:
        return {}
    if config.moe_rank <= 0:
        raise ValueError(f"lora.moe_rank must be a positive integer, got {config.moe_rank}.")
    expert_parameters = discover_moe_expert_parameters(model)
    if not expert_parameters:
        logger.warning(
            "lora.moe_rank=%s but this backbone has no fused MoE expert parameters; "
            "keeping the standard LoRA targets at r=%s.",
            config.moe_rank,
            config.r,
        )
        return {}
    if config.dropout:
        raise ValueError(
            "lora.moe_rank requires dropout: 0. PEFT cannot apply dropout to fused MoE "
            "expert parameters (gate_up_proj / down_proj)."
        )
    if config.r <= 0:
        raise ValueError("lora.r must be positive when lora.moe_rank is set.")
    moe_alpha = int(round(config.moe_rank * config.alpha / config.r))
    rank_pattern = {suffix.split(".")[-1]: config.moe_rank for suffix in expert_parameters}
    alpha_pattern = {suffix.split(".")[-1]: moe_alpha for suffix in expert_parameters}
    logger.info(
        "MoE LoRA: attention r=%s alpha=%s; experts %s at moe_rank=%s alpha=%s; router frozen",
        config.r,
        config.alpha,
        expert_parameters,
        config.moe_rank,
        moe_alpha,
    )
    return {
        "target_parameters": list(expert_parameters),
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
    }


def maybe_apply_lora(model: torch.nn.Module, config: LoraConfigValues, tokenizer=None) -> torch.nn.Module:
    if not config.enabled:
        return model
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise RuntimeError("peft is not installed but LoRA is enabled.")

    target_modules = config.target_modules or discover_lora_targets(model)
    lora_kwargs: dict = {
        "r": config.r,
        "lora_alpha": config.alpha,
        "target_modules": target_modules,
        "lora_dropout": config.dropout,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    }
    lora_kwargs.update(moe_expert_lora_kwargs(model, config))
    if config.tuning_embedding:
        token_ids = resolve_tuning_embedding_token_ids(tokenizer, config.tuning_embedding_tokens)
        lora_kwargs["trainable_token_indices"] = token_ids
        logger.info(
            "tuning_embedding tokens=%s ids=%s",
            resolve_tuning_embedding_tokens(config.tuning_embedding_tokens),
            token_ids,
        )
    return get_peft_model(model, LoraConfig(**lora_kwargs))


def set_trainable_parameters(model, lora_config: LoraConfigValues) -> None:
    if not lora_config.enabled:
        for param in model.llm.parameters():
            param.requires_grad = True
    elif lora_config.freeze_base_model:
        for name, param in model.llm.named_parameters():
            param.requires_grad = "lora_" in name
    else:
        for param in model.llm.parameters():
            param.requires_grad = True

    # Selective digit-token adapters are not named `lora_*`, so reopen them after the freeze.
    # The base embedding table stays frozen; only the PEFT token deltas train.
    if lora_config.tuning_embedding:
        token_params = [
            param for name, param in model.llm.named_parameters() if "trainable_tokens" in name
        ]
        if not token_params:
            raise RuntimeError(
                "tuning_embedding=True but no PEFT trainable-token adapters were found. "
                "Enable LoRA so trainable_token_indices can be applied."
            )
        for param in _word_embedding_parameters(model.llm):
            param.requires_grad = False
        for param in token_params:
            param.requires_grad = True

    uses_learned_geo_modules = model.spatial_division_method not in ("grid", "DIRECT", "geohash", "SCALE")
    for param in model.geo_clip.parameters():
        param.requires_grad = uses_learned_geo_modules
    for param in model.geo_projector.parameters():
        param.requires_grad = uses_learned_geo_modules
    if model.embedding_layer is not None:
        for param in model.embedding_layer.parameters():
            param.requires_grad = True
    for module in (
        model.scale_geo_clip_log,
        model.scale_geo_clip_linear,
        model.scale_geo_projector_log,
        model.scale_geo_projector_linear,
    ):
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


def freeze_llm_parameters(model) -> None:
    """Stage-1 multimodal alignment freeze: stop training the entire LLM side.

    Covers base weights, LoRA adapters, and PEFT trainable-token deltas alike, so during the
    encoder-only stage only the modules outside `model.llm` that `set_trainable_parameters`
    left open (geo_clip / geo_projector / grid cell embeddings / SCALE encoders) get updates.
    Calling `set_trainable_parameters` again restores the configured joint-stage state.
    """
    base = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    for param in base.llm.parameters():
        param.requires_grad = False


def build_optimizer_param_groups(
    model,
    *,
    learning_rate: float,
    geo_learning_rate: float,
    weight_decay: float,
) -> list[dict]:
    """Split trainable parameters into an LLM/LoRA group and a geo-module group.

    The geo modules (`geo_clip`, `geo_projector`, grid cell embeddings, the SCALE encoders) train
    from random initialization, while the LoRA adapters only nudge an already-pretrained LLM, so a
    single shared learning rate is necessarily wrong for one of them. `model.embed_tokens` aliases a
    module owned by `model.llm`, and `named_parameters()` de-duplicates shared tensors, so those
    weights are only ever visited once, under the `llm.` prefix.
    """
    base = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    llm_params: list[torch.nn.Parameter] = []
    geo_params: list[torch.nn.Parameter] = []
    for name, param in base.named_parameters():
        if not param.requires_grad:
            continue
        target = llm_params if name.startswith("llm.") else geo_params
        target.append(param)

    groups: list[dict] = []
    if llm_params:
        groups.append({"params": llm_params, "lr": learning_rate, "weight_decay": weight_decay})
    if geo_params:
        groups.append({"params": geo_params, "lr": geo_learning_rate, "weight_decay": weight_decay})
    if not groups:
        raise ValueError("No trainable parameters found; check lora.enabled / freeze_base_model.")
    return groups


def trainable_parameter_summary(model: torch.nn.Module) -> tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, total, pct


def trainable_parameter_breakdown(model) -> dict[str, tuple[int, int]]:
    """Return trainable/total counts per major module group."""
    groups: dict[str, list[torch.nn.Parameter]] = {
        "llm": list(model.llm.parameters()),
        "geo_clip": list(model.geo_clip.parameters()),
        "geo_projector": list(model.geo_projector.parameters()),
    }
    if model.embedding_layer is not None:
        groups["embedding_layer"] = list(model.embedding_layer.parameters())
    if model.scale_geo_clip_log is not None:
        groups["scale_geo_clip_log"] = list(model.scale_geo_clip_log.parameters())
    if model.scale_geo_clip_linear is not None:
        groups["scale_geo_clip_linear"] = list(model.scale_geo_clip_linear.parameters())
    if model.scale_geo_projector_log is not None:
        groups["scale_geo_projector_log"] = list(model.scale_geo_projector_log.parameters())
    if model.scale_geo_projector_linear is not None:
        groups["scale_geo_projector_linear"] = list(model.scale_geo_projector_linear.parameters())
    return {
        name: (
            sum(p.numel() for p in params if p.requires_grad),
            sum(p.numel() for p in params),
        )
        for name, params in groups.items()
    }
