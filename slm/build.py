"""Build an SLM from a base causal LM and save the geo modules beside the LoRA adapter."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slm.config import LoraConfigValues, ModelConfig
from slm.lora import maybe_apply_lora
from slm.modeling import Geo2VecEncoder, SLM

logger = logging.getLogger(__name__)

GEO_MODULES_FILENAME = "geo_modules.pt"


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def build_tokenizer(config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_tokens([config.geo_token], special_tokens=False)
    geo_token_id = tokenizer.convert_tokens_to_ids(config.geo_token)
    if geo_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Failed to add geo token {config.geo_token!r}.")
    return tokenizer, geo_token_id, added


def apply_runtime_options(llm, config: ModelConfig):
    if config.gradient_checkpointing and hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable()
        if hasattr(llm, "enable_input_require_grads"):
            llm.enable_input_require_grads()
    if hasattr(llm.config, "use_cache"):
        llm.config.use_cache = config.use_cache_during_training
    return llm


def build_encoder(config: ModelConfig, geo_input_dim: int) -> torch.nn.Module:
    if geo_input_dim <= 0:
        raise ValueError("Geo2Vec features need a positive width from the embedding table.")
    return Geo2VecEncoder(geo_input_dim, config.geo_hidden_dim, config.geo_out_dim)


def build_slm(
    config: ModelConfig,
    lora_config: LoraConfigValues,
    *,
    geo_input_dim: int,
) -> SLM:
    tokenizer, geo_token_id, added = build_tokenizer(config)
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": resolve_dtype(config.torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    llm = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    if added:
        llm.resize_token_embeddings(len(tokenizer))
    llm = maybe_apply_lora(llm, lora_config)
    llm = apply_runtime_options(llm, config)
    return SLM(
        llm=llm,
        tokenizer=tokenizer,
        geo_encoder=build_encoder(config, geo_input_dim),
        geo_token_id=geo_token_id,
        geo_projector_hidden_dim=config.geo_projector_hidden_dim,
    )


def save_checkpoint(model: SLM, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(str(output_dir / "llm"))
    model.tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    torch.save(
        {
            "geo_encoder": model.geo_encoder.state_dict(),
            "geo_projector": model.geo_projector.state_dict(),
            "geo_token_id": model.geo_token_id,
            "geo_input_dim": int(getattr(model.geo_encoder, "input_dim", 0)),
            "geo_out_dim": model.geo_out_dim,
        },
        output_dir / GEO_MODULES_FILENAME,
    )


def load_checkpoint(config: ModelConfig, checkpoint_dir: str | Path, *, geo_input_dim: int | None = None) -> SLM:
    checkpoint_dir = Path(checkpoint_dir)
    geo_path = checkpoint_dir / GEO_MODULES_FILENAME
    if not geo_path.is_file():
        raise FileNotFoundError(f"Missing {geo_path}.")
    geo_state = torch.load(geo_path, map_location="cpu")
    input_dim = geo_input_dim or int(geo_state.get("geo_input_dim") or 0)
    if input_dim <= 0:
        raise ValueError("Pass geo_input_dim or use a checkpoint that recorded it.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer", trust_remote_code=config.trust_remote_code)
    geo_token_id = tokenizer.convert_tokens_to_ids(config.geo_token)
    dtype = resolve_dtype(config.torch_dtype)
    llm = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    llm.resize_token_embeddings(len(tokenizer))
    adapter_dir = checkpoint_dir / "llm"
    try:
        from peft import PeftModel

        llm = PeftModel.from_pretrained(llm, adapter_dir)
    except Exception:
        logger.info("No PEFT adapter in %s; loading it as a full model.", adapter_dir)
        llm = AutoModelForCausalLM.from_pretrained(
            adapter_dir,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    llm = apply_runtime_options(llm, config)
    model = SLM(
        llm=llm,
        tokenizer=tokenizer,
        geo_encoder=build_encoder(config, input_dim),
        geo_token_id=geo_token_id,
        geo_projector_hidden_dim=config.geo_projector_hidden_dim,
    )
    model.geo_encoder.load_state_dict(geo_state["geo_encoder"])
    model.geo_projector.load_state_dict(geo_state["geo_projector"])
    saved_token = geo_state.get("geo_token_id")
    if saved_token is not None and int(saved_token) != int(model.geo_token_id):
        logger.warning(
            "Checkpoint geo token id %s differs from the tokenizer id %s; using the tokenizer.",
            saved_token,
            model.geo_token_id,
        )
    return model
