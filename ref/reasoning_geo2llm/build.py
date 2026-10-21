from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_geo2llm.config import LoraConfigValues, ModelConfig
from reasoning_geo2llm.data import grid_num_cells
from reasoning_geo2llm.lora import maybe_apply_lora
from reasoning_geo2llm.modeling import Geo2LLMLlama, GeoCLIP, SphereLocationEncoder

logger = logging.getLogger(__name__)

GEO_MODULES_FILENAME = "geo_modules.pt"
REQUIRED_LEGACY_GEO_KEYS = ("geo_clip", "geo_projector")


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def build_tokenizer(config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_tokens([config.geo_token], special_tokens=False)
    geo_token_id = tokenizer.convert_tokens_to_ids(config.geo_token)
    if geo_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Failed to add geo token {config.geo_token!r}.")
    return tokenizer, geo_token_id, added


def _apply_llm_runtime_options(llm, config: ModelConfig):
    if config.gradient_checkpointing and hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable()
        if hasattr(llm, "enable_input_require_grads"):
            llm.enable_input_require_grads()
    if hasattr(llm.config, "use_cache"):
        llm.config.use_cache = config.use_cache_during_training
    return llm


def resolve_num_cells(config: ModelConfig) -> int | None:
    if config.spatial_division_method != "grid":
        return None
    if config.num_cells is not None:
        return config.num_cells
    return grid_num_cells(config.grid_num_axis)


def build_scale_geo_clips(config: ModelConfig) -> tuple[GeoCLIP | None, GeoCLIP | None]:
    if config.spatial_division_method != "SCALE":
        return None, None
    return (
        GeoCLIP(
            text_dim=config.text_dim,
            hidden_dim=config.geo_hidden_dim,
            num_freqs=config.num_freqs,
            log_sampling=True,
        ),
        GeoCLIP(
            text_dim=config.text_dim,
            hidden_dim=config.geo_hidden_dim,
            num_freqs=config.num_freqs,
            log_sampling=False,
        ),
    )


def build_geo_clip(config: ModelConfig) -> GeoCLIP | SphereLocationEncoder:
    if config.spatial_division_method == "sphere":
        return SphereLocationEncoder(
            out_dim=config.text_dim,
            legendre_polys=config.sphere_legendre_polys,
            dim_hidden=config.sphere_dim_hidden,
            num_layers=config.sphere_num_layers,
        )
    return GeoCLIP(
        text_dim=config.text_dim,
        hidden_dim=config.geo_hidden_dim,
        num_freqs=config.num_freqs,
        log_sampling=config.geo_log_sampling,
    )


def build_geo2llm(
    config: ModelConfig,
    lora_config: LoraConfigValues,
    device_map: str | dict | None = None,
) -> Geo2LLMLlama:
    tokenizer, geo_token_id, added_tokens = build_tokenizer(config)
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": resolve_dtype(config.torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if config.attn_implementation is not None:
        model_kwargs["attn_implementation"] = config.attn_implementation
    # One replica of GPT-OSS already fills a 48GB GPU, and expert LoRA materializes
    # another ~1GB delta. Split layers across the visible GPUs instead of copying
    # the model. max_memory has no CPU entry, so experts stay on GPU.
    if device_map is not None:
        model_kwargs["device_map"] = device_map
        if torch.cuda.is_available():
            model_kwargs["max_memory"] = {index: "44GiB" for index in range(torch.cuda.device_count())}

    llm = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    if added_tokens:
        llm.resize_token_embeddings(len(tokenizer))
    llm = maybe_apply_lora(llm, lora_config, tokenizer=tokenizer)
    llm = _apply_llm_runtime_options(llm, config)

    geo_clip = build_geo_clip(config)
    scale_geo_clip_log, scale_geo_clip_linear = build_scale_geo_clips(config)
    return Geo2LLMLlama(
        llm=llm,
        tokenizer=tokenizer,
        geo_clip=geo_clip,
        geo_token_id=geo_token_id,
        geo_out_dim=config.text_dim,
        geo_projector_hidden_dim=config.geo_projector_hidden_dim,
        spatial_division_method=config.spatial_division_method,
        num_cells=resolve_num_cells(config),
        scale_geo_clip_log=scale_geo_clip_log,
        scale_geo_clip_linear=scale_geo_clip_linear,
    )


def save_checkpoint(model: Geo2LLMLlama, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(str(output_dir / "llm_lora_or_full"))
    model.tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    geo_state: dict[str, object] = {
        "geo_clip": model.geo_clip.state_dict(),
        "geo_projector": model.geo_projector.state_dict(),
        "geo_token_id": model.geo_token_id,
        "spatial_division_method": model.spatial_division_method,
    }
    if model.embedding_layer is not None:
        geo_state["embedding_layer"] = model.embedding_layer.state_dict()
    if model.scale_geo_clip_log is not None:
        geo_state["scale_geo_clip_log"] = model.scale_geo_clip_log.state_dict()
    if model.scale_geo_clip_linear is not None:
        geo_state["scale_geo_clip_linear"] = model.scale_geo_clip_linear.state_dict()
    if model.scale_geo_projector_log is not None:
        geo_state["scale_geo_projector_log"] = model.scale_geo_projector_log.state_dict()
    if model.scale_geo_projector_linear is not None:
        geo_state["scale_geo_projector_linear"] = model.scale_geo_projector_linear.state_dict()
    torch.save(geo_state, output_dir / GEO_MODULES_FILENAME)


def _load_module_state(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    module_name: str,
) -> tuple[list[str], list[str]]:
    incompatible = module.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing:
        logger.warning(
            "Checkpoint %s is missing %d parameter(s): %s",
            module_name,
            len(missing),
            ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""),
        )
    if unexpected:
        logger.warning(
            "Checkpoint %s has %d unexpected key(s): %s",
            module_name,
            len(unexpected),
            ", ".join(unexpected[:8]) + (" ..." if len(unexpected) > 8 else ""),
        )
    return missing, unexpected


def load_geo_modules(model: Geo2LLMLlama, geo_state: dict[str, object]) -> None:
    """Load geo weights from a checkpoint, including legacy files without grid embeddings."""
    missing_required = [key for key in REQUIRED_LEGACY_GEO_KEYS if key not in geo_state]
    if missing_required:
        raise KeyError(
            f"{GEO_MODULES_FILENAME} is missing required keys: {missing_required}. "
            f"Found keys: {sorted(geo_state.keys())}"
        )

    saved_method = geo_state.get("spatial_division_method")
    if saved_method is not None and saved_method != model.spatial_division_method:
        logger.warning(
            "Checkpoint spatial_division_method=%r differs from config=%r; "
            "using config to build the model and loading compatible weights only.",
            saved_method,
            model.spatial_division_method,
        )

    _load_module_state(model.geo_clip, geo_state["geo_clip"], module_name="geo_clip")
    _load_module_state(model.geo_projector, geo_state["geo_projector"], module_name="geo_projector")

    if model.embedding_layer is not None:
        embedding_state = geo_state.get("embedding_layer")
        if embedding_state is None:
            logger.warning(
                "Checkpoint has no embedding_layer weights (legacy Fourier-only checkpoint). "
                "Grid cell embeddings are randomly initialized; geo_clip and geo_projector were loaded."
            )
        else:
            _load_module_state(
                model.embedding_layer,
                embedding_state,
                module_name="embedding_layer",
            )
    elif "embedding_layer" in geo_state:
        logger.info(
            "Checkpoint contains embedding_layer weights but config spatial_division_method is not 'grid'; "
            "ignoring embedding_layer."
        )

    if model.spatial_division_method == "SCALE":
        scale_modules = {
            "scale_geo_clip_log": model.scale_geo_clip_log,
            "scale_geo_clip_linear": model.scale_geo_clip_linear,
            "scale_geo_projector_log": model.scale_geo_projector_log,
            "scale_geo_projector_linear": model.scale_geo_projector_linear,
        }
        for state_key, module in scale_modules.items():
            module_state = geo_state.get(state_key)
            if module_state is None:
                logger.warning(
                    "Checkpoint has no %s weights; this SCALE module is randomly initialized.",
                    state_key,
                )
                continue
            _load_module_state(module, module_state, module_name=state_key)
    else:
        extra_scale_keys = [
            key
            for key in (
                "scale_geo_clip_log",
                "scale_geo_clip_linear",
                "scale_geo_projector_log",
                "scale_geo_projector_linear",
            )
            if key in geo_state
        ]
        if extra_scale_keys:
            logger.info(
                "Checkpoint contains SCALE weights but config spatial_division_method is not 'SCALE'; "
                "ignoring %s.",
                ", ".join(extra_scale_keys),
            )

    saved_geo_token_id = geo_state.get("geo_token_id")
    if saved_geo_token_id is not None and int(saved_geo_token_id) != int(model.geo_token_id):
        logger.warning(
            "Checkpoint geo_token_id=%s differs from tokenizer id=%s; using tokenizer id.",
            saved_geo_token_id,
            model.geo_token_id,
        )


def load_geo_modules_file(model: Geo2LLMLlama, checkpoint_dir: Path) -> None:
    geo_path = checkpoint_dir / GEO_MODULES_FILENAME
    if not geo_path.is_file():
        raise FileNotFoundError(f"Missing geo checkpoint file: {geo_path}")
    geo_state = torch.load(geo_path, map_location="cpu")
    if not isinstance(geo_state, dict):
        raise TypeError(f"Expected dict in {geo_path}, got {type(geo_state).__name__}")
    load_geo_modules(model, geo_state)


def load_checkpoint(config: ModelConfig, checkpoint_dir: str | Path) -> Geo2LLMLlama:
    checkpoint_dir = Path(checkpoint_dir)
    #checkpoint_dir + "_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer", trust_remote_code=config.trust_remote_code)
    geo_token_id = tokenizer.convert_tokens_to_ids(config.geo_token)
    llm = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=resolve_dtype(config.torch_dtype),
        low_cpu_mem_usage=True,
    )
    llm.resize_token_embeddings(len(tokenizer))
    adapter_dir = checkpoint_dir / "llm_lora_or_full"
    try:
        from peft import PeftModel

        llm = PeftModel.from_pretrained(llm, adapter_dir)
    except Exception:
        llm = AutoModelForCausalLM.from_pretrained(
            adapter_dir,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=resolve_dtype(config.torch_dtype),
            low_cpu_mem_usage=True,
        )
    llm = _apply_llm_runtime_options(llm, config)

    geo_clip = build_geo_clip(config)
    scale_geo_clip_log, scale_geo_clip_linear = build_scale_geo_clips(config)
    model = Geo2LLMLlama(
        llm=llm,
        tokenizer=tokenizer,
        geo_clip=geo_clip,
        geo_token_id=geo_token_id,
        geo_out_dim=config.text_dim,
        geo_projector_hidden_dim=config.geo_projector_hidden_dim,
        spatial_division_method=config.spatial_division_method,
        num_cells=resolve_num_cells(config),
        scale_geo_clip_log=scale_geo_clip_log,
        scale_geo_clip_linear=scale_geo_clip_linear,
    )
    load_geo_modules_file(model, checkpoint_dir)
    return model
