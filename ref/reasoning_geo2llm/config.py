from __future__ import annotations

import logging
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    training_data: str = "/home/chenchu/projects/GeoCoding/data/london_training.csv"
    test_data: str = "/home/chenchu/projects/GeoCoding/data/London_test.csv"
    input_format: str = "csv"
    id_col: str = "id"
    latitude_col: str = "latitude"
    longitude_col: str = "longitude"
    description_col: str = "description"
    normalize_coordinates: bool = True
    max_train_samples: int | None = 6000
    max_eval_samples: int | None = 256


@dataclass
class ModelConfig:
    base_model: str = "/home/chenchu/LLMs/Qwen3.5-9B"
    geo_token: str = "<GEO>"
    text_dim: int = 2560
    geo_hidden_dim: int = 4096
    geo_projector_hidden_dim: int = 2048
    num_freqs: int = 36
    geo_log_sampling: bool = False
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    use_cache_during_training: bool = False
    # None = Fourier/GeoCLIP; "grid" = cell embedding; "DIRECT" = textual coordinates;
    # "SCALE" = two Fourier/GeoCLIP embeddings with log and linear frequency sampling;
    # "sphere" = spherical-harmonics + SIREN location encoder (vendored LocationEncoder repo);
    # "HYBRID" = unnormalized lat/lon as prompt text *and* the learned <GEO> embedding right
    # after them. The location encoder still receives the normalized [-1, 1] representation.
    # "geohash" = Base32 geohash of length `geohash_length` as prompt text (ReaGeo / arXiv:2604.21357),
    # no learned <GEO> encoder. Requires raw (unnormalized) lat/lon.
    spatial_division_method: str | None = None
    grid_num_axis: int = 100
    num_cells: int | None = None
    geohash_length: int = 9
    # "sphere" hyperparameters (see locationencoder.LocationEncoder("sphericalharmonics", "siren", ...)).
    sphere_legendre_polys: int = 40
    sphere_dim_hidden: int = 64
    sphere_num_layers: int = 2


@dataclass
class LoraConfigValues:
    enabled: bool = True
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] | None = None
    freeze_base_model: bool = True
    # Expert feed-forward rank for MoE backbones. None keeps the usual target list at `r`.
    # When set on a model whose experts are fused parameters (GPT-OSS `gate_up_proj` /
    # `down_proj`), attention stays at `r` and every expert is adapted at this rank. The
    # router stays frozen. PEFT cannot apply dropout to those parameters, so `dropout`
    # must be 0. Ignored when the backbone has no such expert parameters.
    moe_rank: int | None = None
    # When True, only the embeddings of `tuning_embedding_tokens` are trained (via PEFT
    # trainable_token_indices), not the full word-embedding table. Default tokens are the
    # digit characters 0-9 that spell out lat/lon in DIRECT/HYBRID prompts.
    tuning_embedding: bool = False
    tuning_embedding_tokens: list[str] | None = None


@dataclass
class RuntimeConfig:
    # Same format as CUDA_VISIBLE_DEVICES, for example "0" or "2,3,7".
    cuda_visible_devices: str | None = None


@dataclass
class ReasoningConfig:
    """Configures the 'reasoning' training method.

    When enabled, each training record's label is built from *other* descriptions
    that share the same `data.id_col` value, framed as a thinking trace, followed by
    the record's own description as the final answer:

        <thinking> other description 1 other description 2 ... </thinking> actually <description>
    """

    enabled: bool = False
    # Cap on how many sibling descriptions (same id) go into the thinking trace. None = no cap.
    max_thinking_descriptions: int | None = 5
    shuffle_thinking_descriptions: bool = True
    # None = auto-detect from the loaded tokenizer's own chat template (via
    # `resolve_reasoning_config`), so the same config works across different base models without
    # hard-coding one model's convention. Set explicitly here to pin/override the detected tags.
    thinking_open_tag: str | None = None
    thinking_close_tag: str | None = None
    thinking_join: str = " "
    answer_prefix: str = "actually"


@dataclass
class TrainConfig:
    output_dir: str = "outputs/description_generation_spatial_partition100_3_5"
    init_from_checkpoint: str | None = None
    epochs: int = 30
    batch_size: int = 48
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    num_workers: int = 4
    mixed_precision: str = "bf16"
    log_every: int = 20
    save_every_epochs: int = 1
    eval_every_epochs: int = 5
    eval_num_samples: int = 64  # <= 0 (e.g. -1) evaluates every location instead of sampling.
    eval_sentence_model: str = "all-MiniLM-L6-v2"
    save_on_eval_improvement: bool = True
    seed: int = 42
    compile_model: bool = False
    # The geo modules (geo_clip / geo_projector / grid cell embeddings) start from random weights,
    # while the LoRA adapters only perturb an already-pretrained LLM, so a single shared step size
    # is wrong for one of the two groups. `geo_learning_rate` pins an absolute LR for the geo group;
    # otherwise it is `learning_rate * geo_learning_rate_multiplier`.
    geo_learning_rate: float | None = None
    geo_learning_rate_multiplier: float = 1.0
    # Any name accepted by `transformers.get_scheduler` ("constant", "cosine", "linear", ...).
    # The defaults here ("constant", no warmup, 1.0 multiplier) reproduce the historical flat
    # single-LR behavior, so existing configs stay comparable with previously logged runs.
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0
    warmup_steps: int | None = None
    # Two-stage multimodal tuning (LLaVA-style alignment pre-training). For spatial methods with
    # a learned encoder (Fourier/GeoCLIP default, HYBRID, grid, SCALE, sphere) the run starts with
    # this many *extra* epochs during which the entire LLM side (base weights, LoRA adapters,
    # trainable-token deltas) is frozen and only the geo encoder/projector/cell-embedding modules
    # train, so their random init is aligned to the frozen LLM's embedding space first. After that
    # the trainable set configured under `lora:` is restored for the usual `epochs` joint epochs.
    # 0 disables the stage; DIRECT/geohash (text-only, no geo modules) ignore it.
    encoder_only_epochs: int = 0


@dataclass
class GenerationConfig:
    batch_size: int = 32
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    generations_per_coordinate: int = 4
    output: str = "/home/chenchu/projects/GeoCoding/data/augmented4_50_london.pkl"
    # True = official test split; False = train+test; a filesystem path = that file,
    # normalized to training bounds when `data.normalize_coordinates` is on.
    use_test_coordinates: bool | str = True
    checkpoint: str = "/home/chenchu/projects/GeoCoding/reasoning/outputs/description_generation_SpatialPartition50_london_python/best"
    evaluate_after_generation: bool = True
    eval_sentence_model: str = "all-MiniLM-L6-v2"


@dataclass
class AppConfig:
    data: DataConfig
    model: ModelConfig
    lora: LoraConfigValues
    runtime: RuntimeConfig
    train: TrainConfig
    generation: GenerationConfig
    reasoning: ReasoningConfig


def _dataclass_from_dict(cls: type, values: dict[str, Any]):
    valid = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in valid})


def apply_spatial_division_coordinate_policy(data: DataConfig, model: ModelConfig) -> None:
    """Force `data.normalize_coordinates` to what each spatial method actually consumes.

    DIRECT/geohash/sphere need real-world degrees (prompt text / spherical harmonics), so
    normalization is forced off. HYBRID writes those same degrees into the prompt but
    still feeds the location encoder a [-1, 1] tensor; normalization is forced on and
    the original degrees are kept in the `*_raw` columns for the text path.
    """
    if model.spatial_division_method in ("sphere", "DIRECT", "geohash") and data.normalize_coordinates:
        logger.warning(
            "spatial_division_method=%r requires raw (unnormalized) lat/lon; "
            "forcing data.normalize_coordinates=False (was True).",
            model.spatial_division_method,
        )
        data.normalize_coordinates = False
    if model.spatial_division_method == "HYBRID" and not data.normalize_coordinates:
        logger.warning(
            "spatial_division_method='HYBRID' writes unnormalized degrees into the prompt "
            "but passes a [-1, 1] representation to the location encoder; "
            "forcing data.normalize_coordinates=True (was False).",
        )
        data.normalize_coordinates = True


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    data = _dataclass_from_dict(DataConfig, raw.get("data", {}))
    model = _dataclass_from_dict(ModelConfig, raw.get("model", {}))
    apply_spatial_division_coordinate_policy(data, model)
    return AppConfig(
        data=data,
        model=model,
        lora=_dataclass_from_dict(LoraConfigValues, raw.get("lora", {})),
        runtime=_dataclass_from_dict(RuntimeConfig, raw.get("runtime", {})),
        train=_dataclass_from_dict(TrainConfig, raw.get("train", {})),
        generation=_dataclass_from_dict(GenerationConfig, raw.get("generation", {})),
        reasoning=_dataclass_from_dict(ReasoningConfig, raw.get("reasoning", {})),
    )


def to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: to_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value
