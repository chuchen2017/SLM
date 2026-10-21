"""YAML configuration for SLM training, evaluation, and embedding extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def default_tasks_for_city(city: str) -> list[str]:
    """US counties are polygons with a topology file. LA and Beijing also have roads."""
    if city.upper() == "US":
        return ["area", "distance", "topology", "nearest"]
    return ["length", "area", "distance", "nearest"]


@dataclass
class DataConfig:
    root: str = "data"
    city: str = "LosAngeles"
    tasks: list[str] | None = None
    samples_per_task: int = 2000
    # Optional unordered pair of geometry types for the distance task, e.g.
    # ["LineString", "Polygon"]. None samples any two distinct entities.
    distance_geometry_pair: list[str] | None = None
    length_decimals: int = 2
    area_decimals: int = 2
    distance_decimals: int = 0
    include_thinking: bool = False
    thinking_samples: int = 1000
    thinking_open_tag: str | None = None
    thinking_close_tag: str | None = None
    questions_dir: str = "data/instruction_dataset"
    osm_dir: str = "data/osm"
    benchmark_dir: str = "data/benchmark"
    train_geometry: str = "training_data.gpkg"
    test_geometry: str = "testing_data.gpkg"
    train_embeddings: str = "training_embeddings_self.npy"
    test_embeddings: str = "testing_embeddings_self.npy"
    system_prompt: str | None = None


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen3-8B"
    geo_token: str = "<GEO>"
    geo_input_dim: int | None = None
    geo_hidden_dim: int = 256
    geo_out_dim: int = 256
    geo_projector_hidden_dim: int = 2048
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    use_cache_during_training: bool = False


@dataclass
class LoraConfigValues:
    enabled: bool = True
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list[str] | None = None
    freeze_base_model: bool = True


@dataclass
class RuntimeConfig:
    cuda_visible_devices: str | None = None


@dataclass
class TrainConfig:
    output_dir: str = "outputs/slm"
    init_from_checkpoint: str | None = None
    epochs: int = 3
    # Extra epochs that train only the geo encoder and projector.
    encoder_only_epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-5
    geo_learning_rate: float | None = None
    geo_learning_rate_multiplier: float = 10.0
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    num_workers: int = 0
    mixed_precision: str = "bf16"
    save_every_epochs: int = 1
    eval_every_epochs: int = 1
    eval_num_samples: int = 32
    seed: int = 42
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int | None = None


@dataclass
class GenerationConfig:
    batch_size: int = 8
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = False


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfigValues = field(default_factory=LoraConfigValues)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def _dataclass_from_dict(cls: type, values: dict[str, Any] | None):
    values = values or {}
    valid = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - valid)
    if unknown:
        logger.warning("Ignoring unknown %s fields: %s", cls.__name__, unknown)
    return cls(**{key: value for key, value in values.items() if key in valid})


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = AppConfig(
        data=_dataclass_from_dict(DataConfig, raw.get("data")),
        model=_dataclass_from_dict(ModelConfig, raw.get("model")),
        lora=_dataclass_from_dict(LoraConfigValues, raw.get("lora")),
        runtime=_dataclass_from_dict(RuntimeConfig, raw.get("runtime")),
        train=_dataclass_from_dict(TrainConfig, raw.get("train")),
        generation=_dataclass_from_dict(GenerationConfig, raw.get("generation")),
    )
    if not config.data.tasks:
        config.data.tasks = default_tasks_for_city(config.data.city)
    return config


def to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: to_dict(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value
