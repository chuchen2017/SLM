from __future__ import annotations

import contextlib
import dataclasses
import os
import pickle
import random
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from reasoning_geo2llm.config import DataConfig, ReasoningConfig

SYSTEM_PROMPT = (
    "You are a helpful assistant for geo-localization. Always only use the information "
    "in the representation to generate the description."
)
GENERATION_PROMPT = "You are at location <GEO>, generate a detailed description of the location:"
GEO_PLACEHOLDER = "<GEO>"
DEFAULT_THINKING_OPEN_TAG = "<think>"
DEFAULT_THINKING_CLOSE_TAG = "</think>"
# Suffix for the columns that keep the pre-normalization lat/lon (see `normalize_coordinates`).
RAW_COORDINATE_SUFFIX = "_raw"
# ReaGeo (arXiv:2604.21357): Base32 geohash of length 9 is ~2.4 m, matching typical search precision.
# Alphabet omits a/i/l/o to avoid visual ambiguity, same as Niemeyer's geohash.
GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
DEFAULT_GEOHASH_LENGTH = 9


def resolve_thinking_tags(
    tokenizer,
    fallback_open: str = DEFAULT_THINKING_OPEN_TAG,
    fallback_close: str = DEFAULT_THINKING_CLOSE_TAG,
) -> tuple[str, str]:
    """Discover a model's own thinking-trace tags from its chat template.

    Many reasoning-capable chat templates (Qwen3/3.5, DeepSeek-R1(-distill), QwQ, ...) accept an
    `enable_thinking` flag in `apply_chat_template` and, when true, insert a `<think>...</think>`-
    style scaffold into the assistant turn. We diff the rendered prompt with/without that flag to
    recover the exact tag strings for whichever model is loaded, instead of hard-coding one
    model's convention (e.g. a custom `<thinking>` word that the model may never have seen).
    Falls back to `fallback_open`/`fallback_close` if detection isn't possible.
    """
    probe_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "probe"},
    ]
    try:
        with_thinking = tokenizer.apply_chat_template(
            probe_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        without_thinking = tokenizer.apply_chat_template(
            probe_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        return fallback_open, fallback_close

    if with_thinking == without_thinking:
        return fallback_open, fallback_close

    # `enable_thinking=False` commonly renders a pre-closed *empty* thinking block (e.g.
    # `<think>\n\n</think>\n\n`) to tell the model to skip reasoning; look for that self-contained
    # pair first since it unambiguously names the tag.
    pair_pattern = re.compile(r"<([^<>/\s]+)>\s*</\1>", flags=re.DOTALL)
    for text in (without_thinking, with_thinking):
        match = pair_pattern.search(text)
        if match:
            return f"<{match.group(1)}>", f"</{match.group(1)}>"

    # Otherwise, the two renderings differ because one leaves the tag open for the model to fill
    # in itself (`enable_thinking=True`) while the other immediately closes it - the opening tag
    # can end up in the shared common prefix, with only the closing tag left in the differing
    # tail. Diff the two renderings and look for either tag in whichever tail has content.
    common_len = len(os.path.commonprefix([with_thinking, without_thinking]))
    candidates = [with_thinking[common_len:], without_thinking[common_len:]]

    for candidate in candidates:
        match = re.search(r"</([^<>\s]+)>", candidate)
        if match:
            return f"<{match.group(1)}>", f"</{match.group(1)}>"

    for candidate in candidates:
        match = re.search(r"<([^<>/\s]+)>", candidate)
        if match:
            return f"<{match.group(1)}>", f"</{match.group(1)}>"

    return fallback_open, fallback_close


def resolve_reasoning_config(reasoning: ReasoningConfig | None, tokenizer) -> ReasoningConfig | None:
    """Fill in unset `thinking_open_tag`/`thinking_close_tag` from the tokenizer's chat template.

    Leaves the config untouched if reasoning is disabled or the tags were already pinned
    explicitly in the YAML config, so this is safe (and cheap) to call unconditionally.
    """
    if reasoning is None or not reasoning.enabled:
        return reasoning
    if reasoning.thinking_open_tag and reasoning.thinking_close_tag:
        return reasoning
    open_tag, close_tag = resolve_thinking_tags(tokenizer)
    return dataclasses.replace(reasoning, thinking_open_tag=open_tag, thinking_close_tag=close_tag)


def resolve_system_prompt(reasoning: ReasoningConfig | None) -> str:
    """Add reasoning-format instructions to the system prompt when the 'reasoning' method is enabled."""
    if reasoning is None or not reasoning.enabled:
        return SYSTEM_PROMPT
    open_tag = reasoning.thinking_open_tag or DEFAULT_THINKING_OPEN_TAG
    close_tag = reasoning.thinking_close_tag or DEFAULT_THINKING_CLOSE_TAG
    return (
        f"{SYSTEM_PROMPT} First reason inside {open_tag} "
        f"{close_tag} about other plausible descriptions of the location, "
        f"then answer with '{reasoning.answer_prefix}' followed by the final description."
    )


def build_reasoning_label(
    other_descriptions: list[str],
    description: str,
    *,
    open_tag: str = DEFAULT_THINKING_OPEN_TAG,
    close_tag: str = DEFAULT_THINKING_CLOSE_TAG,
    join_str: str = " ",
    answer_prefix: str = "actually",
) -> str:
    """Format a training label as a thinking trace of sibling descriptions plus the final answer."""
    thinking_text = join_str.join(other_descriptions)
    return f"{open_tag} {thinking_text} {close_tag} {answer_prefix} {description}"


def extract_reasoning_answer(
    text: str,
    *,
    close_tag: str = DEFAULT_THINKING_CLOSE_TAG,
    answer_prefix: str = "actually",
) -> str:
    """Strip a reasoning-formatted generation down to the final answer, for scoring/saving."""
    idx = text.find(close_tag)
    if idx == -1:
        return text.strip()
    remainder = text[idx + len(close_tag) :].strip()
    if answer_prefix and remainder.startswith(answer_prefix):
        remainder = remainder[len(answer_prefix) :].strip()
    return remainder


def canonical_coordinate_key(
    latitude: float,
    longitude: float,
    precision: int = 6,
) -> tuple[float, float]:
    """Match coordinates the same way as the float32 generation tensor pipeline."""
    lat = float(np.float32(latitude))
    lon = float(np.float32(longitude))
    return (round(lat, precision), round(lon, precision))


def coordinate_pair_as_list(latitude: float, longitude: float) -> list[float]:
    """Store lat/lon in the same representation used during generation."""
    return [float(np.float32(latitude)), float(np.float32(longitude))]


def unique_coordinate_tensor(
    dataframe: pd.DataFrame,
    latitude_col: str,
    longitude_col: str,
) -> torch.Tensor:
    """Unique coordinates in float32, matching load_coordinates / generation."""
    coords, _ = unique_coordinate_tensors(dataframe, latitude_col, longitude_col)
    return coords


def unique_coordinate_tensors(
    dataframe: pd.DataFrame,
    latitude_col: str,
    longitude_col: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Unique model-facing coordinates plus their row-aligned real-world degrees.

    The second tensor is `None` when the frame has no `*_raw` columns, i.e. normalization was off
    and the model-facing coordinates already *are* the real-world degrees. It stays float64 because
    it is only ever formatted into prompt text, where float32 would visibly truncate the decimals
    that `format_coordinates_text` prints.
    """
    raw_latitude_col, raw_longitude_col = raw_coordinate_columns(latitude_col, longitude_col)
    deduped = dataframe.drop_duplicates(subset=[latitude_col, longitude_col])
    coords = torch.from_numpy(
        deduped[[latitude_col, longitude_col]].to_numpy(dtype=np.float32, copy=False)
    )
    if raw_latitude_col not in deduped.columns or raw_longitude_col not in deduped.columns:
        return coords, None
    raw_coords = torch.from_numpy(
        deduped[[raw_latitude_col, raw_longitude_col]].to_numpy(dtype=np.float64, copy=False)
    )
    return coords, raw_coords


def format_coordinates_text(latitude: float, longitude: float, precision: int = 6) -> str:
    return f"({latitude:.{precision}f}, {longitude:.{precision}f})"


def encode_geohash(
    latitude: float,
    longitude: float,
    length: int = DEFAULT_GEOHASH_LENGTH,
) -> str:
    """Encode WGS-84 degrees to a Base32 geohash (Niemeyer), default length 9 (~2.4 m).

    Same recursive lon/lat bisection used by ReaGeo: even bits encode longitude, odd bits
    encode latitude; each character is 5 bits from `GEOHASH_BASE32`.
    """
    if length <= 0:
        raise ValueError("geohash length must be positive")
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError(f"latitude {latitude} is outside [-90, 90]")
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError(f"longitude {longitude} is outside [-180, 180]")

    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    bits = [16, 8, 4, 2, 1]
    chars: list[str] = []
    bit = 0
    ch = 0
    even = True
    while len(chars) < length:
        if even:
            mid = (lon_min + lon_max) / 2.0
            if longitude > mid:
                ch |= bits[bit]
                lon_min = mid
            else:
                lon_max = mid
        else:
            mid = (lat_min + lat_max) / 2.0
            if latitude > mid:
                ch |= bits[bit]
                lat_min = mid
            else:
                lat_max = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            chars.append(GEOHASH_BASE32[ch])
            bit = 0
            ch = 0
    return "".join(chars)


def generation_prompt_with_coordinates(latitude: float, longitude: float, precision: int = 6) -> str:
    return GENERATION_PROMPT.replace(
        GEO_PLACEHOLDER,
        format_coordinates_text(latitude, longitude, precision=precision),
    )


def generation_prompt_with_geohash(
    latitude: float,
    longitude: float,
    length: int = DEFAULT_GEOHASH_LENGTH,
) -> str:
    """DIRECT-style prompt whose location slot is a Base32 geohash instead of (lat, lon)."""
    return GENERATION_PROMPT.replace(GEO_PLACEHOLDER, encode_geohash(latitude, longitude, length=length))


def hybrid_prompt_with_coordinates(latitude: float, longitude: float, precision: int = 6) -> str:
    """DIRECT's textual coordinates, immediately followed by the learned `<GEO>` embedding slot."""
    coordinates = format_coordinates_text(latitude, longitude, precision=precision)
    return GENERATION_PROMPT.replace(GEO_PLACEHOLDER, f"{coordinates} {GEO_PLACEHOLDER}")


def build_geo_sample(
    latitude: float,
    longitude: float,
    *,
    spatial_division_method: str | None = None,
    grid_bounds: tuple[float, float, float, float] | None = None,
    grid_num_axis: int = 100,
    text_latitude: float | None = None,
    text_longitude: float | None = None,
    geohash_length: int = DEFAULT_GEOHASH_LENGTH,
) -> dict[str, Any]:
    """Build user prompt and optional geo tensor for one location.

    `latitude`/`longitude` feed the geo tensor, so they are whatever the location encoder expects
    (normalized to [-1, 1] when `data.normalize_coordinates` is on). `text_latitude`/`text_longitude`
    are written into the prompt text instead and should be real-world degrees; they default to
    `latitude`/`longitude`. For "HYBRID" they must differ: the prompt stays in degrees while the
    tensor passed to the location encoder is the normalized representation. For "geohash" the prompt
    is a length-`geohash_length` Base32 string computed from those degrees (no `<GEO>` encoder).
    """
    prompt_latitude = latitude if text_latitude is None else text_latitude
    prompt_longitude = longitude if text_longitude is None else text_longitude
    if spatial_division_method == "DIRECT":
        return {
            "prompt": generation_prompt_with_coordinates(prompt_latitude, prompt_longitude),
            "geo_feature": torch.zeros(0, 2, dtype=torch.float32),
        }
    if spatial_division_method == "geohash":
        return {
            "prompt": generation_prompt_with_geohash(
                prompt_latitude, prompt_longitude, length=geohash_length
            ),
            "geo_feature": torch.zeros(0, 2, dtype=torch.float32),
        }
    if spatial_division_method == "HYBRID":
        return {
            "prompt": hybrid_prompt_with_coordinates(prompt_latitude, prompt_longitude),
            "geo_feature": torch.tensor([[latitude, longitude]], dtype=torch.float32),
        }
    if spatial_division_method == "grid":
        if grid_bounds is None:
            raise ValueError("grid_bounds is required when spatial_division_method='grid'.")
        min_lat, max_lat, min_lon, max_lon = grid_bounds
        cell_id = lat_lon_to_grid_cell(
            latitude,
            longitude,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            num_axis=grid_num_axis,
        )
        return {
            "prompt": GENERATION_PROMPT,
            "geo_feature": torch.tensor([[cell_id]], dtype=torch.long),
        }
    return {
        "prompt": GENERATION_PROMPT,
        "geo_feature": torch.tensor([[latitude, longitude]], dtype=torch.float32),
    }


def _infer_input_format(path: Path, configured_format: str) -> str:
    if configured_format:
        return configured_format
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return "csv"
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    if suffix in {".json"}:
        return "json"
    raise ValueError(f"Could not infer input format for {path}; set data.input_format explicitly.")


def _read_raw_dataframe(path: Path, config: DataConfig) -> pd.DataFrame:
    input_format = _infer_input_format(path, config.input_format)
    if input_format == "pickle":
        with path.open("rb") as f:
            records = pickle.load(f)
        return pd.DataFrame(
            records,
            columns=[
                config.id_col,
                config.latitude_col,
                config.longitude_col,
                config.description_col,
            ],
        )
    if input_format == "csv":
        return pd.read_csv(path)
    if input_format == "json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported input_format={input_format!r}")


def _prepare_location_dataframe(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    if "longtitude" in df.columns and config.longitude_col not in df.columns:
        df[config.longitude_col] = df["longtitude"]

    required = [
        config.id_col,
        config.latitude_col,
        config.longitude_col,
        config.description_col,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    df[config.latitude_col] = df[config.latitude_col].astype(float)
    df[config.longitude_col] = df[config.longitude_col].astype(float)
    df[config.description_col] = df[config.description_col].astype(str)
    return df.dropna(subset=required).reset_index(drop=True)


def _coordinate_bounds(
    df: pd.DataFrame,
    latitude_col: str,
    longitude_col: str,
) -> tuple[float, float, float, float]:
    return (
        float(df[latitude_col].min()),
        float(df[latitude_col].max()),
        float(df[longitude_col].min()),
        float(df[longitude_col].max()),
    )


def raw_coordinate_columns(latitude_col: str, longitude_col: str) -> tuple[str, str]:
    """Names of the columns holding the pre-normalization lat/lon (see `normalize_coordinates`)."""
    return f"{latitude_col}{RAW_COORDINATE_SUFFIX}", f"{longitude_col}{RAW_COORDINATE_SUFFIX}"


def raw_coordinates_from_record(
    record,
    latitude_col: str,
    longitude_col: str,
) -> tuple[float | None, float | None]:
    """Real-world degrees for one row, or `(None, None)` when the frame was never normalized."""
    raw_latitude_col, raw_longitude_col = raw_coordinate_columns(latitude_col, longitude_col)
    if raw_latitude_col not in record or raw_longitude_col not in record:
        return None, None
    return float(record[raw_latitude_col]), float(record[raw_longitude_col])


def normalize_coordinates(
    df: pd.DataFrame,
    latitude_col: str,
    longitude_col: str,
    *,
    bounds: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    # Keep the real-world degrees around: "HYBRID" writes them into the prompt text (unnormalized)
    # while feeding the normalized value to the location encoder. Inverting later would depend
    # on re-deriving the exact same bounds.
    raw_latitude_col, raw_longitude_col = raw_coordinate_columns(latitude_col, longitude_col)
    if raw_latitude_col not in df.columns:
        df[raw_latitude_col] = df[latitude_col]
    if raw_longitude_col not in df.columns:
        df[raw_longitude_col] = df[longitude_col]
    if bounds is None:
        min_lat, max_lat, min_lon, max_lon = _coordinate_bounds(df, latitude_col, longitude_col)
    else:
        min_lat, max_lat, min_lon, max_lon = bounds
    lon_range = max_lon - min_lon if max_lon > min_lon else 1.0
    lat_range = max_lat - min_lat if max_lat > min_lat else 1.0
    df[longitude_col] = (df[longitude_col] - min_lon) / lon_range * 2 - 1
    df[latitude_col] = (df[latitude_col] - min_lat) / lat_range * 2 - 1
    return df


def load_location_records_from_path(path: str | Path, config: DataConfig) -> pd.DataFrame:
    """Load and validate one location-records file without normalization."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = _read_raw_dataframe(path, config)
    return _prepare_location_dataframe(df, config)


def training_coordinate_bounds(config: DataConfig) -> tuple[float, float, float, float]:
    train_df = load_location_records_from_path(config.training_data, config)
    return _coordinate_bounds(train_df, config.latitude_col, config.longitude_col)


def _normalize_to_train_bounds(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    if not config.normalize_coordinates:
        return df
    bounds = training_coordinate_bounds(config)
    return normalize_coordinates(df, config.latitude_col, config.longitude_col, bounds=bounds)


def load_training_records(config: DataConfig) -> pd.DataFrame:
    df = load_location_records_from_path(config.training_data, config)
    if config.normalize_coordinates:
        bounds = _coordinate_bounds(df, config.latitude_col, config.longitude_col)
        df = normalize_coordinates(df, config.latitude_col, config.longitude_col, bounds=bounds)
    return df


def load_test_records(config: DataConfig) -> pd.DataFrame:
    df = load_location_records_from_path(config.test_data, config)
    return _normalize_to_train_bounds(df, config)


def load_coordinate_records(path: str | Path, config: DataConfig) -> pd.DataFrame:
    """Load a coordinate table and normalize it to training bounds when enabled.

    Used when `generation.use_test_coordinates` is a filesystem path rather than a bool.
    Full location files (id/lat/lon/description) go through the usual loader; a lat/lon-only
    table is accepted for generation queries. Normalization preserves `*_raw` degrees so
    HYBRID can write unnormalized text while the encoder still sees [-1, 1].
    """
    path = Path(path)
    try:
        df = load_location_records_from_path(path, config)
    except ValueError:
        df = _read_raw_dataframe(path, config)
        lat_col, lon_col = config.latitude_col, config.longitude_col
        missing = [col for col in (lat_col, lon_col) if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}") from None
        df = df.copy()
        df[lat_col] = df[lat_col].astype(float)
        df[lon_col] = df[lon_col].astype(float)
    return _normalize_to_train_bounds(df, config)


def load_generation_records(config: DataConfig, source: bool | str) -> pd.DataFrame:
    """Records for generate.py: True=test split, False=train+test, str=path."""
    if isinstance(source, str):
        return load_coordinate_records(source, config)
    if source:
        return load_test_records(config)
    return load_all_records(config)


def load_train_test_records(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-split training and test datasets."""
    train_df = load_location_records_from_path(config.training_data, config)
    test_df = load_location_records_from_path(config.test_data, config)
    if config.normalize_coordinates:
        bounds = _coordinate_bounds(train_df, config.latitude_col, config.longitude_col)
        train_df = normalize_coordinates(train_df, config.latitude_col, config.longitude_col, bounds=bounds)
        test_df = normalize_coordinates(test_df, config.latitude_col, config.longitude_col, bounds=bounds)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_all_records(config: DataConfig) -> pd.DataFrame:
    """Concatenate training and test records (e.g. for global grid bounds)."""
    train_df, test_df = load_train_test_records(config)
    return pd.concat([train_df, test_df], ignore_index=True)


def spatial_train_test_split(
    df: pd.DataFrame,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    num_axis: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by grid cells so nearby duplicated descriptions do not leak across sets."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if num_axis <= 0:
        raise ValueError("num_axis must be positive")

    total_cells = num_axis * num_axis
    n_test = max(1, int(total_cells * test_size))
    rng = np.random.RandomState(random_state)
    test_cells = rng.choice(np.arange(total_cells), size=n_test, replace=False)

    lat = df[latitude_col].to_numpy(dtype=np.float64)
    lon = df[longitude_col].to_numpy(dtype=np.float64)
    lat_range = lat.max() - lat.min() if lat.max() > lat.min() else 1.0
    lon_range = lon.max() - lon.min() if lon.max() > lon.min() else 1.0

    x_idx = np.floor((lon - lon.min()) / lon_range * num_axis).astype(int)
    y_idx = np.floor((lat - lat.min()) / lat_range * num_axis).astype(int)
    x_idx = np.clip(x_idx, 0, num_axis - 1)
    y_idx = np.clip(y_idx, 0, num_axis - 1)

    cell_ids = y_idx * num_axis + x_idx
    mask_test = np.isin(cell_ids, test_cells)
    train_df = df.loc[~mask_test].reset_index(drop=True)
    test_df = df.loc[mask_test].reset_index(drop=True)
    return move_overlapping_coordinates_to_train(train_df, test_df, latitude_col, longitude_col)


def grid_num_cells(num_axis: int, reserve_padding_index: bool = True) -> int:
    total = num_axis * num_axis
    return total + 1 if reserve_padding_index else total


def lat_lon_to_grid_cell(
    latitude: float,
    longitude: float,
    *,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    num_axis: int,
) -> int:
    """Map normalized coordinates to a flat grid cell id in [0, num_axis * num_axis)."""
    if num_axis <= 0:
        raise ValueError("num_axis must be positive")
    lon_range = max_lon - min_lon if max_lon > min_lon else 1.0
    lat_range = max_lat - min_lat if max_lat > min_lat else 1.0
    x_idx = int(np.floor((longitude - min_lon) / lon_range * num_axis))
    y_idx = int(np.floor((latitude - min_lat) / lat_range * num_axis))
    x_idx = int(np.clip(x_idx, 0, num_axis - 1))
    y_idx = int(np.clip(y_idx, 0, num_axis - 1))
    return y_idx * num_axis + x_idx


def grid_bounds_from_records(
    records: pd.DataFrame,
    latitude_col: str,
    longitude_col: str,
) -> tuple[float, float, float, float]:
    return (
        float(records[latitude_col].min()),
        float(records[latitude_col].max()),
        float(records[longitude_col].min()),
        float(records[longitude_col].max()),
    )


def move_overlapping_coordinates_to_train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    latitude_col: str,
    longitude_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_coords = set(zip(train_df[latitude_col].values, train_df[longitude_col].values))
    overlap_mask = test_df.apply(
        lambda row: (row[latitude_col], row[longitude_col]) in train_coords,
        axis=1,
    )
    if not overlap_mask.any():
        return train_df, test_df
    return (
        pd.concat([train_df, test_df[overlap_mask]], ignore_index=True),
        test_df.loc[~overlap_mask].reset_index(drop=True),
    )


class LocationDescriptionDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        latitude_col: str = "latitude",
        longitude_col: str = "longitude",
        description_col: str = "description",
        id_col: str = "id",
        num_samples: int | None = None,
        random_sample: bool = True,
        spatial_division_method: str | None = None,
        grid_num_axis: int = 100,
        geohash_length: int = DEFAULT_GEOHASH_LENGTH,
        reasoning: ReasoningConfig | None = None,
    ):
        self.records = records.reset_index(drop=True)
        self.latitude_col = latitude_col
        self.longitude_col = longitude_col
        self.description_col = description_col
        self.id_col = id_col
        self.num_samples = num_samples if num_samples is not None else len(self.records)
        self.random_sample = random_sample
        self.spatial_division_method = spatial_division_method
        self.grid_num_axis = grid_num_axis
        self.geohash_length = geohash_length
        self.reasoning = reasoning

        #print('Using random')

        if self.spatial_division_method == "HYBRID":
            raw_lat_col, raw_lon_col = raw_coordinate_columns(self.latitude_col, self.longitude_col)
            if raw_lat_col not in self.records.columns or raw_lon_col not in self.records.columns:
                raise ValueError(
                    "spatial_division_method='HYBRID' requires preserved raw coordinate columns "
                    f"({raw_lat_col!r}, {raw_lon_col!r}) for the prompt text, while "
                    f"{self.latitude_col!r}/{self.longitude_col!r} hold the [-1, 1] values "
                    "passed to the location encoder. Load data with normalize_coordinates=True."
                )

        if self.spatial_division_method == "grid":
            self.num_axis = grid_num_axis
            self.min_y, self.max_y, self.min_x, self.max_x = grid_bounds_from_records(
                self.records,
                self.latitude_col,
                self.longitude_col,
            )
            self.num_cells = grid_num_cells(self.num_axis)

        self._sibling_indices_by_id: dict[Any, np.ndarray] | None = None
        if self.reasoning is not None and self.reasoning.enabled:
            if self.id_col not in self.records.columns:
                raise ValueError(
                    f"reasoning.enabled requires an {self.id_col!r} column in the training data."
                )
            self._sibling_indices_by_id = self.records.groupby(self.id_col).indices

        print('sample data')
        print(self[0])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.random_sample:
            idx = random.randint(0, len(self.records) - 1)
        else:
            idx = idx % len(self.records)
        record = self.records.iloc[idx]
        latitude = float(record[self.latitude_col])
        longitude = float(record[self.longitude_col])
        
        # # randomly assign a description to the record
        # idx_random = random.randint(0, len(self.records) - 1)
        # latitude = float(1) #float(self.records.iloc[idx_random][self.latitude_col])
        # longitude = float(1) #float(self.records.iloc[idx_random][self.longitude_col])

        grid_bounds = None
        if self.spatial_division_method == "grid":
            grid_bounds = (self.min_y, self.max_y, self.min_x, self.max_x)
        text_latitude, text_longitude = raw_coordinates_from_record(
            record,
            self.latitude_col,
            self.longitude_col,
        )
        sample = build_geo_sample(
            latitude,
            longitude,
            spatial_division_method=self.spatial_division_method,
            grid_bounds=grid_bounds,
            grid_num_axis=self.num_axis if self.spatial_division_method == "grid" else self.grid_num_axis,
            text_latitude=text_latitude,
            text_longitude=text_longitude,
            geohash_length=self.geohash_length,
        )
        description = str(record[self.description_col])
        return {
            **sample,
            "labels": self._build_label(idx, record, description),
        }

    def _build_label(self, idx: int, record: pd.Series, description: str) -> str:
        if self._sibling_indices_by_id is None:
            return description
        location_id = record[self.id_col]
        sibling_positions = self._sibling_indices_by_id.get(location_id, [])
        other_descriptions = [
            str(self.records.iloc[pos][self.description_col])
            for pos in sibling_positions
            if pos != idx
        ]
        if self.reasoning.shuffle_thinking_descriptions:
            random.shuffle(other_descriptions)
        if self.reasoning.max_thinking_descriptions is not None:
            other_descriptions = other_descriptions[: self.reasoning.max_thinking_descriptions]
        return build_reasoning_label(
            other_descriptions,
            description,
            open_tag=self.reasoning.thinking_open_tag or DEFAULT_THINKING_OPEN_TAG,
            close_tag=self.reasoning.thinking_close_tag or DEFAULT_THINKING_CLOSE_TAG,
            join_str=self.reasoning.thinking_join,
            answer_prefix=self.reasoning.answer_prefix,
        )


class CoordinateGenerationDataset(Dataset):
    def __init__(
        self,
        coordinates: np.ndarray | torch.Tensor,
        spatial_division_method: str | None = None,
        grid_bounds: tuple[float, float, float, float] | None = None,
        grid_num_axis: int = 100,
        geohash_length: int = DEFAULT_GEOHASH_LENGTH,
        text_coordinates: np.ndarray | torch.Tensor | None = None,
    ):
        self.coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
        self.spatial_division_method = spatial_division_method
        self.grid_bounds = grid_bounds
        self.grid_num_axis = grid_num_axis
        self.geohash_length = geohash_length
        # Real-world degrees for the prompt text, row-aligned with `coordinates` (see
        # `unique_coordinate_tensors`). Only "DIRECT"/"HYBRID"/"geohash" prompts read them.
        self.text_coordinates = (
            None if text_coordinates is None else torch.as_tensor(text_coordinates, dtype=torch.float64)
        )
        if self.spatial_division_method == "HYBRID" and self.text_coordinates is None:
            raise ValueError(
                "spatial_division_method='HYBRID' requires text_coordinates (unnormalized "
                "degrees for the prompt); `coordinates` should be the normalized values "
                "passed to the location encoder."
            )
        if self.text_coordinates is not None and self.text_coordinates.shape[0] != self.coordinates.shape[0]:
            raise ValueError(
                f"text_coordinates has {self.text_coordinates.shape[0]} rows but coordinates has "
                f"{self.coordinates.shape[0]}; they must be row-aligned."
            )

    def __len__(self) -> int:
        return self.coordinates.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        latitude = float(self.coordinates[idx, 0])
        longitude = float(self.coordinates[idx, 1])
        text_latitude = None
        text_longitude = None
        if self.text_coordinates is not None:
            text_latitude = float(self.text_coordinates[idx, 0])
            text_longitude = float(self.text_coordinates[idx, 1])
        sample = build_geo_sample(
            latitude,
            longitude,
            spatial_division_method=self.spatial_division_method,
            grid_bounds=self.grid_bounds,
            grid_num_axis=self.grid_num_axis,
            text_latitude=text_latitude,
            text_longitude=text_longitude,
            geohash_length=self.geohash_length,
        )
        return {
            **sample,
            "coordinate": coordinate_pair_as_list(latitude, longitude),
            "labels": "",
        }


@contextlib.contextmanager
def tokenizer_padding_side(tokenizer, side: str) -> Iterator[None]:
    previous = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = side
    try:
        yield
    finally:
        tokenizer.padding_side = previous


def tokenizer_uses_gpt_oss_harmony(tokenizer) -> bool:
    """True for GPT-OSS Harmony tokenizers (`<|channel|>` / `<|return|>` special tokens)."""
    unk = getattr(tokenizer, "unk_token_id", None)
    channel_id = tokenizer.convert_tokens_to_ids("<|channel|>")
    return_id = tokenizer.convert_tokens_to_ids("<|return|>")
    if channel_id is None or return_id is None:
        return False
    if unk is not None and (channel_id == unk or return_id == unk):
        return False
    return True


def resolve_inference_chat_options(
    tokenizer,
    spatial_division_method: str | None = None,
) -> dict[str, Any]:
    """Inference-only chat-template options for GPT-OSS; empty for other models.

    GPT-OSS defaults to medium reasoning and often emits a long
    ``<|channel|>analysis|>`` CoT before the answer. That dominates wall-clock
    for DIRECT/geohash (text location codes). SLM/grid/sphere/HYBRID finetunes already emit the
    description right after ``<|start|>assistant``, so prefilling the final
    channel makes them early-EOS with empty text — only DIRECT/geohash get that prefill.
    HYBRID has textual coordinates too, but it is trained with an injected geo
    embedding like SLM, so it follows the SLM behavior here, not DIRECT's.
    Training must keep the defaults (no reasoning_effort / prefer_final_channel).
    """
    if not tokenizer_uses_gpt_oss_harmony(tokenizer):
        return {}
    options: dict[str, Any] = {"reasoning_effort": "low"}
    if spatial_division_method in ("DIRECT", "geohash"):
        options["prefer_final_channel"] = True
    return options


def apply_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
    reasoning_effort: str | None = None,
    prefer_final_channel: bool = False,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    def _call(**call_kwargs) -> str:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **call_kwargs)
        except TypeError:
            # Drop kwargs the template/tokenizer does not accept (e.g. enable_thinking,
            # reasoning_effort) and retry with the remainder.
            supported = dict(call_kwargs)
            for key in ("enable_thinking", "reasoning_effort"):
                supported.pop(key, None)
            try:
                return tokenizer.apply_chat_template(
                    messages, enable_thinking=enable_thinking, **supported
                )
            except TypeError:
                return tokenizer.apply_chat_template(messages, **supported)

    text = _call(**kwargs)
    # Prefill Harmony final channel so generation skips the analysis CoT. Only valid on
    # an open assistant turn (inference / training prompt), never on a completed reply.
    if prefer_final_channel and add_generation_prompt and text.endswith("<|start|>assistant"):
        text += "<|channel|>final<|message|>"
    return text


def _tokenize_ids(tokenizer, text: str) -> list[int]:
    """Token ids for a chat-template string, with no extra BOS/EOS wrapping."""
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = None
        if ids is not None:
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if not ids or isinstance(ids[0], int):
                return list(ids)
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def chat_completion_suffix(tokenizer, prompt_text: str, full_text: str) -> str:
    """Assistant continuation: tokens in `full_text` that are not part of the inference prompt.

    When the completed-turn template is a string prefix of the generation prompt (Qwen, GPT-OSS),
    this is a simple slice. Templates that inject extra generation-only scaffolding (Gemma's empty
    thought channel) diverge before the reply; fall back to the token-level remainder of `full_text`
    after the shared prefix so the label still contains the template's end-of-turn token.
    """
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :]
    prompt_ids = _tokenize_ids(tokenizer, prompt_text)
    full_ids = _tokenize_ids(tokenizer, full_text)
    n = 0
    n_max = min(len(prompt_ids), len(full_ids))
    while n < n_max and prompt_ids[n] == full_ids[n]:
        n += 1
    if n >= len(full_ids):
        return ""
    return tokenizer.decode(full_ids[n:], skip_special_tokens=False)


def render_chat_prompt_and_completion(
    tokenizer,
    *,
    system_prompt: str,
    user_prompt: str,
    assistant_content: str = "",
    reasoning_effort: str | None = None,
    prefer_final_channel: bool = False,
) -> tuple[str, str]:
    """Render the inference prefix and the supervised assistant suffix.

    The prompt is `apply_chat_template(..., add_generation_prompt=True)`, matching `generate()`.
    The completion is the remainder of the same template with the assistant message included
    (`add_generation_prompt=False`), so closers (`<|im_end|>`, `<turn|>`, Harmony `<|return|>`)
    come from the chat template instead of a hand-appended `tokenizer.eos_token`.
    """
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = apply_chat_template(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
        prefer_final_channel=prefer_final_channel,
    )
    if not assistant_content:
        return prompt_text, ""
    full_text = apply_chat_template(
        tokenizer,
        prompt_messages + [{"role": "assistant", "content": assistant_content}],
        add_generation_prompt=False,
        reasoning_effort=reasoning_effort,
        prefer_final_channel=False,
    )
    return prompt_text, chat_completion_suffix(tokenizer, prompt_text, full_text)


def decode_generated_completion(tokenizer, token_ids) -> str:
    """Decode a `generate()` continuation into the assistant text we score/save.

    Harmony channel names (`final`, …) sit between special tokens, so
    `skip_special_tokens=True` would leave them in the string. Prefer the last
    `<|message|>` body when the tokenizer is GPT-OSS.
    """
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    if tokenizer_uses_gpt_oss_harmony(tokenizer):
        raw = tokenizer.decode(token_ids, skip_special_tokens=False)
        if "<|message|>" in raw:
            raw = raw.split("<|message|>")[-1]
        for stop in ("<|return|>", "<|end|>", "<|call|>", "<|endoftext|>"):
            raw = raw.split(stop)[0]
        return raw.strip()
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text.replace("assistant\n", "").strip()


def decode_generated_completions(tokenizer, outputs) -> list[str]:
    return [decode_generated_completion(tokenizer, row) for row in outputs]


def collate_geo_chat_template(
    batch: list[dict[str, Any]],
    tokenizer,
    system_prompt: str = SYSTEM_PROMPT,
    prompt_padding_side: str = "right",
    reasoning_effort: str | None = None,
    prefer_final_channel: bool = False,
) -> dict[str, Any]:
    """Collate prompts/labels for geo2llm.

    The prompt is the chat-template generation prefix (same string `generate()` sees).
    Labels are the remainder of the *completed* chat-template turn — the assistant body
    plus the template's end token — so loss is completion-only. Generation paths pass
    empty `labels` and ignore that tensor.

    `prompt_padding_side` defaults to ``"right"`` for training (teacher forcing).
    Use ``"left"`` for batched decoder-only generation so decode starts from the
    real last prompt token rather than pad embeddings.

    `reasoning_effort` / `prefer_final_channel` are inference-only GPT-OSS knobs;
    leave them at defaults during training.
    """
    max_geo_len = max(item["geo_feature"].size(0) for item in batch)
    geo_dim = 2
    geo_dtype = torch.float32
    for item in batch:
        if item["geo_feature"].numel() > 0:
            geo_dim = item["geo_feature"].size(1)
            geo_dtype = item["geo_feature"].dtype
            break
    geos = torch.zeros(
        len(batch),
        max_geo_len,
        geo_dim,
        dtype=geo_dtype,
    )
    geo_mask = torch.zeros(len(batch), geos.size(1), dtype=torch.bool)
    for i, item in enumerate(batch):
        geos[i, : item["geo_feature"].size(0), :] = item["geo_feature"]
        geo_mask[i, : item["geo_feature"].size(0)] = True

    coordinates = None
    if "coordinate" in batch[0]:
        coordinates = [item["coordinate"] for item in batch]

    prompt_texts: list[str] = []
    labels_texts: list[str] = []
    for item in batch:
        prompt_text, completion_text = render_chat_prompt_and_completion(
            tokenizer,
            system_prompt=system_prompt,
            user_prompt=item["prompt"],
            assistant_content=str(item.get("labels") or ""),
            reasoning_effort=reasoning_effort,
            prefer_final_channel=prefer_final_channel,
        )
        prompt_texts.append(prompt_text)
        labels_texts.append(completion_text)

    with tokenizer_padding_side(tokenizer, prompt_padding_side):
        prompt_enc = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    # Labels stay right-padded for the training concat(prompt, labels) layout.
    with tokenizer_padding_side(tokenizer, "right"):
        if any(labels_texts):
            labels_enc = tokenizer(
                labels_texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        else:
            labels_enc = {
                "input_ids": torch.zeros(len(batch), 0, dtype=torch.long),
                "attention_mask": torch.zeros(len(batch), 0, dtype=torch.long),
            }

    result = {
        "prompt_input_ids": prompt_enc["input_ids"],
        "prompt_texts": prompt_texts,
        "prompt_attention_mask": prompt_enc["attention_mask"],
        "geo_feature": geos,
        "geo_attention_mask": geo_mask,
        "labels": labels_enc["input_ids"],
        "labels_texts": labels_texts,
        "labels_attention_mask": labels_enc["attention_mask"],
    }
    if coordinates is not None:
        result["coordinates"] = coordinates
    return result
