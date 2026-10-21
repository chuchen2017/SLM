"""City reasoning data and the fixed benchmarks."""

from __future__ import annotations

import json
import logging
import pickle
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from slm.answers import format_distance_trace, format_trace
from slm.chat import render_chat_prompt_and_completion, tokenizer_padding_side
from slm.config import DataConfig, default_tasks_for_city
from slm.geometry import LINE_TYPES, POLYGON_TYPES, format_measure, topological_relation
from slm.prompts import (
    GEO_TOKEN,
    LOCATION_PHRASES,
    SYSTEM_PROMPT,
    questions_with_entity_count,
    rewrite_location_slots,
    strip_location_tags,
    unique_location_ids,
)

logger = logging.getLogger(__name__)

TASK_ENTITY_COUNTS = {
    "length": 1,
    "area": 1,
    "distance": 2,
    "topology": 2,
    "nearest": 5,
}
QUESTION_FILES = {
    "length": "polyline_questions.json",
    "area": "polygon_questions.json",
    "distance": "distance_questions.json",
    "topology": "topo_questions.json",
}


def system_prompt_for(config: DataConfig) -> str:
    if config.system_prompt:
        return config.system_prompt
    return SYSTEM_PROMPT


def task_decimals(config: DataConfig, task: str) -> int:
    if task == "length":
        return config.length_decimals
    if task == "area":
        return config.area_decimals
    if task == "distance":
        return config.distance_decimals
    return 0


class SplitGeometries:
    """One city split: geometries aligned with Geo2Vec rows."""

    def __init__(self, frame, embeddings: np.ndarray):
        self.geometries = list(frame.geometry)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings of shape (N, D), got {self.embeddings.shape}.")
        if len(self.geometries) != self.embeddings.shape[0]:
            raise ValueError(
                f"{len(self.geometries)} geometries but {self.embeddings.shape[0]} embedding rows."
            )
        self.names = _name_column(frame)
        by_type: dict[str, list[int]] = {}
        for index, geometry in enumerate(self.geometries):
            by_type.setdefault(geometry.geom_type, []).append(index)
        self.by_type = {key: np.asarray(value, dtype=np.int64) for key, value in by_type.items()}
        self.line_indices = _concat_indices(self.by_type, LINE_TYPES)
        self.polygon_indices = _concat_indices(self.by_type, POLYGON_TYPES)
        self.topology_pairs: list[tuple[int, int, str]] = []

    def __len__(self) -> int:
        return len(self.geometries)

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])


def _concat_indices(by_type: dict[str, np.ndarray], types: set[str]) -> np.ndarray:
    parts = [by_type[name] for name in types if name in by_type]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(parts)


def _name_column(frame) -> list[str]:
    for column in ("name", "NAME", "Name"):
        if column in frame.columns:
            return ["" if value is None else str(value) for value in frame[column].tolist()]
    return [""] * len(frame)


def _read_geodata(path: Path):
    import geopandas as gpd

    frame = gpd.read_file(path)
    return frame.reset_index(drop=True)


def load_aligned_embeddings(path: Path, count: int) -> np.ndarray:
    """Load a Geo2Vec table and drop the unused padding rows the encoder appends."""
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"{path} has shape {array.shape}; expected (N, D).")
    if array.shape[0] < count:
        raise ValueError(f"{path} has {array.shape[0]} rows for {count} geometries.")
    if array.shape[0] > count:
        logger.info(
            "Using the first %d of %d rows in %s. Geo2Vec keeps unused embedding slots.",
            count,
            array.shape[0],
            path.name,
        )
        array = array[:count]
    return np.asarray(array, dtype=np.float32)


def load_split(geometry_path: Path, embedding_path: Path) -> SplitGeometries:
    if not geometry_path.is_file():
        raise FileNotFoundError(f"Geometry file not found: {geometry_path}")
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    frame = _read_geodata(geometry_path)
    embeddings = load_aligned_embeddings(embedding_path, len(frame))
    return SplitGeometries(frame, embeddings)


def _topology_pairs_for_split(all_frame, split: SplitGeometries, relationships: dict) -> list[tuple[int, int, str]]:
    if "GEOID" not in all_frame.columns:
        raise ValueError("Topology relationships need a GEOID column on all_osm_with_names.gpkg.")
    wkb_to_all = {geometry.wkb: index for index, geometry in enumerate(all_frame.geometry)}
    all_geoids = all_frame["GEOID"].astype(str).tolist()
    geoid_to_split: dict[str, int] = {}
    for split_index, geometry in enumerate(split.geometries):
        all_index = wkb_to_all.get(geometry.wkb)
        if all_index is None:
            continue
        geoid_to_split[all_geoids[all_index]] = split_index
    pairs: list[tuple[int, int, str]] = []
    for relation, items in relationships.items():
        name = str(relation).strip().lower()
        for left, right in items:
            left_index = geoid_to_split.get(str(left))
            right_index = geoid_to_split.get(str(right))
            if left_index is None or right_index is None:
                continue
            pairs.append((left_index, right_index, name))
    return pairs


def attach_topology(city_dir: Path, train: SplitGeometries, test: SplitGeometries) -> None:
    relation_path = city_dir / "topo_relationships.json"
    layer_path = city_dir / "all_osm_with_names.gpkg"
    if not relation_path.is_file():
        return
    if not layer_path.is_file():
        logger.warning("Found %s but not %s; topology pairs will be computed on the fly.", relation_path, layer_path)
        return
    relationships = json.loads(relation_path.read_text(encoding="utf-8"))
    all_frame = _read_geodata(layer_path)
    train.topology_pairs = _topology_pairs_for_split(all_frame, train, relationships)
    test.topology_pairs = _topology_pairs_for_split(all_frame, test, relationships)
    logger.info(
        "Topology pairs in %s: train=%d test=%d",
        city_dir.name,
        len(train.topology_pairs),
        len(test.topology_pairs),
    )


def load_city(config: DataConfig) -> tuple[SplitGeometries, SplitGeometries]:
    city_dir = Path(config.osm_dir) / config.city
    train = load_split(city_dir / config.train_geometry, city_dir / config.train_embeddings)
    test = load_split(city_dir / config.test_geometry, city_dir / config.test_embeddings)
    if train.embedding_dim != test.embedding_dim:
        raise ValueError(
            f"{config.city} train embedding dim {train.embedding_dim} != test dim {test.embedding_dim}."
        )
    attach_topology(city_dir, train, test)
    logger.info(
        "%s geometries: train=%d test=%d embedding_dim=%d",
        config.city,
        len(train),
        len(test),
        train.embedding_dim,
    )
    return train, test


def load_task_questions(config: DataConfig) -> dict[str, list[str]]:
    questions: dict[str, list[str]] = {}
    question_dir = Path(config.questions_dir)
    for task in config.tasks or []:
        if task == "nearest":
            continue
        filename = QUESTION_FILES.get(task)
        if filename is None:
            raise ValueError(f"Unknown task {task!r}. Expected one of {sorted(TASK_ENTITY_COUNTS)}.")
        path = question_dir / filename
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError(f"{path} should be a JSON list of question strings.")
        kept = questions_with_entity_count(raw, TASK_ENTITY_COUNTS[task])
        if not kept:
            raise ValueError(f"No {task} questions with {TASK_ENTITY_COUNTS[task]} entities in {path}.")
        questions[task] = kept
        logger.info("Loaded %d %s questions from %s (%d raw).", len(kept), task, path.name, len(raw))
    return questions


def _choice_index(rng, pool: np.ndarray) -> int:
    if len(pool) == 0:
        raise ValueError("Cannot sample from an empty geometry pool.")
    return int(pool[rng.randrange(len(pool))])


def _sample_distinct(rng, pool: np.ndarray, count: int) -> list[int]:
    if len(pool) < count:
        raise ValueError(f"Need {count} geometries but the pool has {len(pool)}.")
    picked = rng.sample(pool.tolist(), count)
    return [int(value) for value in picked]


class SpatialReasoningDataset(Dataset):
    """Random spatial questions whose `<GEO>` tokens line up with entity rows."""

    def __init__(
        self,
        split: SplitGeometries,
        questions: list[str],
        task: str,
        *,
        num_samples: int,
        decimals: int = 0,
        distance_geometry_pair: list[str] | None = None,
        phrases: list[str] | None = None,
        random_sample: bool = True,
        seed: int = 0,
    ):
        if task == "nearest":
            raise ValueError("Nearest-neighbor questions use DistanceTraceDataset.")
        if task not in TASK_ENTITY_COUNTS:
            raise ValueError(f"Unknown task {task!r}.")
        if not questions:
            raise ValueError(f"No questions for task {task}.")
        self.split = split
        self.questions = list(questions)
        self.task = task
        self.num_samples = num_samples
        self.decimals = decimals
        self.distance_geometry_pair = distance_geometry_pair
        self.phrases = list(phrases or LOCATION_PHRASES)
        self.random_sample = random_sample
        self._validate_pools()
        self._fixed: list[dict[str, Any]] | None = None
        if not random_sample:
            rng = random.Random(seed)
            self._fixed = [self._example(rng) for _ in range(num_samples)]

    def _validate_pools(self) -> None:
        if self.task == "length" and len(self.split.line_indices) == 0:
            raise ValueError("The length task needs LineString geometries. Remove it for this city.")
        if self.task == "area" and len(self.split.polygon_indices) == 0:
            raise ValueError("The area task needs polygon geometries.")
        if self.task == "nearest" and len(self.split) < TASK_ENTITY_COUNTS["nearest"]:
            raise ValueError("The nearest-neighbor task needs at least 5 geometries.")
        if self.task in {"distance", "topology"} and len(self.split) < 2 and not self.split.topology_pairs:
            raise ValueError(f"The {self.task} task needs at least two geometries.")
        if self.task == "topology" and not self.split.topology_pairs:
            logger.info("No precomputed topology pairs; relations will be computed from sampled pairs.")
        if self.distance_geometry_pair and self.task == "distance":
            for geom_type in self.distance_geometry_pair:
                if geom_type not in self.split.by_type:
                    raise ValueError(f"distance_geometry_pair type {geom_type!r} is not in this split.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._fixed is not None:
            return self._fixed[index % len(self._fixed)]
        return self._example(random)

    def _example(self, rng) -> dict[str, Any]:
        question = rng.choice(self.questions)
        slot_count = len(unique_location_ids(question))
        slot_to_index, relation = self._sample_slots(rng, slot_count)

        def entity_name(slot: int) -> str:
            name = self.split.names[slot_to_index[slot]].strip()
            return name or f"entity {slot}"

        prompt, sequence = rewrite_location_slots(
            question,
            lambda: rng.choice(self.phrases),
            entity_name,
        )
        rows = np.stack([self.split.embeddings[slot_to_index[slot]] for slot in sequence], axis=0)
        answer, query_slot = self._answer(slot_to_index, relation)
        return {
            "prompt": prompt,
            "geo_feature": torch.from_numpy(np.ascontiguousarray(rows)),
            "labels": format_trace(self.task, answer, query_slot=query_slot),
            "answer": answer,
            "task": self.task,
        }

    def _sample_slots(self, rng, slot_count: int) -> tuple[dict[int, int], str | None]:
        if self.task == "length":
            return {0: _choice_index(rng, self.split.line_indices)}, None
        if self.task == "area":
            return {0: _choice_index(rng, self.split.polygon_indices)}, None
        if self.task == "topology" and self.split.topology_pairs:
            left, right, relation = rng.choice(self.split.topology_pairs)
            return {0: int(left), 1: int(right)}, relation
        if self.task == "distance" and self.distance_geometry_pair:
            return self._sample_type_pair(rng), None
        pool = np.arange(len(self.split), dtype=np.int64)
        chosen = _sample_distinct(rng, pool, slot_count)
        return {slot: index for slot, index in enumerate(chosen)}, None

    def _sample_type_pair(self, rng) -> dict[int, int]:
        first_type, second_type = self.distance_geometry_pair
        if first_type == second_type:
            chosen = _sample_distinct(rng, self.split.by_type[first_type], 2)
            return {0: chosen[0], 1: chosen[1]}
        left = _choice_index(rng, self.split.by_type[first_type])
        right = _choice_index(rng, self.split.by_type[second_type])
        if rng.random() < 0.5:
            left, right = right, left
        return {0: left, 1: right}

    def _answer(self, slot_to_index: dict[int, int], relation: str | None) -> tuple[str, int | None]:
        geometries = self.split.geometries
        if self.task == "length":
            return format_measure(geometries[slot_to_index[0]].length, self.decimals), None
        if self.task == "area":
            return format_measure(geometries[slot_to_index[0]].area, self.decimals), None
        if self.task == "distance":
            distance = geometries[slot_to_index[0]].distance(geometries[slot_to_index[1]])
            return format_measure(distance, self.decimals), None
        if self.task == "topology":
            if relation is None:
                relation = topological_relation(geometries[slot_to_index[0]], geometries[slot_to_index[1]])
            return relation, None
        raise ValueError(f"No answer rule for task {self.task!r}.")


class ThinkingDataset(Dataset):
    """Text supervision from the thinking-trace JSON. Those rows have no geometries."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        num_samples: int,
        embedding_dim: int,
        open_tag: str,
        close_tag: str,
        random_sample: bool = True,
    ):
        if not records:
            raise ValueError("Thinking dataset is empty.")
        self.records = records
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim
        self.open_tag = open_tag
        self.close_tag = close_tag
        self.random_sample = random_sample

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index % len(self.records)] if not self.random_sample else random.choice(self.records)
        steps = row.get("thinking") or []
        if isinstance(steps, list):
            trace = " ".join(str(step) for step in steps)
        else:
            trace = str(steps)
        answer = str(row.get("final_answer", "")).strip()
        label = f"{self.open_tag} {trace} {self.close_tag} {answer}".strip()
        return {
            "prompt": strip_location_tags(str(row["question"])),
            "geo_feature": torch.zeros(0, self.embedding_dim, dtype=torch.float32),
            "labels": label,
            "answer": answer,
            "task": "thinking",
        }


def load_thinking_records(config: DataConfig) -> list[dict[str, Any]]:
    path = Path(config.questions_dir) / "triplet_distance_QA_thinking_steps_total.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} should be a JSON list.")
    return records


class DistanceTemplate:
    """One nearest-neighbor question and the distance calls in its thinking trace."""

    def __init__(self, question: str, pairs: list[tuple[int, int]], n_entities: int):
        self.question = question
        self.pairs = pairs
        self.n_entities = n_entities


def _distance_pair(step: dict[str, Any]) -> tuple[int, int] | None:
    if step.get("name") != "distance_tool":
        return None
    arguments = step.get("arguments") or {}
    left = re.findall(r"(\d+)", str(arguments.get("location_1", "")))
    right = re.findall(r"(\d+)", str(arguments.get("location_2", "")))
    if not left or not right:
        return None
    first, second = sorted((int(left[0]), int(right[0])))
    return first, second


def load_distance_templates(path: Path) -> list[DistanceTemplate]:
    """Question wordings whose labels are a sequence of distance calculations."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    templates: list[DistanceTemplate] = []
    for item in records:
        question = item.get("question") if isinstance(item, dict) else None
        steps = item.get("thinking") if isinstance(item, dict) else None
        if not isinstance(question, str) or not isinstance(steps, list):
            continue
        ids = {int(value) for value in re.findall(r"location_(\d+)", question)}
        if not ids or ids != set(range(max(ids) + 1)):
            continue
        pairs: list[tuple[int, int]] = []
        for step in steps:
            if not isinstance(step, str) or not step.strip():
                continue
            try:
                parsed = json.loads(step)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            pair = _distance_pair(parsed)
            if pair is not None:
                pairs.append(pair)
        if not pairs:
            continue
        used = {entity for pair in pairs for entity in pair}
        if not used <= ids:
            continue
        templates.append(DistanceTemplate(question, pairs, max(ids) + 1))
    if not templates:
        raise ValueError(f"No distance-trace questions in {path}.")
    logger.info("Loaded %d nearest-neighbor question templates from %s.", len(templates), Path(path).name)
    return templates


class DistanceTraceDataset(Dataset):
    """Nearest-neighbor queries sampled on the fly from geometries and question templates.

    Each item draws a new template and a new set of entities. The label is the
    notebook trace ``Distance( <i> , <j> ) = ...`` for every thinking step, not a
    single neighbor index.
    """

    def __init__(
        self,
        split: SplitGeometries,
        templates: list[DistanceTemplate],
        *,
        num_samples: int,
        decimals: int = 0,
        phrases: list[str] | None = None,
        random_sample: bool = True,
        seed: int = 0,
    ):
        if not templates:
            raise ValueError("Nearest-neighbor training needs question templates.")
        widest = max(template.n_entities for template in templates)
        if len(split) < widest:
            raise ValueError(f"Need at least {widest} geometries for nearest-neighbor questions.")
        self.split = split
        self.templates = list(templates)
        self.num_samples = num_samples
        self.decimals = decimals
        self.phrases = list(phrases or LOCATION_PHRASES)
        self.random_sample = random_sample
        self._fixed: list[dict[str, Any]] | None = None
        if not random_sample:
            rng = random.Random(seed)
            self._fixed = [self._example(rng) for _ in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._fixed is not None:
            return self._fixed[index % len(self._fixed)]
        return self._example(random)

    def _example(self, rng) -> dict[str, Any]:
        template = rng.choice(self.templates)
        chosen = _sample_distinct(rng, np.arange(len(self.split), dtype=np.int64), template.n_entities)

        def entity_name(slot: int) -> str:
            name = self.split.names[chosen[slot]].strip()
            return name or f"entity {slot}"

        prompt, sequence = rewrite_location_slots(
            template.question,
            lambda: rng.choice(self.phrases),
            entity_name,
        )
        rows = np.stack([self.split.embeddings[chosen[slot]] for slot in sequence], axis=0)
        measurements = []
        for left, right in template.pairs:
            distance = self.split.geometries[chosen[left]].distance(self.split.geometries[chosen[right]])
            measurements.append((left, right, format_measure(distance, self.decimals)))
        return {
            "prompt": prompt,
            "geo_feature": torch.from_numpy(np.ascontiguousarray(rows)),
            "labels": format_distance_trace(measurements),
            "answer": format_distance_trace(measurements),
            "task": "nearest",
        }


def build_geo_datasets(
    config: DataConfig,
    *,
    thinking_tags: tuple[str, str] = ("<think>", "</think>"),
    build_test: bool = False,
) -> tuple[Dataset, Dataset | None]:
    tasks = list(config.tasks or default_tasks_for_city(config.city))
    config.tasks = tasks
    questions = load_task_questions(config)
    train_split, test_split = load_city(config)
    distance_templates = (
        load_distance_templates(Path(config.questions_dir) / "triplet_distance_QA_thinking_steps_total.json")
        if "nearest" in tasks
        else []
    )
    train_parts: list[Dataset] = []
    test_parts: list[Dataset] = []
    for task_index, task in enumerate(tasks):
        if task == "nearest":
            train_parts.append(
                DistanceTraceDataset(
                    train_split,
                    distance_templates,
                    num_samples=config.samples_per_task,
                    decimals=task_decimals(config, "distance"),
                    random_sample=True,
                    seed=task_index,
                )
            )
            if build_test:
                test_parts.append(
                    DistanceTraceDataset(
                        test_split,
                        distance_templates,
                        num_samples=config.samples_per_task,
                        decimals=task_decimals(config, "distance"),
                        random_sample=True,
                        seed=10_000 + task_index,
                    )
                )
            continue
        common = {
            "questions": questions[task],
            "task": task,
            "num_samples": config.samples_per_task,
            "decimals": task_decimals(config, task),
            "distance_geometry_pair": config.distance_geometry_pair if task == "distance" else None,
        }
        train_parts.append(SpatialReasoningDataset(train_split, random_sample=True, seed=task_index, **common))
        if build_test:
            test_parts.append(
                SpatialReasoningDataset(test_split, random_sample=False, seed=10_000 + task_index, **common)
            )
    if config.include_thinking:
        records = load_thinking_records(config)
        open_tag, close_tag = thinking_tags
        train_parts.append(
            ThinkingDataset(
                records,
                num_samples=config.thinking_samples,
                embedding_dim=train_split.embedding_dim,
                open_tag=open_tag,
                close_tag=close_tag,
                random_sample=True,
            )
        )
    test_dataset = ConcatDataset(test_parts) if test_parts else None
    return ConcatDataset(train_parts), test_dataset


def benchmark_task(path: Path) -> str:
    name = path.name.lower()
    if "nearest" in name:
        return "nearest"
    if "topo" in name:
        return "topology"
    if "distance" in name:
        return "distance"
    if "polyline" in name:
        return "length"
    if "area" in name:
        return "area"
    raise ValueError(f"Cannot tell which task {path.name} evaluates.")


def benchmark_paths(config: DataConfig) -> list[Path]:
    directory = Path(config.benchmark_dir)
    if not directory.is_dir():
        return []
    city = config.city
    return [path for path in sorted(directory.glob("*.pkl")) if city in path.stem]


class BenchmarkDataset(Dataset):
    """Fixed prompts, embeddings, and answers shipped under ``data/benchmark``."""

    def __init__(self, path: Path, *, limit: int | None = None):
        with Path(path).open("rb") as handle:
            records = pickle.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"{path} should be a list of benchmark records.")
        if limit is not None and limit > 0:
            records = records[:limit]
        self.records = records
        self.path = Path(path)
        self.task = benchmark_task(self.path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        rows = [np.asarray(embedding, dtype=np.float32).reshape(-1) for embedding in row["geo_embeddings"]]
        features = torch.from_numpy(np.stack(rows, axis=0))
        prompt = str(row["prompt"])
        if prompt.count(GEO_TOKEN) != features.shape[0]:
            raise ValueError(
                f"{self.path.name} row {index} has {prompt.count(GEO_TOKEN)} <GEO> tokens "
                f"and {features.shape[0]} embeddings."
            )
        return {
            "prompt": prompt,
            "geo_feature": features,
            "labels": "",
            "answer": str(row["labels"]),
            "task": self.task,
        }


def load_benchmarks(config: DataConfig, *, limit: int | None = None) -> list[BenchmarkDataset]:
    datasets = []
    for path in benchmark_paths(config):
        datasets.append(BenchmarkDataset(path, limit=limit))
        logger.info("Benchmark %s (%s, %d rows)", path.name, datasets[-1].task, len(datasets[-1]))
    return datasets


def collate_slm(
    batch: list[dict[str, Any]],
    tokenizer,
    system_prompt: str,
    prompt_padding_side: str = "right",
    reasoning_effort: str | None = None,
    prefer_final_channel: bool = False,
) -> dict[str, Any]:
    """Pad entity rows independently of the text, and supervise only the assistant suffix."""
    width = batch[0]["geo_feature"].shape[-1]
    max_entities = max(item["geo_feature"].shape[0] for item in batch)
    features = torch.zeros(len(batch), max_entities, width, dtype=torch.float32)
    feature_mask = torch.zeros(len(batch), max_entities, dtype=torch.bool)
    for index, item in enumerate(batch):
        count = item["geo_feature"].shape[0]
        if count:
            features[index, :count] = item["geo_feature"]
            feature_mask[index, :count] = True

    prompt_texts: list[str] = []
    label_texts: list[str] = []
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
        label_texts.append(completion_text)

    with tokenizer_padding_side(tokenizer, prompt_padding_side):
        prompt_encoded = tokenizer(prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False)
    with tokenizer_padding_side(tokenizer, "right"):
        if any(label_texts):
            label_encoded = tokenizer(label_texts, return_tensors="pt", padding=True, add_special_tokens=False)
        else:
            label_encoded = {
                "input_ids": torch.zeros(len(batch), 0, dtype=torch.long),
                "attention_mask": torch.zeros(len(batch), 0, dtype=torch.long),
            }

    return {
        "prompt_input_ids": prompt_encoded["input_ids"],
        "prompt_attention_mask": prompt_encoded["attention_mask"],
        "prompt_texts": prompt_texts,
        "geo_feature": features,
        "geo_attention_mask": feature_mask,
        "labels": label_encoded["input_ids"],
        "labels_attention_mask": label_encoded["attention_mask"],
        "labels_texts": label_texts,
        "answers": [str(item.get("answer", "")) for item in batch],
        "tasks": [str(item.get("task", "")) for item in batch],
    }
