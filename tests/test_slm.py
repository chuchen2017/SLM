"""Tests for entity alignment, injection, and the city datasets."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from slm.runtime import prefer_env_libstdcxx

prefer_env_libstdcxx()

import pytest
import torch
from shapely.geometry import box
from torch import nn

from slm.answers import format_distance_trace, format_trace, nearest_entity_from_distances, parse_answer, score_pair
from slm.chat import render_chat_prompt_and_completion
from slm.config import DataConfig, load_config
from slm.data import (
    BenchmarkDataset,
    DistanceTraceDataset,
    SpatialReasoningDataset,
    collate_slm,
    load_benchmarks,
    load_city,
    load_distance_templates,
    load_task_questions,
    task_decimals,
)
from slm.geometry import de9im_to_nl, format_measure, topological_relation
from slm.modeling import SLM, Geo2VecEncoder, inject_geo_token_embeddings
from slm.prompts import GEO_TOKEN, rewrite_location_slots
from utils.geo2vec import normalize_geometries

ROOT = Path(__file__).resolve().parents[1]


class RecordingTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.pad_token = "<pad>"
        self.padding_side = "right"
        self.vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "<GEO>": 3}
        self._next = 4

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _pieces(self, text: str) -> list[str]:
        import re

        return re.findall(r"<[^>]+>|[^<]+", text)

    def _ids(self, text: str) -> list[int]:
        ids = []
        for piece in self._pieces(text):
            if piece not in self.vocab:
                self.vocab[piece] = self._next
                self._next += 1
            ids.append(self.vocab[piece])
        return ids

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def decode(self, ids, skip_special_tokens=False):
        inverse = {value: key for key, value in self.vocab.items()}
        return "".join(inverse.get(int(token), "?") for token in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = "".join(f"<|{message['role']}|>{message['content']}<|end|>" for message in messages)
        if add_generation_prompt:
            text += "<|assistant|>"
        return text

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False):
        rows = [self._ids(text) for text in texts]
        width = max((len(row) for row in rows), default=0)
        ids = []
        mask = []
        for row in rows:
            pad = width - len(row)
            if self.padding_side == "left":
                ids.append([self.pad_token_id] * pad + row)
                mask.append([0] * pad + [1] * len(row))
            else:
                ids.append(row + [self.pad_token_id] * pad)
                mask.append([1] * len(row) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class TinyLLM(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.config = SimpleNamespace(hidden_size=hidden, use_cache=False)
        self.generation_config = SimpleNamespace(eos_token_id=1, pad_token_id=0)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        return SimpleNamespace(loss=inputs_embeds.float().square().mean())


def test_location_slots_follow_sorted_ids():
    names = {0: "Pier", 1: "Road"}
    text, sequence = rewrite_location_slots(
        "How far is <location_1> from <location_0>?",
        lambda: "at",
        lambda slot: names[slot],
    )
    assert sequence == [1, 0]
    assert text == f"How far is road at {GEO_TOKEN} from pier at {GEO_TOKEN}?"

    single, single_sequence = rewrite_location_slots(
        "Area of <location_1>?",
        lambda: "is located at",
        lambda slot: "Main Street",
    )
    assert single_sequence == [0]
    assert single == f"Area of main street is located at {GEO_TOKEN}?"

    zero_based, zero_sequence = rewrite_location_slots(
        "Length of <location_0>?",
        lambda: "at",
        lambda slot: "",
    )
    assert zero_sequence == [0]
    assert zero_based == f"Length of entity 0 at {GEO_TOKEN}?"

    repeated, repeated_sequence = rewrite_location_slots(
        "<location_2> then <location_2>",
        lambda: "at",
        lambda slot: "Loop",
    )
    assert repeated_sequence == [0, 0]
    assert repeated.count(f"loop at {GEO_TOKEN}") == 2


def test_relations_and_measures():
    assert topological_relation(box(0, 0, 1, 1), box(10, 10, 11, 11)) == "disjoint"
    assert topological_relation(box(0.2, 0.2, 0.4, 0.4), box(0, 0, 1, 1)) == "within"
    assert topological_relation(box(0, 0, 1, 1), box(0.2, 0.2, 0.4, 0.4)) == "contains"
    square = box(0, 0, 1, 1)
    assert topological_relation(square, square) == "equals"
    assert topological_relation(box(0, 0, 1, 1), box(1, 0, 2, 1)) == "touches"
    assert de9im_to_nl("FF*FF****") == "disjoint"
    assert de9im_to_nl("TFFFTFFFT") == "equals"
    assert format_measure(1828.001, 2) == "1828.0"
    assert format_measure(12.4, 0) == "12"


def test_answer_traces_round_trip():
    trace = format_trace("distance", "19")
    assert parse_answer("distance", trace) == "19"
    us_style = "<geo_think> TopoRelation( <0> , within , <1> ) </geo_think>"
    assert parse_answer("topology", us_style) == "within"
    assert parse_answer("topology", format_trace("topology", "contains")) == "contains"
    trace = format_distance_trace([(0, 4, "10"), (1, 4, "3"), (2, 4, "8"), (3, 4, "40")])
    assert "NearestNeighbor" not in trace
    assert nearest_entity_from_distances(trace) == "<1>"
    assert parse_answer("nearest", trace) == "<1>"
    scored = score_pair("length", "10.00", "10.002", decimals=2)
    assert scored["parsed"] == 1.0 and scored["correct"] == 1.0
    off = score_pair("length", "10.0", "10.02", decimals=2)
    assert off["parsed"] == 1.0 and off["correct"] == 0.0 and off["abs_error"] == pytest.approx(0.02)
    missed = score_pair("length", None, "10", decimals=0)
    assert missed["parsed"] == 0.0


def test_chat_suffix_is_only_the_assistant_continuation():
    tokenizer = RecordingTokenizer()
    prompt, completion = render_chat_prompt_and_completion(
        tokenizer,
        system_prompt="system",
        user_prompt=f"where is {GEO_TOKEN}",
        assistant_content="LineLength( <0> ) = 3",
    )
    assert prompt.endswith("<|assistant|>")
    assert completion == "LineLength( <0> ) = 3<|end|>"
    assert "<|user|>" not in completion
    from slm.chat import chat_template_end_token_ids

    end_ids = chat_template_end_token_ids(tokenizer)
    assert tokenizer.decode(end_ids, skip_special_tokens=False) == "<|end|>"


def test_injection_uses_the_mask_not_a_nonzero_sum():
    query_ids = torch.tensor([[5, 3, 6, 3]])
    query_embeds = torch.zeros(1, 4, 4)
    # Row 0 sums to 0 and is still a real entity. Row 1 is padding and is not.
    geo_embeds = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [9.0, 9.0, 9.0, 9.0], [4.0, 0.0, 0.0, 1.0]]])
    mask = torch.tensor([[True, False, True]])
    updated = inject_geo_token_embeddings(query_ids, query_embeds, geo_embeds, mask, geo_token_id=3)
    assert torch.equal(updated[0, 1], geo_embeds[0, 0])
    assert torch.equal(updated[0, 3], geo_embeds[0, 2])
    with pytest.raises(ValueError):
        inject_geo_token_embeddings(
            query_ids,
            query_embeds,
            geo_embeds,
            torch.tensor([[True, False, False]]),
            geo_token_id=3,
        )


def test_slm_forward_trains_the_projector_at_every_geo_token():
    torch.manual_seed(0)
    tokenizer = RecordingTokenizer()
    llm = TinyLLM()
    model = SLM(llm, tokenizer, Geo2VecEncoder(4, 16, 8), geo_token_id=3, geo_projector_hidden_dim=16)
    query_ids = torch.tensor([[5, 3, 6, 3]])
    query_mask = torch.ones_like(query_ids)
    geo = torch.randn(1, 2, 4)
    geo_mask = torch.ones(1, 2, dtype=torch.bool)
    prepared = model.prepare_inputs(query_ids, query_mask, geo, geo_mask)
    encoded = model.encode_entities(geo, geo_mask)
    assert torch.allclose(prepared[0, 1], encoded[0, 0])
    assert torch.allclose(prepared[0, 3], encoded[0, 1])
    assert torch.allclose(prepared[0, 0], model.embed_tokens.weight[5])

    labels = torch.tensor([[7, 8]])
    output = model(query_ids, geo, labels, query_mask, torch.ones_like(labels), geo_mask)
    output.loss.backward()
    assert model.geo_projector.net[0].weight.grad is not None
    assert model.geo_encoder.net[0].weight.grad is not None


def test_collate_pads_entities_and_supervises_the_suffix_only():
    tokenizer = RecordingTokenizer()
    batch = [
        {
            "prompt": f"one {GEO_TOKEN}",
            "geo_feature": torch.ones(1, 4),
            "labels": "Area( <0> ) = 2",
            "answer": "2",
            "task": "area",
        },
        {
            "prompt": f"left {GEO_TOKEN} right {GEO_TOKEN}",
            "geo_feature": torch.full((2, 4), 3.0),
            "labels": "Distance( <0> , <1> ) = 5",
            "answer": "5",
            "task": "distance",
        },
    ]
    collated = collate_slm(batch, tokenizer, system_prompt="sys")
    assert collated["geo_feature"].shape == (2, 2, 4)
    assert collated["geo_attention_mask"].tolist() == [[True, False], [True, True]]
    assert collated["labels_texts"][0] == "Area( <0> ) = 2<|end|>"
    assert "<|user|>" not in collated["labels_texts"][1]
    assert torch.count_nonzero(collated["prompt_input_ids"][1] == 3) == 2


def test_normalize_geometries_keeps_relative_position():
    normalized, (width, height) = normalize_geometries(
        [box(10, 20, 12, 24), box(12, 24, 14, 28)]
    )
    assert width == pytest.approx(4)
    assert height == pytest.approx(8)
    assert normalized[0].bounds == pytest.approx((0, 0, 0.5, 0.5))
    assert normalized[1].bounds == pytest.approx((0.5, 0.5, 1, 1))


def test_config_selects_city_tasks():
    config = load_config(ROOT / "configs" / "united_states.yaml")
    assert config.data.city == "US"
    assert "topology" in config.data.tasks
    assert "length" not in config.data.tasks
    assert config.model.base_model == "Qwen/Qwen3-8B"


def _city_config(city: str, tasks: list[str]) -> DataConfig:
    return DataConfig(
        city=city,
        tasks=tasks,
        osm_dir=str(ROOT / "data" / "osm"),
        questions_dir=str(ROOT / "data" / "instruction_dataset"),
        benchmark_dir=str(ROOT / "data" / "benchmark"),
        samples_per_task=2,
    )


@pytest.mark.parametrize("task", ["length", "area", "distance"])
def test_los_angeles_examples_align_entities(task: str):
    config = _city_config("LosAngeles", [task])
    questions = load_task_questions(config)
    train, _ = load_city(config)
    dataset = SpatialReasoningDataset(
        train,
        questions[task],
        task,
        num_samples=3,
        decimals=task_decimals(config, task),
        random_sample=False,
        seed=2,
    )
    if task == "length":
        slot = {0: int(dataset.split.line_indices[0])}
        answer, _ = dataset._answer(slot, None)
        geometry = dataset.split.geometries[slot[0]]
        assert answer == format_measure(geometry.length, dataset.decimals)
    for index in range(len(dataset)):
        item = dataset[index]
        assert item["prompt"].count(GEO_TOKEN) == item["geo_feature"].shape[0]
        assert "<location_" not in item["prompt"]
        assert item["prompt"].replace(GEO_TOKEN, "").count("<") == 0
        assert item["geo_feature"].shape[-1] == 256
        assert parse_answer(task, item["labels"]) == item["answer"]


def test_nearest_trace_is_sampled_each_call():
    config = _city_config("LosAngeles", ["nearest"])
    templates = load_distance_templates(ROOT / "data/instruction_dataset/triplet_distance_QA_thinking_steps_total.json")
    train, _ = load_city(config)
    dataset = DistanceTraceDataset(train, templates, num_samples=4, decimals=0, random_sample=True)
    assert dataset._fixed is None
    first = dataset[0]
    second = dataset[0]
    for item in (first, second):
        assert item["task"] == "nearest"
        assert "NearestNeighbor" not in item["labels"]
        assert "Distance(" in item["labels"]
        assert item["prompt"].count(GEO_TOKEN) == item["geo_feature"].shape[0]
        assert item["prompt"].replace(GEO_TOKEN, "").count("<") == 0
        assert parse_answer("nearest", item["labels"]) is not None
    assert first["labels"] != second["labels"] or first["prompt"] != second["prompt"]


def test_us_topology_uses_the_split_pairs():
    config = _city_config("US", ["topology", "area"])
    questions = load_task_questions(config)
    train, test = load_city(config)
    assert train.topology_pairs
    assert test.topology_pairs
    assert train.embedding_dim == 512
    dataset = SpatialReasoningDataset(
        train,
        questions["topology"],
        "topology",
        num_samples=2,
        random_sample=False,
        seed=3,
    )
    item = dataset[0]
    assert item["geo_feature"].shape[-1] == 512
    assert item["geo_feature"].shape[0] == item["prompt"].count(GEO_TOKEN) >= 2
    assert parse_answer("topology", item["labels"]) == item["answer"]


def test_benchmarks_match_their_embeddings():
    config = _city_config("LosAngeles", ["nearest"])
    datasets = load_benchmarks(config)
    names = {dataset.path.name for dataset in datasets}
    assert "benchmark_nearest_neighbor_Point_PolyLine_LosAngeles.pkl" in names
    nearest = BenchmarkDataset(ROOT / "data/benchmark/benchmark_nearest_neighbor_Point_PolyLine_LosAngeles.pkl")
    item = nearest[0]
    assert item["geo_feature"].shape == (5, 256)
    assert item["answer"].startswith("<")
    us = BenchmarkDataset(ROOT / "data/benchmark/benchmark_topo_US.pkl")
    assert us[0]["geo_feature"].shape == (2, 512)


def test_thinking_rows_have_no_geo_token():
    from slm.data import ThinkingDataset

    dataset = ThinkingDataset(
        [
            {
                "question": "How far is <location_1> from <location_0>?",
                "thinking": ['{"name": "distance_tool"}'],
                "final_answer": "12",
            }
        ],
        num_samples=1,
        embedding_dim=8,
        open_tag="<think>",
        close_tag="</think>",
        random_sample=False,
    )
    item = dataset[0]
    assert item["geo_feature"].shape == (0, 8)
    assert "<location_" not in item["prompt"]
    assert "location 1" in item["prompt"] and "location 0" in item["prompt"]
    assert item["labels"].startswith("<think>")
    assert item["labels"].endswith("12")
    collated = collate_slm(
        [
            item,
            {
                "prompt": f"road {GEO_TOKEN}",
                "geo_feature": torch.ones(1, 8),
                "labels": "LineLength( <0> ) = 4",
                "answer": "4",
                "task": "length",
            },
        ],
        RecordingTokenizer(),
        system_prompt="sys",
    )
    assert collated["geo_attention_mask"].tolist() == [[False], [True]]
    assert torch.count_nonzero(collated["prompt_input_ids"][0] == 3) == 0


def test_lora_targets_are_full_module_paths():
    from slm.lora import discover_lora_targets

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(2, 2)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])
            self.lm_head = nn.Linear(2, 2)

    assert discover_lora_targets(Toy()) == ["layers.0.q_proj", "layers.1.q_proj"]


def test_help_imports():
    # Importing the entry points catches a broken package layout without downloading a model.
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    import slm.evaluate
    import slm.train

    assert callable(slm.train.main)
    assert callable(slm.evaluate.main)
