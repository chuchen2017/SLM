from __future__ import annotations

import argparse
import pickle
from functools import partial
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from reasoning_geo2llm.build import load_checkpoint
from reasoning_geo2llm.config import ReasoningConfig, load_config
from reasoning_geo2llm.data import (
    DEFAULT_GEOHASH_LENGTH,
    build_geo_sample,
    canonical_coordinate_key,
    collate_geo_chat_template,
    decode_generated_completions,
    extract_reasoning_answer,
    grid_bounds_from_records,
    load_all_records,
    load_test_records,
    load_training_records,
    raw_coordinates_from_record,
    resolve_inference_chat_options,
    resolve_reasoning_config,
    resolve_system_prompt,
)
from reasoning_geo2llm.runtime import apply_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Geo2LLM checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--sentence-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of locations to evaluate. Use -1 (or any value <= 0) to evaluate all locations.",
    )
    parser.add_argument(
        "--match-reference-count",
        dest="match_reference_count",
        action="store_true",
        default=True,
        help=(
            "Generate exactly as many samples per location as it has ground-truth descriptions, "
            "batched across locations for throughput (default; matches baselines/KNN.py)."
        ),
    )
    parser.add_argument(
        "--no-match-reference-count",
        dest="match_reference_count",
        action="store_false",
        help="Use the original behavior: generate a fixed batch_size samples for every location.",
    )
    return parser.parse_args()


def coordinate_key(latitude: float, longitude: float, precision: int = 6) -> tuple[float, float]:
    return canonical_coordinate_key(latitude, longitude, precision=precision)


def references_by_coordinate(
    dataframe: pd.DataFrame,
    latitude_col: str,
    longitude_col: str,
    description_col: str,
    precision: int = 6,
) -> dict[tuple[float, float], list[str]]:
    refs: dict[tuple[float, float], list[str]] = {}
    for _, row in dataframe.iterrows():
        key = coordinate_key(row[latitude_col], row[longitude_col], precision=precision)
        text = str(row[description_col])
        refs.setdefault(key, []).append(text)
    return refs


def group_generated_by_coordinate(
    generated_results: list[tuple[list[float] | tuple[float, float], str]],
    precision: int = 6,
) -> tuple[dict[tuple[float, float], list[str]], int]:
    grouped: dict[tuple[float, float], list[str]] = {}
    dropped_invalid = 0
    for coord, text in generated_results:
        if len(coord) < 2:
            dropped_invalid += 1
            continue
        key = coordinate_key(coord[0], coord[1], precision=precision)
        grouped.setdefault(key, []).append(str(text))
    return grouped, dropped_invalid


def _sample_unmatched_keys(
    generated_keys: set[tuple[float, float]],
    reference_keys: set[tuple[float, float]],
    limit: int = 5,
) -> list[tuple[tuple[float, float], list[tuple[float, float]]]]:
    """Return (generated_key, nearest reference keys) for debugging."""
    unmatched = list(generated_keys - reference_keys)[:limit]
    samples: list[tuple[tuple[float, float], list[tuple[float, float]]]] = []
    if not unmatched or not reference_keys:
        return samples
    ref_array = np.array(list(reference_keys), dtype=np.float64)
    for key in unmatched:
        dists = np.sum((ref_array - np.array(key)) ** 2, axis=1)
        nearest_idx = np.argsort(dists)[:3]
        nearest = [tuple(ref_array[i]) for i in nearest_idx]
        samples.append((key, nearest))
    return samples


def match_generated_to_references(
    generated_results: list[tuple[list[float] | tuple[float, float], str]],
    references: dict[tuple[float, float], list[str]],
    precision: int = 6,
) -> tuple[list[tuple[list[str], list[str]]], int, int]:
    """Return (generated, refs) pairs per coordinate, plus skipped/unmatched counts."""
    generated_by_coord, dropped_invalid = group_generated_by_coordinate(
        generated_results,
        precision=precision,
    )
    pairs: list[tuple[list[str], list[str]]] = []
    skipped = 0
    for key, generated in generated_by_coord.items():
        refs = references.get(key)
        if not refs:
            skipped += len(generated)
            continue
        pairs.append((generated, refs))
    unmatched_coords = len(generated_by_coord) - len(pairs)
    return pairs, skipped, unmatched_coords, dropped_invalid


def aggregate_generation_metrics(
    pairs: list[tuple[list[str], list[str]]],
    sentence_model,
) -> dict[str, float]:
    """Same metrics as evaluate_generation, without running the model."""
    if not pairs:
        return {
            "semantic_precision": 0.0,
            "semantic_recall": 0.0,
            "semantic_f1": 0.0,
            "bleu": 0.0,
            "self_bleu": 0.0,
            "num_locations": 0.0,
            "num_generations": 0.0,
        }
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    semantic_scores = [semantic_alignment_scores(generated, refs, sentence_model) for generated, refs in pairs]
    rouge_scores = [_rouge_batch(rouge, generated, refs) for generated, refs in pairs]
    bleu_scores = [_bleu_batch(generated, refs) for generated, refs in pairs]
    self_bleu_scores = [score for generated, _ in pairs if (score := _self_bleu_batch(generated)) is not None]
    num_generations = sum(len(generated) for generated, _ in pairs)
    result = {
        "semantic_precision": float(np.mean([score["precision"] for score in semantic_scores])),
        "semantic_recall": float(np.mean([score["recall"] for score in semantic_scores])),
        "semantic_f1": float(np.mean([score["f1"] for score in semantic_scores])),
        "bleu": float(np.mean(bleu_scores)),
        "self_bleu": float(np.mean(self_bleu_scores)) if self_bleu_scores else 0.0,
        "num_locations": float(len(pairs)),
        "num_generations": float(num_generations),
    }
    for key in rouge_scores[0].keys():
        result[key] = float(np.mean([score[key] for score in rouge_scores]))
    return result


def evaluate_generated_results(
    generated_results: list[tuple[list[float] | tuple[float, float], str]],
    dataframe: pd.DataFrame,
    sentence_model,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    description_col: str = "description",
    coordinate_precision: int = 6,
    debug_unmatched: bool = True,
) -> dict[str, float]:
    """Evaluate pickled generation output against ground-truth descriptions by coordinate."""
    references = references_by_coordinate(
        dataframe,
        latitude_col,
        longitude_col,
        description_col,
        precision=coordinate_precision,
    )
    pairs, skipped, unmatched_coords, dropped_invalid = match_generated_to_references(
        generated_results,
        references,
        precision=coordinate_precision,
    )
    metrics = aggregate_generation_metrics(pairs, sentence_model)
    metrics["skipped_generations_no_ref"] = float(skipped)
    metrics["unmatched_coordinate_groups"] = float(unmatched_coords)
    metrics["dropped_invalid_coordinates"] = float(dropped_invalid)

    if debug_unmatched and (skipped > 0 or dropped_invalid > 0):
        generated_by_coord, _ = group_generated_by_coordinate(
            generated_results,
            precision=coordinate_precision,
        )
        ref_keys = set(references.keys())
        gen_keys = set(generated_by_coord.keys())
        samples = _sample_unmatched_keys(gen_keys, ref_keys)
        if samples:
            print("Coordinate match debug (generated_key -> nearest reference keys):")
            for gen_key, nearest in samples:
                print(f"  gen {gen_key} -> nearest refs {nearest}")
        if dropped_invalid:
            print(
                f"Warning: {dropped_invalid} generations had invalid coordinates "
                "(e.g. grid cell ids used instead of lat/lon)."
            )
    return metrics


def print_metrics(metrics: dict[str, float], title: str | None = None) -> None:
    if title:
        print(title)
    for key, value in metrics.items():
        if key.startswith("num_") or key.endswith("_no_ref") or key.endswith("_groups"):
            if value == int(value):
                print(f"{key}: {int(value)}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:.4f}")


def semantic_alignment_scores(generated: list[str], refs: list[str], sentence_model) -> dict[str, float]:
    if not generated or not refs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    texts = generated + refs
    embeddings = sentence_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    gen_emb = embeddings[: len(generated)]
    ref_emb = embeddings[len(generated) :]
    sim = gen_emb @ ref_emb.T
    precision = float(sim.max(axis=1).mean())
    recall = float(sim.max(axis=0).mean())
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def _geo_feature_for_coordinates(
    latitude: float,
    longitude: float,
    *,
    spatial_division_method: str | None,
    grid_bounds: tuple[float, float, float, float] | None,
    grid_num_axis: int,
    text_latitude: float | None = None,
    text_longitude: float | None = None,
    geohash_length: int = DEFAULT_GEOHASH_LENGTH,
) -> tuple[str, torch.Tensor]:
    sample = build_geo_sample(
        latitude,
        longitude,
        spatial_division_method=spatial_division_method,
        grid_bounds=grid_bounds,
        grid_num_axis=grid_num_axis,
        text_latitude=text_latitude,
        text_longitude=text_longitude,
        geohash_length=geohash_length,
    )
    return sample["prompt"], sample["geo_feature"]


def _max_new_tokens_for_refs(
    refs: list[str],
    tokenizer,
    reasoning_enabled: bool,
    reasoning: ReasoningConfig | None,
) -> int:
    if reasoning_enabled:
        thinking_source = " ".join(refs)
        max_ref_tokens = len(tokenizer.encode(thinking_source, add_special_tokens=False))
        return max(256, max_ref_tokens + 60)
    max_ref_tokens = len(tokenizer.encode(refs[0] if refs else "", add_special_tokens=False))
    return max(128, max_ref_tokens + 30)


def _decode_generations(
    outputs: torch.Tensor,
    tokenizer,
    reasoning_enabled: bool,
    reasoning: ReasoningConfig | None,
) -> list[str]:
    generated = decode_generated_completions(tokenizer, outputs)
    if reasoning_enabled:
        generated = [
            extract_reasoning_answer(
                text,
                close_tag=reasoning.thinking_close_tag,
                answer_prefix=reasoning.answer_prefix,
            )
            for text in generated
        ]
    return generated


def _collect_evaluation_locations(
    dataframe: pd.DataFrame,
    num_samples: int,
    spatial_division_method: str | None,
    grid_bounds: tuple[float, float, float, float] | None,
    grid_num_axis: int,
    seed: int = 42,
    geohash_length: int = DEFAULT_GEOHASH_LENGTH,
) -> list[dict]:
    """Group `dataframe` by id, sample up to `num_samples` locations, and build the
    prompt/geo_feature/refs needed to generate+evaluate each one. `num_samples <= 0` (e.g. -1)
    evaluates every location in `dataframe` instead of sampling a subset."""
    grouped = dataframe.groupby("id")
    group_ids = list(grouped.groups.keys())
    if num_samples <= 0:
        sampled_ids = group_ids
    else:
        rng = np.random.RandomState(seed)
        sampled_ids = rng.choice(group_ids, size=min(num_samples, len(group_ids)), replace=False)

    locations: list[dict] = []
    for location_id in sampled_ids:
        group = grouped.get_group(location_id)
        latitude = float(group.iloc[0]["latitude"])
        longitude = float(group.iloc[0]["longitude"])
        refs = group["description"].astype(str).tolist()
        text_latitude, text_longitude = raw_coordinates_from_record(
            group.iloc[0],
            "latitude",
            "longitude",
        )
        if spatial_division_method == "HYBRID" and text_latitude is None:
            raise ValueError(
                "spatial_division_method='HYBRID' needs unnormalized degrees for the prompt "
                "and a [-1, 1] geo_feature for the location encoder; the evaluation frame is "
                "missing latitude_raw/longitude_raw. Load data with normalize_coordinates=True."
            )
        prompt, geo_feature = _geo_feature_for_coordinates(
            latitude,
            longitude,
            spatial_division_method=spatial_division_method,
            grid_bounds=grid_bounds,
            grid_num_axis=grid_num_axis,
            text_latitude=text_latitude,
            text_longitude=text_longitude,
            geohash_length=geohash_length,
        )
        locations.append(
            {
                "id": location_id,
                "latitude": latitude,
                "longitude": longitude,
                "prompt": prompt,
                "geo_feature": geo_feature,
                "refs": refs,
            }
        )
    return locations


def _generate_fixed_batch_per_location(
    model,
    tokenizer,
    locations: list[dict],
    collate_fn,
    device: torch.device,
    batch_size: int,
    temperature: float,
    top_k: int,
    top_p: float,
    reasoning_enabled: bool,
    reasoning: ReasoningConfig | None,
) -> list[tuple[list[str], list[str]]]:
    """Original evaluation logic: for every location, run one `model.generate` call producing
    `batch_size` samples, regardless of how many ground-truth descriptions that location has.
    Kept for comparison/backwards-compatibility; see `match_reference_count` in
    `evaluate_generation`.
    """
    pairs: list[tuple[list[str], list[str]]] = []
    for location in tqdm(locations, desc="Evaluating locations", disable=__name__ != "__main__"):
        refs = location["refs"]
        batch = [
            {
                "prompt": location["prompt"],
                "geo_feature": location["geo_feature"],
                "labels": "",
            }
        ] * batch_size
        batch = collate_fn(batch)
        max_new_tokens = _max_new_tokens_for_refs(refs, tokenizer, reasoning_enabled, reasoning)

        with torch.no_grad():
            outputs = model.generate(
                query_ids=batch["prompt_input_ids"].to(device),
                geo_tensor=batch["geo_feature"].to(device),
                geo_attention_mask=batch["geo_attention_mask"].to(device),
                query_mask=batch["prompt_attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
            )
        generated = _decode_generations(outputs, tokenizer, reasoning_enabled, reasoning)
        pairs.append((generated, refs))
    return pairs


def _generate_matched_reference_count(
    model,
    tokenizer,
    locations: list[dict],
    collate_fn,
    device: torch.device,
    batch_size: int,
    temperature: float,
    top_k: int,
    top_p: float,
    reasoning_enabled: bool,
    reasoning: ReasoningConfig | None,
) -> list[tuple[list[str], list[str]]]:
    """For each location, generate exactly as many samples as it has ground-truth descriptions
    (matching `baselines/KNN.py`'s methodology), instead of a fixed `batch_size` per location.

    To keep generation efficient, queries are first flattened across *all* locations (each
    location's prompt repeated once per needed sample), then generated in chunks of `batch_size`
    mixing queries from different locations together, and finally regrouped back by location.
    """
    queries: list[dict] = []
    query_location_idx: list[int] = []
    query_max_new_tokens: list[int] = []
    generated_by_location: list[list[str]] = [[] for _ in locations]

    for loc_idx, location in enumerate(locations):
        refs = location["refs"]
        num_needed = len(refs)
        if num_needed <= 0:
            continue
        max_new_tokens = _max_new_tokens_for_refs(refs, tokenizer, reasoning_enabled, reasoning)
        for _ in range(num_needed):
            queries.append(
                {
                    "prompt": location["prompt"],
                    "geo_feature": location["geo_feature"],
                    "labels": "",
                }
            )
            query_location_idx.append(loc_idx)
            query_max_new_tokens.append(max_new_tokens)

    chunk_starts = list(range(0, len(queries), batch_size))
    for start in tqdm(chunk_starts, desc="Generating batches", disable=__name__ != "__main__"):
        chunk = queries[start : start + batch_size]
        chunk_location_idx = query_location_idx[start : start + batch_size]
        chunk_max_new_tokens = max(query_max_new_tokens[start : start + batch_size])

        batch = collate_fn(chunk)
        with torch.no_grad():
            outputs = model.generate(
                query_ids=batch["prompt_input_ids"].to(device),
                geo_tensor=batch["geo_feature"].to(device),
                geo_attention_mask=batch["geo_attention_mask"].to(device),
                query_mask=batch["prompt_attention_mask"].to(device),
                max_new_tokens=chunk_max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
            )
        decoded = _decode_generations(outputs, tokenizer, reasoning_enabled, reasoning)
        for loc_idx, text in zip(chunk_location_idx, decoded):
            generated_by_location[loc_idx].append(text)

    return [(generated_by_location[i], locations[i]["refs"]) for i in range(len(locations))]


def _pairs_to_flat_generations(
    locations: list[dict],
    pairs: list[tuple[list[str], list[str]]],
) -> list[tuple[list[float], str]]:
    """Flatten per-location (generated, refs) pairs into the same
    `list[tuple[[latitude, longitude], generated_text]]` format that `generate.py` writes to its
    output pkl - one entry per generated sample, coordinates repeated across a location's
    generations."""
    results: list[tuple[list[float], str]] = []
    for location, (generated, _refs) in zip(locations, pairs):
        coords = [location["latitude"], location["longitude"]]
        for text in generated:
            results.append((coords, text))
    return results


def _save_generation_results(results: list[tuple[list[float], str]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(results, f)
    print(f"Saved {len(results)} generated descriptions to {output}")


def evaluate_generation(
    model,
    tokenizer,
    dataframe: pd.DataFrame,
    sentence_model,
    batch_size: int = 32,
    num_samples: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    spatial_division_method: str | None = None,
    grid_num_axis: int = 100,
    grid_bounds: tuple[float, float, float, float] | None = None,
    reasoning: ReasoningConfig | None = None,
    match_reference_count: bool = True,
    output_path: str | Path | None = None,
    geohash_length: int = DEFAULT_GEOHASH_LENGTH,
) -> dict[str, float]:
    """Generate descriptions per test location and score them against the ground truth.

    `match_reference_count` is the turning point between two generation strategies:
      - True (default): generate exactly as many samples per location as it has ground-truth
        descriptions, matching how `baselines/KNN.py` is evaluated, batching queries *across*
        locations (up to `batch_size` at a time) for throughput. This fixes the original logic,
        which always generated `batch_size` samples per location regardless of how many
        ground-truth descriptions existed to compare against.
      - False: the original behavior, kept for comparison - one `model.generate` call per
        location producing `batch_size` samples every time.

    If `output_path` is given, the raw generations are also written to a pkl file in the same
    `list[tuple[[latitude, longitude], generated_text]]` format that `generate.py` writes to
    `generation.output` (one entry per generated sample).
    """

    if isinstance(sentence_model, tuple):
        sentence_model, device = sentence_model
        sentence_model = SentenceTransformer(sentence_model, device=device)
    else:
        sentence_model = sentence_model#SentenceTransformer(sentence_model, device=str(device))


    reasoning = resolve_reasoning_config(reasoning, tokenizer)
    reasoning_enabled = reasoning is not None and reasoning.enabled

    locations = _collect_evaluation_locations(
        dataframe,
        num_samples,
        spatial_division_method,
        grid_bounds,
        grid_num_axis,
        geohash_length=geohash_length,
    )
    collate_fn = partial(
        collate_geo_chat_template,
        tokenizer=tokenizer,
        system_prompt=resolve_system_prompt(reasoning),
        prompt_padding_side="left",
        **resolve_inference_chat_options(tokenizer, spatial_division_method),
    )

    device = next(model.parameters()).device
    model.eval()

    generate_fn = _generate_matched_reference_count if match_reference_count else _generate_fixed_batch_per_location
    pairs = generate_fn(
        model,
        tokenizer,
        locations,
        collate_fn,
        device,
        batch_size,
        temperature,
        top_k,
        top_p,
        reasoning_enabled,
        reasoning,
    )

    if output_path is not None:
        _save_generation_results(_pairs_to_flat_generations(locations, pairs), output_path)

    result = aggregate_generation_metrics(pairs, sentence_model)
    result.pop("num_locations", None)
    result.pop("num_generations", None)
    return result


def _rouge_batch(rouge, generated: list[str], refs: list[str]) -> dict[str, float]:
    per_sample = []
    for pred in generated:
        best = None
        best_mean = -1.0
        for ref in refs or [""]:
            score = rouge.score(ref, pred)
            mean_f1 = np.mean([score["rouge1"].fmeasure, score["rouge2"].fmeasure, score["rougeL"].fmeasure])
            if mean_f1 > best_mean:
                best = score
                best_mean = mean_f1
        per_sample.append(best)
    return {
        "rouge1_p": float(np.mean([s["rouge1"].precision for s in per_sample])),
        "rouge1_r": float(np.mean([s["rouge1"].recall for s in per_sample])),
        "rouge1_f": float(np.mean([s["rouge1"].fmeasure for s in per_sample])),
        "rouge2_p": float(np.mean([s["rouge2"].precision for s in per_sample])),
        "rouge2_r": float(np.mean([s["rouge2"].recall for s in per_sample])),
        "rouge2_f": float(np.mean([s["rouge2"].fmeasure for s in per_sample])),
        "rougeL_p": float(np.mean([s["rougeL"].precision for s in per_sample])),
        "rougeL_r": float(np.mean([s["rougeL"].recall for s in per_sample])),
        "rougeL_f": float(np.mean([s["rougeL"].fmeasure for s in per_sample])),
    }


def _bleu_tokenize(text: str) -> list[str]:
    return text.lower().split()


_BLEU_SMOOTHING = SmoothingFunction().method1


def _bleu_batch(generated: list[str], refs: list[str]) -> float:
    """Multi-reference BLEU-4 (smoothed) of each generated sample against all ground-truth
    references for a location, averaged over generated samples. Measures how well generations
    align with the ground truth."""
    if not generated or not refs:
        return 0.0
    ref_tokens = [_bleu_tokenize(ref) for ref in refs]
    scores = [
        sentence_bleu(ref_tokens, _bleu_tokenize(gen), smoothing_function=_BLEU_SMOOTHING)
        for gen in generated
    ]
    return float(np.mean(scores))


def _self_bleu_batch(generated: list[str]) -> float | None:
    """Self-BLEU: BLEU of each generated sample against the *other* generated samples for the
    same location (used as multi-references), averaged over samples. Measures diversity among
    generations for a location - higher self-BLEU means generations are more repetitive/similar
    to each other, lower means more diverse. Returns `None` when fewer than 2 samples were
    generated for the location (self-BLEU is undefined in that case)."""
    if len(generated) < 2:
        return None
    tokens = [_bleu_tokenize(gen) for gen in generated]
    scores = [
        sentence_bleu([tokens[j] for j in range(len(tokens)) if j != i], hyp, smoothing_function=_BLEU_SMOOTHING)
        for i, hyp in enumerate(tokens)
    ]
    return float(np.mean(scores))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    apply_runtime_config(cfg.runtime)
    dataframe = load_test_records(cfg.data) if args.split == "test" else load_training_records(cfg.data)
    model = load_checkpoint(cfg.model, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if cfg.reasoning.enabled:
        cfg.reasoning = resolve_reasoning_config(cfg.reasoning, model.tokenizer)
        print(
            "reasoning thinking tags resolved from tokenizer chat template: "
            f"open={cfg.reasoning.thinking_open_tag!r} close={cfg.reasoning.thinking_close_tag!r}"
        )
    sentence_model = SentenceTransformer(args.sentence_model, device=str(device))
    grid_bounds = None
    if cfg.model.spatial_division_method == "grid":
        grid_bounds = grid_bounds_from_records(
            load_all_records(cfg.data),
            cfg.data.latitude_col,
            cfg.data.longitude_col,
        )
    eval_output_path = None
    if cfg.generation.output:
        generation_output = Path(cfg.generation.output)
        eval_output_path = generation_output.with_name(f"{generation_output.stem}_eval{generation_output.suffix}")

    result = evaluate_generation(
        model,
        model.tokenizer,
        dataframe,
        sentence_model,
        batch_size=cfg.generation.batch_size,
        num_samples=args.num_samples,
        temperature=cfg.generation.temperature,
        top_k=cfg.generation.top_k,
        top_p=cfg.generation.top_p,
        spatial_division_method=cfg.model.spatial_division_method,
        grid_num_axis=cfg.model.grid_num_axis,
        grid_bounds=grid_bounds,
        reasoning=cfg.reasoning,
        match_reference_count=args.match_reference_count,
        output_path=eval_output_path,
        geohash_length=cfg.model.geohash_length,
    )
    for key, value in result.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
