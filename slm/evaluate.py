"""Evaluate an SLM checkpoint on the fixed benchmark pickles."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slm.runtime import apply_runtime_config, prefer_env_libstdcxx

prefer_env_libstdcxx()

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from slm.answers import combined_score, parse_answer, score_pair, summarize
from slm.build import load_checkpoint
from slm.chat import decode_generated_completion, resolve_inference_chat_options
from slm.config import load_config, to_dict
from slm.data import BenchmarkDataset, collate_slm, load_benchmarks, system_prompt_for, task_decimals

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SLM on the geospatial benchmarks.")
    parser.add_argument("--config", required=True, help="YAML config used for the run.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory saved by slm.train.")
    parser.add_argument("--output", default=None, help="Where to write metrics JSON. Defaults to the checkpoint dir.")
    parser.add_argument("--num-samples", type=int, default=0, help="Examples per benchmark file. 0 means all of them.")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def evaluate_benchmarks(
    model,
    tokenizer,
    datasets: list[BenchmarkDataset],
    *,
    generation: GenerationConfig,
    system_prompt: str,
    decimals_for_task,
    device: torch.device | None = None,
) -> dict:
    """Generate on each benchmark file and score parsed answers against its labels."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    chat_options = resolve_inference_chat_options(tokenizer)
    per_file: dict[str, dict] = {}
    predictions: list[dict] = []
    for dataset in datasets:
        collate = partial(
            collate_slm,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            prompt_padding_side="left",
            **chat_options,
        )
        loader = DataLoader(dataset, batch_size=generation.batch_size, shuffle=False, collate_fn=collate)
        rows: list[dict[str, float]] = []
        for batch in tqdm(loader, desc=dataset.path.name, leave=False):
            batch = move_batch(batch, device)
            generated = model.generate(
                query_ids=batch["prompt_input_ids"],
                geo_tensor=batch["geo_feature"],
                query_mask=batch["prompt_attention_mask"],
                geo_attention_mask=batch["geo_attention_mask"],
                max_new_tokens=generation.max_new_tokens,
                temperature=generation.temperature,
                top_k=generation.top_k,
                top_p=generation.top_p,
                do_sample=generation.do_sample,
            )
            for index in range(len(batch["answers"])):
                text = decode_generated_completion(tokenizer, generated[index])
                task = batch["tasks"][index]
                gold = batch["answers"][index]
                parsed = parse_answer(task, text)
                rows.append(score_pair(task, parsed, gold, decimals=decimals_for_task(task)))
                predictions.append(
                    {
                        "benchmark": dataset.path.name,
                        "task": task,
                        "gold": gold,
                        "parsed": parsed,
                        "generation": text,
                    }
                )
        summary = summarize(dataset.task, rows)
        summary["task"] = dataset.task
        per_file[dataset.path.name] = summary
    numeric_view = {name: {key: value for key, value in summary.items() if key != "task"} for name, summary in per_file.items()}
    return {"files": per_file, "score": combined_score(numeric_view), "predictions": predictions}


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = load_config(args.config)
    apply_runtime_config(config.runtime)
    if args.batch_size:
        config.generation.batch_size = args.batch_size
    limit = args.num_samples if args.num_samples > 0 else None
    datasets = load_benchmarks(config.data, limit=limit)
    if not datasets:
        raise FileNotFoundError(f"No benchmark pickles for city {config.data.city!r} in {config.data.benchmark_dir}.")
    model = load_checkpoint(config.model, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    result = evaluate_benchmarks(
        model,
        model.tokenizer,
        datasets,
        generation=config.generation,
        system_prompt=system_prompt_for(config.data),
        decimals_for_task=lambda task: task_decimals(config.data, task),
        device=device,
    )
    output = Path(args.output) if args.output else Path(args.checkpoint) / "benchmark_metrics.json"
    payload = {"score": result["score"], "files": result["files"], "config": to_dict(config)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    prediction_path = output.with_name(output.stem + "_predictions.jsonl")
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in result["predictions"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("score=%.4f", result["score"])
    for name, summary in result["files"].items():
        rendered = " ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in summary.items())
        logger.info("%s %s", name, rendered)
    logger.info("Wrote %s", output)


if __name__ == "__main__":
    main()
