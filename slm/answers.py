"""Render and parse the ``<geo_think>`` traces the model is trained to emit."""

from __future__ import annotations

import re

KNOWN_RELATIONS = {
    "disjoint",
    "touches",
    "intersects",
    "crosses",
    "within",
    "contains",
    "overlaps",
    "equals",
    "covers",
}

_NUMERIC = r"([+-]?\d+(?:\.\d+)?)"
_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "length": [re.compile(rf"LineLength\(\s*<\d+>\s*\)\s*=\s*{_NUMERIC}")],
    "area": [re.compile(rf"Area\(\s*<\d+>\s*\)\s*=\s*{_NUMERIC}")],
    "distance": [re.compile(rf"Distance\(\s*<\d+>\s*,\s*<\d+>\s*\)\s*=\s*{_NUMERIC}")],
    "topology": [
        re.compile(r"TopoRelation\(\s*<\d+>\s*,\s*<\d+>\s*\)\s*=\s*([A-Za-z]+)"),
        # The US notebook wrote the relation between the two entity slots.
        re.compile(r"TopoRelation\(\s*<\d+>\s*,\s*([A-Za-z]+)\s*,\s*<\d+>\s*\)"),
    ],
    "nearest": [],
}

_DISTANCE_CALL = re.compile(
    rf"Distance\(\s*<(\d+)>\s*,\s*<(\d+)>\s*\)\s*=\s*{_NUMERIC}"
)


def format_trace(task: str, answer: str, *, query_slot: int | None = None) -> str:
    """Supervised answer string for one spatial task."""
    if task == "length":
        return f"<geo_think> LineLength( <0> ) = {answer} </geo_think>"
    if task == "area":
        return f"<geo_think> Area( <0> ) = {answer} </geo_think>"
    if task == "distance":
        return f"<geo_think> Distance( <0> , <1> ) = {answer} </geo_think>"
    if task == "topology":
        return f"<geo_think> TopoRelation( <0> , <1> ) = {answer} </geo_think>"
    if task == "thinking":
        return str(answer)
    raise ValueError(f"Unknown task {task!r}.")


def format_distance_trace(measurements: list[tuple[int, int, str]]) -> str:
    """Nearest-neighbor label: one rounded distance call per thinking step.

    The smaller entity index is written first, matching the notebook:
    ``Distance( <0> , <4> ) = 12 , Distance( <1> , <4> ) = 30``.
    """
    parts = [f"Distance( <{left}> , <{right}> ) = {value}" for left, right, value in measurements]
    return " <geo_think> " + " , ".join(parts) + " </geo_think> "


def nearest_entity_from_distances(text: str) -> str | None:
    """Entity index with the smallest distance to the shared reference entity."""
    calls = [(int(left), int(right), float(value)) for left, right, value in _DISTANCE_CALL.findall(text or "")]
    if not calls:
        return None
    counts: dict[int, int] = {}
    for left, right, _ in calls:
        counts[left] = counts.get(left, 0) + 1
        counts[right] = counts.get(right, 0) + 1
    hub = max(counts, key=lambda entity: (counts[entity], entity))
    best_entity = None
    best_distance = None
    for left, right, distance in calls:
        if hub not in (left, right):
            continue
        other = right if left == hub else left
        if best_distance is None or distance < best_distance or (distance == best_distance and other < best_entity):
            best_entity = other
            best_distance = distance
    if best_entity is None:
        return None
    return f"<{best_entity}>"


def _normalize_nearest(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"\d+", value):
        return f"<{value}>"
    return value


def parse_answer(task: str, text: str) -> str | None:
    """Pull the task answer out of a generation. Returns ``None`` when it is absent."""
    if text is None:
        return None
    if task == "nearest":
        from_distances = nearest_entity_from_distances(text)
        if from_distances is not None:
            return from_distances
    patterns = _PATTERNS.get(task, [])
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            if task == "nearest":
                return _normalize_nearest(value)
            if task == "topology":
                return value.lower()
            return value

    stripped = text.strip()
    if task == "topology" and stripped.lower() in KNOWN_RELATIONS:
        return stripped.lower()
    if task == "nearest" and re.fullmatch(r"<\d+>", stripped):
        return stripped
    if task in {"length", "area", "distance"} and re.fullmatch(_NUMERIC, stripped):
        return stripped
    return None


def numbers_close(predicted: str, gold: str, decimals: int) -> bool:
    tolerance = 0.5 if decimals <= 0 else 0.5 * (10 ** (-decimals))
    return abs(float(predicted) - float(gold)) <= tolerance


def score_pair(task: str, predicted: str | None, gold: str, *, decimals: int = 0) -> dict[str, float]:
    """Score one prediction. ``parsed`` is 0 when the trace could not be read."""
    if predicted is None:
        return {"parsed": 0.0, "correct": 0.0, "abs_error": float("nan")}
    if task in {"length", "area", "distance"}:
        error = abs(float(predicted) - float(gold))
        correct = 1.0 if numbers_close(predicted, gold, decimals) else 0.0
        return {"parsed": 1.0, "correct": correct, "abs_error": error}
    if task == "topology":
        correct = 1.0 if predicted.lower() == str(gold).strip().lower() else 0.0
        return {"parsed": 1.0, "correct": correct, "abs_error": float("nan")}
    if task == "nearest":
        correct = 1.0 if _normalize_nearest(predicted) == _normalize_nearest(str(gold)) else 0.0
        return {"parsed": 1.0, "correct": correct, "abs_error": float("nan")}
    correct = 1.0 if predicted.strip() == str(gold).strip() else 0.0
    return {"parsed": 1.0, "correct": correct, "abs_error": float("nan")}


def summarize(task: str, rows: list[dict[str, float]]) -> dict[str, float]:
    total = len(rows)
    parsed_rows = [row for row in rows if row["parsed"] > 0]
    parsed = len(parsed_rows)
    summary = {
        "n": float(total),
        "parsed": float(parsed),
        "parse_rate": (parsed / total) if total else 0.0,
    }
    if task in {"length", "area", "distance"}:
        errors = [row["abs_error"] for row in parsed_rows]
        summary["mae"] = sum(errors) / len(errors) if errors else float("nan")
        summary["exact"] = (
            sum(row["correct"] for row in parsed_rows) / len(parsed_rows) if parsed_rows else 0.0
        )
    else:
        summary["accuracy"] = (
            sum(row["correct"] for row in parsed_rows) / len(parsed_rows) if parsed_rows else 0.0
        )
    return summary


def combined_score(metrics: dict[str, dict[str, float]]) -> float:
    """Higher is better. Numeric tasks contribute ``1 / (1 + MAE)``."""
    scores: list[float] = []
    for summary in metrics.values():
        if summary.get("parsed", 0) <= 0:
            continue
        if "accuracy" in summary:
            scores.append(float(summary["accuracy"]))
        elif "mae" in summary and summary["mae"] == summary["mae"]:
            scores.append(1.0 / (1.0 + float(summary["mae"])))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
