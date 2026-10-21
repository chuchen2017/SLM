"""Question rewriting and the phrases the notebooks sampled at each entity."""

from __future__ import annotations

import re

# The notebooks drew one of these for every <LOCATION> placeholder.
LOCATION_PHRASES = [
    "is situated at",
    "is located at",
    "lies at",
    "is positioned at",
    "can be found at",
    "is geographically positioned at",
    "is mapped at",
    "is set at the geographic coordinates",
    "is located at latitude",
    "is pinpointed at",
    "is defined by the coordinates",
    "is referenced at",
    "is anchored at",
    "is recorded at",
    "is plotted at",
    "is marked at",
    "is registered at",
    "is specified at",
    "is denoted by",
    "is identified at",
    "at",
    "located at",
    ":",
]

GEO_TOKEN = "<GEO>"
LOCATION_PATTERN = re.compile(r"<location_(\d+)>")

SYSTEM_PROMPT = (
    "You are SLM, a spatial language model. Each <GEO> token is the embedding of one "
    "geospatial entity. Answer from those embeddings. Put the spatial predicate and the "
    "answer in a single <geo_think> trace."
)

def unique_location_ids(question: str) -> list[int]:
    return sorted({int(value) for value in LOCATION_PATTERN.findall(question)})


def marker_text(value: str) -> str:
    """Lowercase name or phrase with the angle brackets removed."""
    return " ".join(str(value).replace("<", "").replace(">", "").split()).lower()


def rewrite_location_slots(question: str, phrase_fn, name_fn) -> tuple[str, list[int]]:
    """Replace ``<location_k>`` with ``name phrase <GEO>``.

    A place named Main Street reads ``main street is located at <GEO>``. The name
    and the phrase are plain words. ``<GEO>`` is the only bracketed token.

    Slots are the location ids ranked from smallest to largest, so a question that
    only mentions ``<location_1>`` becomes entity 0, and ``<location_0>`` /
    ``<location_1>`` stay 0 and 1. The returned list has one slot per ``<GEO>``
    token, in left-to-right order. A repeated location id repeats that entity.
    """
    found = [int(match.group(1)) for match in LOCATION_PATTERN.finditer(question)]
    if not found:
        raise ValueError(f"Question has no <location_k> placeholder: {question}")
    mapping = {old: new for new, old in enumerate(sorted(set(found)))}
    sequence: list[int] = []

    def replace(match: re.Match[str]) -> str:
        slot = mapping[int(match.group(1))]
        sequence.append(slot)
        name = marker_text(name_fn(slot)) or f"entity {slot}"
        phrase = marker_text(phrase_fn()) or "at"
        return f"{name} {phrase} {GEO_TOKEN}"

    return LOCATION_PATTERN.sub(replace, question), sequence


def strip_location_tags(question: str) -> str:
    """Text-only rendering for the ungrounded thinking set, which has no geometries."""
    return LOCATION_PATTERN.sub(lambda match: f"location {match.group(1)}", question)


def questions_with_entity_count(questions: list[str], count: int) -> list[str]:
    return [question for question in questions if len(unique_location_ids(question)) == count]
