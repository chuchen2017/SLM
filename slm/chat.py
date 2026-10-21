"""Chat-template rendering shared by training and generation.

The notebooks tokenized the user turn and the assistant turn as two separate chat
templates and concatenated the ids. That duplicates the assistant header and never
teaches the template's end-of-turn token. Training here renders one completion: the
generation prompt, then only the suffix that `generate` is supposed to produce.
"""

from __future__ import annotations

import contextlib
import os
import re
from typing import Any, Iterator


DEFAULT_THINKING_OPEN_TAG = "<think>"
DEFAULT_THINKING_CLOSE_TAG = "</think>"


def tokenizer_padding_side(tokenizer, side: str) -> Iterator[None]:
    previous = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = side
    try:
        yield
    finally:
        tokenizer.padding_side = previous


tokenizer_padding_side = contextlib.contextmanager(tokenizer_padding_side)


def _as_token_id_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        ids: list[int] = []
        for item in value:
            ids.extend(_as_token_id_list(item))
        return ids
    token_id = int(value)
    return [token_id] if token_id >= 0 else []


def chat_template_end_token_ids(tokenizer) -> list[int]:
    """Token ids of the closer ``apply_chat_template`` writes after the assistant text.

    Qwen3 uses ``<|im_end|>``. Another family uses a different marker. The string
    comes from rendering one completed turn and keeping whatever follows the answer,
    so training and generation stop on that model's own token.
    """
    _, completion = render_chat_prompt_and_completion(
        tokenizer,
        system_prompt="system",
        user_prompt="user",
        assistant_content="ANSWER",
    )
    if not completion.startswith("ANSWER"):
        return []
    marker = completion[len("ANSWER") :]
    if not marker.strip():
        return []
    return _tokenize_ids(tokenizer, marker)


def resolve_generation_stop_token_ids(tokenizer, llm=None) -> list[int]:
    """EOS ids `generate` should stop on.

    The chat-template closer is included even when it differs from
    ``eos_token_id`` or ``generation_config.eos_token_id``.
    """
    ids: list[int] = []
    seen: set[int] = set()

    def add(value) -> None:
        for token_id in _as_token_id_list(value):
            if token_id not in seen:
                seen.add(token_id)
                ids.append(token_id)

    add(getattr(tokenizer, "eos_token_id", None))
    generation_config = getattr(llm, "generation_config", None) if llm is not None else None
    add(getattr(generation_config, "eos_token_id", None))
    end_of_turn = getattr(tokenizer, "eot_token", None)
    if isinstance(end_of_turn, str) and hasattr(tokenizer, "convert_tokens_to_ids"):
        end_id = tokenizer.convert_tokens_to_ids(end_of_turn)
        unknown = getattr(tokenizer, "unk_token_id", None)
        if end_id is not None and end_id != unknown:
            add(end_id)
    add(chat_template_end_token_ids(tokenizer))
    return ids


def tokenizer_uses_gpt_oss_harmony(tokenizer) -> bool:
    unknown = getattr(tokenizer, "unk_token_id", None)
    channel_id = tokenizer.convert_tokens_to_ids("<|channel|>")
    return_id = tokenizer.convert_tokens_to_ids("<|return|>")
    if channel_id is None or return_id is None:
        return False
    if unknown is not None and (channel_id == unknown or return_id == unknown):
        return False
    return True


def resolve_inference_chat_options(tokenizer) -> dict[str, Any]:
    """GPT-OSS Harmony defaults for generation. Empty for other tokenizers."""
    if not tokenizer_uses_gpt_oss_harmony(tokenizer):
        return {}
    return {"reasoning_effort": "low"}


def resolve_thinking_tags(
    tokenizer,
    fallback_open: str = DEFAULT_THINKING_OPEN_TAG,
    fallback_close: str = DEFAULT_THINKING_CLOSE_TAG,
) -> tuple[str, str]:
    """Read the think tags a chat template actually inserts, when it has them."""
    probe = [
        {"role": "system", "content": "probe"},
        {"role": "user", "content": "probe"},
    ]
    try:
        with_thinking = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        without_thinking = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        return fallback_open, fallback_close
    if with_thinking == without_thinking:
        return fallback_open, fallback_close
    pair_pattern = re.compile(r"<([^<>/\s]+)>\s*</\1>", flags=re.DOTALL)
    for text in (without_thinking, with_thinking):
        match = pair_pattern.search(text)
        if match:
            return f"<{match.group(1)}>", f"</{match.group(1)}>"
    common = len(os.path.commonprefix([with_thinking, without_thinking]))
    for candidate in (with_thinking[common:], without_thinking[common:]):
        match = re.search(r"</([^<>\s]+)>", candidate)
        if match:
            return f"<{match.group(1)}>", f"</{match.group(1)}>"
    return fallback_open, fallback_close


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

    def call(**call_kwargs) -> str:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **call_kwargs)
        except TypeError:
            supported = dict(call_kwargs)
            for key in ("enable_thinking", "reasoning_effort"):
                supported.pop(key, None)
            try:
                return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **supported)
            except TypeError:
                return tokenizer.apply_chat_template(messages, **supported)

    text = call(**kwargs)
    if prefer_final_channel and add_generation_prompt and text.endswith("<|start|>assistant"):
        text += "<|channel|>final<|message|>"
    return text


def _tokenize_ids(tokenizer, text: str) -> list[int]:
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
    """Tokens of the completed chat that are not part of the generation prompt."""
    if full_text.startswith(prompt_text):
        return full_text[len(prompt_text) :]
    prompt_ids = _tokenize_ids(tokenizer, prompt_text)
    full_ids = _tokenize_ids(tokenizer, full_text)
    shared = 0
    limit = min(len(prompt_ids), len(full_ids))
    while shared < limit and prompt_ids[shared] == full_ids[shared]:
        shared += 1
    if shared >= len(full_ids):
        return ""
    return tokenizer.decode(full_ids[shared:], skip_special_tokens=False)


def render_chat_prompt_and_completion(
    tokenizer,
    *,
    system_prompt: str,
    user_prompt: str,
    assistant_content: str = "",
    reasoning_effort: str | None = None,
    prefer_final_channel: bool = False,
) -> tuple[str, str]:
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
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
