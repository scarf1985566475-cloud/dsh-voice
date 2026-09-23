"""Lenient JSON extraction from model answers.

Both the translator and the summarizer ask the chat model for JSON. Models
sometimes wrap it in a code fence or add a sentence around it, and a batch
translation that fails to parse must degrade to per-item calls rather than lose
the meeting. This helper is the one place that knows how forgiving to be.
"""

from __future__ import annotations

import json
from typing import Any


def strip_code_fence(text: str) -> str:
    """Remove a Markdown code fence a model added despite being told not to.

    Args:
        text: Raw model output.

    Returns:
        The text without a leading/trailing fence.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_value(text: str) -> Any | None:
    """Parse the JSON value out of a model answer.

    Tries the answer as-is, then without a code fence, then the outermost brace
    or bracket pair inside surrounding prose.

    Args:
        text: Raw model output.

    Returns:
        The parsed value, or ``None`` when nothing parseable is present.
    """
    direct = text.strip()
    fenced = strip_code_fence(text)
    candidates: list[str] = [direct, fenced]
    for source in (fenced, direct):
        start_brace, start_bracket = source.find("{"), source.find("[")
        starts = [index for index in (start_brace, start_bracket) if index != -1]
        if not starts:
            continue
        start = min(starts)
        end = max(source.rfind("}"), source.rfind("]"))
        if end > start:
            candidates.append(source[start:end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object out of a model answer.

    Args:
        text: Raw model output.

    Returns:
        The parsed object, or ``None`` when the answer holds no JSON object.
    """
    value = parse_json_value(text)
    return value if isinstance(value, dict) else None
