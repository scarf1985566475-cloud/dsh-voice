"""Meeting summarization.

The model is asked for a JSON object rather than prose, and the Markdown minutes
are rendered here from that object. Two reasons: a panel can then show action
items as a list instead of re-parsing text, and a malformed answer degrades into
a readable fallback rather than into a broken document.

Long meetings are summarized map-reduce style: each chunk becomes bullet notes,
then the notes are merged into the final minutes. Nothing is invented at either
step — the prompt says so explicitly, and unknown owners are recorded as
``未提及`` rather than guessed.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from .config import load_settings
from .jsonio import parse_json_object
from .meetings import Meeting

LOGGER = logging.getLogger("dsh_voice.summarize")

#: Characters per map step. Sized so a chunk is comfortably inside the model's
#: context once the prompt and the answer budget are added.
CHUNK_CHARS = 12_000

#: Fallback marker the prompt mandates for facts the transcript does not contain.
UNKNOWN = "未提及"

SUMMARY_SYSTEM = """你是资深会议纪要专家。你会收到一份会议的逐句记录（英文原文，部分带中文译文），
请据此写出准确、可执行的中文会议纪要。

铁律：
1. 只能依据记录内容。记录没说的，必须写「未提及」，绝不臆造人名、数字、日期、结论。
2. 保留所有专有名词、产品名、缩写、数字与单位，不要意译。
3. 输出必须是**合法 JSON**（json），不要 Markdown 代码块，不要任何解释文字。
4. 全部使用简体中文（专有名词保留原文）。"""

SUMMARY_SCHEMA = """{
  "title": "会议标题，不超过 20 字，概括主题",
  "overview": "2-4 句话说明这是什么会议、讨论了什么、总体结论倾向",
  "key_points": ["关键要点，每条一句完整的话，含具体数字或名称"],
  "decisions": ["已经明确做出的决定；没有则给空数组"],
  "action_items": [{"task": "要做的事", "owner": "负责人，未提及则填「未提及」", "due": "时间点，未提及则填「未提及」"}],
  "risks": ["风险、阻塞、未决问题；没有则给空数组"],
  "topics": ["3-6 个话题标签词"]
}"""

MAP_SYSTEM = """你在为一场会议的纪要做分块预处理。请把这一段记录压缩成中文要点笔记。

铁律：
1. 只依据记录内容，不补充任何外部信息或推测。
2. 数字、人名、产品名、日期原样保留。
3. 输出必须是**合法 JSON**，不要代码块，不要解释。

格式：
{"notes": ["要点1", "要点2"], "decisions": ["决定1"], "action_items": [{"task": "...", "owner": "...", "due": "..."}], "topics": ["话题"]}"""


@dataclass
class SummaryResult:
    """A summarization attempt and its product."""

    ok: bool
    markdown: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    chunks: int = 0
    ms: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize for the wire."""
        return {
            "ok": self.ok,
            "markdown": self.markdown,
            "structured": self.structured,
            "chunks": self.chunks,
            "ms": self.ms,
            **({"error": self.error} if self.error else {}),
        }


def _as_list(value: Any) -> list[Any]:
    """Coerce a model field into a list (a lone string becomes a one-item list)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def _as_str_list(value: Any) -> list[str]:
    """Coerce a model field into a list of non-empty strings."""
    return [text for text in (str(item).strip() for item in _as_list(value)) if text]


def _as_action_items(value: Any) -> list[dict[str, str]]:
    """Coerce the action-item field into ``{task, owner, due}`` records."""
    items: list[dict[str, str]] = []
    for entry in _as_list(value):
        if isinstance(entry, dict):
            task = str(entry.get("task") or entry.get("item") or "").strip()
            if not task:
                continue
            items.append({
                "task": task,
                "owner": str(entry.get("owner") or UNKNOWN).strip() or UNKNOWN,
                "due": str(entry.get("due") or entry.get("deadline") or UNKNOWN).strip() or UNKNOWN,
            })
        elif isinstance(entry, str) and entry.strip():
            items.append({"task": entry.strip(), "owner": UNKNOWN, "due": UNKNOWN})
    return items


def normalize_structured(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce a model answer into the canonical summary shape.

    Args:
        payload: Parsed JSON from the model.

    Returns:
        A dict with exactly the documented keys, every value the right type.
    """
    return {
        "title": str(payload.get("title") or "").strip(),
        "overview": str(payload.get("overview") or "").strip(),
        "key_points": _as_str_list(payload.get("key_points")),
        "decisions": _as_str_list(payload.get("decisions")),
        "action_items": _as_action_items(payload.get("action_items")),
        "risks": _as_str_list(payload.get("risks")),
        "topics": _as_str_list(payload.get("topics")),
    }


def render_markdown(structured: dict[str, Any], meeting: Meeting | None = None) -> str:
    """Render canonical structured minutes as Markdown.

    Args:
        structured: Output of :func:`normalize_structured`.
        meeting: Meeting the minutes describe, used for the footer and a title
            fallback.

    Returns:
        Markdown minutes.
    """
    title = structured.get("title") or (meeting.meta.title if meeting else "会议纪要")
    lines: list[str] = [f"# {title}", ""]

    overview = structured.get("overview") or ""
    if overview:
        lines += ["## 概览", "", overview, ""]

    key_points = structured.get("key_points") or []
    lines += ["## 关键要点", ""]
    lines += [f"- {point}" for point in key_points] if key_points else ["- 未提及"]
    lines.append("")

    decisions = structured.get("decisions") or []
    lines += ["## 决定事项", ""]
    lines += [f"- {item}" for item in decisions] if decisions else ["- 无"]
    lines.append("")

    actions = structured.get("action_items") or []
    lines += ["## 待办事项", ""]
    if actions:
        lines += ["| 事项 | 负责人 | 时间点 |", "| --- | --- | --- |"]
        lines += [f"| {item['task']} | {item['owner']} | {item['due']} |" for item in actions]
    else:
        lines.append("- 无")
    lines.append("")

    risks = structured.get("risks") or []
    lines += ["## 风险与未决问题", ""]
    lines += [f"- {item}" for item in risks] if risks else ["- 无"]
    lines.append("")

    topics = structured.get("topics") or []
    if topics:
        lines += ["## 话题", "", "　".join(f"`{topic}`" for topic in topics), ""]

    if meeting is not None:
        meta = meeting.meta
        footer = (
            f"*记录：{meta.segments} 段 · 时长 {_format_duration(meta.wall_seconds or meta.duration_s)}"
            f" · 生成于 {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}*"
        )
        lines += ["---", "", footer]
    return "\n".join(lines).rstrip() + "\n"


def _format_duration(seconds: float) -> str:
    """Format a duration as ``Hh MMm`` / ``MMm SSs`` / ``SSs``."""
    total = max(0, int(seconds))
    if total >= 3600:
        return f"{total // 3600}h{(total % 3600) // 60:02d}m"
    if total >= 60:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total}s"


def chunk_transcript(segments: list[Any], limit: int = CHUNK_CHARS) -> list[str]:
    """Split a transcript into chunks that fit the map step.

    Segmentation happens on utterance boundaries so a chunk never cuts a
    sentence in half.

    Args:
        segments: Meeting segments, in order.
        limit: Maximum characters per chunk.

    Returns:
        Chunk texts; a short transcript yields exactly one.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for segment in segments:
        line = segment.en.strip()
        if segment.zh.strip():
            line = f"{line}\n{segment.zh.strip()}"
        if not line:
            continue
        if size + len(line) > limit and current:
            chunks.append("\n\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [""]


class MeetingSummarizer:
    """Turns a transcript into structured minutes over the DeepSeek API."""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        settings = load_settings()
        self.api_base = (api_base or settings.api_base).rstrip("/")
        self.api_model = api_model or settings.api_model
        self.api_key = api_key if api_key is not None else settings.api_key
        self._timeout = timeout_s
        self._client: httpx.AsyncClient | None = None

    @property
    def available(self) -> bool:
        """Whether an API key was resolved."""
        return bool(self.api_key)

    async def _http(self) -> httpx.AsyncClient:
        """Lazily create the shared async client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Release the connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def _chat(self, system: str, user: str, *, max_tokens: int = 3000) -> dict[str, Any]:
        """Run one JSON-mode chat completion.

        Args:
            system: System prompt.
            user: User content.
            max_tokens: Answer budget.

        Returns:
            The parsed JSON object.

        Raises:
            RuntimeError: When no key is configured or the answer is unusable.
        """
        if not self.api_key:
            raise RuntimeError("no DeepSeek API key configured; cannot summarize")
        client = await self._http()
        response = await client.post(
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": self.api_model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "stream": False,
            },
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        parsed = parse_json_object(str(content))
        if parsed is None:
            raise RuntimeError(f"model returned no parsable JSON object: {str(content)[:200]}")
        return parsed

    async def summarize_meeting_async(self, meeting: Meeting) -> SummaryResult:
        """Summarize a recorded meeting.

        Args:
            meeting: The meeting to summarize.

        Returns:
            The result, including rendered Markdown and structured fields.
        """
        started = time.monotonic()
        if not meeting.segments:
            return SummaryResult(
                ok=False,
                error="meeting has no recorded segments",
                ms=int((time.monotonic() - started) * 1000),
            )
        chunks = chunk_transcript(meeting.segments)
        try:
            if len(chunks) == 1:
                structured = normalize_structured(await self._final_call(chunks[0], meeting))
            else:
                structured = normalize_structured(await self._reduce_call(chunks, meeting))
        except Exception as error:  # noqa: BLE001 - any failure is reported as data
            LOGGER.warning("summarization failed for %s: %s", meeting.meta.id, error)
            return SummaryResult(
                ok=False,
                error=f"{type(error).__name__}: {error}",
                chunks=len(chunks),
                ms=int((time.monotonic() - started) * 1000),
            )
        return SummaryResult(
            ok=True,
            markdown=render_markdown(structured, meeting),
            structured=structured,
            chunks=len(chunks),
            ms=int((time.monotonic() - started) * 1000),
        )

    async def _final_call(self, transcript: str, meeting: Meeting) -> dict[str, Any]:
        """One-shot summarization for a transcript that fits in the context."""
        meta = meeting.meta
        user = (
            f"会议标题：{meta.title}\n"
            f"开始时间：{meta.started_at}\n"
            f"录音时长：{_format_duration(meta.wall_seconds or meta.duration_s)}\n"
            f"记录段数：{meta.segments}\n\n"
            f"逐句记录如下（英文为识别原文，中文为译文）：\n\n{transcript}\n\n"
            f"请按以下 JSON 结构输出会议纪要：\n{SUMMARY_SCHEMA}"
        )
        return await self._chat(SUMMARY_SYSTEM, user)

    async def _reduce_call(self, chunks: list[str], meeting: Meeting) -> dict[str, Any]:
        """Map-reduce summarization for a transcript that does not fit."""
        notes: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            user = (
                f"这是会议的第 {index + 1}/{len(chunks)} 段记录：\n\n{chunk}\n\n"
                '请输出 {"notes": [...], "decisions": [...], '
                '"action_items": [{"task": "...", "owner": "...", "due": "..."}], "topics": [...]}'
            )
            notes.append(await self._chat(MAP_SYSTEM, user, max_tokens=2000))
        merged = json.dumps(notes, ensure_ascii=False)
        meta = meeting.meta
        user = (
            f"会议标题：{meta.title}\n开始时间：{meta.started_at}\n"
            f"记录段数：{meta.segments}（已分 {len(chunks)} 段预处理）\n\n"
            f"各段要点笔记（JSON 数组）：\n{merged}\n\n"
            f"请合并去重，按以下 JSON 结构输出最终会议纪要：\n{SUMMARY_SCHEMA}"
        )
        return await self._chat(SUMMARY_SYSTEM, user)

    def summarize_meeting(self, meeting: Meeting) -> SummaryResult:
        """Synchronous wrapper for tool and CLI callers.

        The pool is created and closed inside one event loop, so no cleanup is
        needed by the caller.

        Args:
            meeting: The meeting to summarize.

        Returns:
            The summarization result.
        """

        async def once() -> SummaryResult:
            try:
                return await self.summarize_meeting_async(meeting)
            finally:
                await self.aclose()

        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(once())
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(once())).result()


def transcript_for_prompt(meeting: Meeting, *, bilingual: bool = True) -> str:
    """Render a meeting transcript for a prompt, with a hard character cap.

    The cap is a guard against a pathologically long meeting blowing the context;
    :func:`chunk_transcript` is the mechanism that actually handles length, so
    this only ever truncates when map-reduce is disabled by the caller.
    """
    text = meeting.transcript_markdown(bilingual=bilingual)
    if len(text) <= CHUNK_CHARS * 4:
        return text
    return text[: CHUNK_CHARS * 4] + "\n\n…（记录过长，已截断）"


def extract_title_from_summary(markdown: str) -> str:
    """Pull the first ``# `` heading out of rendered minutes."""
    match = re.search(r"^#\s+(.+)$", markdown, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""
