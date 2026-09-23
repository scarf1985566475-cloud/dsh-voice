"""English-to-Chinese translation over the DeepSeek chat API.

The prompt is a simultaneous-interpreter brief rather than a generic
"translate this" instruction: subtitle-length output, no commentary, and a
rolling glossary of the previous sentence so terminology stays stable across
segment boundaries — the failure a per-sentence translator shows most.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from .config import load_settings
from .jsonio import parse_json_object

LOGGER = logging.getLogger("dsh_voice.translate")

SYSTEM_PROMPT = (
    "You are a professional simultaneous interpreter producing live Chinese subtitles "
    "from spoken English.\n"
    "Rules:\n"
    "1. Output ONLY the Chinese translation — no quotes, no pinyin, no explanations, no notes.\n"
    "2. Translate faithfully and completely; never summarize away content.\n"
    "3. Use natural, concise spoken Chinese suited to subtitles.\n"
    "4. Keep proper nouns, product names, numbers, units and acronyms accurate; keep a well-known "
    "English acronym as-is when Chinese speakers normally say it.\n"
    "5. The input comes from automatic speech recognition: it may lack punctuation or contain a "
    "misheard word. Infer the intended meaning from context and translate that.\n"
    "6. If the input is empty, or clearly not speech (noise, a stray symbol), output nothing at all."
)

#: Sentence-final punctuation that ends a context-worthy unit.
_SENTENCE_ENDINGS = (".", "!", "?", "。", "！", "？")


@dataclass
class TranslationResult:
    """One translation attempt and whether it succeeded."""

    text: str
    ok: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize for the wire."""
        return {"text": self.text, "ok": self.ok, **({"error": self.error} if self.error else {})}


class Translator:
    """A reusable DeepSeek translation client with a small rolling context.

    Attributes:
        api_base: DeepSeek-compatible API base URL.
        api_model: Chat model name.
        target_language: Target language code (only ``zh`` is wired to a prompt).
        context_turns: How many previous source sentences to keep as reference.
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_model: str | None = None,
        api_key: str | None = None,
        target_language: str | None = None,
        context_turns: int = 3,
        timeout_s: float = 20.0,
    ) -> None:
        settings = load_settings()
        self.api_base = (api_base or settings.api_base).rstrip("/")
        self.api_model = api_model or settings.api_model
        self.api_key = api_key if api_key is not None else settings.api_key
        self.target_language = target_language or settings.target_language
        self.context_turns = context_turns
        self._timeout = timeout_s
        self._client: httpx.AsyncClient | None = None

    @property
    def available(self) -> bool:
        """Whether an API key was resolved."""
        return bool(self.api_key)

    async def _http(self) -> httpx.AsyncClient:
        """Lazily create the shared async client (one connection pool per process)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Release the connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _messages(self, text: str, context: list[str] | None) -> list[dict[str, str]]:
        """Build the chat messages, folding recent source sentences into the brief."""
        system = SYSTEM_PROMPT
        if context:
            recent = [item for item in context if item.strip()][-self.context_turns:]
            if recent:
                system += (
                    "\n\nPrevious source sentences, for terminology continuity only "
                    "(do NOT translate or repeat them):\n"
                    + "\n".join(f"- {item}" for item in recent)
                )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]

    async def translate_async(self, text: str, context: list[str] | None = None) -> TranslationResult:
        """Translate one English fragment into Chinese.

        Args:
            text: Source text (typically one ASR segment).
            context: Recent source sentences used only as terminology context.

        Returns:
            The translation, or a failed result carrying a human-readable error.
        """
        stripped = text.strip()
        if not stripped:
            return TranslationResult(text="")
        if not self.api_key:
            return TranslationResult(text="", ok=False, error="no DeepSeek API key configured")
        payload = {
            "model": self.api_model,
            "messages": self._messages(stripped, context),
            "temperature": 0.0,
            "max_tokens": 800,
            "stream": False,
        }
        try:
            client = await self._http()
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:300] if error.response is not None else str(error)
            LOGGER.warning("translation HTTP error: %s", detail)
            return TranslationResult(text="", ok=False, error=f"HTTP {error.response.status_code}: {detail}")
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as error:
            LOGGER.warning("translation failed: %s", error)
            return TranslationResult(text="", ok=False, error=str(error))
        return TranslationResult(text=str(content).strip())

    async def translate_many_async(
        self,
        texts: list[str],
        *,
        batch_size: int = 8,
        context: list[str] | None = None,
    ) -> list[TranslationResult]:
        """Translate many fragments, batching them into few API calls.

        The live path must translate one sentence at a time because latency is
        the whole point there. A *file* has no such constraint, and one call per
        sentence would make summarizing a one-hour recording both slow and
        needlessly expensive — so fragments go out in batches and are matched
        back by index.

        A batch whose answer does not parse cleanly falls back to per-item
        translation, so speed never costs correctness.

        Args:
            texts: Source fragments, in order.
            batch_size: Fragments per request.
            context: Recent source sentences used only as terminology context.

        Returns:
            One result per input, in the same order.
        """
        results: list[TranslationResult] = [TranslationResult(text="") for _ in texts]
        if not self.api_key:
            return [TranslationResult(text="", ok=False, error="no DeepSeek API key configured") for _ in texts]
        step = max(1, batch_size)
        for start in range(0, len(texts), step):
            window = texts[start:start + step]
            pending = [(offset, text) for offset, text in enumerate(window) if text.strip()]
            if not pending:
                continue
            if len(pending) == 1:
                results[start + pending[0][0]] = await self.translate_async(pending[0][1], context)
                continue
            parsed = await self._translate_batch(pending, context)
            if parsed is None:
                for offset, text in pending:
                    results[start + offset] = await self.translate_async(text, context)
                continue
            for (offset, _text), translation in zip(pending, parsed):
                results[start + offset] = TranslationResult(text=translation)
        return results

    async def _translate_batch(
        self,
        pending: list[tuple[int, str]],
        context: list[str] | None,
    ) -> list[str] | None:
        """Translate one batch in a single JSON-mode request.

        Args:
            pending: ``(index, text)`` pairs to translate.
            context: Recent source sentences used only as terminology context.

        Returns:
            Translations in input order, or ``None`` when the answer is unusable.
        """
        items = [text.strip() for _offset, text in pending]
        system = (
            SYSTEM_PROMPT
            + f"\n\n你这次会收到一个 JSON 数组，共 {len(items)} 条独立句子。"
            + "请逐条翻译，保持顺序与条数完全一致，输出必须是合法 JSON："
            + '{"translations": ["第一条译文", "第二条译文", ...]}。不要输出任何其它文字。'
        )
        if context:
            recent = [item for item in context if item.strip()][-self.context_turns:]
            if recent:
                system += "\n\n术语参考（不要翻译这些）：\n" + "\n".join(f"- {item}" for item in recent)
        payload = {
            "model": self.api_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "max_tokens": 4000,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        try:
            client = await self._http()
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as error:
            LOGGER.warning("batch translation failed, falling back to per-item: %s", error)
            return None
        parsed = parse_json_object(str(content))
        if parsed is None:
            return None
        translations = parsed.get("translations")
        if not isinstance(translations, list) or len(translations) != len(items):
            LOGGER.warning("batch translation answer had %s items, expected %s", 
                           len(translations) if isinstance(translations, list) else "no", len(items))
            return None
        return [str(item).strip() for item in translations]

    def translate(self, text: str, context: list[str] | None = None) -> TranslationResult:
        """Synchronous translation for tool and CLI callers.

        The connection pool is created and closed inside one event loop, so this
        method is self-contained: callers must not follow it with
        :meth:`aclose` (the loop it used is already gone).

        Args:
            text: Source text.
            context: Recent source sentences used only as terminology context.

        Returns:
            The translation, or a failed result carrying a human-readable error.
        """

        async def once() -> TranslationResult:
            try:
                return await self.translate_async(text, context)
            finally:
                await self.aclose()

        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(once())
        # Already inside a loop (an async host): run on a private loop in a worker
        # thread rather than deadlocking on the running one.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(once())).result()


def is_sentence_end(text: str) -> bool:
    """Whether a fragment looks complete enough to anchor terminology context."""
    stripped = text.strip()
    return bool(stripped) and stripped.endswith(_SENTENCE_ENDINGS)


def push_context(context: list[str], text: str, limit: int = 6) -> list[str]:
    """Append a source sentence to the rolling context, trimming from the front.

    Args:
        context: Mutable rolling list.
        text: New source sentence.
        limit: Maximum retained entries.

    Returns:
        The same list, for chaining.
    """
    stripped = text.strip()
    if stripped:
        context.append(stripped)
        del context[:-limit]
    return context
