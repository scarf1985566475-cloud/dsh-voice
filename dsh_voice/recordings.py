"""Turn an existing recording into a stored, summarized meeting.

The live path records as it listens; this is the other half of "record and
summarize" — a meeting that already happened, sitting in an audio file. It
produces exactly the same record shape (metadata, transcript journal, minutes)
so the panel and the tools treat both identically.

Translation here is batched rather than per-sentence: a file has no latency
budget, and one API call per sentence would make a one-hour recording slow and
needlessly expensive.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from .config import InterpreterSettings, load_settings
from .meetings import MeetingStore
from .summarize import MeetingSummarizer
from .translate import Translator

LOGGER = logging.getLogger("dsh_voice.recordings")

#: Fragments per translation request when translating a file.
TRANSLATE_BATCH = 8


async def build_meeting_from_file(
    path: str | Path,
    *,
    settings: InterpreterSettings | None = None,
    store: MeetingStore | None = None,
    language: str | None = None,
    title: str | None = None,
    translate: bool = True,
    summarize: bool = True,
    initial_prompt: str | None = None,
) -> dict[str, Any]:
    """Transcribe a recording into a meeting record, then summarize it.

    Args:
        path: Source media file (any ffmpeg-readable format).
        settings: Effective settings; loaded from the environment when omitted.
        store: Meeting store to write into; the configured one when omitted.
        language: Source language override.
        title: Meeting title; defaults to the file name.
        translate: Translate each segment into the target language.
        summarize: Generate minutes after recording.
        initial_prompt: Vocabulary hint passed to the ASR model.

    Returns:
        A payload with the meeting metadata, the transcript, and the minutes
        (or the summarization error, since a failed summary must not discard a
        successful transcript).
    """
    from .asr import transcribe_file

    settings = settings or load_settings()
    store = store or MeetingStore(settings.meetings_dir)
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"audio file not found: {source}")

    started = time.monotonic()
    loop = asyncio.get_running_loop()
    transcript = await loop.run_in_executor(
        None, lambda: transcribe_file(source, language=language, initial_prompt=initial_prompt),
    )
    asr_ms = int((time.monotonic() - started) * 1000)
    LOGGER.info(
        "transcribed %s: %d segments in %dms",
        source.name, len(transcript.segments), asr_ms,
    )

    recorder = store.create(
        title=title or source.stem,
        source="file",
        source_path=str(source),
        language=language or settings.language,
        target_language=settings.target_language,
        model=settings.model,
    )

    texts = [segment.text for segment in transcript.segments]
    translations: list[str] = ["" for _ in texts]
    translate_ms = 0
    translations_ok = True
    if translate and texts and settings.api_key:
        translator = Translator(
            api_base=settings.api_base,
            api_model=settings.api_model,
            api_key=settings.api_key,
            target_language=settings.target_language,
        )
        translate_started = time.monotonic()
        try:
            results = await translator.translate_many_async(texts, batch_size=TRANSLATE_BATCH)
            translations = [result.text for result in results]
            translations_ok = all(result.ok for result in results)
        finally:
            await translator.aclose()
        translate_ms = int((time.monotonic() - translate_started) * 1000)

    for index, segment in enumerate(transcript.segments):
        segment_id = index + 1
        recorder.record_segment(segment_id, segment.start, segment.end, segment.text, asr_ms)
        if translations[index]:
            recorder.record_translation(segment_id, translations[index], translate_ms)
    recorder.finalize(wall_seconds=transcript.duration)

    meeting = store.read(recorder.meta.id)
    payload: dict[str, Any] = {
        "ok": True,
        # Flat metadata, not the nested read shape: this is what a tool caller
        # wants to pass straight on to read_meeting / summarize_meeting.
        "meeting": meeting.meta.as_dict(),
        "transcript": meeting.transcript_text(),
        "translated": bool(translate and texts and settings.api_key and translations_ok),
        "asr_ms": asr_ms,
        "translate_ms": translate_ms,
    }

    if summarize and meeting.segments:
        summarizer = MeetingSummarizer(
            api_base=settings.api_base,
            api_model=settings.api_model,
            api_key=settings.api_key,
        )
        try:
            result = await summarizer.summarize_meeting_async(meeting)
        finally:
            await summarizer.aclose()
        if result.ok:
            store.save_summary(meeting.meta.id, result.markdown, result.structured)
            payload["summary"] = result.markdown
            payload["summary_structured"] = result.structured
            payload["summary_ms"] = result.ms
        else:
            # The transcript is the valuable part; a failed summary is a warning.
            store.mark_summary_failed(meeting.meta.id)
            payload["summary_error"] = result.error
    return payload
