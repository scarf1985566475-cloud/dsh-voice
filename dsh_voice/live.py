"""Live interpretation session: streaming audio in, captions and translations out.

Pipeline, per connection::

    browser mic -> PCM frames -> rolling window -> Qwen3-ASR
                                               -> commit stable sentences
                                               -> DeepSeek (EN->ZH) -> JSON events

The hard part of simultaneous interpretation is *when to commit*. A silence
detector alone is not enough: a speaker who never pauses would produce nothing
until the hard cap. So the session re-decodes a short rolling window and commits
each decoded sentence whose end has fallen far enough behind the audio frontier
that a re-decode would not change it. Sentences therefore commit continuously
while the speaker is still talking, which is what makes the output simultaneous
rather than consecutive.

Three further rules shape the design. Decoding is serialized process-wide (MLX
owns one GPU stream), so a *preview* decode is dropped whenever work is already
queued — a stale preview is worth less than the commit it would delay. Every
event for one span carries the same id, so a client replaces a preview line in
place. And the audio buffer is trimmed against the uncommitted window, so a long
meeting stays memory-flat.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np

from .config import SAMPLE_RATE, InterpreterSettings
from .meetings import MeetingRecorder
from .translate import Translator, is_sentence_end, push_context
from .vad import Segment, StreamingSegmenter

LOGGER = logging.getLogger("dsh_voice.live")

#: One shared decode worker: MLX decodes on a single GPU stream, so a second
#: worker would only contend with the first.
_DECODE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dsh-voice-asr")

#: Audio retained per session beyond the uncommitted window.
BUFFER_KEEP_S = 60.0

#: Pending decode jobs per session. Flushes block when full; previews are dropped.
QUEUE_LIMIT = 12

#: Preview cadence: how often the ticker considers re-decoding the window.
TICK_S = 0.2

#: A decoded sentence is committed once its end trails the audio frontier by
#: this much, because a later decode would land on the same words. Shorter than
#: a second: every extra moment here is caption latency the viewer feels.
COMMIT_MARGIN_S = 0.6

#: Longest span a *preview* decode may cover. A preview re-decodes the open
#: window every partial interval, so on a slow (or memory-pressured) machine an
#: uncapped window turns the preview loop into a backlog that never drains.
#: Oversized windows are shown as their tail only: display only, while the
#: commit still comes from a flush or a full-window preview.
PREVIEW_MAX_WINDOW_S = 6.0

#: Preview gain applied to every decode window. A far-field or quiet microphone
#: hands the recognizer amplitudes well under its speech range, and on that signal the
#: decoder falls into repetition loops; lifting the window toward -20 dBFS
#: before decoding removes them. The peak ceiling and the gain cap keep a silent
#: window from turning its noise floor into a shout.
TARGET_RMS_DB = -20.0
MAX_GAIN_DB = 18.0
PEAK_CEILING = 0.98

#: The recognizer's stock answers to non-speech audio. A span is dropped only when its
#: text is one of these *and* its audio is markedly quieter than speech in the
#: same window, so a real "thank you" at normal level survives.
_HALLUCINATION_FILLERS = frozenset({
    "thank you",
    "thanks",
    "thank you very much",
    "thank you so much",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "cheers",
    "bye",
})

#: Sentence-final characters used to avoid committing an unfinished clause.
_SENTENCE_ENDINGS = (".", "!", "?", "。", "！", "？", "…")


def _rms(audio: np.ndarray) -> float:
    """Root-mean-square level of a waveform (0.0 for empty input)."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _normalized_text(text: str) -> str:
    """Collapse a caption to its comparable form: lowercase, no edge punctuation."""
    return " ".join(text.lower().split()).strip(" .,!?…:;\"'")


def _boost_for_decode(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale a decode window toward :data:`TARGET_RMS_DB`, gain only and peak-safe.

    Args:
        audio: The window about to be decoded, mono float32 in [-1, 1].

    Returns:
        The audio to decode and the applied gain in dB (``0.0`` when the window
        was already at or above the target and was left untouched).
    """
    level = _rms(audio)
    if level <= 1e-6:
        return audio, 0.0
    gain = (10.0 ** (TARGET_RMS_DB / 20.0)) / level
    if gain <= 1.0:
        return audio, 0.0
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        gain = min(gain, PEAK_CEILING / peak)
    gain = min(gain, 10.0 ** (MAX_GAIN_DB / 20.0))
    if gain <= 1.0:
        return audio, 0.0
    return (audio * gain).astype(np.float32), 20.0 * float(np.log10(gain))

#: Bound on how long shutdown waits for an in-flight decode and translation.
SHUTDOWN_TIMEOUT_S = 45.0


@dataclass
class _Job:
    """One queued decode request, positioned in absolute stream samples."""

    kind: str  # 'preview' | 'flush'
    start_sample: int
    end_sample: int
    flush_at: int = 0
    enqueued_at: float = field(default_factory=time.monotonic)
    #: Set by the worker once the job (including its translations) is done, so a
    #: caller can wait for a flush instead of guessing at a grace period.
    done: asyncio.Future[None] | None = None


@dataclass
class SessionStats:
    """Rolling counters a client may display."""

    segments: int = 0
    previews: int = 0
    translations: int = 0
    failed_translations: int = 0
    dropped_stale: int = 0
    dropped_hallucinations: int = 0
    audio_seconds: float = 0.0
    last_asr_ms: int = 0
    last_translate_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Serialize for the wire."""
        return {
            "segments": self.segments,
            "previews": self.previews,
            "translations": self.translations,
            "failed_translations": self.failed_translations,
            "dropped_stale": self.dropped_stale,
            "dropped_hallucinations": self.dropped_hallucinations,
            "audio_seconds": round(self.audio_seconds, 2),
            "last_asr_ms": self.last_asr_ms,
            "last_translate_ms": self.last_translate_ms,
        }


@dataclass
class _DecodedSentence:
    """One decoded sentence, in absolute stream coordinates."""

    start_sample: int
    end_sample: int
    text: str
    #: Detected language code (``en``/``zh``/...), carried to the client badge.
    language: str | None = None


class LiveSession:
    """One client's live interpretation session.

    Coordinates: ``self._origin`` is the stream offset of ``self._buffer[0]``,
    ``self._consumed`` is the offset the segmenter has already seen, and
    ``self._committed`` is the offset of the first sample that is not yet part of
    a committed sentence — the start of the uncommitted window.

    Attributes:
        settings: Effective configuration for the session.
        stats: Rolling counters.
    """

    def __init__(
        self,
        settings: InterpreterSettings,
        send: Callable[[dict[str, Any]], Awaitable[None]],
        *,
        translator: Translator | None = None,
        recorder: MeetingRecorder | None = None,
    ) -> None:
        self.settings = settings
        self.send = send
        self.stats = SessionStats()
        self.recorder = recorder
        self._started_monotonic = time.monotonic()
        self.translator = translator if translator is not None else Translator(
            api_base=settings.api_base,
            api_model=settings.api_model,
            api_key=settings.api_key,
            target_language=settings.target_language,
        )

        self._segmenter = StreamingSegmenter(
            sample_rate=SAMPLE_RATE,
            start_speech_ms=settings.start_speech_ms,
            end_silence_ms=settings.end_silence_ms,
            max_segment_s=settings.max_segment_s,
            min_segment_s=settings.min_segment_s,
        )
        self._buffer = np.zeros(0, dtype=np.float32)
        self._origin = 0
        self._consumed = 0
        self._committed = 0
        self._commit_margin = int(COMMIT_MARGIN_S * SAMPLE_RATE)
        self._jobs: asyncio.Queue[_Job | None] = asyncio.Queue(maxsize=QUEUE_LIMIT)
        self._tasks: list[asyncio.Task[None]] = []
        self._worker_task: asyncio.Task[None] | None = None
        self._closed = False

        self._segment_id = 0
        self._open_id = 1
        self._last_empty_preview_id = 0
        self._last_preview_end = 0
        self._partial_translated_at = 0.0
        self._context: list[str] = []
        self._translate_partials = settings.translate_partials
        self._translate_tasks: set[asyncio.Task[None]] = set()
        self._last_committed_text = ""
        #: The language the decoder reported for the newest window, reused for
        #: previews and flush tails that were decoded from older windows.
        self._last_language: str | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Report readiness and start the decode worker and preview ticker."""
        await self.send({
            "type": "ready",
            "sample_rate": SAMPLE_RATE,
            "settings": self.settings.as_public_dict(),
            **({"meeting": self.recorder.meta.as_dict()} if self.recorder is not None else {}),
        })
        self._worker_task = asyncio.create_task(self._worker_loop(), name="dsh-voice-worker")
        self._tasks = [
            self._worker_task,
            asyncio.create_task(self._preview_loop(), name="dsh-voice-preview"),
        ]

    async def stop(self) -> None:
        """Commit the tail, let the worker finish it, and shut down.

        The worker is stopped with a sentinel rather than cancelled: a plain
        cancel would abort a translation already in flight, which silently drops
        the last sentence's caption — exactly the moment a user is watching.
        """
        if self._closed:
            return
        self._closed = True
        for task in self._tasks:
            if task is not self._worker_task:
                task.cancel()
        for task in self._tasks:
            if task is not self._worker_task:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._tasks = []

        self._segmenter.flush()
        if self._consumed > self._committed:
            await self._enqueue_flush(self._consumed)
        # Awaited rather than put_nowait: a full queue would otherwise drop the
        # sentinel and leave shutdown waiting out its whole timeout. The worker
        # is still draining, so this resolves as soon as it takes one job.
        await self._jobs.put(None)
        if self._worker_task is not None:
            try:
                await asyncio.wait_for(self._worker_task, timeout=SHUTDOWN_TIMEOUT_S)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # A hung decode must not hold shutdown open; drain what we can.
                self._worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._worker_task
            self._worker_task = None
        while True:
            try:
                job = self._jobs.get_nowait()
            except asyncio.QueueEmpty:
                break
            if job is None:
                continue
            with contextlib.suppress(Exception):
                await self._process(job)
            if job.done is not None and not job.done.done():
                job.done.set_result(None)
        if self._translate_tasks:
            # Translations now run off the decode worker; let the last ones land
            # before the record is finalized and the HTTP client is closed.
            await asyncio.gather(*list(self._translate_tasks), return_exceptions=True)
        if self.recorder is not None:
            # Finalized last, so the record's segment count and end time include
            # everything the drain just committed.
            self.recorder.finalize(time.monotonic() - self._started_monotonic)
        await self.translator.aclose()

    @property
    def meeting_id(self) -> str | None:
        """Id of the meeting being recorded, or ``None`` when recording is off."""
        return self.recorder.meta.id if self.recorder is not None else None

    # ── input ────────────────────────────────────────────────────────────────

    async def process_audio(self, samples: np.ndarray) -> None:
        """Append client audio, advance the segmenter, and queue any closed span.

        Args:
            samples: Mono float32 samples at :data:`config.SAMPLE_RATE`.
        """
        if self._closed or samples.size == 0:
            return
        self._buffer = np.concatenate((self._buffer, np.ascontiguousarray(samples, dtype=np.float32)))
        self.stats.audio_seconds += samples.size / SAMPLE_RATE

        index = max(0, self._consumed - self._origin)
        if index < self._buffer.size:
            spans = self._segmenter.feed(self._buffer[index:])
            self._consumed = self._origin + self._buffer.size
            for span in spans:
                await self._enqueue_flush(span.end_sample)
        self._trim()

    # ── queueing ─────────────────────────────────────────────────────────────

    async def _enqueue_flush(self, end_sample: int) -> None:
        """Queue a decode that commits the whole window up to ``end_sample``."""
        if end_sample <= self._committed:
            return
        job = _Job(kind="flush", start_sample=self._committed, end_sample=end_sample, flush_at=end_sample)
        try:
            self._jobs.put_nowait(job)
        except asyncio.QueueFull:
            self._drop_previews()
            await self._jobs.put(job)

    def _drop_previews(self) -> None:
        """Discard queued preview jobs while preserving flush order."""
        kept: list[_Job | None] = []
        while True:
            try:
                job = self._jobs.get_nowait()
            except asyncio.QueueEmpty:
                break
            if job is None or job.kind == "flush":
                kept.append(job)
        for job in kept:
            with contextlib.suppress(asyncio.QueueFull):
                self._jobs.put_nowait(job)

    async def _preview_loop(self) -> None:
        """Re-decode the uncommitted window while speech continues."""
        while True:
            await asyncio.sleep(TICK_S)
            if self._closed:
                continue
            window = self._consumed - self._committed
            if window < self.settings.partial_interval_s * SAMPLE_RATE * 0.5:
                continue
            if self._consumed - self._last_preview_end < self.settings.partial_interval_s * SAMPLE_RATE:
                continue
            if not self._jobs.empty():
                # A decode is already pending; a stale preview is not worth queueing.
                continue
            self._last_preview_end = self._consumed
            # Bound the preview decode. An unbounded rolling window is the
            # failure mode on a machine that cannot decode faster than it
            # listens: the queue never drains and captions arrive seconds late.
            # An oversized window is previewed as its tail only (display-only in
            # _process); the commit still comes from a flush.
            preview_start = max(self._committed, self._consumed - int(PREVIEW_MAX_WINDOW_S * SAMPLE_RATE))
            await self._jobs.put(_Job(kind="preview", start_sample=preview_start, end_sample=self._consumed))

    async def _worker_loop(self) -> None:
        """Decode and translate queued jobs in order until the sentinel arrives."""
        while True:
            job = await self._jobs.get()
            if job is None:
                return
            try:
                await self._process(job)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - one bad span must not kill the session
                LOGGER.exception("live segment failed")
                await self._send_error(f"{type(error).__name__}: {error}")
            finally:
                if job.done is not None and not job.done.done():
                    job.done.set_result(None)

    # ── decoding ─────────────────────────────────────────────────────────────

    def _slice(self, start_sample: int, end_sample: int) -> tuple[np.ndarray | None, int]:
        """Resolve a stream range against the buffer.

        Args:
            start_sample: Absolute start of the wanted range.
            end_sample: Absolute end of the wanted range.

        Returns:
            The audio and the absolute offset it starts at, or ``(None, 0)`` when
            trimming already discarded part of the range.
        """
        lo = max(0, start_sample - self._origin)
        hi = min(self._buffer.size, end_sample - self._origin)
        if hi <= lo:
            return None, 0
        expected = end_sample - start_sample
        audio = np.ascontiguousarray(self._buffer[lo:hi], dtype=np.float32)
        if audio.size < expected * 0.95:
            return None, 0
        return audio, self._origin + lo

    def _sentences(self, job: _Job, window_start: int, transcript: Any) -> list[_DecodedSentence]:
        """Convert a decode result into absolute-coordinate sentences.

        Args:
            job: The job being processed.
            window_start: Absolute offset of the decoded audio's first sample.
            transcript: The ASR result.

        Returns:
            Sentences in stream order.
        """
        sentences: list[_DecodedSentence] = []
        for segment in transcript.segments:
            start = window_start + int(segment.start * SAMPLE_RATE)
            end = window_start + int(segment.end * SAMPLE_RATE)
            if end <= start:
                end = min(job.end_sample, start + int(0.2 * SAMPLE_RATE))
            sentences.append(_DecodedSentence(
                start_sample=start, end_sample=end, text=segment.text,
                language=transcript.language,
            ))
        if not sentences and transcript.text.strip():
            sentences.append(_DecodedSentence(
                start_sample=window_start, end_sample=job.end_sample, text=transcript.text.strip(),
                language=transcript.language,
            ))
        return sentences

    def _split_committable(self, sentences: list[_DecodedSentence], job: _Job) -> tuple[list[_DecodedSentence], str]:
        """Split a decode into the sentences to commit now and the preview text.

        On a flush everything commits. On a preview only sentences that end well
        behind the audio frontier commit, and a trailing clause without terminal
        punctuation is held back so a half-sentence is never published.

        Args:
            sentences: Decoded sentences in stream order.
            job: The job being processed.

        Returns:
            The committable sentences and the remaining preview text.
        """
        if job.kind == "flush":
            return sentences, ""
        frontier = job.end_sample - self._commit_margin
        stable = [sentence for sentence in sentences if sentence.end_sample <= frontier]
        while stable and not stable[-1].text.rstrip().endswith(_SENTENCE_ENDINGS):
            stable.pop()
        committed_until = max((sentence.end_sample for sentence in stable), default=self._committed)
        tail = " ".join(
            sentence.text for sentence in sentences if sentence.end_sample > committed_until
        ).strip()
        return stable, tail

    def _drop_hallucinations(
        self,
        sentences: list[_DecodedSentence],
        audio: np.ndarray,
        window_start: int,
    ) -> list[_DecodedSentence]:
        """Discard filler output on noise and immediate repeats.

        On a quiet or far-field microphone the decoder answers near-silence with
        a small vocabulary of stock phrases. A span is dropped only when its text
        is one of those phrases *and* its audio is markedly quieter than the
        window's speech level, so a genuine thank-you at normal volume survives.
        A sentence identical to the one just committed is also dropped: that is
        the other shape of the same repetition loop.
        """
        if not sentences:
            return sentences
        window_level = _rms(audio)
        kept: list[_DecodedSentence] = []
        previous = self._last_committed_text
        for sentence in sentences:
            text = _normalized_text(sentence.text)
            lo = max(0, sentence.start_sample - window_start)
            hi = min(audio.size, sentence.end_sample - window_start)
            span_level = _rms(audio[lo:hi]) if hi > lo else 0.0
            duration = (sentence.end_sample - sentence.start_sample) / SAMPLE_RATE
            if (
                text in _HALLUCINATION_FILLERS
                and window_level > 0
                and span_level < window_level * 0.45
            ):
                self.stats.dropped_hallucinations += 1
                continue
            if text and text == previous and duration < 3.0:
                self.stats.dropped_hallucinations += 1
                continue
            kept.append(sentence)
            previous = text
        return kept

    async def _process(self, job: _Job) -> None:
        """Decode one job, commit its stable sentences, and publish the preview."""
        # Never decode audio that is already committed. A job can sit in the
        # queue while a later one commits past it, and re-decoding that audio
        # would republish sentences the viewer has already read.
        start = max(job.start_sample, self._committed)
        if start >= job.end_sample:
            self.stats.dropped_stale += 1
            return
        audio, window_start = self._slice(start, job.end_sample)
        if audio is None:
            self.stats.dropped_stale += 1
            return

        # A truncated preview is display-only: committing from it would skip the
        # audio between the last commit and its start.
        display_only = job.kind == "preview" and job.start_sample > self._committed

        started = time.monotonic()
        loop = asyncio.get_running_loop()
        from .asr import transcribe_array

        decode_audio, _gain_db = _boost_for_decode(audio)
        transcript = await loop.run_in_executor(
            _DECODE_POOL,
            lambda: transcribe_array(decode_audio, streaming=True),
        )
        asr_ms = int((time.monotonic() - started) * 1000)
        self.stats.last_asr_ms = asr_ms
        self._last_language = transcript.language or self._last_language
        if job.kind == "preview":
            self.stats.previews += 1

        sentences = [
            sentence for sentence in self._sentences(job, window_start, transcript)
            if sentence.end_sample > self._committed  # a stale job may predate the last commit
        ]
        sentences = self._drop_hallucinations(sentences, audio, window_start)
        if display_only:
            stable, tail = [], " ".join(sentence.text for sentence in sentences).strip()
        else:
            stable, tail = self._split_committable(sentences, job)

        for sentence in stable:
            await self._commit(sentence, asr_ms)
        if job.kind == "flush":
            await self._commit_tail(sentences, job)
        await self._publish_preview(tail, job, asr_ms)

    async def _commit(self, sentence: _DecodedSentence, asr_ms: int) -> None:
        """Publish one committed sentence and translate it."""
        segment_id = self._open_id
        self._open_id += 1
        self._segment_id = max(self._segment_id, segment_id)
        self._committed = max(self._committed, sentence.end_sample)
        self.stats.segments += 1
        await self.send({
            "type": "final",
            "id": segment_id,
            "start": sentence.start_sample / SAMPLE_RATE,
            "end": sentence.end_sample / SAMPLE_RATE,
            "duration": round((sentence.end_sample - sentence.start_sample) / SAMPLE_RATE, 2),
            "text": sentence.text,
            "language": sentence.language or self._last_language,
            "asr_ms": asr_ms,
            "stats": self.stats.as_dict(),
        })
        if self.recorder is not None:
            # Journalled before the translation exists: the transcript must
            # survive a crash even though the Chinese arrives later.
            self.recorder.record_segment(
                segment_id,
                sentence.start_sample / SAMPLE_RATE,
                sentence.end_sample / SAMPLE_RATE,
                sentence.text,
                asr_ms,
            )
        self._last_committed_text = _normalized_text(sentence.text)
        self._schedule_translation(segment_id, sentence.text, final=True)

    async def _commit_tail(self, sentences: list[_DecodedSentence], job: _Job) -> None:
        """Commit whatever remains in a flush that produced no usable segments."""
        if self._committed >= job.flush_at:
            return
        # Only the still-uncommitted part counts: a flush that overlaps sentences
        # already published by a preview must not repeat them.
        remaining = [sentence for sentence in sentences if sentence.end_sample > self._committed]
        text = " ".join(sentence.text for sentence in remaining).strip()
        end = max(job.flush_at, self._committed)
        if not text:
            # Silence: advance the window without publishing an empty caption.
            self._committed = end
            return
        await self._commit(_DecodedSentence(self._committed, end, text), self.stats.last_asr_ms)

    async def _publish_preview(self, text: str, job: _Job, asr_ms: int) -> None:
        """Publish the not-yet-committed tail under the id its commit will reuse.

        The id is deliberately the *next* commit id, so the row a viewer is
        reading is replaced in place by its final rather than duplicated.
        """
        if job.kind == "flush":
            return
        if not text and self._open_id == self._last_empty_preview_id:
            return  # nothing new to say, and the row already shows an empty preview
        self._last_empty_preview_id = self._open_id if not text else 0
        await self.send({
            "type": "partial",
            "id": self._open_id,
            "start": self._committed / SAMPLE_RATE,
            "text": text,
            "language": self._last_language,
            "asr_ms": asr_ms,
        })
        self._schedule_translation(self._open_id, text, final=False)

    # ── translation ──────────────────────────────────────────────────────────

    def _schedule_translation(self, segment_id: int, text: str, *, final: bool) -> None:
        """Run one translation off the decode worker.

        The DeepSeek round-trip is a second of network; awaiting it inline stalls
        the next decode by that much, which is exactly how a caption queue starts
        growing. Order does not matter to the client: it merges translation
        events into their row by id.
        """
        task = asyncio.create_task(
            self._maybe_translate(segment_id, text, final=final),
            name="dsh-voice-translate",
        )
        self._translate_tasks.add(task)
        task.add_done_callback(self._translate_tasks.discard)

    async def _maybe_translate(self, segment_id: int, text: str, *, final: bool) -> None:
        """Translate a fragment when enabled, emitting a result event either way."""
        if not self.settings.translate:
            return
        if not text:
            if final:
                await self.send({"type": "translation", "id": segment_id, "text": "", "final": True})
            return
        if not final:
            if not self._translate_partials:
                return
            now = time.monotonic()
            if now - self._partial_translated_at < self.settings.partial_translate_interval_s:
                return
            if not self._jobs.empty():
                # A flush is pending: spend the API call on that instead of a preview.
                return
            self._partial_translated_at = now
        started = time.monotonic()
        result = await self.translator.translate_async(text, self._context)
        elapsed = int((time.monotonic() - started) * 1000)
        self.stats.last_translate_ms = elapsed
        if final:
            if result.ok:
                self.stats.translations += 1
            else:
                self.stats.failed_translations += 1
            if is_sentence_end(text):
                push_context(self._context, text)
            if self.recorder is not None and result.text:
                self.recorder.record_translation(segment_id, result.text, elapsed)
        await self.send({
            "type": "translation",
            "id": segment_id,
            "text": result.text,
            "final": final,
            "ok": result.ok,
            **({"error": result.error} if result.error else {}),
            "ms": elapsed,
            "stats": self.stats.as_dict() if final else None,
        })

    # ── buffer bookkeeping ───────────────────────────────────────────────────

    def _trim(self) -> None:
        """Drop audio nothing can need any more, never cutting the open window.

        The uncommitted window is the whole working set: flushes decode from
        ``_committed`` forward and never re-read committed audio, so everything
        before it is dead weight.
        """
        keep = int(BUFFER_KEEP_S * SAMPLE_RATE)
        keep = max(keep, self._consumed - self._committed + SAMPLE_RATE)
        excess = self._buffer.size - keep
        if excess <= 0:
            return
        self._buffer = self._buffer[excess:].copy()
        self._origin += excess

    async def _send_error(self, message: str) -> None:
        """Report a non-fatal problem to the client."""
        with contextlib.suppress(Exception):
            await self.send({"type": "error", "message": message})

    # ── client control ───────────────────────────────────────────────────────

    def configure(self, *, translate_partials: bool | None = None) -> None:
        """Apply a runtime toggle from the client.

        Args:
            translate_partials: Whether to translate still-open previews.
        """
        if translate_partials is not None:
            self._translate_partials = translate_partials

    async def flush(self) -> None:
        """Commit the uncommitted window and wait until its captions are sent.

        Waiting matters: a client that closes the socket the moment it asks to
        stop would otherwise lose the last sentence's translation, which is the
        one a user is most likely to be reading.
        """
        self._segmenter.flush()
        if self._closed or self._consumed <= self._committed:
            return
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        job = _Job(
            kind="flush",
            start_sample=self._committed,
            end_sample=self._consumed,
            flush_at=self._consumed,
            done=done,
        )
        try:
            self._jobs.put_nowait(job)
        except asyncio.QueueFull:
            self._drop_previews()
            await self._jobs.put(job)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(done, timeout=SHUTDOWN_TIMEOUT_S)
