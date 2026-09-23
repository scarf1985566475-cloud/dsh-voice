"""Streaming voice-activity segmentation for live interpretation.

An energy detector with an adaptive noise floor is enough here: the goal is not
speech/silence classification for its own sake but *when to commit a decode*.
Committing at natural pauses keeps Whisper's context window small (so latency
stays low) and stops a decoder from rewriting text a viewer already read.

The segmenter only decides boundaries. It never stores audio; the caller owns
the buffer and resolves the returned absolute sample offsets against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SAMPLE_RATE


@dataclass(frozen=True)
class Segment:
    """One committed span, positioned in the stream by absolute sample offset."""

    start_sample: int
    end_sample: int
    sample_rate: int = SAMPLE_RATE
    reason: str = "silence"

    @property
    def start_s(self) -> float:
        """Start offset in seconds from the beginning of the stream."""
        return self.start_sample / self.sample_rate

    @property
    def end_s(self) -> float:
        """End offset in seconds from the beginning of the stream."""
        return self.end_sample / self.sample_rate

    @property
    def duration_s(self) -> float:
        """Span length in seconds."""
        return (self.end_sample - self.start_sample) / self.sample_rate


class StreamingSegmenter:
    """Turn a continuous mono stream into utterance-sized spans.

    Attributes:
        hop_samples: Analysis hop in samples.
        end_silence_ms: Trailing silence that closes a span.
        start_speech_ms: Leading voiced audio that opens a span.
        max_segment_s: Hard cap after which an open span is force-committed.
        min_segment_s: Spans shorter than this are discarded as noise.
    """

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        hop_ms: int = 10,
        start_speech_ms: int = 140,
        end_silence_ms: int = 700,
        max_segment_s: float = 24.0,
        min_segment_s: float = 0.45,
        pre_roll_ms: int = 240,
        noise_ratio: float = 2.8,
        abs_floor: float = 0.0035,
        noise_decay: float = 0.995,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_samples = max(1, sample_rate * hop_ms // 1000)
        self.start_speech_samples = sample_rate * start_speech_ms // 1000
        self.end_silence_samples = sample_rate * end_silence_ms // 1000
        self.max_segment_samples = int(max_segment_s * sample_rate)
        self.min_segment_samples = int(min_segment_s * sample_rate)
        self.pre_roll_samples = sample_rate * pre_roll_ms // 1000
        self.noise_ratio = noise_ratio
        self.abs_floor = abs_floor
        self.noise_decay = noise_decay

        self._cursor = 0
        self._tail = np.zeros(0, dtype=np.float32)
        self._noise_floor = abs_floor
        self._speech_run = 0
        self._silence_run = 0
        self._segment_start: int | None = None

    @property
    def open_segment_start(self) -> int | None:
        """Absolute start sample of the span being accumulated, if any."""
        return self._segment_start

    def open_duration_s(self) -> float:
        """Length of the span being accumulated, or ``0.0`` when idle."""
        if self._segment_start is None:
            return 0.0
        return (self._cursor - self._segment_start) / self.sample_rate

    def reset(self) -> None:
        """Forget all stream state (a new session or an explicit flush)."""
        self._tail = np.zeros(0, dtype=np.float32)
        self._noise_floor = self.abs_floor
        self._speech_run = 0
        self._silence_run = 0
        self._segment_start = None

    def feed(self, chunk: np.ndarray) -> list[Segment]:
        """Consume audio and return every span that closed during this call.

        Args:
            chunk: Mono float32 samples continuing the stream.

        Returns:
            Committed spans in stream order. Short blips are already discarded.
        """
        if chunk.size == 0:
            return []
        data = np.ascontiguousarray(chunk, dtype=np.float32)
        if self._tail.size:
            data = np.concatenate((self._tail, data))
        hop = self.hop_samples
        frames = data.size // hop
        committed: list[Segment] = []
        for index in range(frames):
            frame = data[index * hop:(index + 1) * hop]
            committed.extend(self._consume_frame(frame))
        self._tail = data[frames * hop:].copy()
        self._cursor += frames * hop
        return committed

    def flush(self) -> Segment | None:
        """Commit the open span at the end of a stream.

        Returns:
            The final span, or ``None`` when nothing worth decoding is open.
        """
        start = self._segment_start
        self._segment_start = None
        self._speech_run = 0
        self._silence_run = 0
        if start is None:
            return None
        end = self._cursor
        if end - start < self.min_segment_samples:
            return None
        return Segment(start_sample=start, end_sample=end, sample_rate=self.sample_rate, reason="flush")

    def _consume_frame(self, frame: np.ndarray) -> list[Segment]:
        """Advance the state machine by one hop and return a span when it closes."""
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
        threshold = max(self._noise_floor * self.noise_ratio, self.abs_floor)
        voiced = rms > threshold
        hop = self.hop_samples
        closed: list[Segment] = []

        if not voiced and self._segment_start is None:
            # Idle: follow the room tone so the threshold tracks the environment
            # instead of a fixed level that a noisy room would defeat.
            self._noise_floor = max(
                self.abs_floor,
                self._noise_floor * self.noise_decay + rms * (1.0 - self.noise_decay),
            )

        if voiced:
            self._silence_run = 0
            self._speech_run += hop
            if self._segment_start is None and self._speech_run >= self.start_speech_samples:
                # Reach back over the run that opened the span plus a short pre-roll,
                # so the decoder sees the onset consonant instead of clipping it.
                self._segment_start = max(0, self._cursor - self._speech_run - self.pre_roll_samples)
        else:
            self._speech_run = 0
            if self._segment_start is not None:
                self._silence_run += hop
                if self._silence_run >= self.end_silence_samples:
                    # Trim the trailing silence the detector itself introduced.
                    end = self._cursor - self._silence_run + hop
                    span = Segment(
                        start_sample=self._segment_start,
                        end_sample=end,
                        sample_rate=self.sample_rate,
                        reason="silence",
                    )
                    self._segment_start = None
                    self._silence_run = 0
                    if span.end_sample - span.start_sample >= self.min_segment_samples:
                        closed.append(span)

        if (
            self._segment_start is not None
            and self._cursor - self._segment_start >= self.max_segment_samples
        ):
            # A monologue never pauses: commit anyway so captions keep flowing.
            span = Segment(
                start_sample=self._segment_start,
                end_sample=self._cursor,
                sample_rate=self.sample_rate,
                reason="max-length",
            )
            self._segment_start = self._cursor
            closed.append(span)
        return closed

    @staticmethod
    def slice_audio(buffer: np.ndarray, origin_sample: int, start_sample: int, end_sample: int) -> np.ndarray:
        """Resolve an absolute sample range against a trimmed buffer.

        Args:
            buffer: Mono float32 samples whose first element is ``origin_sample``.
            origin_sample: Absolute stream offset of ``buffer[0]``.
            start_sample: Absolute start of the wanted range.
            end_sample: Absolute end of the wanted range.

        Returns:
            The overlapping slice, clamped to what the buffer still holds.
        """
        lo = max(0, start_sample - origin_sample)
        hi = min(buffer.size, end_sample - origin_sample)
        if hi <= lo:
            return np.zeros(0, dtype=np.float32)
        return np.ascontiguousarray(buffer[lo:hi], dtype=np.float32)
