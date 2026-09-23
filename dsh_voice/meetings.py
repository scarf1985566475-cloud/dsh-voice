"""Durable meeting records.

A live session is worthless five minutes after it ends unless something wrote it
down. This module owns that: one directory per meeting, an append-only JSONL
journal inside it, and a metadata file that a listing can read without parsing
the journal.

The journal is append-only on purpose. Segments commit before their translation
exists, and a crashed or killed process must still leave a readable transcript —
so instead of rewriting lines, translation arrives as a second record that the
reader folds onto its segment by id::

    {"kind": "segment", "id": 3, "start": 8.3, "end": 12.9, "en": "...", "asr_ms": 430}
    {"kind": "translation", "id": 3, "zh": "...", "ms": 520}

Nothing here needs the network, so a recording can always be read back.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

LOGGER = logging.getLogger("dsh_voice.meetings")

#: Journal record kinds. Unknown kinds are preserved on read so a newer writer
#: cannot corrupt an older reader's view.
KIND_SEGMENT = "segment"
KIND_TRANSLATION = "translation"

META_FILENAME = "meta.json"
JOURNAL_FILENAME = "transcript.jsonl"
SUMMARY_MD_FILENAME = "summary.md"
SUMMARY_JSON_FILENAME = "summary.json"


class MeetingError(RuntimeError):
    """Raised for an unreadable or unknown meeting."""


def _utc_now() -> str:
    """Current UTC time as a second-resolution ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_meeting_id(when: datetime | None = None) -> str:
    """Build a sortable, collision-resistant meeting id.

    Fixed-width to the millisecond, so plain lexicographic order *is*
    chronological order — which is what lets a listing sort by directory name
    without reading every metadata file.

    Args:
        when: Timestamp to encode; defaults to now (local time, so a listing
            reads in the operator's own clock).

    Returns:
        An id shaped ``YYYYMMDD-HHMMSS-mmm-xxxx``.
    """
    moment = when or datetime.now()
    stamp = f"{moment.strftime('%Y%m%d-%H%M%S')}-{moment.microsecond // 1000:03d}"
    return f"{stamp}-{secrets.token_hex(2)}"


#: Marker separating a same-millisecond counter from the timestamp of an id.
_COUNTER_MARKER = "#"


def successor_id(previous: str) -> str:
    """Build an id that sorts immediately after ``previous``.

    Used when two meetings start inside the same millisecond: the random suffix
    is not ordered, so a counter is appended instead. This is order-safe by
    construction — ``A#001`` is greater than ``A`` because ``A`` is its prefix —
    and the zero-padded counter keeps ``A#002`` above ``A#001``.

    The counter is bounded at 999 acquisitions within one millisecond, which
    each requiring a directory creation is far beyond what a store will see.

    Args:
        previous: The most recently issued id.

    Returns:
        An id strictly greater than ``previous``.
    """
    base, marker, counter = previous.partition(_COUNTER_MARKER)
    if marker and counter.isdigit():
        return f"{base}{_COUNTER_MARKER}{int(counter) + 1:03d}"
    return f"{previous}{_COUNTER_MARKER}001"


@dataclass
class Segment:
    """One committed utterance, with its translation when it has arrived."""

    id: int
    start: float
    end: float
    en: str
    zh: str = ""
    asr_ms: int = 0
    translate_ms: int = 0
    at: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize for the wire."""
        return asdict(self)


@dataclass
class MeetingMeta:
    """Everything a listing needs, without touching the journal."""

    id: str
    title: str
    started_at: str
    ended_at: str | None = None
    duration_s: float = 0.0
    wall_seconds: float = 0.0
    segments: int = 0
    language: str | None = None
    target_language: str = "zh"
    model: str = ""
    source: str = "live"
    source_path: str | None = None
    summary_status: str = "none"
    summary_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize for the wire."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MeetingMeta:
        """Rebuild metadata defensively: an unknown key must not break listing."""
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in payload.items() if key in known})


@dataclass
class Meeting:
    """A meeting as read back: metadata plus its folded transcript."""

    meta: MeetingMeta
    segments: list[Segment] = field(default_factory=list)
    summary: str = ""
    summary_structured: dict[str, Any] | None = None

    def as_dict(self, include_segments: bool = True) -> dict[str, Any]:
        """Serialize for the wire.

        Args:
            include_segments: Whether to embed the transcript; a listing only
                needs the metadata.
        """
        payload: dict[str, Any] = {"meta": self.meta.as_dict(), "summary": self.summary}
        if include_segments:
            payload["segments"] = [segment.as_dict() for segment in self.segments]
        if self.summary_structured is not None:
            payload["summary_structured"] = self.summary_structured
        return payload

    def transcript_text(self, *, bilingual: bool = True) -> str:
        """Render the transcript as plain text.

        Args:
            bilingual: Include the Chinese line under each English one.

        Returns:
            Newline-separated text, one blank line between turns.
        """
        blocks: list[str] = []
        for segment in self.segments:
            lines = [segment.en.strip()]
            if bilingual and segment.zh.strip():
                lines.append(segment.zh.strip())
            blocks.append("\n".join(line for line in lines if line))
        return "\n\n".join(blocks)

    def transcript_markdown(self, *, bilingual: bool = True) -> str:
        """Render the transcript as Markdown with timestamps.

        Args:
            bilingual: Include the Chinese line under each English one.

        Returns:
            Markdown text suitable for a summary prompt or a saved note.
        """
        lines: list[str] = []
        for segment in self.segments:
            stamp = _format_clock(segment.start)
            lines.append(f"**[{stamp}]** {segment.en.strip()}")
            if bilingual and segment.zh.strip():
                lines.append(segment.zh.strip())
            lines.append("")
        return "\n".join(lines).strip()


def _format_clock(seconds: float) -> str:
    """Format a stream offset as ``MM:SS``."""
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


class MeetingRecorder:
    """Append-only writer for one meeting.

    Attributes:
        meta: The live metadata, rewritten on every meaningful change so a
            listing reflects a still-running meeting.
    """

    def __init__(self, directory: Path, meta: MeetingMeta) -> None:
        self.directory = directory
        self.meta = meta
        self._journal = directory / JOURNAL_FILENAME
        self._handle = self._journal.open("a", encoding="utf-8")
        self.closed = False
        self._write_meta()

    # ── writing ──────────────────────────────────────────────────────────────

    def _write_meta(self) -> None:
        """Persist metadata atomically (a killed process must not truncate it)."""
        target = self.directory / META_FILENAME
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.meta.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def _append(self, record: dict[str, Any]) -> None:
        """Append one journal record and flush it to the OS."""
        if self.closed:
            return
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def record_segment(self, segment_id: int, start: float, end: float, en: str, asr_ms: int = 0) -> None:
        """Journal one committed utterance.

        Args:
            segment_id: The id its events were published under.
            start: Stream start offset in seconds.
            end: Stream end offset in seconds.
            en: Committed English text.
            asr_ms: Decode time for this segment.
        """
        self._append({
            "kind": KIND_SEGMENT,
            "id": segment_id,
            "start": round(start, 3),
            "end": round(end, 3),
            "en": en,
            "asr_ms": asr_ms,
            "at": _utc_now(),
        })
        self.meta.segments += 1
        self.meta.duration_s = max(self.meta.duration_s, round(end, 3))
        self._write_meta()

    def record_translation(self, segment_id: int, zh: str, ms: int = 0) -> None:
        """Journal the translation of an already-journaled utterance."""
        self._append({"kind": KIND_TRANSLATION, "id": segment_id, "zh": zh, "ms": ms})

    def note(self, text: str) -> None:
        """Append a free-form note to the journal (markers, inaudible gaps)."""
        self._append({"kind": "note", "text": text, "at": _utc_now()})

    def finalize(self, wall_seconds: float | None = None) -> MeetingMeta:
        """Close the journal and stamp the end of the meeting.

        Args:
            wall_seconds: Elapsed wall-clock time, recorded alongside audio time
                because a meeting's real length includes the silence in it.

        Returns:
            The finalized metadata.
        """
        if self.closed:
            return self.meta
        self.closed = True
        self.meta.ended_at = _utc_now()
        if wall_seconds is not None:
            self.meta.wall_seconds = round(wall_seconds, 1)
        self._write_meta()
        try:
            self._handle.close()
        except OSError:  # pragma: no cover - closing an already-closed handle
            pass
        return self.meta

    def set_summary_status(self, status: str) -> None:
        """Record the summarization outcome in the metadata."""
        self.meta.summary_status = status
        self.meta.summary_at = _utc_now()
        self._write_meta()

    def set_title(self, title: str) -> None:
        """Rename the meeting (a panel's "name this meeting" action)."""
        cleaned = title.strip()
        if not cleaned:
            return
        self.meta.title = cleaned[:120]
        self._write_meta()


class MeetingStore:
    """A directory of meetings.

    Attributes:
        root: Directory holding one subdirectory per meeting.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        #: Last id issued by this store, so back-to-back creates stay ordered.
        self._last_id: str | None = None

    def ensure(self) -> Path:
        """Create the root directory if needed and return it."""
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def _next_id(self) -> str:
        """Issue a strictly increasing meeting id."""
        candidate = new_meeting_id()
        if self._last_id is not None and candidate <= self._last_id:
            candidate = successor_id(self._last_id)
        self._last_id = candidate
        return candidate

    def directory(self, meeting_id: str) -> Path:
        """Resolve a meeting directory, rejecting ids that escape the root."""
        if not meeting_id or "/" in meeting_id or "\\" in meeting_id or meeting_id.startswith("."):
            raise MeetingError(f"invalid meeting id: {meeting_id!r}")
        return self.root / meeting_id

    def create(
        self,
        *,
        title: str | None = None,
        source: str = "live",
        source_path: str | None = None,
        language: str | None = None,
        target_language: str = "zh",
        model: str = "",
    ) -> MeetingRecorder:
        """Start a new meeting.

        Args:
            title: Human title; defaults to a timestamp label.
            source: How the audio arrived (``live`` or ``file``).
            source_path: Recorded file, when the meeting came from a file.
            language: Source language code.
            target_language: Translation target code.
            model: ASR model identifier.

        Returns:
            A recorder writing into the new meeting directory.
        """
        self.ensure()
        meeting_id = self._next_id()
        directory = self.directory(meeting_id)
        directory.mkdir(parents=True, exist_ok=False)
        meta = MeetingMeta(
            id=meeting_id,
            title=title or f"会议 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            started_at=_utc_now(),
            language=language,
            target_language=target_language,
            model=model,
            source=source,
            source_path=source_path,
        )
        return MeetingRecorder(directory, meta)

    def list_ids(self) -> list[str]:
        """List meeting ids, newest first, ignoring anything unreadable."""
        if not self.root.is_dir():
            return []
        ids = [
            entry.name
            for entry in self.root.iterdir()
            if entry.is_dir() and (entry / META_FILENAME).is_file()
        ]
        return sorted(ids, reverse=True)

    def list(self, limit: int | None = None, *, with_summary: bool = False) -> list[MeetingMeta]:
        """List meeting metadata, newest first.

        Args:
            limit: Maximum entries to return.
            with_summary: Attach the first 200 characters of each summary, for a
                listing that previews rather than just names.

        Returns:
            Metadata records; unreadable meetings are skipped, not fatal.
        """
        metas: list[MeetingMeta] = []
        for meeting_id in self.list_ids():
            try:
                metas.append(self.read_meta(meeting_id))
            except MeetingError:
                LOGGER.warning("skipping unreadable meeting %s", meeting_id)
                continue
        if limit is not None:
            metas = metas[:limit]
        return metas

    def read_meta(self, meeting_id: str) -> MeetingMeta:
        """Read one meeting's metadata.

        Raises:
            MeetingError: When the meeting is missing or malformed.
        """
        path = self.directory(meeting_id) / META_FILENAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise MeetingError(f"meeting not found: {meeting_id}") from error
        except (OSError, ValueError) as error:
            raise MeetingError(f"meeting {meeting_id} is unreadable: {error}") from error
        return MeetingMeta.from_dict(payload)

    def iter_records(self, meeting_id: str) -> Iterator[dict[str, Any]]:
        """Yield journal records in file order, ignoring malformed lines.

        A torn final line is expected after a hard kill, so it is skipped rather
        than thrown: one bad byte must not cost the whole transcript.
        """
        path = self.directory(meeting_id) / JOURNAL_FILENAME
        try:
            handle = path.open("r", encoding="utf-8")
        except FileNotFoundError:
            return
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    LOGGER.warning("skipping malformed journal line in %s", meeting_id)
                    continue
                if isinstance(record, dict):
                    yield record

    def fold_segments(self, meeting_id: str) -> list[Segment]:
        """Fold the journal into ordered segments with their translations applied.

        Args:
            meeting_id: Meeting to read.

        Returns:
            Segments in commit order. A translation whose segment is missing is
            dropped, because a caption with no source text has nothing to say.
        """
        by_id: dict[int, Segment] = {}
        order: list[int] = []
        for record in self.iter_records(meeting_id):
            kind = record.get("kind")
            if kind == KIND_SEGMENT:
                segment_id = record.get("id")
                if not isinstance(segment_id, int):
                    continue
                if segment_id not in by_id:
                    order.append(segment_id)
                by_id[segment_id] = Segment(
                    id=segment_id,
                    start=float(record.get("start", 0.0)),
                    end=float(record.get("end", 0.0)),
                    en=str(record.get("en", "")),
                    asr_ms=int(record.get("asr_ms", 0) or 0),
                    at=str(record.get("at", "")),
                )
            elif kind == KIND_TRANSLATION:
                segment_id = record.get("id")
                if isinstance(segment_id, int) and segment_id in by_id:
                    by_id[segment_id].zh = str(record.get("zh", ""))
                    by_id[segment_id].translate_ms = int(record.get("ms", 0) or 0)
        return [by_id[segment_id] for segment_id in order]

    def read(self, meeting_id: str, *, include_segments: bool = True) -> Meeting:
        """Read a full meeting: metadata, transcript, and any saved summary.

        Raises:
            MeetingError: When the meeting is missing or malformed.
        """
        meta = self.read_meta(meeting_id)
        segments = self.fold_segments(meeting_id) if include_segments else []
        summary = ""
        summary_path = self.directory(meeting_id) / SUMMARY_MD_FILENAME
        try:
            summary = summary_path.read_text(encoding="utf-8")
        except OSError:
            summary = ""
        structured: dict[str, Any] | None = None
        structured_path = self.directory(meeting_id) / SUMMARY_JSON_FILENAME
        try:
            loaded = json.loads(structured_path.read_text(encoding="utf-8"))
            structured = loaded if isinstance(loaded, dict) else None
        except (OSError, ValueError):
            structured = None
        return Meeting(meta=meta, segments=segments, summary=summary, summary_structured=structured)

    def save_summary(self, meeting_id: str, markdown: str, structured: dict[str, Any] | None = None) -> None:
        """Persist a generated summary beside its transcript."""
        directory = self.directory(meeting_id)
        (directory / SUMMARY_MD_FILENAME).write_text(markdown, encoding="utf-8")
        if structured is not None:
            (directory / SUMMARY_JSON_FILENAME).write_text(
                json.dumps(structured, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        meta = self.read_meta(meeting_id)
        meta.summary_status = "ready"
        meta.summary_at = _utc_now()
        (directory / META_FILENAME).write_text(
            json.dumps(meta.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def mark_summary_failed(self, meeting_id: str) -> None:
        """Record that summarization was attempted and did not succeed."""
        try:
            meta = self.read_meta(meeting_id)
        except MeetingError:
            return
        meta.summary_status = "failed"
        meta.summary_at = _utc_now()
        (self.directory(meeting_id) / META_FILENAME).write_text(
            json.dumps(meta.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def delete(self, meeting_id: str) -> None:
        """Remove a meeting and everything in it.

        Raises:
            MeetingError: When the meeting does not exist.
        """
        directory = self.directory(meeting_id)
        if not (directory / META_FILENAME).is_file():
            raise MeetingError(f"meeting not found: {meeting_id}")
        shutil.rmtree(directory)

    def set_title(self, meeting_id: str, title: str) -> MeetingMeta:
        """Rename a stored meeting.

        Args:
            meeting_id: Meeting to rename.
            title: New title; blank input is ignored.

        Returns:
            The updated metadata.

        Raises:
            MeetingError: When the meeting does not exist.
        """
        meta = self.read_meta(meeting_id)
        cleaned = title.strip()
        if cleaned:
            meta.title = cleaned[:120]
            (self.directory(meeting_id) / META_FILENAME).write_text(
                json.dumps(meta.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return meta

    def total_segments(self) -> int:
        """Count segments across every meeting (a cheap index for a listing)."""
        return sum(meta.segments for meta in self.list())


def meeting_summary_row(meta: MeetingMeta) -> dict[str, Any]:
    """Project metadata into the compact shape a listing row needs."""
    return {
        "id": meta.id,
        "title": meta.title,
        "started_at": meta.started_at,
        "ended_at": meta.ended_at,
        "duration_s": meta.duration_s,
        "wall_seconds": meta.wall_seconds,
        "segments": meta.segments,
        "source": meta.source,
        "summary_status": meta.summary_status,
    }


def merge_segment_text(segments: Iterable[Segment], *, bilingual: bool = True) -> str:
    """Join segments into plain text (the shape a summary prompt wants)."""
    meeting = Meeting(meta=MeetingMeta(id="", title="", started_at=""), segments=list(segments))
    return meeting.transcript_text(bilingual=bilingual)
