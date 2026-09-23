"""Event-level harness for the live session.

Runs a real :class:`~dsh_voice.live.LiveSession` against a synthesized sample at
real-time pace and prints every event, then checks the invariants a viewer would
notice: no sentence is published twice, every committed span is translated, and
previews are always superseded by a commit under the same id.

Usage::

    .venv-audio/bin/python tests/live_stream_test.py [--audio FILE] [--text TEXT]
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsh_voice.audio_io import decode_to_mono  # noqa: E402
from dsh_voice.cli import _say_to_file, pick_say_voice  # noqa: E402
from dsh_voice.config import PROJECT_DIR, SAMPLE_RATE, load_settings  # noqa: E402
from dsh_voice.live import LiveSession  # noqa: E402
from dsh_voice.meetings import MeetingStore  # noqa: E402

DEFAULT_TEXT = (
    "Good morning everyone. Today we will review the quarterly results. "
    "Revenue grew faster than we guided, but supply chain costs remain a headwind. "
    "Before we discuss the next release, let me hand over to the platform team. "
    "Thank you all for joining on such short notice."
)


def normalize(text: str) -> str:
    """Collapse whitespace and case so duplicate detection is text-meaningful."""
    return " ".join(text.lower().split())


def audit(events: list[dict], transcript_text: str) -> list[str]:
    """Check the viewer-visible invariants over one run's event stream.

    Args:
        events: Every event the session emitted, in order.
        transcript_text: The reference transcript of the source audio.

    Returns:
        One message per violated invariant; empty when the run is clean.
    """
    problems: list[str] = []
    finals = [event for event in events if event.get("type") == "final"]
    previews = [event for event in events if event.get("type") == "partial"]
    translations = [event for event in events if event.get("type") == "translation"]

    committed_text = " ".join(normalize(event.get("text", "")) for event in finals)
    reference = normalize(transcript_text)
    published = [event.get("text", "") for event in finals if event.get("text")]

    if len(published) > 1:
        # A duplicated sentence is the failure a stale decode queue produces.
        for index, text in enumerate(published):
            normalized = normalize(text)
            if normalized and committed_text.count(normalized) > 1:
                problems.append(f"final #{finals[index].get('id')} is published more than once: {text!r}")

    reference_words = reference.replace(".", " ").split()
    committed_words = committed_text.replace(".", " ").split()
    if committed_words and reference_words[:len(committed_words)] != committed_words:
        problems.append(
            "committed text is not a prefix of the reference transcript:\n"
            f"  committed: {' '.join(committed_words)[:200]}\n"
            f"  reference: {' '.join(reference_words)[:200]}",
        )

    final_ids = {event.get("id") for event in finals}
    translated_ids = {event.get("id") for event in translations if event.get("final")}
    missing = sorted(final_ids - translated_ids)
    if missing:
        problems.append(f"committed spans with no final translation: {missing}")

    for preview in previews:
        successor = [event for event in finals if event.get("id") == preview.get("id")]
        if not successor:
            problems.append(f"preview id {preview.get('id')} was never superseded by a commit")
    return problems


def audit_record(meeting: object, events: list[dict], store_root: Path, meeting_id: str) -> list[str]:
    """Check that the session wrote the meeting down, faithfully.

    A live panel that shows captions but records nothing misses the point of the
    feature, so the journal is checked against the events rather than merely for
    existence.

    Args:
        meeting: The meeting read back from the store.
        events: Every event the session emitted.
        store_root: Store root, checked for the journal on disk.
        meeting_id: The meeting id the session reported.

    Returns:
        One message per violated invariant; empty when the record is faithful.
    """
    problems: list[str] = []
    finals = [event for event in events if event.get("type") == "final"]
    translations = {
        event["id"]: event.get("text", "")
        for event in events
        if event.get("type") == "translation" and event.get("final") and event.get("text")
    }

    if len(meeting.segments) != len(finals):
        problems.append(f"recorded {len(meeting.segments)} segments but {len(finals)} were committed")
    if meeting.meta.segments != len(finals):
        problems.append(f"metadata claims {meeting.meta.segments} segments, events show {len(finals)}")
    if meeting.meta.ended_at is None:
        problems.append("meeting was never finalized (ended_at is empty)")

    for event, segment in zip(finals, meeting.segments):
        if normalize(segment.en) != normalize(event.get("text", "")):
            problems.append(f"segment #{segment.id} text differs from the event that produced it")
        expected_zh = translations.get(event.get("id"))
        if expected_zh and normalize(segment.zh) != normalize(expected_zh):
            problems.append(f"segment #{segment.id} translation differs from the event")

    untranslated = [
        segment.id for segment in meeting.segments
        if not segment.zh and segment.id in translations
    ]
    if untranslated:
        problems.append(f"segments with a translation event but no stored translation: {untranslated}")

    if not (store_root / meeting_id / "transcript.jsonl").is_file():
        problems.append("no transcript journal on disk")
    if not (store_root / meeting_id / "meta.json").is_file():
        problems.append("no metadata file on disk")
    return problems


async def run(audio: np.ndarray, settings: object, pace: float, store: MeetingStore) -> tuple[list[dict], str | None]:
    """Stream audio through a live session at a realistic pace, recording it."""
    events: list[dict] = []

    async def send(event: dict) -> None:
        events.append(event)
        kind = event.get("type")
        if kind == "ready":
            meeting = event.get("meeting") or {}
            print(f"  ready            model_ready={event['settings']['model_ready']} "
                  f"meeting={meeting.get('id', '未记录')}")
        elif kind == "partial":
            print(f"  preview  #{event['id']:<3} {event.get('text', '')[:88]}")
        elif kind == "final":
            print(f"  COMMIT   #{event['id']:<3} [{event['start']:.1f}s] {event.get('text', '')[:88]}")
        elif kind == "translation":
            tag = "final" if event.get("final") else "preview"
            detail = event.get("error") or event.get("text", "")
            print(f"  zh/{tag:<7} #{event['id']:<3} {detail[:88]}")
        elif kind == "error":
            print(f"  ERROR           {event.get('message')}")

    recorder = store.create(title="harness", language=settings.language, target_language="zh")
    session = LiveSession(settings, send, recorder=recorder)
    await session.start()
    frame = SAMPLE_RATE // 10
    for offset in range(0, audio.size, frame):
        await session.process_audio(audio[offset:offset + frame])
        await asyncio.sleep(pace)
    await session.stop()
    return events, session.meeting_id


def main() -> int:
    """Run the harness and report invariant violations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--pace", type=float, default=0.1, help="sleep per 100ms frame (0.1 = real time)")
    parser.add_argument("--reference", help="reference transcript; defaults to the synthesized text")
    parser.add_argument("--dump", help="write the raw event stream to this JSON file")
    args = parser.parse_args()

    settings = load_settings()
    if args.audio:
        source = Path(args.audio).expanduser()
    else:
        source = PROJECT_DIR / "selftest" / "harness.aiff"
        voice = pick_say_voice(None)
        print(f"synthesizing sample with voice {voice}...")
        _say_to_file(args.text, source, voice)

    audio = decode_to_mono(source, SAMPLE_RATE)
    print(f"streaming {audio.size / SAMPLE_RATE:.2f}s of audio at pace {args.pace}...\n")

    store_root = Path(tempfile.mkdtemp(prefix="dsh-voice-harness-"))
    store = MeetingStore(store_root)
    try:
        events, meeting_id = asyncio.run(run(audio, settings, args.pace, store))
        if args.dump:
            import json

            Path(args.dump).write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nraw events written to {args.dump}")

        finals = [event for event in events if event.get("type") == "final"]
        translations = [event for event in events if event.get("type") == "translation" and event.get("final")]
        previews = [event for event in events if event.get("type") == "partial"]
        audio_seconds = audio.size / SAMPLE_RATE
        last_commit = finals[-1]["end"] if finals else 0.0

        print("\nsummary")
        print(f"  previews     {len(previews)}")
        print(f"  commits      {len(finals)}")
        print(f"  translations {len(translations)}")
        print(f"  audio        {audio_seconds:.2f}s")
        print(f"  last commit  {last_commit:.2f}s  (lag behind end-of-audio: {audio_seconds - last_commit:.2f}s)")
        print(f"  english      {' '.join(event.get('text', '') for event in finals)}")
        print(f"  chinese      {' '.join(event.get('text', '') for event in translations)}")

        problems = audit(events, args.reference or args.text)

        print("\nmeeting record")
        if meeting_id is None:
            problems.append("the session reported no meeting id")
        else:
            meeting = store.read(meeting_id)
            print(f"  id           {meeting.meta.id}")
            print(f"  title        {meeting.meta.title}")
            print(f"  segments     {meeting.meta.segments}")
            print(f"  audio        {meeting.meta.duration_s:.2f}s")
            print(f"  wall         {meeting.meta.wall_seconds:.1f}s")
            print(f"  finalized    {meeting.meta.ended_at is not None}")
            print(f"  transcript   {meeting.transcript_text()[:110]}")
            problems.extend(audit_record(meeting, events, store_root, meeting_id))
    finally:
        shutil.rmtree(store_root, ignore_errors=True)

    if problems:
        print("\nFAILED invariants:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nall invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
