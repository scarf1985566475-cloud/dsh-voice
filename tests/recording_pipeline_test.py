"""End-to-end check of the "record and summarize a meeting" objective.

Synthesizes a meeting-shaped recording, runs it through the real pipeline
(ffmpeg decode -> MLX Whisper -> batched DeepSeek translation -> stored meeting
-> DeepSeek minutes), and then checks the product both structurally and for the
facts that were actually said.

This is the test that answers the objective. A summary that renders the right
headings but invents its content would still fail: the decision, the owner, and
the deadline below come from the synthesized script, so they are checkable.

Requires the network (translation and summarization) and the local model.

Usage::

    .venv-audio/bin/python tests/recording_pipeline_test.py [--keep]
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsh_voice.cli import _say_to_file, pick_say_voice  # noqa: E402
from dsh_voice.config import PROJECT_DIR, load_settings  # noqa: E402
from dsh_voice.meetings import MeetingStore  # noqa: E402
from dsh_voice.recordings import build_meeting_from_file  # noqa: E402

# The script is deliberately fact-dense: a decision, an owner, a deadline, and
# two numbers. Each is separately checkable in the generated minutes.
MEETING_SCRIPT = (
    "Good morning everyone. Today we are reviewing the third quarter results. "
    "Revenue grew twelve percent year over year, which is above our guidance. "
    "However, supply chain costs increased, so gross margin dropped to forty one percent. "
    "We decided to delay the migration to November. "
    "Sarah will prepare the vendor comparison by next Friday. "
    "Does anyone have questions before we close?"
)

REQUIRED_SECTIONS = ("## 概览", "## 关键要点", "## 决定事项", "## 待办事项", "## 风险与未决问题")

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record one assertion."""
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    """Whether any needle appears, ignoring whitespace and case.

    Whitespace-insensitive on purpose: Chinese output spaces numbers and units
    freely ("11 月" and "11月" are the same fact), so a literal match would
    report a missing deadline that is plainly present.
    """
    squashed = ''.join(haystack.lower().split())
    return any(''.join(needle.lower().split()) in squashed for needle in needles)


def main() -> int:
    """Run the pipeline once and audit its product."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the synthesized audio and meeting record")
    args = parser.parse_args()

    settings = load_settings()
    store_root = Path(tempfile.mkdtemp(prefix="dsh-voice-pipeline-"))
    sample = PROJECT_DIR / "selftest" / "pipeline.aiff"
    try:
        voice = pick_say_voice(None)
        print(f"synthesizing a meeting recording with voice {voice}...")
        _say_to_file(MEETING_SCRIPT, sample, voice)

        print("\nrunning the pipeline (transcribe -> translate -> store -> summarize)")
        payload = asyncio.run(build_meeting_from_file(
            sample,
            settings=settings,
            store=MeetingStore(store_root),
            title="Q3 复盘会",
            translate=True,
            summarize=True,
        ))

        store = MeetingStore(store_root)
        print("\nrecord")
        check(payload.get("ok") is True, "the pipeline reported success")
        meeting_id = payload["meeting"]["id"]
        check(bool(meeting_id), f"a meeting id was issued ({meeting_id})")
        meeting = store.read(meeting_id)
        check(meeting.meta.source == "file", "the record is marked as coming from a file")
        check(meeting.meta.title == "Q3 复盘会", "the supplied title is kept")
        check(meeting.meta.ended_at is not None, "the record is finalized")
        check(meeting.meta.duration_s > 5, f"audio duration was measured ({meeting.meta.duration_s:.1f}s)")
        check(len(meeting.segments) >= 4, f"the recording split into segments ({len(meeting.segments)})")

        print("\ntranscript")
        english = " ".join(segment.en for segment in meeting.segments)
        chinese = " ".join(segment.zh for segment in meeting.segments)
        check(contains_any(english, ("quarter", "revenue", "margin")),
              "the English transcript carries the meeting's substance")
        check(sum(1 for segment in meeting.segments if segment.zh) >= len(meeting.segments) - 1,
              "essentially every segment was translated")
        check(contains_any(chinese, ("营收", "收入", "毛利", "季度")),
              "the Chinese translation carries the same substance")
        check(store.list_ids() == [meeting_id], "the meeting is listed")

        print("\nminutes")
        summary = payload.get("summary", "")
        if not summary:
            check(False, f"no minutes were produced ({payload.get('summary_error')})")
        else:
            for section in REQUIRED_SECTIONS:
                check(section in summary, f"minutes contain the section {section}")
            structured = payload.get("summary_structured") or {}
            check(bool(structured.get("overview")), "the overview is non-empty")
            check(len(structured.get("key_points") or []) >= 2,
                  f"key points were extracted ({len(structured.get('key_points') or [])})")
            check(bool(structured.get("decisions")), "the decision was captured")
            actions = structured.get("action_items") or []
            check(bool(actions), f"action items were extracted ({len(actions)})")

            flat_summary = summary + " " + str(structured)
            check(contains_any(flat_summary, ("11 月", "11月", "November", "十一月")),
                  "the decision deadline (November) survived into the minutes")
            check(contains_any(flat_summary, ("供应商", "vendor", "对比", "比价")),
                  "the action item (vendor comparison) survived into the minutes")
            check(contains_any(flat_summary, ("Sarah", "萨拉", "莎拉")),
                  "the owner named in the recording appears in the minutes")
            check(contains_any(flat_summary, ("12%", "12 %", "十二")),
                  "the revenue figure survived into the minutes")
            check(contains_any(flat_summary, ("41%", "41 %", "四十一")),
                  "the margin figure survived into the minutes")
            if actions:
                first = actions[0]
                check(set(first) == {"task", "owner", "due"}, "action items carry task/owner/due")
                check(all(value.strip() for value in first.values()), "action items have no empty fields")

        print("\npersistence and reuse")
        summary_path = store_root / meeting_id / "summary.md"
        check(summary_path.is_file(), "minutes are saved next to the transcript")
        check(store.read(meeting_id).meta.summary_status == "ready", "the summary status is recorded")
        check(store.read(meeting_id).summary.strip() == summary.strip(), "saved minutes read back unchanged")

        print("\nsecond run reuses the record")
        again = asyncio.run(build_meeting_from_file(
            sample,
            settings=settings,
            store=MeetingStore(store_root),
            title="第二次",
            translate=False,
            summarize=False,
        ))
        check(again["meeting"]["id"] != meeting_id, "a second run creates a separate meeting")
        check(len(MeetingStore(store_root).list_ids()) == 2, "both meetings are listed")
        check(not again.get("summary"), "the second run skipped summarization when asked")
    finally:
        if args.keep:
            print(f"\nkept: {store_root} and {sample}")
        else:
            shutil.rmtree(store_root, ignore_errors=True)

    print("")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("recording -> transcript -> minutes works end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
