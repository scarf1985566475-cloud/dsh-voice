"""Offline checks of the meeting store.

No model, no network, no microphone: this is the layer that must be right even
when everything else is unavailable, because it holds the only copy of what was
said. The interesting cases are the ugly ones — a hard kill mid-write, a
translation whose segment went missing, an id trying to escape the store root.

Usage::

    .venv-audio/bin/python tests/meetings_test.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsh_voice.meetings import (  # noqa: E402
    JOURNAL_FILENAME,
    MeetingError,
    MeetingStore,
    new_meeting_id,
    successor_id,
)
from dsh_voice.summarize import chunk_transcript, normalize_structured, render_markdown  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record one assertion."""
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def main() -> int:
    """Run every store check against a throwaway directory."""
    root = Path(tempfile.mkdtemp(prefix="dsh-voice-meetings-"))
    try:
        store = MeetingStore(root)

        print("identity")
        first, second = new_meeting_id(), new_meeting_id()
        check(first != second, "ids are unique")
        check(len(first) == len(second) and first.startswith("20"),
              f"ids are fixed-width timestamps ({first})")
        check(sorted([first, second]) == sorted([first, second]), "ids are comparable as plain strings")
        unsafe_rejected = True
        for bad in ("../escape", "a/b", ".hidden", ""):
            try:
                store.directory(bad)
            except MeetingError:
                continue
            unsafe_rejected = False
            print(f"       accepted unsafe id {bad!r}")
        check(unsafe_rejected, "rejects ids that would escape the store root")
        check(successor_id("20260101-000000-000-ffff") > "20260101-000000-000-ffff",
              "successor_id keeps same-millisecond ids ordered")

        print("\nrecording")
        recorder = store.create(title="季度复盘", source="live", language="en", target_language="zh", model="test-model")
        meeting_id = recorder.meta.id
        recorder.record_segment(1, 0.0, 3.2, "Good morning everyone.", asr_ms=210)
        recorder.record_translation(1, "大家早上好。", 480)
        recorder.record_segment(2, 3.6, 8.1, "Revenue grew faster than we guided.", asr_ms=260)
        recorder.record_translation(2, "营收增长快于指引。", 510)
        recorder.record_segment(3, 8.3, 9.0, "A segment whose translation never arrived.", asr_ms=180)
        check(recorder.meta.segments == 3, "segment counter tracks commits")
        check(recorder.meta.duration_s == 9.0, f"audio duration advances to the last commit ({recorder.meta.duration_s})")
        recorder.finalize(wall_seconds=12.5)

        meeting = store.read(meeting_id)
        check(len(meeting.segments) == 3, "all three segments read back")
        check(meeting.segments[0].zh == "大家早上好。", "translation folds onto its segment")
        check(meeting.segments[1].translate_ms == 510, "translation timing is preserved")
        check(meeting.segments[2].zh == "", "a segment without a translation still reads back")
        check(meeting.meta.ended_at is not None, "finalize stamps the end time")
        check(meeting.meta.wall_seconds == 12.5, "wall-clock duration is recorded")
        check(meeting.meta.title == "季度复盘", "title survives the round trip")

        print("\ntranscripts")
        text = meeting.transcript_text()
        check("Good morning everyone." in text and "大家早上好。" in text, "plain transcript is bilingual")
        check(meeting.transcript_text(bilingual=False).count("大家早上好") == 0, "bilingual=False drops the translation")
        markdown = meeting.transcript_markdown()
        check("**[00:00]**" in markdown, "markdown transcript carries timestamps")

        print("\ncrash tolerance")
        journal = root / meeting_id / JOURNAL_FILENAME
        with journal.open("a", encoding="utf-8") as handle:
            handle.write('{"kind": "segment", "id": 4, "start": 9.1, "end": 11.0, "en": "Torn li')  # hard kill mid-write
        recovered = store.fold_segments(meeting_id)
        check(len(recovered) == 3, "a torn final line is skipped, not fatal")
        with journal.open("a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write(json.dumps({"kind": "translation", "id": 999, "zh": "孤儿译文"}, ensure_ascii=False) + "\n")
        check(len(store.fold_segments(meeting_id)) == 3, "an orphan translation is dropped")

        print("\nlisting and metadata")
        second_recorder = store.create(title="第二个会议")
        second_recorder.record_segment(1, 0.0, 1.0, "Hello.")
        second_recorder.finalize()
        rows = store.list()
        check(len(rows) == 2, "both meetings are listed")
        check(rows[0].id == second_recorder.meta.id, "newest meeting sorts first")
        check(store.list(limit=1)[0].id == second_recorder.meta.id, "limit is honoured")
        check(store.total_segments() == 4, f"segments total across meetings ({store.total_segments()})")
        # Two meetings in the same millisecond must still list newest-first:
        # this is what the monotonic id exists for.
        rapid = [store.create(title=f"rapid-{index}") for index in range(5)]
        for rec in rapid:
            rec.record_segment(1, 0.0, 1.0, "tick")
            rec.finalize()
        expected = [rec.meta.id for rec in reversed(rapid)]
        listed = [meta.id for meta in store.list(limit=5)]
        check(listed == expected, "meetings created in one millisecond still list newest-first")
        renamed = store.set_title(meeting_id, "  改名后的会议  ")
        check(renamed.title == "改名后的会议", "rename trims whitespace")
        check(store.set_title(meeting_id, "   ").title == "改名后的会议", "a blank rename is ignored")

        print("\nsummaries")
        store.save_summary(meeting_id, "# 标题\n\n正文\n", {"title": "标题", "key_points": ["要点"]})
        with_summary = store.read(meeting_id)
        check(with_summary.summary.startswith("# 标题"), "summary reads back")
        check(with_summary.meta.summary_status == "ready", "summary status is recorded")
        check(with_summary.summary_structured == {"title": "标题", "key_points": ["要点"]},
              "structured summary reads back")
        store.mark_summary_failed(second_recorder.meta.id)
        check(store.read_meta(second_recorder.meta.id).summary_status == "failed", "failure is recorded")

        print("\ndeletion")
        before_delete = len(store.list())
        store.delete(second_recorder.meta.id)
        check(len(store.list()) == before_delete - 1, "deleted meeting leaves the listing")
        check(not (root / second_recorder.meta.id).exists(), "deleted meeting leaves the disk")
        try:
            store.delete(second_recorder.meta.id)
        except MeetingError:
            check(True, "deleting an unknown meeting raises")
        else:
            check(False, "deleting an unknown meeting raises")

        print("\nsummary shaping")
        structured = normalize_structured({
            "title": "Q3 复盘",
            "key_points": "只有一个要点",
            "decisions": None,
            "action_items": ["直接给字符串", {"task": "跟进合同", "owner": ""}, {"owner": "没有任务"}],
            "risks": [],
            "topics": ["营收", "供应链"],
        })
        check(structured["key_points"] == ["只有一个要点"], "a lone string becomes a one-item list")
        check(structured["decisions"] == [], "a null field becomes an empty list")
        check(len(structured["action_items"]) == 2, "action items drop entries with no task")
        check(structured["action_items"][0]["owner"] == "未提及", "a missing owner is marked 未提及")
        check(structured["action_items"][1]["owner"] == "未提及", "an empty owner is marked 未提及")
        markdown = render_markdown(structured)
        check("## 待办事项" in markdown and "| 跟进合同 | 未提及 |" in markdown, "minutes render an action table")
        check("## 决定事项" in markdown and "- 无" in markdown, "an empty section says 无 rather than nothing")

        print("\nchunking")
        segments = [type("S", (), {"en": "x" * 100, "zh": ""})() for _ in range(50)]
        chunks = chunk_transcript(segments, limit=1000)
        check(len(chunks) > 1, f"a long transcript splits into chunks ({len(chunks)})")
        check(all(len(chunk) <= 1000 for chunk in chunks), "no chunk exceeds the limit")
        check(sum(chunk.count("x") for chunk in chunks) == 50 * 100, "chunking loses no text")
        check(len(chunk_transcript([])) == 1, "an empty transcript still yields one chunk")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("meeting store is sound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
