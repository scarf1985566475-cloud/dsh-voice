"""Offline checks of the Obsidian exporter.

Builds a meeting directly through the store (no model, no network, no
microphone), then verifies both halves of the export: the note Obsidian will
render, and the file layout it lands in. The details worth pinning are the ones
a vault owner would notice — frontmatter a dataview query can use, action items
that are real checkboxes rather than a table, a filename that cannot break
Obsidian's indexer, and a re-export that updates one note instead of stacking
copies.

Usage::

    .venv-audio/bin/python tests/obsidian_export_test.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsh_voice.meetings import MeetingError, MeetingStore  # noqa: E402
from dsh_voice.obsidian import (  # noqa: E402
    DEFAULT_FOLDER,
    export_meeting,
    list_vaults,
    note_filename,
    render_note,
    resolve_vault,
)

SUMMARY_MARKDOWN = """# Q3 复盘会

## 概览

本次会议回顾第三季度业绩。

## 关键要点

- 营收同比增长 12%。

## 决定事项

- 把迁移推迟到 11 月。

## 待办事项

| 事项 | 负责人 | 时间点 |
| --- | --- | --- |
| 准备供应商对比 | Sarah | 下周五之前 |
| 更新路线图 | 未提及 | 未提及 |

## 风险与未决问题

- 供应链成本上升。

---

*记录：3 段 · 时长 22s · 生成于 2026-09-23T07:06:34+00:00*
"""

SUMMARY_STRUCTURED = {
    "title": "Q3 复盘会",
    "overview": "本次会议回顾第三季度业绩。",
    "key_points": ["营收同比增长 12%。"],
    "decisions": ["把迁移推迟到 11 月。"],
    "action_items": [
        {"task": "准备供应商对比", "owner": "Sarah", "due": "下周五之前"},
        {"task": "更新路线图", "owner": "未提及", "due": "未提及"},
    ],
    "risks": ["供应链成本上升。"],
    "topics": ["Q3 复盘", "营收"],
}

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record one assertion."""
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def build_meeting(store: MeetingStore, title: str = "Q3 复盘会") -> str:
    """Record a small meeting through the public store API."""
    recorder = store.create(title=title, source="live", language="en", target_language="zh", model="test")
    recorder.record_segment(1, 0.0, 4.0, "Good morning everyone.", asr_ms=200)
    recorder.record_translation(1, "大家早上好。", 400)
    recorder.record_segment(2, 4.2, 9.0, "Revenue grew twelve percent year over year.", asr_ms=250)
    recorder.record_translation(2, "营收同比增长 12%。", 450)
    recorder.record_segment(3, 9.4, 14.0, "We decided to delay the migration to November.", asr_ms=300)
    recorder.record_translation(3, "我们决定把迁移推迟到 11 月。", 500)
    recorder.finalize(wall_seconds=22.0)
    store.save_summary(recorder.meta.id, SUMMARY_MARKDOWN, SUMMARY_STRUCTURED)
    return recorder.meta.id


def main() -> int:
    """Run every exporter check against throwaway directories."""
    root = Path(tempfile.mkdtemp(prefix="dsh-voice-obsidian-"))
    vault = Path(tempfile.mkdtemp(prefix="dsh-voice-vault-"))
    try:
        store = MeetingStore(root / "meetings")
        meeting_id = build_meeting(store)

        print("filename")
        meeting = store.read(meeting_id)
        name = note_filename(meeting)
        check(name.endswith(".md"), f"derives a .md filename ({name})")
        check(name.startswith("2026-") or name[:2].isdigit(), "prefixes the note with its date")
        hostile = store.read(meeting_id)
        hostile.meta.title = 'A/B: *test* ? "quoted" <tag> | #hash ^caret [bracket]'
        sanitized = note_filename(hostile)
        check(not any(char in sanitized for char in '\\/:*?"<>|#^[]'),
              f"strips characters Obsidian rejects ({sanitized})")

        print("\nnote content")
        note = render_note(store.read(meeting_id))
        check(note.startswith("---\n"), "opens with YAML frontmatter")
        check('\ntags:\n  - 会议记录\n' in note, "frontmatter carries the default tags")
        check('topics:\n  - "Q3 复盘"\n' in note, "frontmatter carries the topics")
        check(f'meeting_id: "{meeting_id}"' in note, "frontmatter identifies the source meeting")
        check("segments: 3" in note, "frontmatter records the segment count")
        check("> [!info] 会议信息" in note, "renders an info callout with the meeting facts")
        check("> [!quote]- 逐句记录" in note, "puts the transcript in a collapsed callout")
        check("**[00:00]** Good morning everyone." in note, "transcript lines carry a timestamp")
        check("> 大家早上好。" in note, "transcript lines carry the translation")
        check("| 事项 | 负责人 |" not in note, "does not leave the action table in the note")
        check("- [ ] 准备供应商对比 — @Sarah — 📅 下周五之前" in note,
              "renders action items as checkboxes with owner and due")
        check("- [ ] 更新路线图" in note, "renders an action item with no owner without a dangling dash")
        check("未提及" not in note.split("逐句记录")[0].split("## 待办事项")[-1],
              "drops placeholder owners from the task lines")
        check("*记录：3 段" not in note, "strips the stored footer (the note has its own metadata)")
        check(note.count("# Q3 复盘会") == 1, "renders the title exactly once, not twice")

        print("\nwithout a transcript")
        lean = render_note(store.read(meeting_id), include_transcript=False)
        check("逐句记录" not in lean, "omits the transcript when asked")
        check("## 概览" in lean, "still carries the minutes")

        print("\nexport to the vault")
        result = export_meeting(meeting_id, store=store, vault=vault)
        check(result.ok, f"export reported success ({result.error or 'ok'})")
        check(result.via == "vault-file", f"used the vault-file route ({result.via})")
        expected = vault / DEFAULT_FOLDER / note_filename(store.read(meeting_id))
        check(Path(result.path) == expected, f"wrote into {DEFAULT_FOLDER}/")
        check(expected.is_file(), "the note exists on disk")
        check(expected.read_text(encoding="utf-8") == render_note(store.read(meeting_id)),
              "the file matches the rendered note byte for byte")

        print("\nre-export and overwrite policy")
        again = export_meeting(meeting_id, store=store, vault=vault)
        check(Path(again.path) == expected, "a re-export updates the same note")
        check(len(list((vault / DEFAULT_FOLDER).iterdir())) == 1, "no duplicate note is created")
        refused = export_meeting(meeting_id, store=store, vault=vault, overwrite=False)
        check(not refused.ok and "already exists" in (refused.error or ""),
              "refuses to overwrite when asked not to")

        print("\nfolder and filename overrides")
        custom = export_meeting(meeting_id, store=store, vault=vault, folder="Meetings/2026", filename="自定义.md")
        check(custom.ok and Path(custom.path) == vault / "Meetings" / "2026" / "自定义.md",
              "honours an explicit folder and filename")

        print("\nvault resolution")
        check(resolve_vault(vault) == vault, "an explicit vault wins")
        os.environ["DSH_VOICE_OBSIDIAN_VAULT"] = str(vault)
        check(resolve_vault() == vault, "the environment variable is honoured")
        del os.environ["DSH_VOICE_OBSIDIAN_VAULT"]
        check(isinstance(list_vaults(), list), "the registry read never raises")
        try:
            resolve_vault(vault / "does-not-exist")
        except MeetingError:
            check(True, "a missing vault is an error, not a silent fallback")
        else:
            check(False, "a missing vault is an error, not a silent fallback")

        print("\nunknown meeting")
        try:
            export_meeting("20200101-000000-000-0000", store=store, vault=vault)
        except MeetingError:
            check(True, "exporting an unknown meeting raises")
        else:
            check(False, "exporting an unknown meeting raises")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(vault, ignore_errors=True)

    print("")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Obsidian export is sound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
