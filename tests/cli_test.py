"""End-to-end checks of the ``dsh-voice`` command line.

The CLI is how a human drives this project without the GUI, and it had a bug
that no other test could see: ``summarize`` closed its HTTP pool from a second
event loop, so the command printed a traceback *after* doing the work and never
saved the result. Unit tests covered the summarizer and the store; nothing
covered the command that wires them together.

Everything here runs offline: the network-backed calls are stubbed at the
summarizer, and the rest is real file I/O against a temporary store.

Usage::

    .venv-audio/bin/python tests/cli_test.py
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsh_voice import cli  # noqa: E402
from dsh_voice.meetings import MeetingStore  # noqa: E402
from dsh_voice.summarize import SummaryResult  # noqa: E402

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record one assertion."""
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    """Invoke the CLI in-process and capture its streams."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as exit_code:  # argparse --help and friends
            code = int(exit_code.code or 0)
    return code, out.getvalue(), err.getvalue()


def seed_meeting(store: MeetingStore) -> str:
    """Write a small meeting with minutes so the read paths have data."""
    recorder = store.create(title="CLI 测试会", source="live", language="en", target_language="zh", model="test")
    recorder.record_segment(1, 0.0, 4.0, "Good morning everyone.", asr_ms=200)
    recorder.record_translation(1, "大家早上好。", 400)
    recorder.record_segment(2, 4.2, 9.0, "We decided to delay the migration.", asr_ms=250)
    recorder.record_translation(2, "我们决定推迟迁移。", 450)
    recorder.finalize(wall_seconds=11.0)
    return recorder.meta.id


def main() -> int:
    """Exercise each command against a temporary store."""
    root = Path(tempfile.mkdtemp(prefix="dsh-voice-cli-"))
    vault = Path(tempfile.mkdtemp(prefix="dsh-voice-cli-vault-"))
    previous = {key: os.environ.get(key) for key in ("DSH_VOICE_MEETINGS_DIR", "DSH_VOICE_OBSIDIAN_VAULT")}
    os.environ["DSH_VOICE_MEETINGS_DIR"] = str(root)
    os.environ["DSH_VOICE_OBSIDIAN_VAULT"] = str(vault)
    try:
        store = MeetingStore(root)
        meeting_id = seed_meeting(store)

        print("meetings")
        code, out, _err = run_cli(["meetings", "--text"])
        check(code == 0, "exits zero")
        check("CLI 测试会" in out, "lists the seeded meeting")
        check("2段" in out, "reports the segment count")
        code, out, _err = run_cli(["meetings"])
        payload = json.loads(out)
        check(payload["count"] == 1, "json listing reports the count")

        print("\nshow")
        code, out, _err = run_cli(["show", meeting_id])
        check(code == 0, "exits zero")
        check("Good morning everyone." in out and "大家早上好。" in out, "prints the bilingual transcript")
        code, out, _err = run_cli(["show", meeting_id, "--english-only"])
        check("大家早上好。" not in out, "--english-only drops the translation")
        code, out, _err = run_cli(["show", meeting_id, "--json"])
        check(json.loads(out)["meta"]["id"] == meeting_id, "--json emits the full record")
        code, _out, err = run_cli(["show", meeting_id, "--summary"])
        check(code == 1 and "no summary yet" in err, "asking for absent minutes fails clearly")

        print("\nsummarize")
        # Stub the network call: this test is about the command's plumbing.
        from dsh_voice import summarize as summarize_module

        calls: list[str] = []

        def fake_sync(self, meeting):  # noqa: ANN001, ARG001
            calls.append(meeting.meta.id)
            result = SummaryResult(ok=True, chunks=1, ms=12,
                                   markdown="# CLI 纪要\n\n## 概览\n\n一次测试会议。\n",
                                   structured={"title": "CLI 纪要", "overview": "一次测试会议。"})
            return result

        original = summarize_module.MeetingSummarizer.summarize_meeting
        summarize_module.MeetingSummarizer.summarize_meeting = fake_sync
        try:
            code, out, err = run_cli(["summarize", meeting_id])
            check(code == 0, f"exits zero (stderr: {err.strip()[:120]})")
            check("CLI 纪要" in out, "prints the generated minutes")
            check(calls == [meeting_id], "called the summarizer exactly once")
            # The bug this test exists for: the command must finish, not raise.
            check("Traceback" not in err and "Event loop is closed" not in err,
                  "does not raise after generating")
            check(store.read(meeting_id).meta.summary_status == "ready", "saves the minutes")
            check(store.read(meeting_id).summary.startswith("# CLI 纪要"), "the saved minutes are the generated ones")

            code, out, _err = run_cli(["summarize", meeting_id])
            check(code == 0 and out.startswith("# CLI 纪要"), "a second run returns the cached minutes")
            check(len(calls) == 1, "and does not call the model again")

            code, out, _err = run_cli(["summarize", meeting_id, "--force"])
            check(code == 0 and len(calls) == 2, "--force regenerates")
        finally:
            summarize_module.MeetingSummarizer.summarize_meeting = original

        print("\nshow --summary after generation")
        code, out, _err = run_cli(["show", meeting_id, "--summary"])
        check(code == 0 and "CLI 纪要" in out, "the minutes are readable through show")

        print("\nexport")
        code, out, err = run_cli(["export", "--vaults"])
        check(code == 0 and str(vault) in out, "lists the configured vault")
        code, out, _err = run_cli(["export", meeting_id])
        check(code == 0, "exports without error")
        exported = json.loads(out)
        check(exported["ok"] and exported["via"] == "vault-file", "reports the vault-file route")
        note = Path(exported["path"])
        check(note.is_file(), "writes the note")
        check("CLI 纪要" in note.read_text(encoding="utf-8"), "the note carries the minutes")

        print("\nunknown meeting")
        code, _out, err = run_cli(["show", "20200101-000000-000-0000"])
        check(code == 1 and "not found" in err, "show fails clearly on an unknown id")
        code, _out, err = run_cli(["summarize", "20200101-000000-000-0000"])
        check(code == 1 and "not found" in err, "summarize fails clearly on an unknown id")
        code, _out, err = run_cli(["export", "20200101-000000-000-0000"])
        check(code == 1, "export fails on an unknown id")

        print("\nhelp surface")
        for command in ("serve", "transcribe", "translate", "record", "devices", "meetings",
                        "show", "summarize", "ingest", "export", "doctor", "benchmark",
                        "download-model", "selftest"):
            code, out, _err = run_cli([command, "--help"])
            check(code == 0 and command in out, f"{command} --help is wired")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(vault, ignore_errors=True)

    print("")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("command line is sound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
