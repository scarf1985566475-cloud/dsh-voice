"""Offline checks of offline re-transcription.

The recording is the safety net: after a meeting whose captions were switched
off, the archived WAV is all that is left, so re-transcription must rebuild a
complete record from it — English and Chinese — without ever touching the audio.
The model is stubbed here on purpose: what this pins is the wiring, not the ASR.

Usage::

    .venv-audio/bin/python tests/retranscribe_test.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsh_voice import asr  # noqa: E402
from dsh_voice.asr import Transcript, TranscriptSegment  # noqa: E402
from dsh_voice.meetings import AUDIO_FILENAME  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
failures: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record one assertion."""
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def run_cli(args: list[str], env: dict[str, str]) -> tuple[int, str, str]:
    """Run the CLI in a subprocess so the real parser and exit codes are covered."""
    proc = subprocess.run(
        [PYTHON, "-m", "dsh_voice.cli", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return proc.returncode, proc.stdout, proc.stderr


def main() -> int:
    """Rebuild a captions-off meeting from its archived audio."""
    root = Path(tempfile.mkdtemp(prefix="dsh-voice-retranscribe-"))
    env = dict(os.environ, DSH_VOICE_MEETINGS_DIR=str(root / "meetings"))
    try:
        from dsh_voice.config import load_settings
        from dsh_voice.meetings import MeetingStore

        meetings = root / "meetings"
        store = MeetingStore(meetings)

        print("record built")
        recorder = store.create(title="没有字幕的一场", source="live", language="en", target_language="zh")
        audio_path = store.audio_path(recorder.meta.id)
        audio_path.write_bytes(b"RIFF" + b"\0" * 64)
        recorder.finalize(wall_seconds=42.0)
        meeting_id = recorder.meta.id
        check(recorder.meta.segments == 0, "a captions-off meeting starts with zero segments")
        check(store.audio_path(meeting_id).is_file(), f"the archived audio exists ({AUDIO_FILENAME})")

        print("\nstub model")
        original = asr.transcribe_file
        calls: list[Path] = []

        def fake_transcribe(path: Path, *, language: str | None = None, initial_prompt: str | None = None) -> Transcript:
            calls.append(Path(path))
            return Transcript(
                text="Good morning. We shipped it.",
                language="en",
                segments=[
                    TranscriptSegment(start=0.0, end=3.0, text="Good morning."),
                    TranscriptSegment(start=3.5, end=6.0, text="We shipped it."),
                ],
            )

        asr.transcribe_file = fake_transcribe
        os.environ["DSH_VOICE_MEETINGS_DIR"] = str(meetings)
        try:
            settings = load_settings()
            from dsh_voice import cli

            args = cli.build_parser().parse_args(["retranscribe", meeting_id, "--no-translate"])
            code = args.func(args)
        finally:
            asr.transcribe_file = original
            os.environ.pop("DSH_VOICE_MEETINGS_DIR", None)

        print("\nrewrite")
        check(code == 0, f"retranscribe exits 0 ({code})")
        check(len(calls) == 1 and calls[0].name == AUDIO_FILENAME, "it transcribed the archived audio")
        readback = store.read(meeting_id)
        check([segment.en for segment in readback.segments] == ["Good morning.", "We shipped it."],
              "the transcript is rebuilt from the archive")
        check([segment.id for segment in readback.segments] == [1, 2], "segments are renumbered from 1")
        check(readback.meta.segments == 2, "metadata follows the new transcript")
        check(readback.meta.model == settings.model, "the model that produced it is recorded")
        check(store.audio_path(meeting_id).is_file(), "the archived audio is left alone")

        print("\nmissing audio")
        plain = store.create(title="没有录音的一场", source="live")
        plain.finalize(wall_seconds=1.0)
        code, _out, err = run_cli(["retranscribe", plain.meta.id], env)
        check(code == 1, "a meeting without audio fails clearly")
        check(AUDIO_FILENAME in err, "the error names the missing file")

        print("\nunknown meeting")
        code, _out, err = run_cli(["retranscribe", "20200101-000000-000-0000"], env)
        check(code == 1 and "not found" in err, "an unknown id fails clearly")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("offline re-transcription is sound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
