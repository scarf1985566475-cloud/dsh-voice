"""Operator entry point: ``dsh-voice <command>``.

Commands cover the whole surface an operator needs: bring the live server up,
transcribe or translate a file, record from the microphone, check readiness, and
run a self-test that simulates a live session from synthesized speech (so the
whole pipeline is verifiable on a machine where the microphone cannot be
granted to a shell).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .config import MODEL_CATALOG, PROJECT_DIR, SAMPLE_RATE, load_settings
from .obsidian import DEFAULT_FOLDER as DEFAULT_OBSIDIAN_FOLDER


def _print(payload: Any) -> None:
    """Print one JSON payload in a readable, non-ASCII-preserving form."""
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


#: Voices that actually synthesize arbitrary text. A fresh macOS profile can
#: resolve its default voice to a novelty sound effect (Bells, Zarvox, ...),
#: which silently truncates long input — so the self-test names a real voice.
_REAL_VOICE_PREFERENCE = ("Samantha", "Alex", "Fred", "Kathy", "Ralph")

_NOVELTY_VOICES = frozenset({
    "Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Deranged",
    "Good News", "Jester", "Junior", "Organ", "Superstar", "Trinoids", "Whisper",
    "Wobble", "Zarvox",
})


def _say_voices() -> list[tuple[str, str]]:
    """List installed voices as ``(name, locale)`` pairs."""
    result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=False)  # noqa: S603
    voices: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            voices.append((parts[0], parts[1]))
    return voices


def pick_say_voice(explicit: str | None = None) -> str | None:
    """Choose a speech-synthesis voice whose output is not truncated.

    Args:
        explicit: A voice name from the command line; used verbatim when present.

    Returns:
        A voice name, or ``None`` to let ``say`` use its own default.
    """
    if explicit:
        return explicit
    from_env = os.environ.get("DSH_VOICE_SAY_VOICE")
    if from_env:
        return from_env
    voices = _say_voices()
    available = {name for name, _locale in voices}
    for candidate in _REAL_VOICE_PREFERENCE:
        if candidate in available:
            return candidate
    english = [name for name, locale in voices if locale.startswith("en") and name not in _NOVELTY_VOICES]
    return english[0] if english else None


def _say_to_file(text: str, destination: Path, voice: str | None = None) -> Path:
    """Synthesize speech with the macOS ``say`` command.

    Args:
        text: Utterance to synthesize.
        destination: Output file path (AIFF; ffmpeg reads it directly).
        voice: Voice name; ``None`` picks a reliable one via :func:`pick_say_voice`.

    Returns:
        The written path.

    Raises:
        RuntimeError: When ``say`` is unavailable or produced nothing.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if shutil.which("say") is None:
        raise RuntimeError("the macOS 'say' command is unavailable; pass --audio instead")
    chosen = pick_say_voice(voice)
    command = ["say"] + (["-v", chosen] if chosen else []) + ["-o", str(destination), text]
    result = subprocess.run(command, capture_output=True, check=False)  # noqa: S603 - fixed argument vector
    if result.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(
            f"'say' failed (voice={chosen or 'default'}): "
            f"{result.stderr.decode('utf-8', 'replace').strip()}",
        )
    return destination


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the live HTTP/WebSocket service."""
    from . import server

    overrides: dict[str, Any] = {}
    if args.port:
        overrides["port"] = args.port
    if args.host:
        overrides["host"] = args.host
    if args.no_translate:
        overrides["translate"] = False
    if args.language:
        overrides["language"] = args.language
    server.STATE.settings = load_settings(**overrides)
    settings = server.STATE.settings
    print(f"dsh-voice {__version__} serving on http://{settings.host}:{settings.port}", flush=True)
    print(f"  microphone test page: http://{settings.host}:{settings.port}/", flush=True)
    print(f"  websocket endpoint:   ws://{settings.host}:{settings.port}/live", flush=True)
    if args.reload:
        print("  reload:               watching Python sources (a restart drops a live session)", flush=True)
    server.main(reload=bool(args.reload))
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    """Transcribe a file, optionally translating it."""
    from .asr import transcribe_file
    from .translate import Translator

    transcript = transcribe_file(args.path, language=args.language, initial_prompt=args.prompt)
    payload: dict[str, Any] = transcript.as_dict()
    if args.translate and transcript.text:
        translator = Translator()
        # translate() is self-contained: it opens and closes its own pool.
        payload["translation"] = translator.translate(transcript.text).as_dict()
    if args.text:
        print(transcript.text)
    else:
        _print(payload)
    return 0


def _cmd_translate(args: argparse.Namespace) -> int:
    """Translate English text into Chinese."""
    from .translate import Translator

    translator = Translator()
    result = translator.translate(args.text, args.context)
    if not result.ok:
        print(f"translation failed: {result.error}", file=sys.stderr)
        return 1
    print(result.text)
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    """Record from the microphone, optionally transcribing the result."""
    from .audio_io import record_to_file

    destination = Path(args.out).expanduser() if args.out else (
        PROJECT_DIR / "recordings" / f"recording-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    )
    print(f"recording {args.seconds}s from device :{args.device} -> {destination}", flush=True)
    written = record_to_file(destination, args.seconds, device_index=args.device)
    payload: dict[str, Any] = {"path": str(written)}
    if args.transcribe:
        from .asr import transcribe_file

        transcript = transcribe_file(written, language=args.language)
        payload["transcript"] = transcript.as_dict()
        print(transcript.text)
    _print(payload)
    return 0


def _cmd_devices(_args: argparse.Namespace) -> int:
    """List capture devices."""
    from .audio_io import list_input_devices

    _print({"devices": list_input_devices()})
    return 0


def _cmd_meetings(args: argparse.Namespace) -> int:
    """List recorded meetings."""
    from .meetings import MeetingStore, meeting_summary_row

    store = MeetingStore(load_settings().meetings_dir)
    rows = [meeting_summary_row(meta) for meta in store.list(limit=args.limit)]
    if args.text:
        for row in rows:
            print(
                f"{row['id']}  {row['title']}  {row['segments']}段  "
                f"{row['duration_s']:.1f}s  纪要:{row['summary_status']}"
            )
        return 0
    _print({"root": str(store.root), "count": len(rows), "meetings": rows})
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    """Show one meeting's transcript or minutes."""
    from .meetings import MeetingError, MeetingStore

    store = MeetingStore(load_settings().meetings_dir)
    try:
        meeting = store.read(args.meeting_id)
    except MeetingError as error:
        print(str(error), file=sys.stderr)
        return 1
    if args.summary:
        if not meeting.summary:
            print(f"meeting {args.meeting_id} has no summary yet; "
                  f"run: dsh-voice summarize {args.meeting_id}", file=sys.stderr)
            return 1
        print(meeting.summary)
        return 0
    if args.json:
        _print(meeting.as_dict())
        return 0
    print(f"# {meeting.meta.title}  ({meeting.meta.id})")
    print(f"  开始 {meeting.meta.started_at}  录音 {meeting.meta.duration_s:.1f}s  "
          f"时长 {meeting.meta.wall_seconds:.0f}s  段数 {meeting.meta.segments}  "
          f"纪要 {meeting.meta.summary_status}")
    print()
    print(meeting.transcript_text(bilingual=not args.english_only))
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    """Generate Chinese minutes for a recorded meeting."""
    from .meetings import MeetingError, MeetingStore
    from .summarize import MeetingSummarizer

    settings = load_settings()
    store = MeetingStore(settings.meetings_dir)
    try:
        meeting = store.read(args.meeting_id)
    except MeetingError as error:
        print(str(error), file=sys.stderr)
        return 1
    if not meeting.segments:
        print(f"meeting {args.meeting_id} has no recorded segments", file=sys.stderr)
        return 1
    if meeting.summary and not args.force:
        print(meeting.summary)
        return 0
    summarizer = MeetingSummarizer(
        api_base=settings.api_base, api_model=settings.api_model, api_key=settings.api_key,
    )
    # The synchronous wrapper owns the event loop and the connection pool
    # together; closing the pool from a second asyncio.run would run against a
    # loop that is already gone.
    result = summarizer.summarize_meeting(meeting)
    if not result.ok:
        store.mark_summary_failed(args.meeting_id)
        print(f"summarization failed: {result.error}", file=sys.stderr)
        return 1
    store.save_summary(args.meeting_id, result.markdown, result.structured)
    print(result.markdown)
    print(f"\n(已保存到 {store.root / args.meeting_id / 'summary.md'} · "
          f"{result.ms}ms · {result.chunks} chunk(s))", file=sys.stderr)
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Turn an existing recording into a stored, summarized meeting."""
    import asyncio

    from .meetings import MeetingStore
    from .recordings import build_meeting_from_file

    settings = load_settings()
    payload = asyncio.run(build_meeting_from_file(
        args.path,
        settings=settings,
        store=MeetingStore(settings.meetings_dir),
        language=args.language,
        title=args.title,
        translate=not args.no_translate,
        summarize=not args.no_summarize,
    ))
    meeting = payload["meeting"]
    print(f"meeting {meeting['id']} · {meeting['segments']} 段 · {meeting['duration_s']:.1f}s", file=sys.stderr)
    if payload.get("summary_error"):
        print(f"summarization failed: {payload['summary_error']}", file=sys.stderr)
    if args.text:
        print(payload.get("summary") or payload.get("transcript", ""))
    else:
        _print(payload)
    return 0 if not payload.get("summary_error") else 1


def _cmd_export(args: argparse.Namespace) -> int:
    """Export a recorded meeting into an Obsidian vault."""
    from .meetings import MeetingStore
    from .obsidian import export_meeting, list_vaults, resolve_vault

    if args.vaults:
        # Report both what Obsidian knows about and what is explicitly
        # configured, so a mismatch between the two is visible rather than
        # surfacing later as "the note went somewhere unexpected".
        configured = os.environ.get("DSH_VOICE_OBSIDIAN_VAULT") or os.environ.get("OBSIDIAN_VAULT")
        _print({
            "vaults": [str(path) for path in list_vaults()],
            "configured": configured or None,
            "default_folder": DEFAULT_OBSIDIAN_FOLDER,
        })
        return 0

    store = MeetingStore(load_settings().meetings_dir)
    try:
        result = export_meeting(
            args.meeting_id,
            store=store,
            vault=args.vault,
            folder=args.folder,
            filename=args.filename,
            include_transcript=not args.no_transcript,
            overwrite=not args.no_overwrite,
        )
    except Exception as error:  # noqa: BLE001 - report the reason, not a traceback
        print(f"export failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    if not result.ok:
        print(f"export failed: {result.error}", file=sys.stderr)
        return 1
    print(f"已导出到 {result.path}（{result.via}）", file=sys.stderr)
    _print(result.as_dict())
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Compare local ASR models on latency, memory, and accuracy."""
    from .benchmark import format_table, run_benchmark
    from .config import PROJECT_DIR

    models = [Path(item).expanduser() for item in args.models] if args.models else [
        path for path in sorted((PROJECT_DIR / "models").iterdir()) if path.is_dir()
    ]
    if not models:
        print(f"no models found under {PROJECT_DIR / 'models'}", file=sys.stderr)
        return 1
    script = args.text
    if args.text_file:
        script = Path(args.text_file).expanduser().read_text(encoding="utf-8").strip()
    payload = run_benchmark(
        models,
        text=script,
        voice=args.voice,
        snr_db=args.snr,
        language=args.language or None,
        repeats=args.repeats,
    )
    if args.json:
        _print(payload)
    else:
        print(format_table(payload))
    return 0


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """Check every dependency and report readiness."""
    from .asr import model_present
    from .audio_io import AudioToolError, ffmpeg_version
    from .meetings import MeetingStore

    settings = load_settings()
    try:
        ffmpeg = ffmpeg_version()
        ffmpeg_ok = True
    except AudioToolError as error:
        ffmpeg = str(error)
        ffmpeg_ok = False
    store = MeetingStore(settings.meetings_dir)
    report = {
        "version": __version__,
        "python": sys.version.split()[0],
        "ffmpeg_ok": ffmpeg_ok,
        "ffmpeg": ffmpeg,
        "model_path": settings.model,
        "model_present": model_present(),
        "model_warning": settings.as_public_dict()["model_warning"],
        "models_installed": [
            {"tag": choice.tag, "present": choice.present, "english_only": choice.english_only,
             "note": choice.note}
            for choice in MODEL_CATALOG
        ],
        "translation_configured": settings.api_key is not None,
        "recording_enabled": settings.record,
        "meetings_dir": str(store.root),
        "meetings_recorded": len(store.list_ids()),
        "has_say": subprocess.run(["which", "say"], capture_output=True, check=False).returncode == 0,
        "settings": settings.as_public_dict(),
    }
    _print(report)
    return 0 if ffmpeg_ok and report["model_present"] else 1


def _cmd_download_model(_args: argparse.Namespace) -> int:
    """Fetch the Whisper weights into the project directory."""
    from .asr import ensure_model

    path = ensure_model()
    _print({"ok": True, "model": str(path)})
    return 0


def _cmd_selftest(args: argparse.Namespace) -> int:
    """Run an end-to-end check of the pipeline, live streaming included.

    Without ``--audio`` the check synthesizes speech with ``say``, so it needs no
    microphone and no human. With ``--live`` the same audio is pushed through a
    real :class:`~dsh_voice.live.LiveSession` in real-time-sized frames, which
    exercises segmentation, decode, and translation exactly as the GUI does.
    """
    import asyncio

    import numpy as np

    settings = load_settings()
    print(f"dsh-voice {__version__} self-test", flush=True)
    steps: list[dict[str, Any]] = []

    started = time.monotonic()
    source = Path(args.audio).expanduser() if args.audio else PROJECT_DIR / "selftest" / "sample.aiff"
    if args.audio:
        print(f"[1/4] using supplied audio {source}", flush=True)
    else:
        text = args.text or (
            "Good morning everyone. Today we will review the quarterly results. "
            "Revenue grew faster than we guided, but supply chain costs remain a headwind. "
            "Before we discuss the next release, let me hand over to the platform team. "
            "Thank you all for joining on such short notice."
        )
        voice = pick_say_voice(args.voice)
        print(f"[1/4] synthesizing speech with 'say' (voice: {voice or 'system default'})...", flush=True)
        _say_to_file(text, source, voice)
    steps.append({"step": "prepare-audio", "path": str(source), "ms": int((time.monotonic() - started) * 1000)})

    from .audio_io import decode_to_mono

    started = time.monotonic()
    audio = decode_to_mono(source, SAMPLE_RATE)
    steps.append({
        "step": "decode", "samples": int(audio.size),
        "seconds": round(audio.size / SAMPLE_RATE, 2),
        "ms": int((time.monotonic() - started) * 1000),
    })
    print(f"[2/4] decoded {audio.size / SAMPLE_RATE:.2f}s of audio", flush=True)

    if not args.live:
        from .asr import transcribe_file

        started = time.monotonic()
        transcript = transcribe_file(source, language=args.language)
        steps.append({"step": "transcribe-file", "ms": int((time.monotonic() - started) * 1000),
                      "text": transcript.text, "segments": len(transcript.segments)})
        print(f"[3/4] transcript: {transcript.text}", flush=True)
    else:
        events: list[dict[str, Any]] = []

        async def run_live() -> None:
            from .live import LiveSession

            async def send(event: dict[str, Any]) -> None:
                events.append(event)
                kind = event.get("type")
                if kind == "final":
                    print(f"      final#{event['id']} ({event['start']:.1f}s): {event['text']}", flush=True)
                elif kind == "translation" and event.get("final"):
                    print(f"      zh#{event['id']}: {event['text']}", flush=True)
                elif kind == "error":
                    print(f"      error: {event['message']}", flush=True)

            session = LiveSession(settings, send)
            await session.start()
            # Push in 100 ms frames at real-time pace: the segmenter and the
            # partial ticker behave differently under a time-warped feed, and a
            # faithful test is the whole point of this mode.
            frame = SAMPLE_RATE // 10
            for offset in range(0, audio.size, frame):
                await session.process_audio(audio[offset:offset + frame])
                await asyncio.sleep(0.1)
            await session.stop()

        started = time.monotonic()
        asyncio.run(run_live())
        finals = [event for event in events if event.get("type") == "final"]
        translations = [event for event in events if event.get("type") == "translation" and event.get("final")]
        steps.append({
            "step": "transcribe-live",
            "ms": int((time.monotonic() - started) * 1000),
            "finals": len(finals),
            "translations": len(translations),
            "text": " ".join(event.get("text", "") for event in finals),
            "zh": " ".join(event.get("text", "") for event in translations),
        })
        print(f"[3/4] live transcript: {steps[-1]['text']}", flush=True)
        if translations:
            print(f"      live translation: {steps[-1]['zh']}", flush=True)

    text_for_translation = steps[-1].get("text", "")
    if args.no_translate:
        print("[4/4] translation skipped (--no-translate)", flush=True)
    elif text_for_translation:
        from .translate import Translator

        started = time.monotonic()
        translator = Translator()
        result = translator.translate(text_for_translation)
        steps.append({
            "step": "translate", "ok": result.ok, "ms": int((time.monotonic() - started) * 1000),
            "zh": result.text, **({"error": result.error} if result.error else {}),
        })
        print(f"[4/4] translation: {result.text or result.error}", flush=True)
    else:
        print("[4/4] translation skipped (no transcript text)", flush=True)

    _print({"ok": True, "steps": steps})
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``dsh-voice`` argument parser."""
    parser = argparse.ArgumentParser(prog="dsh-voice", description="Local transcription and live EN->ZH interpretation")
    parser.add_argument("--version", action="version", version=f"dsh-voice {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the live HTTP/WebSocket service")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--language", help="source language code (default: en)")
    serve.add_argument("--no-translate", action="store_true", help="captions only, no Chinese translation")
    serve.add_argument("--reload", action="store_true",
                       help="restart on Python source changes (development; drops a live session)")
    serve.set_defaults(func=_cmd_serve)

    transcribe = sub.add_parser("transcribe", help="transcribe an audio or video file")
    transcribe.add_argument("path")
    transcribe.add_argument("--language")
    transcribe.add_argument("--prompt", help="vocabulary hint (names, jargon)")
    transcribe.add_argument("--translate", action="store_true", help="also translate into Chinese")
    transcribe.add_argument("--text", action="store_true", help="print only the transcript text")
    transcribe.set_defaults(func=_cmd_transcribe)

    translate = sub.add_parser("translate", help="translate English text into Chinese")
    translate.add_argument("text")
    translate.add_argument("--context", nargs="*", help="recent sentences for terminology continuity")
    translate.set_defaults(func=_cmd_translate)

    record = sub.add_parser("record", help="record from the microphone")
    record.add_argument("--seconds", type=float, default=10.0)
    record.add_argument("--device", default="0", help="avfoundation audio device index")
    record.add_argument("--out", help="destination .wav path")
    record.add_argument("--transcribe", action="store_true")
    record.add_argument("--language")
    record.set_defaults(func=_cmd_record)

    devices = sub.add_parser("devices", help="list capture devices")
    devices.set_defaults(func=_cmd_devices)

    meetings = sub.add_parser("meetings", help="list recorded meetings")
    meetings.add_argument("--limit", type=int, default=50)
    meetings.add_argument("--text", action="store_true", help="print a compact table instead of JSON")
    meetings.set_defaults(func=_cmd_meetings)

    show = sub.add_parser("show", help="show one meeting's transcript or minutes")
    show.add_argument("meeting_id")
    show.add_argument("--summary", action="store_true", help="print the minutes instead of the transcript")
    show.add_argument("--english-only", action="store_true", help="omit the Chinese lines")
    show.add_argument("--json", action="store_true", help="print the whole record as JSON")
    show.set_defaults(func=_cmd_show)

    summarize = sub.add_parser("summarize", help="generate Chinese minutes for a recorded meeting")
    summarize.add_argument("meeting_id")
    summarize.add_argument("--force", action="store_true", help="regenerate even when minutes exist")
    summarize.set_defaults(func=_cmd_summarize)

    ingest = sub.add_parser("ingest", help="transcribe + translate + summarize an existing recording")
    ingest.add_argument("path")
    ingest.add_argument("--title")
    ingest.add_argument("--language")
    ingest.add_argument("--no-translate", action="store_true")
    ingest.add_argument("--no-summarize", action="store_true")
    ingest.add_argument("--text", action="store_true", help="print the minutes instead of JSON")
    ingest.set_defaults(func=_cmd_ingest)

    export = sub.add_parser("export", help="export a meeting into an Obsidian vault")
    export.add_argument("meeting_id", nargs="?", help="meeting id; omit with --vaults")
    export.add_argument("--vault", help="vault directory (default: the last vault Obsidian opened)")
    export.add_argument("--folder", help=f"folder inside the vault (default: {DEFAULT_OBSIDIAN_FOLDER})")
    export.add_argument("--filename", help="explicit note filename")
    export.add_argument("--no-transcript", action="store_true", help="minutes only, no sentence log")
    export.add_argument("--no-overwrite", action="store_true", help="refuse to replace an existing note")
    export.add_argument("--vaults", action="store_true", help="list the vaults Obsidian knows about")
    export.set_defaults(func=_cmd_export)

    doctor = sub.add_parser("doctor", help="check dependencies and readiness")
    doctor.set_defaults(func=_cmd_doctor)

    benchmark = sub.add_parser("benchmark", help="compare local ASR models (speed / memory / accuracy)")
    benchmark.add_argument("--models", nargs="*", help="model directories; default: everything under models/")
    benchmark.add_argument("--text", help="script to synthesize for the benchmark audio")
    benchmark.add_argument("--text-file", help="read the benchmark script from this file "
                                                "(it is also used as the reference)")
    benchmark.add_argument("--voice", help="macOS 'say' voice for the sample")
    benchmark.add_argument("--snr", type=float, default=10.0, help="noise level for the noisy variant (dB)")
    benchmark.add_argument("--language", default="en")
    benchmark.add_argument("--repeats", type=int, default=3, help="slices timed per model")
    benchmark.add_argument("--json", action="store_true", help="print the raw payload")
    benchmark.set_defaults(func=_cmd_benchmark)

    download = sub.add_parser("download-model", help="fetch the Whisper weights")
    download.set_defaults(func=_cmd_download_model)

    selftest = sub.add_parser("selftest", help="end-to-end check without a microphone")
    selftest.add_argument("--audio", help="use this file instead of synthesized speech")
    selftest.add_argument("--text", help="text to synthesize when --audio is omitted")
    selftest.add_argument("--voice", help="macOS 'say' voice used for the synthesized sample")
    selftest.add_argument("--live", action="store_true", help="stream through a real live session")
    selftest.add_argument("--language")
    selftest.add_argument("--no-translate", action="store_true")
    selftest.set_defaults(func=_cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one CLI command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
