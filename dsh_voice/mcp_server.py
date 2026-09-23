"""Stdio MCP server exposing the transcription tools to the DSH agent.

Runs in the ``.venv-audio`` interpreter and speaks MCP over stdio, the same
shape as the Zotero bridge already wired into the profile. The agent gets:

* file transcription (any ffmpeg-readable format, optionally translated),
* ad-hoc English-to-Chinese translation,
* microphone recording through the bundled ffmpeg,
* and a status probe for the local live server the GUI panel talks to.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from . import __version__
from .config import PROJECT_DIR, load_settings
from .meetings import MeetingError, MeetingStore, meeting_summary_row

LOGGER = logging.getLogger("dsh_voice.mcp")

mcp: FastMCP = FastMCP(
    name="dsh-voice",
    instructions=(
        "Local speech tools: transcribe audio files with MLX Whisper, translate English "
        "into Chinese, record from the microphone, keep meeting records, and produce "
        "Chinese meeting minutes. Transcription runs fully offline on the local machine; "
        "only translation and summarization call the DeepSeek API."
    ),
)


def _default_output_path() -> Path:
    """Where a new recording lands when the caller does not name a file."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return PROJECT_DIR / "recordings" / f"recording-{stamp}.wav"


def _health(url: str, timeout: float = 1.5) -> dict[str, Any] | None:
    """Probe a running local server, returning its JSON payload or ``None``."""
    import httpx

    try:
        response = httpx.get(f"{url}/health", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:  # noqa: BLE001 - an unreachable server is an expected state
        return None
    return payload if isinstance(payload, dict) else None


@mcp.tool()
def voice_status() -> dict[str, Any]:
    """Report the state of local transcription: model files, ffmpeg, translation key, and live server.

    Call this first when a voice task fails, to tell a missing model download
    apart from an unreachable server or a missing API key.

    Returns:
        A status object with model readiness, the ffmpeg build, whether the
        DeepSeek translation key resolved, and whether the live server answers.
    """
    from .asr import model_present
    from .audio_io import AudioToolError, ffmpeg_version

    settings = load_settings()
    try:
        ffmpeg = ffmpeg_version()
    except AudioToolError as error:
        ffmpeg = f"unavailable: {error}"
    url = f"http://{settings.host}:{settings.port}"
    health = _health(url)
    store = MeetingStore(settings.meetings_dir)
    return {
        "version": __version__,
        "model_path": settings.model,
        "model_present": model_present(),
        "ffmpeg": ffmpeg,
        "translation_configured": settings.api_key is not None,
        "translation_model": settings.api_model,
        "source_language": settings.language,
        "target_language": settings.target_language,
        "recording_enabled": settings.record,
        "meetings_dir": str(store.root),
        "meetings_recorded": len(store.list_ids()),
        "live_server_url": url,
        "live_server_running": health is not None,
        "live_server_warmup": (health or {}).get("warmup"),
        "recordings_dir": str(PROJECT_DIR / "recordings"),
    }


@mcp.tool()
def transcribe_audio(
    path: str,
    language: str | None = None,
    translate: bool = False,
    initial_prompt: str | None = None,
) -> dict[str, Any]:
    """Transcribe an audio or video file with the local Whisper model.

    Args:
        path: Absolute or ~-relative path to any ffmpeg-readable file
            (wav, mp3, m4a, mp4, flac, aac, ...).
        language: Source language code such as "en" or "zh"; omit to auto-detect.
        translate: Also translate the transcript into Chinese.
        initial_prompt: Optional hint listing names or jargon that occur in the
            audio; this improves spelling of proper nouns.

    Returns:
        The transcript text, detected language, duration, per-segment timings,
        and the Chinese translation when requested.
    """
    from .asr import ModelUnavailableError, transcribe_file

    try:
        transcript = transcribe_file(path, language=language, initial_prompt=initial_prompt)
    except ModelUnavailableError as error:
        return {"ok": False, "error": str(error)}
    except Exception as error:  # noqa: BLE001 - report decode failures as data
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}

    result: dict[str, Any] = {"ok": True, "path": str(Path(path).expanduser()), **transcript.as_dict()}
    if translate and transcript.text:
        from .translate import Translator

        # translate() opens and closes its own pool, so no cleanup is needed here.
        translation = Translator().translate(transcript.text)
        result["translation"] = translation.as_dict()
    return result


@mcp.tool()
def translate_to_chinese(text: str, context: list[str] | None = None) -> dict[str, Any]:
    """Translate English text into natural Chinese using the DeepSeek API.

    Args:
        text: English text to translate.
        context: Optional recent English sentences, used only to keep
            terminology consistent across a series of translations.

    Returns:
        The Chinese translation, or an error describing why it failed.
    """
    from .translate import Translator

    # translate() opens and closes its own pool, so no cleanup is needed here.
    result = Translator().translate(text, context)
    return {"ok": result.ok, **result.as_dict()}


@mcp.tool()
def list_audio_devices() -> list[dict[str, str]]:
    """List microphone devices the bundled ffmpeg can capture from.

    Returns:
        One entry per audio device with the ``index`` to pass to
        :func:`record_audio` and its human-readable name. An empty list means
        the machine exposes no avfoundation capture device.
    """
    from .audio_io import list_input_devices

    return list_input_devices()


@mcp.tool()
def record_audio(
    seconds: float = 10.0,
    device_index: str = "0",
    out_path: str | None = None,
    transcribe: bool = True,
    translate: bool = False,
) -> dict[str, Any]:
    """Record from the microphone into a WAV file, optionally transcribing it.

    Recording blocks for the requested duration, so keep ``seconds`` modest or
    ask the user before capturing a long session.

    Args:
        seconds: Capture duration.
        device_index: Microphone index from :func:`list_audio_devices`.
        out_path: Destination WAV path; defaults to ``recordings/recording-<timestamp>.wav``.
        transcribe: Transcribe the recording after capture.
        translate: Also translate the transcript into Chinese.

    Returns:
        The written path and, when requested, its transcript and translation.
    """
    from .audio_io import record_to_file

    destination = Path(out_path).expanduser() if out_path else _default_output_path()
    try:
        written = record_to_file(destination, seconds, device_index=device_index)
    except Exception as error:  # noqa: BLE001 - a denied microphone is a normal outcome
        return {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "hint": "macOS microphone permission must be granted to the process running dsh "
                    "(System Settings > Privacy & Security > Microphone).",
        }
    result: dict[str, Any] = {"ok": True, "path": str(written), "seconds": seconds}
    if transcribe:
        result.update(transcribe_audio(str(written), translate=translate))
    return result


@mcp.tool()
def ensure_live_server(start_if_down: bool = True) -> dict[str, Any]:
    """Check the local live-interpretation server, starting it when it is down.

    The server backs the "同传" panel in the DSH web GUI: it accepts streamed
    microphone audio and returns live captions and Chinese translations.

    Args:
        start_if_down: Spawn ``dsh-voice serve`` as a detached background
            process when nothing answers on the configured port.

    Returns:
        The server URL, whether it is healthy, and whether this call started it.
    """
    settings = load_settings()
    url = f"http://{settings.host}:{settings.port}"
    health = _health(url)
    if health is not None:
        return {"ok": True, "url": url, "started": False, "health": health}
    if not start_if_down:
        return {"ok": False, "url": url, "started": False, "error": "server is not running"}

    log_path = PROJECT_DIR / "logs" / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    interpreter = Path(sys.executable)
    with log_path.open("ab") as log:
        process = subprocess.Popen(  # noqa: S603 - fixed argument vector, no shell
            [str(interpreter), "-m", "dsh_voice.server"],
            cwd=str(PROJECT_DIR),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            env={**os.environ},
        )
    deadline = time.time() + 30
    while time.time() < deadline:
        time.sleep(0.5)
        health = _health(url)
        if health is not None:
            return {"ok": True, "url": url, "started": True, "pid": process.pid, "health": health}
        if process.poll() is not None:
            return {
                "ok": False, "url": url, "started": False, "pid": process.pid,
                "error": f"server exited with code {process.returncode}",
                "log": str(log_path),
            }
    return {"ok": False, "url": url, "started": False, "pid": process.pid, "error": "timed out waiting for /health"}


@mcp.tool()
def list_meetings(limit: int = 20) -> dict[str, Any]:
    """List recorded meetings, newest first.

    Every live session in the "同传" panel and every processed recording is
    stored as a meeting with an English transcript, Chinese translation, and
    minutes once they have been generated.

    Args:
        limit: Maximum meetings to return.

    Returns:
        Meeting metadata rows (id, title, time, duration, segment count, summary
        status) plus the directory they live in.
    """
    settings = load_settings()
    store = MeetingStore(settings.meetings_dir)
    metas = store.list(limit=max(1, min(limit, 200)))
    return {
        "ok": True,
        "root": str(store.root),
        "count": len(metas),
        "meetings": [meeting_summary_row(meta) for meta in metas],
    }


@mcp.tool()
def read_meeting(meeting_id: str, include_segments: bool = True) -> dict[str, Any]:
    """Read one recorded meeting: metadata, transcript, translations, and minutes.

    Args:
        meeting_id: Meeting id from :func:`list_meetings`.
        include_segments: Include the per-sentence records; disable for a large
            meeting when only the metadata and minutes are needed.

    Returns:
        The meeting record, or an error when the id is unknown.
    """
    settings = load_settings()
    store = MeetingStore(settings.meetings_dir)
    try:
        meeting = store.read(meeting_id, include_segments=include_segments)
    except MeetingError as error:
        return {"ok": False, "error": str(error)}
    payload = {"ok": True, **meeting.as_dict(include_segments=include_segments)}
    payload["transcript_text"] = meeting.transcript_text()
    return payload


@mcp.tool()
def summarize_meeting(meeting_id: str, force: bool = False) -> dict[str, Any]:
    """Produce Chinese meeting minutes for a recorded meeting.

    The minutes are structured (overview, key points, decisions, action items
    with owners, risks) and saved next to the transcript, so reading them again
    is free. Call :func:`list_meetings` first to find the id.

    Args:
        meeting_id: Meeting id to summarize.
        force: Regenerate even when minutes already exist.

    Returns:
        The Markdown minutes plus their structured fields, or an error.
    """
    from .summarize import MeetingSummarizer

    settings = load_settings()
    store = MeetingStore(settings.meetings_dir)
    try:
        meeting = store.read(meeting_id)
    except MeetingError as error:
        return {"ok": False, "error": str(error)}
    if not meeting.segments:
        return {"ok": False, "error": f"meeting {meeting_id} has no recorded segments"}
    if meeting.summary and not force:
        return {
            "ok": True, "cached": True, "meeting_id": meeting_id,
            "summary": meeting.summary, "structured": meeting.summary_structured,
        }

    summarizer = MeetingSummarizer()
    result = summarizer.summarize_meeting(meeting)
    if not result.ok:
        store.mark_summary_failed(meeting_id)
        return {"ok": False, "error": result.error, "meeting_id": meeting_id}
    store.save_summary(meeting_id, result.markdown, result.structured)
    return {
        "ok": True, "cached": False, "meeting_id": meeting_id,
        "summary": result.markdown, "structured": result.structured,
        "chunks": result.chunks, "ms": result.ms,
    }


@mcp.tool()
def summarize_recording(
    path: str,
    title: str | None = None,
    language: str | None = None,
    translate: bool = True,
    summarize: bool = True,
) -> dict[str, Any]:
    """Process an existing recording end to end: transcribe, translate, store, and summarize.

    Use this for a meeting that already happened — a recorded call, a voice memo,
    a video's audio track. The result is a meeting record identical in shape to a
    live one, so :func:`read_meeting` and :func:`summarize_meeting` work on it too.

    The call blocks for as long as the file takes to transcribe, so prefer
    naming a file over recording a new one when the audio already exists.

    Args:
        path: Absolute or ~-relative path to any ffmpeg-readable file.
        title: Meeting title; defaults to the file name.
        language: Source language override; omit to use the configured language.
        translate: Translate each sentence into Chinese.
        summarize: Generate Chinese minutes.

    Returns:
        The meeting metadata, the full transcript, and the minutes (or the
        summarization error, since a failed summary keeps the transcript).
    """
    import asyncio

    from .recordings import build_meeting_from_file

    settings = load_settings()
    store = MeetingStore(settings.meetings_dir)
    try:
        payload = asyncio.run(build_meeting_from_file(
            path,
            settings=settings,
            store=store,
            language=language,
            title=title,
            translate=translate,
            summarize=summarize,
        ))
    except FileNotFoundError as error:
        return {"ok": False, "error": str(error)}
    except Exception as error:  # noqa: BLE001 - report any pipeline failure as data
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return payload


@mcp.tool()
def rename_meeting(meeting_id: str, title: str) -> dict[str, Any]:
    """Give a recorded meeting a meaningful title.

    Args:
        meeting_id: Meeting id from :func:`list_meetings`.
        title: New title.

    Returns:
        The updated metadata, or an error when the id is unknown.
    """
    settings = load_settings()
    store = MeetingStore(settings.meetings_dir)
    try:
        meta = store.set_title(meeting_id, title)
    except MeetingError as error:
        return {"ok": False, "error": str(error)}
    return {"ok": True, "meta": meta.as_dict()}


@mcp.tool()
def list_obsidian_vaults() -> dict[str, Any]:
    """List the Obsidian vaults this machine knows about.

    Returns:
        Vault directories, most recently opened first, plus the folder meetings
        are exported into by default.
    """
    from .obsidian import DEFAULT_FOLDER, list_vaults

    return {
        "ok": True,
        "vaults": [str(path) for path in list_vaults()],
        "default_folder": DEFAULT_FOLDER,
    }


@mcp.tool()
def export_meeting_to_obsidian(
    meeting_id: str,
    vault: str | None = None,
    folder: str | None = None,
    include_transcript: bool = True,
) -> dict[str, Any]:
    """Export a recorded meeting — minutes and transcript — into an Obsidian vault.

    The note carries YAML frontmatter (title, date, duration, tags, topics), the
    Chinese minutes, action items as ``- [ ]`` checkboxes so Obsidian's task
    queries pick them up, and the sentence-by-sentence record in a collapsed
    callout. Re-exporting the same meeting updates one note rather than
    creating copies.

    Writing into the vault works whether or not Obsidian is running. When a
    Local REST API URL and key are configured, the note is pushed through the
    plugin instead so a running Obsidian reloads it immediately.

    Args:
        meeting_id: Meeting id from :func:`list_meetings`.
        vault: Vault directory; the last vault Obsidian opened when omitted.
        folder: Folder inside the vault (default ``80-会议记录``).
        include_transcript: Append the sentence-by-sentence record.

    Returns:
        The note path, the vault, and which route wrote it, or the reason it failed.
    """
    from .obsidian import export_meeting

    settings = load_settings()
    store = MeetingStore(settings.meetings_dir)
    try:
        result = export_meeting(
            meeting_id,
            store=store,
            vault=vault,
            folder=folder,
            include_transcript=include_transcript,
        )
    except MeetingError as error:
        return {"ok": False, "error": str(error)}
    except Exception as error:  # noqa: BLE001 - report the reason rather than raising
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    return result.as_dict()


def main() -> None:
    """Run the MCP server over stdio (the entry point wired into the DSH profile)."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
