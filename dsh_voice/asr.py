"""Local speech recognition on MLX Whisper (Apple Silicon, fully offline).

The model is loaded from a local directory so steady-state operation never
touches the network: :func:`ensure_model` performs the one-time snapshot, and
every later call addresses the directory directly. ``mlx_whisper`` already
caches the loaded weights per path (``ModelHolder``), so a long interpretation
session pays the load cost once.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_MODEL_REPO, SAMPLE_RATE, load_settings

LOGGER = logging.getLogger("dsh_voice.asr")

#: huggingface.co is unreachable from some networks while this mirror is not.
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

#: Serializes every decode: MLX runs on one GPU stream and the holder caches one model.
_DECODE_LOCK = threading.Lock()

_model_lock = threading.Lock()
_model_ready = False


class ModelUnavailableError(RuntimeError):
    """Raised when the Whisper weights are neither present locally nor downloadable."""


@dataclass
class TranscriptSegment:
    """One decoded span with its position in the source audio."""

    start: float
    end: float
    text: str

    def as_dict(self) -> dict[str, Any]:
        """Serialize for the wire."""
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


@dataclass
class Transcript:
    """A complete decode result."""

    text: str
    language: str | None = None
    segments: list[TranscriptSegment] = field(default_factory=list)
    duration: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Serialize for the wire, keeping segmentation for callers that cite timings."""
        return {
            "text": self.text,
            "language": self.language,
            "duration": round(self.duration, 3),
            "segments": [segment.as_dict() for segment in self.segments],
        }


def configure_hf_endpoint() -> str:
    """Point huggingface_hub at a reachable endpoint before it is first used.

    Returns:
        The endpoint that will be used for a snapshot download.
    """
    endpoint = os.environ.get("DSH_VOICE_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT")
    if not endpoint:
        endpoint = DEFAULT_HF_ENDPOINT
    os.environ["HF_ENDPOINT"] = endpoint
    return endpoint


def model_path() -> Path:
    """Resolve the configured local model directory."""
    return Path(load_settings().model).expanduser()


def model_present() -> bool:
    """Whether the configured model directory already holds weights."""
    directory = model_path()
    if not directory.is_dir():
        return False
    return (directory / "weights.safetensors").exists() or (directory / "weights.npz").exists()


def ensure_model(repo: str | None = None, destination: str | Path | None = None) -> Path:
    """Materialize the Whisper weights locally, downloading them only when absent.

    Args:
        repo: Hugging Face repo id; defaults to the configured repo.
        destination: Target directory; defaults to the configured model path.

    Returns:
        The local directory holding the weights.

    Raises:
        ModelUnavailableError: When the snapshot cannot be fetched.
    """
    settings = load_settings()
    target = Path(destination).expanduser() if destination is not None else model_path()
    if model_present() and destination is None:
        return target
    if (target / "weights.safetensors").exists() or (target / "weights.npz").exists():
        return target
    endpoint = configure_hf_endpoint()
    LOGGER.info("downloading %s into %s via %s", repo or settings.model_repo, target, endpoint)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ModelUnavailableError("huggingface_hub is required to fetch the Whisper weights") from error
    try:
        # Keep the hub cache inside the project so a sandboxed process never
        # needs write access to the user's home cache directory.
        os.environ.setdefault("HF_HOME", str(target.parent / ".hf"))
        snapshot_download(repo_id=repo or settings.model_repo, local_dir=str(target))
    except Exception as error:  # noqa: BLE001 - surface any hub failure as one typed error
        raise ModelUnavailableError(
            f"could not fetch {repo or settings.model_repo}: {error}. "
            f"Set DSH_VOICE_HF_ENDPOINT to a reachable mirror, or place the weights in {target}.",
        ) from error
    return target


def warmup() -> str:
    """Load the weights once so the first live segment is not the slow one.

    Returns:
        The resolved model directory.

    Raises:
        ModelUnavailableError: When the model cannot be materialized.
    """
    global _model_ready  # noqa: PLW0603 - one process-wide latch
    directory = ensure_model()
    with _model_lock:
        if not _model_ready:
            import mlx_whisper

            silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
            mlx_whisper.transcribe(silence, path_or_hf_repo=str(directory), verbose=None, language="en")
            _model_ready = True
    return str(directory)


def _decode_options(language: str | None, *, streaming: bool, initial_prompt: str | None) -> dict[str, Any]:
    """Build the decoding options shared by the file and live paths.

    Two settings here are load-bearing, both chosen from measurement on a hard,
    domain-specific sample (an academic lecture) rather than from the defaults:

    * ``temperature`` keeps Whisper's fallback ladder. Pinning it to ``0.0``
      looks harmless and is a common recipe, but it *removes the retry* that the
      compression-ratio and log-probability guards trigger — and a recognizer
      stuck in a repetition loop is exactly what those guards catch. Measured on
      the sample: 72.6% WER pinned versus 4.7% with the ladder.
    * ``condition_on_previous_text`` is off for **both** paths. Feeding a window
      its own previous output is what starts a runaway repetition on hard
      content; measured 0.9% off versus 4.7% on, with the failure mode being
      hundreds of hallucinated words rather than a small accuracy difference.

    Args:
        language: Source language, or ``None`` to auto-detect.
        streaming: Live mode; currently affects nothing beyond intent, since
            cross-window prompting is off everywhere.
        initial_prompt: Optional domain vocabulary hint.

    Returns:
        Keyword arguments for ``mlx_whisper.transcribe``.
    """
    options: dict[str, Any] = {
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "condition_on_previous_text": False,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "verbose": None,
    }
    if language:
        options["language"] = language
    if initial_prompt:
        options["initial_prompt"] = initial_prompt
    return options


def _run(audio: np.ndarray, *, streaming: bool, language: str | None, initial_prompt: str | None) -> Transcript:
    """Decode one waveform without touching the network or the global model latch."""
    if audio.size == 0:
        return Transcript(text="")
    settings = load_settings()
    directory = ensure_model()
    import mlx_whisper

    options = _decode_options(language, streaming=streaming, initial_prompt=initial_prompt)
    with _DECODE_LOCK:
        result = mlx_whisper.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            path_or_hf_repo=str(directory),
            **options,
        )
    segments = [
        TranscriptSegment(
            start=float(segment.get("start", 0.0)),
            end=float(segment.get("end", 0.0)),
            text=str(segment.get("text", "")).strip(),
        )
        for segment in result.get("segments", [])
    ]
    return Transcript(
        text=str(result.get("text", "")).strip(),
        language=result.get("language") or settings.language,
        segments=[segment for segment in segments if segment.text],
        duration=audio.size / SAMPLE_RATE,
    )


def transcribe_array(
    audio: np.ndarray,
    *,
    language: str | None = None,
    streaming: bool = False,
    initial_prompt: str | None = None,
) -> Transcript:
    """Transcribe an in-memory mono 16 kHz float32 waveform.

    Args:
        audio: Samples in ``[-1, 1]`` at :data:`config.SAMPLE_RATE`.
        language: Source language override; defaults to the configured language.
        streaming: Use live-mode decoding options.
        initial_prompt: Optional domain vocabulary hint.

    Returns:
        The transcript, including segment timings.
    """
    settings = load_settings()
    return _run(
        audio,
        streaming=streaming,
        language=settings.language if language is None else language,
        initial_prompt=initial_prompt,
    )


def transcribe_file(
    path: str | Path,
    *,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> Transcript:
    """Transcribe a media file of any ffmpeg-readable format.

    Args:
        path: Source media file.
        language: Source language override; defaults to the configured language.
        initial_prompt: Optional domain vocabulary hint.

    Returns:
        The transcript, including segment timings.

    Raises:
        audio_io.AudioToolError: When the file cannot be decoded.
    """
    from .audio_io import decode_to_mono

    settings = load_settings()
    audio = decode_to_mono(path, SAMPLE_RATE)
    return _run(
        audio,
        streaming=False,
        language=settings.language if language is None else language,
        initial_prompt=initial_prompt,
    )
