"""Local speech recognition on Qwen3-ASR (Apple Silicon, fully offline).

The model is loaded from a local directory so steady-state operation never
touches the network: :func:`ensure_model` performs the one-time snapshot, and
every later call addresses the directory directly. Weights are cached per path
for the life of the process, so a long interpretation session pays the load
cost once.

Why Qwen3-ASR rather than Whisper: on our own recordings the Whisper family
answered near-silence with stock phrases ("Thank you.", "Loss.") and entered
repetition loops; Qwen3-ASR is trained against exactly that failure, detects
the spoken language per chunk (a lecture with Chinese side-talk no longer gets
translated as English), and covers 30 languages plus 22 Chinese dialects —
which is the accent problem this project exists for.
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

#: Serializes every decode: MLX runs on one GPU stream and the cache holds one model.
_DECODE_LOCK = threading.Lock()

_model_lock = threading.Lock()
_model_ready = False

#: Model instances keyed by resolved directory, so one process loads each once.
_MODEL_CACHE: dict[str, Any] = {}

#: Audio is decoded in these slices. Qwen3-ASR has no token-level timestamps,
#: so the slice length *is* the timestamp granularity a meeting record gets;
#: 30 s keeps citations useful without cutting too many words at a boundary.
CHUNK_DURATION_S = 30.0

#: Very short live windows are padded to this length by the model's chunker.
#: Keeping it small means a 0.6 s span is not decoded with 1.4 s of silence.
MIN_CHUNK_DURATION_S = 0.3

#: Language codes and spellings people actually pass in, mapped to the names
#: Qwen3-ASR's prompt expects. Anything unknown falls back to auto-detection.
_LANGUAGE_ALIASES: dict[str, str] = {
    "en": "English", "eng": "English", "english": "English",
    "zh": "Chinese", "zho": "Chinese", "cmn": "Chinese", "chinese": "Chinese",
    "mandarin": "Chinese", "zh-cn": "Chinese", "zh-hans": "Chinese", "zh-tw": "Chinese",
    "yue": "Cantonese", "cantonese": "Cantonese", "zh-hk": "Cantonese",
    "ja": "Japanese", "jp": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "kr": "Korean", "korean": "Korean",
    "fr": "French", "fra": "French", "fre": "French", "french": "French",
    "de": "German", "deu": "German", "ger": "German", "german": "German",
    "es": "Spanish", "spa": "Spanish", "spanish": "Spanish",
    "pt": "Portuguese", "por": "Portuguese", "portuguese": "Portuguese",
    "it": "Italian", "ita": "Italian", "italian": "Italian",
    "ru": "Russian", "rus": "Russian", "russian": "Russian",
    "ar": "Arabic", "ara": "Arabic", "arabic": "Arabic",
    "th": "Thai", "tha": "Thai", "thai": "Thai",
    "vi": "Vietnamese", "vie": "Vietnamese", "vietnamese": "Vietnamese",
    "id": "Indonesian", "ind": "Indonesian", "indonesian": "Indonesian",
    "tr": "Turkish", "tur": "Turkish", "turkish": "Turkish",
    "hi": "Hindi", "hin": "Hindi", "hindi": "Hindi",
    "nl": "Dutch", "nld": "Dutch", "dut": "Dutch", "dutch": "Dutch",
    "sv": "Swedish", "swe": "Swedish", "swedish": "Swedish",
    "da": "Danish", "dan": "Danish", "danish": "Danish",
    "fi": "Finnish", "fin": "Finnish", "finnish": "Finnish",
    "no": "Norwegian", "nor": "Norwegian", "norwegian": "Norwegian",
    "pl": "Polish", "pol": "Polish", "polish": "Polish",
    "cs": "Czech", "ces": "Czech", "cze": "Czech", "czech": "Czech",
    "el": "Greek", "ell": "Greek", "gre": "Greek", "greek": "Greek",
    "hu": "Hungarian", "hun": "Hungarian", "hungarian": "Hungarian",
    "ro": "Romanian", "ron": "Romanian", "rum": "Romanian", "romanian": "Romanian",
    "sk": "Slovak", "slk": "Slovak", "slo": "Slovak", "slovak": "Slovak",
    "uk": "Ukrainian", "ukr": "Ukrainian", "ukrainian": "Ukrainian",
    "fa": "Persian", "fas": "Persian", "per": "Persian", "persian": "Persian",
    "ur": "Urdu", "urd": "Urdu", "urdu": "Urdu",
    "ms": "Malay", "msa": "Malay", "may": "Malay", "malay": "Malay",
    "he": "Hebrew", "heb": "Hebrew", "hebrew": "Hebrew",
    "bn": "Bengali", "ben": "Bengali", "bengali": "Bengali",
    "ta": "Tamil", "tam": "Tamil", "tamil": "Tamil",
    "te": "Telugu", "tel": "Telugu", "telugu": "Telugu",
    "mr": "Marathi", "mar": "Marathi", "marathi": "Marathi",
    "gu": "Gujarati", "guj": "Gujarati", "gujarati": "Gujarati",
    "kn": "Kannada", "kan": "Kannada", "kannada": "Kannada",
    "ml": "Malayalam", "mal": "Malayalam", "malayalam": "Malayalam",
    "pa": "Punjabi", "pan": "Punjabi", "punjabi": "Punjabi",
    "sw": "Swahili", "swa": "Swahili", "swahili": "Swahili",
    "fil": "Filipino", "tl": "Filipino", "filipino": "Filipino",
}

#: Canonical name back to a two-letter code for the wire.
_LANGUAGE_CODES: dict[str, str] = {
    "Chinese": "zh", "English": "en", "Cantonese": "yue", "Japanese": "ja",
    "Korean": "ko", "French": "fr", "German": "de", "Spanish": "es",
    "Portuguese": "pt", "Italian": "it", "Russian": "ru", "Arabic": "ar",
    "Thai": "th", "Vietnamese": "vi", "Indonesian": "id", "Turkish": "tr",
    "Hindi": "hi", "Dutch": "nl", "Swedish": "sv", "Danish": "da",
    "Finnish": "fi", "Norwegian": "no", "Polish": "pl", "Czech": "cs",
    "Greek": "el", "Hungarian": "hu", "Romanian": "ro", "Slovak": "sk",
    "Ukrainian": "uk", "Persian": "fa", "Urdu": "ur", "Malay": "ms",
    "Hebrew": "he", "Bengali": "bn", "Tamil": "ta", "Telugu": "te",
    "Marathi": "mr", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml",
    "Punjabi": "pa", "Swahili": "sw", "Filipino": "fil",
}


class ModelUnavailableError(RuntimeError):
    """Raised when the Qwen3-ASR weights are neither present locally nor downloadable."""


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


def model_present(directory: str | Path | None = None) -> bool:
    """Whether a model directory already holds weights."""
    target = Path(directory).expanduser() if directory is not None else model_path()
    if not target.is_dir():
        return False
    for pattern in ("*.safetensors", "*.npz"):
        if any(target.glob(pattern)):
            return True
    return False


def ensure_model(repo: str | None = None, destination: str | Path | None = None) -> Path:
    """Materialize the Qwen3-ASR weights locally, downloading them only when absent.

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
    if model_present(target):
        return target
    endpoint = configure_hf_endpoint()
    LOGGER.info("downloading %s into %s via %s", repo or settings.model_repo, target, endpoint)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ModelUnavailableError("huggingface_hub is required to fetch the model weights") from error
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
    if not model_present(target):
        raise ModelUnavailableError(f"the snapshot in {target} contains no weights")
    return target


def _load_model(directory: Path) -> Any:
    """Load (and cache) the Qwen3-ASR model for one directory.

    Args:
        directory: Model directory holding ``config.json`` and the weights.

    Returns:
        The loaded model, ready for ``generate``.

    Raises:
        ModelUnavailableError: When the weights cannot be loaded.
    """
    key = str(directory)
    with _model_lock:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from mlx_audio.stt.utils import load_model
        except ImportError as error:  # pragma: no cover - dependency is declared
            raise ModelUnavailableError("mlx-audio is required to load Qwen3-ASR") from error
        try:
            model = load_model(key)
        except Exception as error:  # noqa: BLE001 - surface one typed error
            raise ModelUnavailableError(f"could not load {key}: {error}") from error
        _MODEL_CACHE[key] = model
        return model


def warmup() -> str:
    """Load the weights and compile one decode so the first live span is not the slow one.

    Returns:
        The resolved model directory.

    Raises:
        ModelUnavailableError: When the model cannot be materialized.
    """
    global _model_ready  # noqa: PLW0603 - one process-wide latch
    directory = ensure_model()
    model = _load_model(directory)
    with _model_lock:
        if not _model_ready:
            with _DECODE_LOCK:
                silence = np.zeros(int(0.6 * SAMPLE_RATE), dtype=np.float32)
                model.generate(silence, language=None, chunk_duration=CHUNK_DURATION_S,
                               min_chunk_duration=MIN_CHUNK_DURATION_S, verbose=False)
            _model_ready = True
    return str(directory)


def _qwen_language(language: str | None) -> str | None:
    """Map a language code or name onto Qwen3-ASR's prompt vocabulary.

    Args:
        language: Two-letter code, full name, or ``None`` for auto-detection.

    Returns:
        The canonical name the model expects, or ``None`` to auto-detect. An
        unknown value is logged and treated as auto-detection rather than
        passed through, because a name the model does not know produces a
        prompt it was not trained on.
    """
    if language is None:
        return None
    cleaned = str(language).strip()
    if not cleaned:
        return None
    mapped = _LANGUAGE_ALIASES.get(cleaned.lower())
    if mapped is None:
        LOGGER.warning("unknown language %r; falling back to auto-detect", cleaned)
    return mapped


def _language_code(name: str | None) -> str | None:
    """Map a detected language name back to a two-letter code for the wire."""
    if not name:
        return None
    return _LANGUAGE_CODES.get(name, str(name).lower())


def _decode_options(language: str | None, *, streaming: bool, initial_prompt: str | None) -> dict[str, Any]:
    """Build the generation options shared by the file and live paths.

    Qwen3-ASR has no temperature fallback ladder — its repetition guard is the
    training itself, and it is deterministically greedy by default. What does
    matter here:

    * ``language`` is either a canonical name or ``None``; auto-detection is
      what lets a bilingual lecture keep its Chinese side-talk in Chinese.
    * ``chunk_duration`` fixes the timestamp granularity of a long file.
    * ``system_prompt`` is the model's vocabulary-bias channel, so a CLI
      ``--prompt`` still works the way it did with Whisper's ``initial_prompt``.

    Args:
        language: Source language, or ``None`` to auto-detect.
        streaming: Live mode; windows there are shorter than one chunk, so this
            changes nothing today but keeps the two call sites symmetric.
        initial_prompt: Optional domain vocabulary hint.

    Returns:
        Keyword arguments for ``model.generate``.
    """
    del streaming  # a live window is always one chunk
    options: dict[str, Any] = {
        "language": _qwen_language(language),
        "chunk_duration": CHUNK_DURATION_S,
        "min_chunk_duration": MIN_CHUNK_DURATION_S,
        "verbose": False,
    }
    if initial_prompt:
        options["system_prompt"] = initial_prompt
    return options


def _run(audio: np.ndarray, *, streaming: bool, language: str | None, initial_prompt: str | None) -> Transcript:
    """Decode one waveform without touching the network or the global model latch."""
    if audio.size == 0:
        return Transcript(text="")
    settings = load_settings()
    directory = ensure_model()
    model = _load_model(directory)

    options = _decode_options(language, streaming=streaming, initial_prompt=initial_prompt)
    with _DECODE_LOCK:
        output = model.generate(np.ascontiguousarray(audio, dtype=np.float32), **options)

    detected: list[str] = []
    raw_language = getattr(output, "language", None)
    if isinstance(raw_language, (list, tuple)):
        detected = [str(item) for item in raw_language if item]
    elif raw_language:
        detected = [str(raw_language)]

    segments = [
        TranscriptSegment(
            start=float(segment.get("start", 0.0)),
            end=float(segment.get("end", 0.0)),
            text=str(segment.get("text", "")).strip(),
        )
        for segment in getattr(output, "segments", None) or []
    ]
    duration = audio.size / SAMPLE_RATE
    if not segments and str(output.text).strip():
        segments = [TranscriptSegment(start=0.0, end=duration, text=str(output.text).strip())]
    return Transcript(
        text=str(output.text).strip(),
        language=_language_code(detected[0] if detected else None) or settings.language,
        segments=[segment for segment in segments if segment.text],
        duration=duration,
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
