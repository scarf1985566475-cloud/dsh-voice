"""Runtime configuration for the dsh-voice transcription and interpretation service.

Every knob is resolvable from the environment so that the DSH host, the MCP
server process, and a hand-run development shell all agree on one source of
truth. Nothing here reads the network or touches the filesystem beyond
resolving paths.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

#: ASR's fixed input rate. Every buffer in this package is mono float32 at this rate.
SAMPLE_RATE = 16_000

#: Package directory (``.../dsh-voice/dsh_voice``).
PACKAGE_DIR = Path(__file__).resolve().parent

#: Project directory (``.../dsh-voice``) — the anchor for models and scratch files.
PROJECT_DIR = PACKAGE_DIR.parent

#: Hugging Face repo the local model directory is populated from (first run only).
DEFAULT_MODEL_REPO = "mlx-community/Qwen3-ASR-0.6B-4bit"

#: Local model directory. Loading from a path keeps every later run fully offline.
DEFAULT_MODEL_DIR = PROJECT_DIR / "models" / "qwen3-asr-0.6b-4bit"


@dataclass(frozen=True)
class ModelChoice:
    """One installable speech-recognition model.

    Attributes:
        tag: Short name accepted by ``DSH_VOICE_MODEL``.
        directory: Local directory the weights live in.
        repo: Hugging Face repo to fetch them from on first use.
        note: One-line description for humans and status payloads.
    """

    tag: str
    directory: Path
    repo: str
    note: str

    @property
    def present(self) -> bool:
        """Whether the weights are already on disk.

        Qwen3-ASR ships ``model.safetensors`` while the older MLX Whisper
        snapshots used ``weights.safetensors``/``weights.npz``; accepting any
        snapshot of that shape keeps this check true for both.
        """
        if not self.directory.is_dir():
            return False
        for pattern in ("*.safetensors", "*.npz"):
            if any(self.directory.glob(pattern)):
                return True
        return False

    def as_public_dict(self) -> dict[str, object]:
        """Serialize for a status payload."""
        return {
            "tag": self.tag,
            "path": str(self.directory),
            "repo": self.repo,
            "present": self.present,
            "note": self.note,
        }


#: Every model this project can run, best-measured first. The ordering is the
#: preference order for a matching language; see ``dsh-voice benchmark`` for the
#: measurements behind it.
MODEL_CATALOG: tuple[ModelChoice, ...] = (
    ModelChoice(
        tag="qwen3",
        directory=DEFAULT_MODEL_DIR,
        repo=DEFAULT_MODEL_REPO,
        note="Qwen3-ASR 0.6B (4-bit): 30 languages, 22 Chinese dialects, auto-detect",
    ),
)


def find_model(tag: str) -> ModelChoice | None:
    """Look up a catalog entry by tag (case-insensitive)."""
    wanted = tag.strip().lower()
    for choice in MODEL_CATALOG:
        if choice.tag == wanted:
            return choice
    return None


def select_model(language: str | None, override: str | None = None) -> ModelChoice:
    """Pick the model to run for one configuration.

    Resolution is deliberately predictable rather than clever: an explicit
    ``DSH_VOICE_MODEL`` wins, and otherwise the single multilingual default is
    used for every language. Qwen3-ASR detects the language per chunk, so no
    routing decision is needed here.

    Args:
        language: Source language code, or ``None`` for auto-detection. Unused
            for selection today; kept in the signature for compatibility with
            callers that pass it.
        override: Explicit model tag, path, or repo id.

    Returns:
        The chosen model. An explicit override always wins, even when its
        weights are absent — the caller then downloads them.
    """
    del language  # selection does not depend on it; see the docstring
    if override:
        tagged = find_model(override)
        if tagged is not None:
            return tagged
        # A path or repo id: honour it as given, with no catalog knowledge.
        return ModelChoice(
            tag=override,
            directory=Path(override).expanduser(),
            repo=override,
            note="explicitly configured",
        )
    default = find_model("qwen3")
    assert default is not None  # the catalog always carries the fallback
    return default


def model_language_warning(model: str | Path, language: str | None) -> str | None:
    """Report a model that cannot handle the configured language.

    Every model in the catalog today is multilingual (Qwen3-ASR covers 30
    languages and 22 Chinese dialects), so there is no bad pairing to report.
    The function stays on the status payload because a model loaded from an
    arbitrary path could still be language-limited; today it returns ``None``.

    Args:
        model: Configured model directory.
        language: Configured source language, or ``None`` for auto-detection.

    Returns:
        A warning sentence, or ``None`` when the pairing is fine.
    """
    del model, language
    return None


def catalog_as_public_dict() -> list[dict[str, object]]:
    """Describe every catalog model, for status payloads and documentation."""
    return [choice.as_public_dict() for choice in MODEL_CATALOG]

#: Loopback only: the service is a local sidecar for the DSH web GUI, never a public server.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8768

#: Where meeting records live: one directory per meeting, each with its transcript.
DEFAULT_MEETINGS_DIR = PROJECT_DIR / "meetings"

#: DeepSeek chat endpoint and model used for the English-to-Chinese leg.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

_ENV_PREFIX = "DSH_VOICE_"


def _env(name: str, default: str | None) -> str | None:
    """Read one prefixed environment variable, treating blanks as unset."""
    value = os.environ.get(_ENV_PREFIX + name) or os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_float(name: str, default: float) -> float:
    """Read one prefixed float, falling back to the default on anything unparseable."""
    raw = _env(name, None)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Read one prefixed integer, falling back to the default on anything unparseable."""
    raw = _env(name, None)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read one prefixed boolean, accepting the usual truthy spellings."""
    raw = _env(name, None)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def dsh_home() -> Path:
    """Resolve the DSH home directory that holds ``.credentials.yaml``."""
    explicit = _env("DSH_HOME", None)
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".dsh-home"


_CREDENTIAL_LINE = re.compile(r"^\s{0,4}([A-Z0-9_]+)\s*:\s*(\S+)\s*$")


def credentials() -> dict[str, str]:
    """Read ``key: value`` pairs from the DSH credentials file.

    The file is a shallow ``version/refs`` mapping, so a line scan is enough and
    keeps this module free of a YAML dependency on the hot path. A missing or
    unreadable file yields an empty mapping rather than an error: the caller
    decides whether the absence is fatal.
    """
    path = dsh_home() / ".credentials.yaml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = _CREDENTIAL_LINE.match(line)
        if match is not None:
            found[match.group(1)] = match.group(2).strip("'\"")
    return found


def deepseek_api_key() -> str | None:
    """Resolve the DeepSeek API key from the environment, then from DSH credentials."""
    explicit = _env("DEEPSEEK_API_KEY", None)
    if explicit:
        return explicit
    return credentials().get("DEEPSEEK_API_KEY")


@dataclass(frozen=True)
class InterpreterSettings:
    """One immutable view of the service configuration.

    Attributes:
        model: Local model directory, or a Hugging Face repo id when the
            directory has not been populated yet.
        model_repo: Repo to snapshot into ``model`` on first use.
        language: Source language forced for decoding; ``None`` auto-detects.
        target_language: Translation target (only Chinese is wired to a prompt).
        host: Bind address for the live server.
        port: Bind port for the live server.
        translate: Whether to translate finalized English segments.
        translate_partials: Whether to translate still-open (partial) segments.
        partial_interval_s: Minimum audio growth before a new partial decode.
        partial_translate_interval_s: Minimum spacing between partial translations.
        end_silence_ms: Trailing silence that closes a segment.
        start_speech_ms: Leading speech that opens a segment.
        max_segment_s: Hard cap on a segment before it is force-closed.
        min_segment_s: Segments shorter than this are dropped as noise.
        record: Whether live sessions are written to the meeting store.
        meetings_dir: Root directory of the meeting store.
        api_base: DeepSeek-compatible API base URL.
        api_model: Chat model used for translation.
        api_key: DeepSeek API key, or ``None`` when translation is unavailable.
    """

    model: str
    model_repo: str
    language: str | None
    target_language: str
    host: str
    port: int
    translate: bool
    translate_partials: bool
    partial_interval_s: float
    partial_translate_interval_s: float
    end_silence_ms: int
    start_speech_ms: int
    max_segment_s: float
    min_segment_s: float
    record: bool
    meetings_dir: str
    api_base: str
    api_model: str
    api_key: str | None

    @property
    def translation_available(self) -> bool:
        """Whether a translation leg can run at all."""
        return self.api_key is not None and self.translate

    def as_public_dict(self) -> dict[str, object]:
        """Describe the settings for a client or model without leaking the API key."""
        tagged = next((item for item in MODEL_CATALOG if item.directory == Path(self.model)), None)
        return {
            "model": self.model,
            "model_tag": tagged.tag if tagged is not None else None,
            "model_ready": Path(self.model).is_dir(),
            "model_warning": model_language_warning(self.model, self.language),
            "language": self.language,
            "target_language": self.target_language,
            "host": self.host,
            "port": self.port,
            "translate": self.translate,
            "translate_partials": self.translate_partials,
            "translation_available": self.translation_available,
            "record": self.record,
            "api_model": self.api_model,
        }


def _env_language() -> str | None:
    """Resolve the source language, where blank and ``auto`` both mean auto-detect.

    Plain :func:`_env` cannot express this: it treats a blank value as unset and
    substitutes the default. It also cannot express "default is auto-detect":
    Qwen3-ASR reports the language per chunk, which is what a bilingual lecture
    needs, so an unset ``DSH_VOICE_LANGUAGE`` means auto rather than English.
    Set ``DSH_VOICE_LANGUAGE=en`` to pin English again.
    """
    raw = os.environ.get(_ENV_PREFIX + "LANGUAGE")
    if raw is None:
        raw = os.environ.get("LANGUAGE")
    if raw is None:
        return None
    cleaned = raw.strip()
    return None if cleaned.lower() in {"", "auto", "none", "detect"} else cleaned


def load_settings(**overrides: object) -> InterpreterSettings:
    """Build the settings for one process.

    Environment variables win over the built-in defaults, and explicit keyword
    overrides win over the environment (the CLI uses them for flags).

    Args:
        **overrides: Field values that replace the resolved defaults.

    Returns:
        The frozen settings for this process.
    """
    values: dict[str, object] = {
        "model": _env("MODEL", None),
        "model_repo": _env("MODEL_REPO", None),
        "language": _env_language(),
        "target_language": _env("TARGET_LANGUAGE", "zh"),
        "host": _env("HOST", DEFAULT_HOST),
        "port": _env_int("PORT", DEFAULT_PORT),
        "translate": _env_bool("TRANSLATE", True),
        "translate_partials": _env_bool("TRANSLATE_PARTIALS", True),
        # Cadence and segment caps are tuned for a busy machine: the decode
        # worker is shared, so a 1.2 s cadence over an uncapped window queues
        # faster than a laptop decodes, which is what makes live captions arrive
        # late. A slower cadence with a short commit window keeps the queue
        # draining while the text still moves.
        "partial_interval_s": _env_float("PARTIAL_INTERVAL", 2.0),
        "partial_translate_interval_s": _env_float("PARTIAL_TRANSLATE_INTERVAL", 2.5),
        # 700 ms 追踪静音尾比 500 ms 少切半句：少掉半截话、少触发幻觉补全，
        # 代价是整句延迟约 +0.2 s（用户选定，2026-10-28）。
        "end_silence_ms": _env_int("END_SILENCE_MS", 700),
        "start_speech_ms": _env_int("START_SPEECH_MS", 140),
        "max_segment_s": _env_float("MAX_SEGMENT", 20.0),
        "min_segment_s": _env_float("MIN_SEGMENT", 0.45),
        # 默认不把实时会话写入会议记录：纪要改由整场录音离线转写产出，
        # 实时记录只会留下同一场会的低质量副本。需要时设 DSH_VOICE_RECORD=1。
        "record": _env_bool("RECORD", False),
        "meetings_dir": _env("MEETINGS_DIR", str(DEFAULT_MEETINGS_DIR)),
        "api_base": _env("API_BASE", DEEPSEEK_BASE_URL),
        "api_model": _env("API_MODEL", DEEPSEEK_MODEL),
        "api_key": deepseek_api_key(),
    }
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in values:
            raise TypeError(f"load_settings: unknown setting {key!r}")
        values[key] = value
    # ``language`` accepts the empty string as "auto-detect".
    if values["language"] == "":
        values["language"] = None
    # Resolve the model through the catalog so a bare tag ("qwen3") becomes its
    # directory, and so model_repo always describes the model actually selected.
    choice = select_model(values["language"], str(values["model"]) if values["model"] else None)  # type: ignore[arg-type]
    values["model"] = str(choice.directory)
    values["model_repo"] = values["model_repo"] or choice.repo
    return InterpreterSettings(**values)  # type: ignore[arg-type]
