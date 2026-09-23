"""Audio decode and capture through the ffmpeg binary bundled with ``imageio-ffmpeg``.

Mel-frequency work and resampling are delegated to ffmpeg so that this package
never needs sox, a system ffmpeg, or a Python sound card binding. Every decode
returns mono float32 at :data:`config.SAMPLE_RATE`, which is exactly what MLX
Whisper accepts as its ``audio`` argument.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE


class AudioToolError(RuntimeError):
    """Raised when no usable ffmpeg binary exists or an ffmpeg run fails."""


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Resolve an ffmpeg executable, preferring the pip-installed static build.

    Returns:
        Absolute path to an ffmpeg binary.

    Raises:
        AudioToolError: When neither ``imageio-ffmpeg`` nor ``PATH`` provides one.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - any import/binary failure falls through to PATH
        found = shutil.which("ffmpeg")
        if found is None:
            raise AudioToolError(
                "no ffmpeg binary available: install imageio-ffmpeg (pip install imageio-ffmpeg) "
                "or put ffmpeg on PATH",
            ) from None
        return found


@functools.lru_cache(maxsize=1)
def ffmpeg_version() -> str:
    """Report the resolved ffmpeg build banner (first line only)."""
    result = subprocess.run(  # noqa: S603 - fixed argument vector, no shell
        [ffmpeg_exe(), "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (result.stdout or "").splitlines()[0] if result.stdout else "unknown"


def _run_error(prefix: str, result: subprocess.CompletedProcess[bytes]) -> AudioToolError:
    """Build an error carrying ffmpeg's own stderr tail."""
    tail = (result.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-6:]
    detail = "\n".join(tail) if tail else "(no stderr)"
    return AudioToolError(f"{prefix}\n{detail}")


def decode_to_mono(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any ffmpeg-readable media into a mono float32 waveform.

    Args:
        path: Source media file (wav, mp3, m4a, mp4, flac, ...).
        sample_rate: Output sample rate in Hz.

    Returns:
        One-dimensional float32 array in ``[-1, 1]``.

    Raises:
        AudioToolError: When the file is missing or ffmpeg fails.
    """
    source = Path(path).expanduser()
    if not source.exists():
        raise AudioToolError(f"audio file not found: {source}")
    result = subprocess.run(  # noqa: S603 - fixed argument vector, no shell
        [
            ffmpeg_exe(), "-nostdin", "-loglevel", "error",
            "-i", str(source),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(sample_rate),
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise _run_error(f"ffmpeg failed to decode {source}", result)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def decode_bytes(data: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode an in-memory encoded audio payload into a mono float32 waveform.

    Args:
        data: Encoded container bytes (what a browser upload or a recording holds).
        sample_rate: Output sample rate in Hz.

    Returns:
        One-dimensional float32 array in ``[-1, 1]``.

    Raises:
        AudioToolError: When ffmpeg cannot decode the payload.
    """
    result = subprocess.run(  # noqa: S603 - fixed argument vector, no shell
        [
            ffmpeg_exe(), "-nostdin", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(sample_rate),
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise _run_error("ffmpeg failed to decode the uploaded audio", result)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def encode_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode mono float32 samples as a 16-bit PCM WAV payload.

    Args:
        audio: Mono float32 samples in ``[-1, 1]``.
        sample_rate: Sample rate in Hz.

    Returns:
        Complete WAV file bytes (RIFF header included).
    """
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    channels = 1
    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    header = b"".join([
        b"RIFF", (36 + len(pcm)).to_bytes(4, "little"), b"WAVE",
        b"fmt ", (16).to_bytes(4, "little"),
        (1).to_bytes(2, "little"), (channels).to_bytes(2, "little"),
        sample_rate.to_bytes(4, "little"), byte_rate.to_bytes(4, "little"),
        (channels * bits // 8).to_bytes(2, "little"), (bits).to_bytes(2, "little"),
        b"data", len(pcm).to_bytes(4, "little"),
    ])
    return header + pcm


def _avfoundation_devices() -> str:
    """Return ffmpeg's avfoundation device listing (empty when unsupported)."""
    result = subprocess.run(  # noqa: S603 - fixed argument vector, no shell
        [ffmpeg_exe(), "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
        check=False,
    )
    return (result.stderr or "") + (result.stdout or "")


def list_input_devices() -> list[dict[str, str]]:
    """List capture-capable audio devices the bundled ffmpeg can address.

    Returns:
        One entry per audio device with its ``index`` (as the avfoundation
        ``":<index>"`` specifier) and human-readable ``name``. Empty when the
        platform or build has no avfoundation support.
    """
    listing = _avfoundation_devices()
    devices: list[dict[str, str]] = []
    in_audio_section = False
    for line in listing.splitlines():
        if "AVFoundation audio devices" in line:
            in_audio_section = True
            continue
        if "AVFoundation video devices" in line:
            in_audio_section = False
            continue
        if not in_audio_section or "AVFoundation" not in line:
            continue
        # ffmpeg prints: [AVFoundation indev @ 0x...] [0] MacBook Pro Microphone
        if "] [" not in line:
            continue
        index = line.rpartition("] [")[2].split("]")[0].strip()
        name = line.rpartition("] ")[2].strip()
        if index:
            devices.append({"index": index, "name": name})
    return devices


def record_to_file(
    out_path: str | Path,
    seconds: float,
    device_index: str = "0",
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Capture microphone audio into a WAV file with ffmpeg's avfoundation input.

    On macOS the *hosting* process needs microphone permission (System Settings
    -> Privacy & Security -> Microphone); a denial surfaces as an ffmpeg error.

    Args:
        out_path: Destination ``.wav`` path.
        seconds: Capture duration.
        device_index: avfoundation audio device index (see :func:`list_input_devices`).
        sample_rate: Output sample rate in Hz.

    Returns:
        The written path.

    Raises:
        AudioToolError: When ffmpeg fails or writes nothing.
    """
    destination = Path(out_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(  # noqa: S603 - fixed argument vector, no shell
        [
            ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-y",
            "-f", "avfoundation", "-i", f":{device_index}",
            "-t", f"{seconds:.3f}",
            "-ac", "1", "-ar", str(sample_rate),
            "-acodec", "pcm_s16le",
            str(destination),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not destination.exists():
        raise _run_error(f"ffmpeg failed to record from audio device :{device_index}", result)
    return destination


def iter_audio_blocks(path: str | Path, block_seconds: float = 0.5, sample_rate: int = SAMPLE_RATE) -> Iterator[np.ndarray]:
    """Stream a media file as fixed-size mono float32 blocks.

    Streaming keeps long-file transcription memory-flat: callers slice the
    model input per window instead of materializing a whole recording.

    Args:
        path: Source media file.
        block_seconds: Block duration in seconds.
        sample_rate: Output sample rate in Hz.

    Yields:
        Consecutive mono float32 blocks, the last one short.
    """
    block = max(1, int(round(block_seconds * sample_rate)))
    offset = 0
    audio = decode_to_mono(path, sample_rate)
    while offset < audio.size:
        yield audio[offset:offset + block]
        offset += block
