"""Compare local ASR models on speed, memory, and accuracy.

Choosing a model by reputation is guesswork. This harness measures the three
things that actually decide the trade-off for live interpretation:

* **Slice latency** — how long one ~6 s commit takes. This, not throughput, is
  what a viewer experiences as subtitle lag.
* **Accuracy (WER)** — word error rate against a known script, measured both on
  clean speech and on the same speech mixed with noise at a fixed SNR, because
  a smaller model often survives clean audio and falls apart in a real room.
* **Peak memory** — what the model costs to keep resident.

Each model is measured in its own subprocess: weights are hundreds of megabytes,
so measuring them in one process would charge every model for its predecessors,
and one model failing to load would take the whole comparison with it.

Accuracy here is measured on synthesized speech, which is easier than a real
room. Treat the *ranking* as transferable and the absolute numbers as an upper
bound on quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio_io import decode_to_mono, encode_wav
from .config import PROJECT_DIR, SAMPLE_RATE

LOGGER = logging.getLogger("dsh_voice.benchmark")

#: Slice lengths timed per model, in seconds. 6 s is one live commit; 20 s and
#: 30 s are the long-form windows a distilled model is supposed to win on, so
#: measuring only the short one would dismiss such a model unfairly.
SLICE_LENGTHS = (6.0, 20.0)

#: The slice length reported as the headline latency.
PRIMARY_SLICE_SECONDS = 6.0

#: Default SNR for the noisy variant. Around 10 dB is a noisy room.
DEFAULT_SNR_DB = 10.0

#: Reference script the synthesized benchmark audio reads aloud. Long enough
#: (~180 words) that one word of difference is not 2% of the score, and seeded
#: with numbers, a name, and a date — where small models break first.
REFERENCE_TEXT = (
    "Good morning everyone, and thank you for joining on such short notice. "
    "Today we are reviewing the third quarter results before we move on to planning. "
    "Revenue grew twelve percent year over year, which is above the guidance we gave in July. "
    "However, supply chain costs increased by eight percent, so gross margin dropped to forty one percent. "
    "The team in Singapore flagged three delayed shipments, and two of them affect the November release. "
    "We decided to delay the migration to November so that we can finish the security review properly. "
    "Sarah will prepare the vendor comparison by next Friday, and Marcus will update the roadmap. "
    "The budget request for the fourth quarter is two hundred thousand dollars, which is ten percent higher than last year. "
    "We also agreed to revisit the pricing model after the customer interviews are complete. "
    "If there are no objections, we will publish the summary on Thursday and close the review. "
    "Does anyone have questions before we finish?"
)

_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
#: Indexed by tens digit, so the empty slot at 0 is explicit — a leading-space
#: `.split()` would silently drop it and shift every entry by one.
_TENS = ("", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def number_to_words(value: int) -> str:
    """Spell an integer 0..999 the way it is spoken.

    Needed because a recognizer writes "12%" where the script says "twelve
    percent"; without normalizing both sides, that counts as three errors and
    makes every model look equally bad at numbers.

    Args:
        value: Integer to spell.

    Returns:
        The spoken form, or the digits when out of range.
    """
    if value < 0 or value > 999:
        return str(value)
    if value < 20:
        return _UNITS[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] + (f" {_UNITS[ones]}" if ones else "")
    hundreds, rest = divmod(value, 100)
    text = f"{_UNITS[hundreds]} hundred"
    if rest:
        text += f" {number_to_words(rest)}"
    return text


def normalize_for_wer(text: str) -> str:
    """Normalize text so WER compares meaning rather than formatting.

    Lowercases, expands integers and ``%`` into words, drops punctuation, and
    collapses whitespace.

    Args:
        text: Raw transcript or reference.

    Returns:
        A space-separated token string.
    """
    lowered = text.lower()
    lowered = lowered.replace("%", " percent ").replace("&", " and ")
    lowered = re.sub(r"\d+", lambda match: f" {number_to_words(int(match.group()))} ", lowered)
    lowered = re.sub(r"[^a-z0-9']+", " ", lowered)
    return " ".join(lowered.split())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word error rate as edit distance over reference length.

    Args:
        reference: The known script.
        hypothesis: What the model produced.

    Returns:
        Errors per reference word; ``0.0`` for an empty reference.
    """
    ref = normalize_for_wer(reference).split()
    hyp = normalize_for_wer(hypothesis).split()
    if not ref:
        return 0.0
    # Standard Levenshtein over tokens, with a rolling row to stay cheap.
    previous = list(range(len(hyp) + 1))
    for row, ref_word in enumerate(ref, start=1):
        current = [row]
        for column, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(
                previous[column] + 1,          # deletion
                current[column - 1] + 1,       # insertion
                previous[column - 1] + cost,   # substitution
            ))
        previous = current
    return previous[-1] / len(ref)


def add_noise(audio: np.ndarray, snr_db: float, seed: int = 7) -> np.ndarray:
    """Mix white noise into a signal at a target signal-to-noise ratio.

    Args:
        audio: Clean mono float32 samples.
        snr_db: Target SNR in decibels (higher is cleaner).
        seed: Noise seed, so a benchmark is reproducible.

    Returns:
        The mixed signal, clipped into ``[-1, 1]``.
    """
    if audio.size == 0:
        return audio
    speech_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    if speech_rms <= 0:
        return audio
    generator = np.random.default_rng(seed)
    noise = generator.standard_normal(audio.size).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    target_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    scaled = noise * (target_rms / noise_rms) if noise_rms > 0 else noise
    return np.clip(audio + scaled.astype(np.float32), -1.0, 1.0)


def measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Report the achieved SNR of a mixed signal."""
    residual = noisy - clean
    speech = float(np.mean(np.square(clean), dtype=np.float64))
    noise = float(np.mean(np.square(residual), dtype=np.float64))
    if noise <= 0:
        return float("inf")
    return 10.0 * np.log10(speech / noise)


@dataclass
class ModelResult:
    """One model's measurements."""

    model: str
    label: str
    ok: bool = True
    error: str | None = None
    load_s: float = 0.0
    rtf: float = 0.0
    slice_ms: float = 0.0
    slice_p90_ms: float = 0.0
    peak_mb: float = 0.0
    rss_mb: float = 0.0
    wer_clean: float = 0.0
    wer_clean_quantized: float = 0.0
    wer_noisy: float = 0.0
    text_clean: str = ""
    text_noisy: str = ""
    size_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def wer_clean_worst(self) -> float:
        """The worse of the two clean runs.

        Whisper can flip into a repetition loop on a numerically identical
        input — a 16-bit rounding difference is enough — so the honest clean
        score is the worse of the float and quantized runs, not the better one.
        """
        return max(self.wer_clean, self.wer_clean_quantized)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for the wire."""
        return {
            "model": self.model, "label": self.label, "ok": self.ok, "error": self.error,
            "load_s": round(self.load_s, 2), "rtf": round(self.rtf, 3),
            "slice_ms": round(self.slice_ms, 1), "slice_p90_ms": round(self.slice_p90_ms, 1),
            "peak_mb": round(self.peak_mb, 1), "rss_mb": round(self.rss_mb, 1),
            "wer_clean": round(self.wer_clean, 4),
            "wer_clean_quantized": round(self.wer_clean_quantized, 4),
            "wer_clean_worst": round(self.wer_clean_worst, 4),
            "wer_noisy": round(self.wer_noisy, 4),
            "text_clean": self.text_clean, "text_noisy": self.text_noisy,
            "size_mb": round(self.size_mb, 1), **self.extra,
        }


def model_size_mb(path: Path) -> float:
    """Total size of a model directory's weight files, in megabytes."""
    total = 0
    for pattern in ("*.safetensors", "*.npz"):
        for file in path.glob(pattern):
            total += file.stat().st_size
    return total / 1e6


def _peak_memory_mb() -> float:
    """Peak MLX GPU memory for this process, in megabytes."""
    try:
        import mlx.core as mx

        getter = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        return float(getter()) / 1e6
    except Exception:  # noqa: BLE001 - measurement is best-effort
        return 0.0


def _rss_mb() -> float:
    """Peak resident set size for this process, in megabytes."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes.
        return float(usage) / 1e6 if sys.platform == "darwin" else float(usage) / 1e3
    except Exception:  # noqa: BLE001 - measurement is best-effort
        return 0.0


def quantize_int16(audio: np.ndarray) -> np.ndarray:
    """Round-trip audio through 16-bit PCM, as a WAV or upload would.

    Used to probe numerical fragility: a recognizer that decodes one version
    cleanly and loops on the other is not accurate, it is lucky.

    Args:
        audio: Mono float32 samples.

    Returns:
        The same audio after 16-bit quantization.
    """
    return (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype(np.int16).astype(np.float32) / 32767.0


def measure_one(
    model_path: Path,
    *,
    clean: np.ndarray,
    noisy: np.ndarray,
    language: str | None,
    reference: str,
    repeats: int = 3,
    slice_lengths: tuple[float, ...] = SLICE_LENGTHS,
) -> ModelResult:
    """Measure one model in the calling process.

    Decoding uses the *product's* options (``dsh_voice.asr._decode_options``)
    rather than a benchmark-local recipe: measuring a configuration the product
    does not run is how a comparison comes to recommend the wrong model.

    Args:
        model_path: Local model directory.
        clean: Clean reference audio (mono float32 at 16 kHz).
        noisy: The same audio with noise mixed in.
        language: Source language override.
        reference: The script both recordings read.
        repeats: How many slices to time per length.
        slice_lengths: Slice durations, in seconds, to time latency at.

    Returns:
        The measurements, or a result carrying the failure reason.
    """
    import mlx_whisper

    from .asr import _decode_options

    result = ModelResult(model=str(model_path), label=model_path.name, size_mb=model_size_mb(model_path))
    options = _decode_options(language, streaming=False, initial_prompt=None)

    try:
        started = time.monotonic()
        # Warm up: the first call pays the weight load, which is not decode speed.
        mlx_whisper.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32),
                               path_or_hf_repo=str(model_path), **options)
        result.load_s = time.monotonic() - started

        started = time.monotonic()
        clean_out = mlx_whisper.transcribe(clean, path_or_hf_repo=str(model_path), **options)
        clean_wall = time.monotonic() - started
        result.text_clean = str(clean_out.get("text", "")).strip()
        result.wer_clean = word_error_rate(reference, result.text_clean)
        result.rtf = clean_wall / (clean.size / SAMPLE_RATE) if clean.size else 0.0

        quantized_out = mlx_whisper.transcribe(
            quantize_int16(clean), path_or_hf_repo=str(model_path), **options,
        )
        result.wer_clean_quantized = word_error_rate(reference, str(quantized_out.get("text", "")).strip())

        noisy_out = mlx_whisper.transcribe(noisy, path_or_hf_repo=str(model_path), **options)
        result.text_noisy = str(noisy_out.get("text", "")).strip()
        result.wer_noisy = word_error_rate(reference, result.text_noisy)

        # Latency per slice length: the live path commits roughly the short
        # window, but a distilled model's advantage only appears on longer ones.
        per_length: dict[str, dict[str, float]] = {}
        for seconds in slice_lengths:
            window = int(seconds * SAMPLE_RATE)
            if clean.size < window:
                continue
            offsets = np.linspace(0, clean.size - window, num=repeats, dtype=int)
            timings: list[float] = []
            for offset in offsets:
                slice_started = time.monotonic()
                mlx_whisper.transcribe(clean[offset:offset + window],
                                       path_or_hf_repo=str(model_path), **options)
                timings.append((time.monotonic() - slice_started) * 1000.0)
            per_length[f"{seconds:g}s"] = {
                "median_ms": round(statistics.median(timings), 1),
                "p90_ms": round(max(timings), 1),
                "samples_ms": [round(value, 1) for value in timings],
            }
        primary = per_length.get(f"{PRIMARY_SLICE_SECONDS:g}s")
        if primary is not None:
            result.slice_ms = primary["median_ms"]
            result.slice_p90_ms = primary["p90_ms"]
        result.extra["slices"] = per_length
        result.peak_mb = _peak_memory_mb()
        result.rss_mb = _rss_mb()
    except Exception as error:  # noqa: BLE001 - a model that will not load is a data point
        result.ok = False
        result.error = f"{type(error).__name__}: {error}"
    return result


def _run_subprocess(model_path: Path, clean_path: Path, noisy_path: Path,
                    language: str | None, reference: str, repeats: int,
                    slice_lengths: tuple[float, ...]) -> ModelResult:
    """Measure one model in a fresh interpreter and read its JSON report."""
    command = [
        sys.executable, "-m", "dsh_voice.benchmark", "--one", str(model_path),
        "--clean", str(clean_path), "--noisy", str(noisy_path),
        "--reference", reference, "--repeats", str(repeats),
        "--slice-lengths", ",".join(f"{value:g}" for value in slice_lengths),
    ]
    if language:
        command += ["--language", language]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    payload: dict[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{") and line.rstrip().endswith("}"):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            break
    if payload is None:
        return ModelResult(
            model=str(model_path), label=model_path.name, ok=False,
            error=(completed.stderr or completed.stdout or "no output").strip()[-400:],
        )
    # ``as_dict`` flattens ``extra`` into the top level, so split it back out
    # rather than handing unknown keys to the constructor.
    known = {name for name in ModelResult.__dataclass_fields__}
    fields = {key: value for key, value in payload.items() if key in known and key != "extra"}
    extras = {key: value for key, value in payload.items() if key not in known}
    return ModelResult(**fields, extra=extras)


def prepare_audio(
    *,
    text: str = REFERENCE_TEXT,
    voice: str | None = None,
    snr_db: float = DEFAULT_SNR_DB,
    workdir: Path | None = None,
) -> tuple[Path, Path, np.ndarray, np.ndarray, float]:
    """Synthesize the benchmark audio: one clean file, one noisy file.

    Args:
        text: Script to read.
        voice: macOS ``say`` voice; a reliable one is chosen when omitted.
        snr_db: Target SNR for the noisy variant.
        workdir: Directory for the generated files.

    Returns:
        ``(clean_path, noisy_path, clean_audio, noisy_audio, achieved_snr_db)``.
    """
    from .cli import _say_to_file, pick_say_voice

    workdir = workdir or PROJECT_DIR / "selftest"
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / "benchmark-clean.aiff"
    _say_to_file(text, source, voice or pick_say_voice(None))
    clean = decode_to_mono(source, SAMPLE_RATE)
    noisy = add_noise(clean, snr_db)
    achieved = measure_snr(clean, noisy)
    noisy_path = workdir / "benchmark-noisy.wav"
    noisy_path.write_bytes(encode_wav(noisy, SAMPLE_RATE))
    clean_path = workdir / "benchmark-clean.wav"
    clean_path.write_bytes(encode_wav(clean, SAMPLE_RATE))
    return clean_path, noisy_path, clean, noisy, achieved


def run_benchmark(
    models: list[Path],
    *,
    reference: str = REFERENCE_TEXT,
    text: str | None = None,
    voice: str | None = None,
    snr_db: float = DEFAULT_SNR_DB,
    language: str | None = "en",
    repeats: int = 3,
    slice_lengths: tuple[float, ...] = SLICE_LENGTHS,
    workdir: Path | None = None,
) -> dict[str, Any]:
    """Benchmark several models against the same audio.

    Args:
        models: Model directories to compare.
        reference: The script the audio reads, used for WER.
        text: Script to synthesize; defaults to the module's reference text.
        voice: ``say`` voice for the sample.
        snr_db: Target SNR of the noisy variant.
        language: Source language override.
        repeats: Slices timed per model per length.
        slice_lengths: Slice durations, in seconds, to time latency at.
        workdir: Where generated audio lands.

    Returns:
        A payload with the measured results, the achieved SNR, and the ranking.
    """
    script = text or reference
    # Whatever the audio reads *is* the reference — passing a separate one would
    # silently score every model against text nobody spoke.
    reference = script
    clean_path, noisy_path, _clean, _noisy, achieved_snr = prepare_audio(
        text=script, voice=voice, snr_db=snr_db, workdir=workdir,
    )
    results: list[ModelResult] = []
    for model in models:
        if not model.is_dir():
            results.append(ModelResult(model=str(model), label=model.name, ok=False,
                                       error=f"model directory not found: {model}"))
            continue
        LOGGER.info("benchmarking %s", model.name)
        results.append(_run_subprocess(model, clean_path, noisy_path, language, script, repeats, slice_lengths))

    usable = [result for result in results if result.ok]
    # Rank by the latency a viewer feels, breaking ties on accuracy under noise.
    ranked = sorted(usable, key=lambda item: (item.slice_ms, item.wer_noisy + item.wer_clean))
    payload: dict[str, Any] = {
        "reference": script,
        "reference_words": len(normalize_for_wer(script).split()),
        "snr_db": round(achieved_snr, 1),
        "slice_lengths": list(slice_lengths),
        "results": [result.as_dict() for result in results],
        "ranking": [result.label for result in ranked],
        "fastest": ranked[0].label if ranked else None,
        "most_accurate_noisy": (min(usable, key=lambda item: item.wer_noisy).label if usable else None),
    }
    payload["recommendation"] = recommend(payload)
    return payload


def format_table(payload: dict[str, Any]) -> str:
    """Render a benchmark payload as a fixed-width comparison table."""
    lengths = payload.get("slice_lengths") or []
    slice_headers = "".join(f"{f'{length:g}s':>9}" for length in lengths)
    header = (
        f"{'model':<28}{'size':>7}{'load':>7}{'RTF':>7}{'peak':>8}"
        f"{'WER clean':>11}{'worst':>8}{'WER noisy':>11}{slice_headers}"
    )
    lines = [header, "-" * len(header)]
    for result in payload["results"]:
        if not result["ok"]:
            lines.append(f"{result['label']:<28}{'--':>7}{'--':>7}{'--':>7}{'--':>8}"
                         f"{'--':>11}{'--':>8}{'--':>11}{'--' * len(lengths):>9}{'  ' + result['error']}")
            continue
        slices = result.get("slices") or {}
        slice_cells = "".join(
            f"{slices.get(f'{length:g}s', {}).get('median_ms', 0):>8.0f}ms" for length in lengths
        )
        lines.append(
            f"{result['label']:<28}{result['size_mb']:>6.0f}M{result['load_s']:>6.1f}s"
            f"{result['rtf']:>7.2f}{result['peak_mb']:>7.0f}M"
            f"{result['wer_clean'] * 100:>10.1f}%{result['wer_clean_worst'] * 100:>7.1f}%"
            f"{result['wer_noisy'] * 100:>10.1f}%{slice_cells}"
        )
    lines.append("")
    lines.append(f"noisy variant SNR: {payload['snr_db']} dB · {payload.get('reference_words', 0)} reference words")
    lines.append("`worst` is the worse of the float and 16-bit-quantized clean runs — a repetition "
                 "loop triggered by rounding shows up here")
    lines.append(f"fastest: {payload['fastest']} · most accurate under noise: {payload['most_accurate_noisy']}")
    recommendation = payload.get("recommendation")
    if recommendation:
        lines.append(f"recommendation: {recommendation}")
    return "\n".join(lines)


def recommend(payload: dict[str, Any]) -> str:
    """Pick the model with the best measured trade-off and say why.

    The rule is deliberately conservative: a challenger replaces the incumbent
    only if it is meaningfully faster at the primary slice length *and* no worse
    under noise. "Meaningfully" is 20%, which is around the point a viewer
    notices subtitle lag at all.

    Args:
        payload: A benchmark payload from :func:`run_benchmark`.

    Returns:
        A one-line recommendation naming the winner.
    """
    usable = [item for item in payload["results"] if item["ok"]]
    if not usable:
        return "no model loaded successfully; nothing to recommend"
    incumbent = next((item for item in usable if item["label"] == "whisper-large-v3-turbo"), None)
    best = min(usable, key=lambda item: item["slice_ms"])
    if incumbent is None:
        return f"{best['label']} is the fastest measured model ({best['slice_ms']:.0f}ms per slice)"
    if best["label"] == incumbent["label"]:
        return f"{incumbent['label']} is already both fastest and most accurate; keep it"
    # A challenger must be clearly faster AND at least as robust: the worst-case
    # clean score is what catches a model that loops when the input is rounded.
    faster = best["slice_ms"] < incumbent["slice_ms"] * 0.8
    no_worse = best["wer_noisy"] <= incumbent["wer_noisy"] and best["wer_clean_worst"] <= incumbent["wer_clean_worst"]
    if faster and no_worse:
        return (f"switch to {best['label']}: {best['slice_ms']:.0f}ms vs {incumbent['slice_ms']:.0f}ms per "
                f"slice with no accuracy loss under noise")
    reason = "not 20% faster" if not faster else "worse accuracy"
    return (f"keep {incumbent['label']}: {best['label']} is {best['slice_ms']:.0f}ms vs "
            f"{incumbent['slice_ms']:.0f}ms per slice, worst-case clean "
            f"{best['wer_clean_worst'] * 100:.1f}% vs {incumbent['wer_clean_worst'] * 100:.1f}% ({reason})")


def main(argv: list[str] | None = None) -> int:
    """Run either one model measurement (internal) or the full comparison."""
    parser = argparse.ArgumentParser(description="Compare local ASR models")
    parser.add_argument("--one", help="internal: measure a single model and print JSON")
    parser.add_argument("--clean", help="internal: clean audio path")
    parser.add_argument("--noisy", help="internal: noisy audio path")
    parser.add_argument("--reference", default=REFERENCE_TEXT)
    parser.add_argument("--language", default="en")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--slice-lengths", default=",".join(f"{value:g}" for value in SLICE_LENGTHS),
                        help="comma-separated slice durations in seconds")
    parser.add_argument("--models", nargs="*", help="model directories to compare")
    parser.add_argument("--text", help="script to synthesize for the benchmark audio")
    parser.add_argument("--text-file", help="read the script from this file instead of --text")
    parser.add_argument("--voice")
    parser.add_argument("--snr", type=float, default=DEFAULT_SNR_DB)
    parser.add_argument("--json", action="store_true", help="print the raw payload instead of a table")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    slice_lengths = tuple(
        float(part) for part in str(args.slice_lengths).split(",") if part.strip()
    ) or SLICE_LENGTHS

    if args.one:
        clean = decode_to_mono(args.clean, SAMPLE_RATE) if args.clean else np.zeros(0, dtype=np.float32)
        noisy = decode_to_mono(args.noisy, SAMPLE_RATE) if args.noisy else clean
        result = measure_one(
            Path(args.one), clean=clean, noisy=noisy, language=args.language or None,
            reference=args.reference, repeats=args.repeats, slice_lengths=slice_lengths,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False))
        return 0 if result.ok else 1

    models = [Path(item).expanduser() for item in (args.models or [])]
    if not models:
        models = sorted(path for path in (PROJECT_DIR / "models").iterdir() if path.is_dir())
    if args.text_file:
        args.text = Path(args.text_file).expanduser().read_text(encoding="utf-8").strip()
        # An explicit script is its own reference: there is no separate truth.
        args.reference = args.text
    payload = run_benchmark(
        models, reference=args.reference, text=args.text, voice=args.voice,
        snr_db=args.snr, language=args.language or None, repeats=args.repeats,
        slice_lengths=slice_lengths,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_table(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
