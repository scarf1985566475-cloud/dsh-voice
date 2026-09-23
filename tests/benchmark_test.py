"""Offline checks of the model benchmark harness.

The harness decides which ASR model this project ships with, so its arithmetic
has to be right: a WER that miscounts numbers would declare the wrong winner,
and a noise mixer that misses its target SNR would make the noisy comparison
meaningless. None of this needs a model, a network, or a microphone.

Usage::

    .venv-audio/bin/python tests/benchmark_test.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsh_voice.benchmark import (  # noqa: E402
    ModelResult,
    add_noise,
    format_table,
    measure_snr,
    model_size_mb,
    normalize_for_wer,
    number_to_words,
    recommend,
    word_error_rate,
)

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record one assertion."""
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def main() -> int:
    """Run every harness check."""
    print("numbers to words")
    cases = {
        0: "zero", 5: "five", 12: "twelve", 19: "nineteen", 20: "twenty",
        21: "twenty one", 41: "forty one", 99: "ninety nine", 100: "one hundred",
        105: "one hundred five", 118: "one hundred eighteen", 999: "nine hundred ninety nine",
    }
    for value, expected in cases.items():
        actual = number_to_words(value)
        check(actual == expected, f"{value} -> {actual!r}")
    # The table bug this test exists for: an off-by-one tens lookup.
    for tens in range(2, 10):
        spelled = number_to_words(tens * 10)
        check(bool(spelled) and spelled != "0", f"{tens * 10} spells ({spelled})")
    check(number_to_words(1234) == "1234", "out-of-range values fall back to digits")
    check(number_to_words(-1) == "-1", "negatives fall back to digits")

    print("\nnormalization")
    check(normalize_for_wer("Revenue grew 12% year over year.") ==
          "revenue grew twelve percent year over year", "expands digits and percent")
    check(normalize_for_wer("A/B  test--case") == "a b test case", "strips punctuation")
    check(normalize_for_wer("R&D") == "r and d", "spells out an ampersand")
    check(normalize_for_wer("") == "", "handles empty input")

    print("\nword error rate")
    check(word_error_rate("hello world", "hello world") == 0.0, "identical text scores zero")
    check(word_error_rate("", "anything") == 0.0, "an empty reference scores zero, not infinity")
    check(abs(word_error_rate("hello world", "hello there") - 0.5) < 1e-9, "one substitution in two words")
    check(abs(word_error_rate("a b c d", "a b c d e") - 0.25) < 1e-9, "one insertion in four words")
    check(abs(word_error_rate("a b c d", "a b c") - 0.25) < 1e-9, "one deletion in four words")
    # The reason normalization exists: a formatter difference must not count.
    check(word_error_rate("Revenue grew twelve percent", "Revenue grew 12%") == 0.0,
          "digits and spelled numbers compare equal")
    check(word_error_rate("margin dropped to forty one percent", "margin dropped to 41%") == 0.0,
          "two-digit expansion compares equal")
    check(word_error_rate("We delay to November", "We delay to November") == 0.0, "case differences ignore")
    check(word_error_rate("hello world", "goodbye world") > 0.0, "a real difference still counts")

    print("\nnoise mixing")
    generator = np.random.default_rng(3)
    speech = generator.standard_normal(16000 * 5).astype(np.float32) * 0.1
    for target in (20.0, 10.0, 5.0):
        mixed = add_noise(speech, target)
        achieved = measure_snr(speech, mixed)
        check(abs(achieved - target) < 0.5, f"hits {target:.0f} dB SNR (achieved {achieved:.2f})")
    check(np.max(np.abs(add_noise(speech * 100, 10.0))) <= 1.0, "clips into [-1, 1]")
    first = add_noise(speech, 10.0, seed=7)
    second = add_noise(speech, 10.0, seed=7)
    check(np.array_equal(first, second), "is reproducible for a given seed")
    check(not np.array_equal(first, add_noise(speech, 10.0, seed=8)), "a different seed gives different noise")
    check(add_noise(np.zeros(0, dtype=np.float32), 10.0).size == 0, "handles empty audio")
    silence = np.zeros(16000, dtype=np.float32)
    check(np.array_equal(add_noise(silence, 10.0), silence), "leaves pure silence alone")
    check(math.isinf(measure_snr(speech, speech)), "reports infinite SNR for identical signals")

    print("\nreport shape")
    result = ModelResult(
        model="/tmp/model", label="whisper-test", load_s=1.234, rtf=0.25,
        slice_ms=321.0, slice_p90_ms=400.0, peak_mb=1500.0, rss_mb=3000.0,
        wer_clean=0.0, wer_noisy=0.1, size_mb=1500.0,
        extra={"slices": {"6s": {"median_ms": 321.0, "p90_ms": 400.0}, "20s": {"median_ms": 900.0, "p90_ms": 950.0}}},
    )
    payload = {
        "reference": "x", "snr_db": 10.0, "slice_lengths": [6.0, 20.0],
        "results": [result.as_dict()], "ranking": ["whisper-test"],
        "fastest": "whisper-test", "most_accurate_noisy": "whisper-test",
        "recommendation": "keep whisper-test",
    }
    table = format_table(payload)
    check("whisper-test" in table, "table names the model")
    check("321ms" in table, "table reports the 6s slice latency")
    check("900ms" in table, "table reports the 20s slice latency")
    check("10.0%" in table, "table reports noisy WER as a percentage")
    check("keep whisper-test" in table, "table surfaces the recommendation")
    failed = ModelResult(model="/tmp/bad", label="broken", ok=False, error="Boom")
    broken_payload = dict(payload, results=[failed.as_dict()])
    check("Boom" in format_table(broken_payload), "a failed model still renders, with its reason")

    print("\nrecommendation rule")
    def row(label, slice_ms, wer_noisy, ok=True):
        return ModelResult(model=label, label=label, ok=ok, slice_ms=slice_ms, wer_noisy=wer_noisy).as_dict()

    base = {"reference": "x", "snr_db": 10.0, "slice_lengths": [6.0], "ranking": []}
    incumbent_only = dict(base, results=[row("whisper-large-v3-turbo", 1600.0, 0.01)])
    check("keep" in recommend(incumbent_only), "keeps the incumbent when it is the only option")
    clearly_better = dict(base, results=[row("whisper-large-v3-turbo", 1600.0, 0.01), row("small", 500.0, 0.01)])
    check("switch to small" in recommend(clearly_better), "switches when a challenger is far faster and as accurate")
    no_speed_gain = dict(base, results=[row("whisper-large-v3-turbo", 1600.0, 0.01), row("distil", 1500.0, 0.01)])
    check("keep whisper-large-v3-turbo" in recommend(no_speed_gain), "keeps the incumbent when the speed gain is marginal")
    less_accurate = dict(base, results=[row("whisper-large-v3-turbo", 1600.0, 0.01), row("small", 500.0, 0.05)])
    check("keep whisper-large-v3-turbo" in recommend(less_accurate), "keeps the incumbent when the challenger loses accuracy")
    none_loaded = dict(base, results=[row("broken", 0.0, 0.0, ok=False)])
    check("nothing to recommend" in recommend(none_loaded), "says so when no model loaded")

    print("\nmodel size")
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "weights.safetensors").write_bytes(b"x" * 1_000_000)
        (root / "weights.npz").write_bytes(b"x" * 500_000)
        (root / "config.json").write_text("{}", encoding="utf-8")
        size = model_size_mb(root)
        check(1.4 < size < 1.6, f"sums only weight files ({size:.2f} MB)")

    print("")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("benchmark harness is sound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
