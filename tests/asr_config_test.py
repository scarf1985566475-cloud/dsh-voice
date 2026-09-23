"""Regression checks for the decoding recipe and model selection.

These two settings are where a silent mistake is most expensive. The decode
options decide whether a recognizer can recover from a repetition loop, and the
model choice decides whether it can understand the audio at all. Both were
gotten wrong once; the numbers in the assertions below are the measurements that
caught it.

No model is loaded and no network is touched — this checks the configuration the
product will run with, not the models themselves.

Usage::

    .venv-audio/bin/python tests/asr_config_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsh_voice.asr import _decode_options  # noqa: E402
from dsh_voice.config import (  # noqa: E402
    MODEL_CATALOG,
    catalog_as_public_dict,
    find_model,
    load_settings,
    model_language_warning,
    select_model,
)

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record one assertion."""
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def with_env(**values: str):
    """Context-free helper: set environment variables and return a restore thunk."""
    previous = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        os.environ[key] = value
    def restore() -> None:
        for key, old in previous.items():
            if old is None:
                del os.environ[key]
            else:
                os.environ[key] = old
    return restore


def main() -> int:
    """Run every configuration check."""
    print("decode options")
    file_options = _decode_options("en", streaming=False, initial_prompt=None)
    live_options = _decode_options("en", streaming=True, initial_prompt=None)

    # Pinning temperature to 0.0 disables the fallback ladder that the
    # compression-ratio guard triggers, which is what let a repetition loop run:
    # measured 72.6% WER pinned against 0.9% with the ladder, same audio.
    ladder = file_options["temperature"]
    check(isinstance(ladder, tuple) and len(ladder) > 1,
          f"keeps a temperature fallback ladder ({ladder})")
    check(ladder[0] == 0.0, "starts the ladder at temperature 0")
    check(live_options["temperature"] == ladder, "live and file paths use the same ladder")

    # Cross-window prompting is what starts the runaway loop on hard content:
    # measured 4.7% WER with it on against 0.9% with it off.
    check(file_options["condition_on_previous_text"] is False, "file path does not feed back its own output")
    check(live_options["condition_on_previous_text"] is False, "live path does not feed back its own output")

    for key, expected in (
        ("compression_ratio_threshold", 2.4),
        ("logprob_threshold", -1.0),
        ("no_speech_threshold", 0.6),
    ):
        check(file_options.get(key) == expected, f"keeps {key} at {expected} so the guard can fire")

    # The ladder is only useful if the guards that trigger it are present; a
    # future edit disabling one silently re-arms the loop.
    check(file_options["compression_ratio_threshold"] is not None and
          file_options["logprob_threshold"] is not None,
          "the guards that escalate the ladder remain enabled")

    for label, options in (("file", file_options), ("live", live_options)):
        check("streaming" not in options, f"{label} options do not leak non-Whisper kwargs")
        check(options.get("verbose") is None, f"{label} options stay quiet")

    check(_decode_options(None, streaming=True, initial_prompt=None).get("language") is None,
          "auto-detect omits the language argument rather than passing None")
    check(_decode_options("en", streaming=True, initial_prompt="Aristotle").get("initial_prompt") == "Aristotle",
          "passes a vocabulary hint through")

    print("\nmodel catalog")
    check(len(MODEL_CATALOG) >= 3, f"catalog carries the variants ({len(MODEL_CATALOG)})")
    check(find_model("turbo") is not None, "looks a model up by tag")
    check(find_model("TURBO") is not None, "tag lookup is case-insensitive")
    check(find_model("nope") is None, "an unknown tag is not invented")
    check(find_model("turbo") is not None and not find_model("turbo").english_only,
          "the default model is multilingual")
    check(any(choice.english_only for choice in MODEL_CATALOG), "an English-only option is offered")
    described = catalog_as_public_dict()
    check(all("tag" in item and "present" in item for item in described), "catalog describes each model")

    print("\nselection")
    restore = with_env(DSH_VOICE_MODEL="", DSH_VOICE_LANGUAGE="en")
    try:
        chosen = load_settings()
        check(chosen.model.endswith("whisper-large-v3-turbo"),
              f"the unset default is the multilingual model ({Path(chosen.model).name})")
        check(chosen.model_repo.endswith("whisper-large-v3-turbo"), "the matching repo is recorded")

        os.environ["DSH_VOICE_MODEL"] = "small"
        small = load_settings()
        check(small.model.endswith("whisper-small-mlx"),
              f"a bare tag resolves to its directory ({Path(small.model).name})")
        check(small.model_repo.endswith("whisper-small-mlx"), "the tag's repo follows the tag")

        os.environ["DSH_VOICE_MODEL"] = "/tmp/some-model"
        custom = load_settings()
        check(custom.model == "/tmp/some-model", "an explicit path is used verbatim")
        check(custom.model_repo == "/tmp/some-model", "a path doubles as its own repo id")

        os.environ["DSH_VOICE_MODEL"] = ""
        os.environ["DSH_VOICE_LANGUAGE"] = "zh"
        check(load_settings().model.endswith("whisper-large-v3-turbo"), "Chinese audio gets the multilingual model")
        os.environ["DSH_VOICE_LANGUAGE"] = "auto"
        check(load_settings().language is None, "'auto' means auto-detect")
        check(load_settings().model.endswith("whisper-large-v3-turbo"),
              "auto-detect gets the multilingual model, never an English-only one")
        os.environ["DSH_VOICE_LANGUAGE"] = ""
        check(load_settings().language is None, "a blank language means auto-detect")
    finally:
        restore()

    print("\nlanguage guard")
    distil = find_model("distil")
    assert distil is not None
    check(model_language_warning(distil.directory, "en") is None, "English-only model + English audio is fine")
    warning = model_language_warning(distil.directory, "zh")
    check(warning is not None and "English only" in warning, "flags an English-only model on Chinese audio")
    check(model_language_warning(distil.directory, None) is not None, "flags it for auto-detect too")
    check(model_language_warning("/tmp/unknown", "zh") is None, "says nothing about a model it does not know")

    print("")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("decoding recipe and model selection are sound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
