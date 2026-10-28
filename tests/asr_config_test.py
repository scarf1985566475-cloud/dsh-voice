"""Regression checks for the decoding recipe and model selection.

Both are where a silent mistake is most expensive. The decode options decide
whether a recognizer gets the language right — Qwen3-ASR's language hint is
prompted, not passed, so a stale code or an English-only model turns Chinese
side-talk into a fluent English hallucination. The model choice decides whether
the weights are even there.

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

from dsh_voice.asr import _decode_options, _qwen_language  # noqa: E402
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

    # Qwen3-ASR has no temperature fallback ladder and no cross-window text
    # feedback; a repetition loop is a training-level failure, not a recipe one.
    check("temperature" not in file_options, "does not pass a Whisper-only temperature ladder")
    check("condition_on_previous_text" not in file_options,
          "does not feed its own output back across windows")
    check(file_options["language"] == "English", "an English code is prompted as English")
    check(live_options["language"] == "English", "live and file paths prompt the same language")

    # The chunker is the timestamp granularity of a long meeting. Too long and
    # a citation points at a ten-minute block; too short and words get cut.
    check(file_options["chunk_duration"] == 30.0, "splits long audio into 30 s timestamped chunks")
    check(file_options["min_chunk_duration"] < 1.0, "a short live window is not padded with silence")
    check(file_options["verbose"] is False, "stays quiet")

    check(_decode_options(None, streaming=True, initial_prompt=None)["language"] is None,
          "auto-detect passes None, not a guessed code")
    check(_decode_options("en", streaming=True, initial_prompt="Aristotle")["system_prompt"] == "Aristotle",
          "passes a vocabulary hint through the model's prompt channel")
    check("initial_prompt" not in file_options, "does not leak Whisper's keyword name")

    print("\nlanguage prompt mapping")
    for given, expected in (
        ("en", "English"), ("eng", "English"), ("zh", "Chinese"), ("cmn", "Chinese"),
        ("zh-cn", "Chinese"), ("yue", "Cantonese"), ("zh-hk", "Cantonese"),
        ("ja", "Japanese"), ("ko", "Korean"), ("fr", "French"),
    ):
        check(_qwen_language(given) == expected, f"{given!r} -> {expected!r}")
    check(_qwen_language("English") == "English", "a canonical name passes through")
    check(_qwen_language(None) is None and _qwen_language("") is None, "None and blank mean auto-detect")
    check(_qwen_language("klingon") is None, "an unknown language falls back to auto-detect, not a bad prompt")

    print("\nmodel catalog")
    check(len(MODEL_CATALOG) == 1, f"the retired Whisper variants are gone ({len(MODEL_CATALOG)} entry)")
    check(find_model("qwen3") is not None, "looks the model up by tag")
    check(find_model("QWEN3") is not None, "tag lookup is case-insensitive")
    check(find_model("nope") is None, "an unknown tag is not invented")
    check(find_model("turbo") is None and find_model("small") is None and find_model("distil") is None,
          "the old whisper tags are gone")
    choices = catalog_as_public_dict()
    check(all("tag" in item and "present" in item and "note" in item for item in choices),
          "catalog describes each model")
    check(all("english_only" not in item for item in choices), "the English-only notion is retired")

    print("\nselection")
    restore = with_env(DSH_VOICE_MODEL="", DSH_VOICE_LANGUAGE="")
    try:
        chosen = load_settings()
        check(chosen.model.endswith("qwen3-asr-0.6b-4bit"),
              f"the unset default is Qwen3-ASR ({Path(chosen.model).name})")
        check(chosen.model_repo.endswith("Qwen3-ASR-0.6B-4bit"), "the matching repo is recorded")
        check(chosen.language is None, "an unset language means auto-detect")

        os.environ["DSH_VOICE_MODEL"] = "qwen3"
        aliased = load_settings()
        check(aliased.model.endswith("qwen3-asr-0.6b-4bit"),
              f"a bare tag resolves to its directory ({Path(aliased.model).name})")
        check(aliased.model_repo.endswith("Qwen3-ASR-0.6B-4bit"), "the tag's repo follows the tag")

        os.environ["DSH_VOICE_MODEL"] = "/tmp/some-model"
        custom = load_settings()
        check(custom.model == "/tmp/some-model", "an explicit path is used verbatim")
        check(custom.model_repo == "/tmp/some-model", "a path doubles as its own repo id")

        os.environ["DSH_VOICE_MODEL"] = ""
        os.environ["DSH_VOICE_LANGUAGE"] = "zh"
        check(load_settings().language == "zh", "a language code is preserved for the prompt")
        os.environ["DSH_VOICE_LANGUAGE"] = "auto"
        check(load_settings().language is None, "'auto' means auto-detect")
        os.environ["DSH_VOICE_LANGUAGE"] = "none"
        check(load_settings().language is None, "'none' means auto-detect")
        os.environ["DSH_VOICE_LANGUAGE"] = "en"
        check(load_settings().language == "en", "an explicit English code stays explicit")
    finally:
        restore()

    print("\nlanguage guard")
    qwen3 = find_model("qwen3")
    assert qwen3 is not None
    # Qwen3-ASR is multilingual; the guard exists so a future catalog entry
    # with one language cannot silently receive another.
    check(model_language_warning(qwen3.directory, "zh") is None, "multilingual model accepts Chinese")
    check(model_language_warning(qwen3.directory, None) is None, "multilingual model accepts auto-detect")
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
