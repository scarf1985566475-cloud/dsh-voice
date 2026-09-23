#!/usr/bin/env bash
# Run every dsh-voice check, fastest first, stopping at the first failure.
#
#   bash tests/run_all.sh            # offline checks only (no model, no network)
#   bash tests/run_all.sh --full     # the whole pipeline (model + DeepSeek API)
#
# The live checks need the service on 127.0.0.1:8768; start it with
#   .venv-audio/bin/dsh-voice serve
#
# The full run creates and then removes one meeting record, so running it does
# not accumulate junk in the operator's own meeting list.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY="../.venv-audio/bin/python"
FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

fail() { echo; echo "FAILED: $1"; exit 1; }

echo "== 1/10 meeting store (records, crash tolerance, ordering) =="
"$PY" tests/meetings_test.py || fail "meeting store"

echo
echo "== 2/10 Obsidian export (note shape, filenames, vault layout) =="
"$PY" tests/obsidian_export_test.py || fail "obsidian export"

echo
echo "== 3/10 decoding recipe + model selection =="
"$PY" tests/asr_config_test.py || fail "asr configuration"

echo
echo "== 4/10 model benchmark harness (WER math, noise mixing, report) =="
"$PY" tests/benchmark_test.py || fail "benchmark harness"

echo
echo "== 5/10 command line (every subcommand, offline) =="
"$PY" tests/cli_test.py || fail "command line"

echo
echo "== 6/10 browser bundle (render, slot registration, minutes renderer) =="
node tests/plugin_render_test.mjs || fail "browser bundle"

echo
echo "== 7/10 host half (supervisor behaviour) =="
node tests/plugin_host_test.mjs || fail "host half"

echo
echo "== 8/10 live session survives navigation (route changes must not stop a recording) =="
node tests/plugin_session_test.mjs || fail "live session lifetime"

if [[ $FULL -eq 0 ]]; then
  echo
  echo "offline checks pass (rerun with --full for the model + API pipeline)"
  exit 0
fi

echo
echo "== 9/10 live session + WebSocket service + meeting API =="
"$PY" tests/live_stream_test.py || fail "live session"
"$PY" tests/server_ws_test.py || fail "websocket service"

echo
echo "== 10/10 recording -> transcript -> minutes =="
"$PY" tests/recording_pipeline_test.py || fail "recording pipeline"

echo
echo "all checks pass"
