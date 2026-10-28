"""Offline checks of the HTTP surface the interpreter panel talks to.

Two failures reported after a real recording motivated this file:

1. The panel's ``fetch`` helper always sends ``Content-Type: application/json``,
   even on GET, so the browser preflights ``OPTIONS /meetings`` first. A service
   without CORS answers 405 and the meeting list can never load.
2. Live sessions used to default to ``RECORD=false``, so a stopped recording
   never appeared in the meeting list at all.

No model, no network: the app is built with a throwaway meeting store.

Usage::

    .venv-audio/bin/python tests/server_http_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.pop("DSH_VOICE_RECORD", None)

from fastapi.testclient import TestClient  # noqa: E402

from dsh_voice import server  # noqa: E402
from dsh_voice.config import load_settings  # noqa: E402

PANEL_ORIGIN = "http://127.0.0.1:61594"

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    """Record one assertion."""
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def main() -> int:
    """Run the checks."""
    settings = load_settings(record=False, meetings_dir=tempfile.mkdtemp(prefix="dsh-voice-http-"))
    check(settings.record is False, "explicit record=False override is honoured")
    check(load_settings().record is True, "live sessions are recorded by default")

    server.STATE.settings = settings
    app = server.create_app()
    client = TestClient(app)

    preflight = client.options(
        "/meetings?limit=100",
        headers={
            "Origin": PANEL_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    check(preflight.status_code == 200, f"preflight OPTIONS /meetings is allowed ({preflight.status_code})")
    check(
        preflight.headers.get("access-control-allow-origin") == PANEL_ORIGIN,
        "preflight echoes the panel origin back",
    )
    allowed = preflight.headers.get("access-control-allow-methods", "")
    check("GET" in allowed, f"preflight lists GET as allowed ({allowed})")

    listed = client.get("/meetings?limit=5", headers={"Origin": PANEL_ORIGIN})
    check(listed.status_code == 200, f"GET /meetings succeeds ({listed.status_code})")
    check(
        listed.headers.get("access-control-allow-origin") == PANEL_ORIGIN,
        "GET /meetings carries CORS headers for the panel",
    )
    payload = listed.json()
    check("meetings" in payload, "GET /meetings returns a meetings array")

    # A non-browser client (curl, the MCP tools) never sends an Origin; make sure
    # adding CORS did not break that path.
    plain = client.get("/meetings?limit=5")
    check(plain.status_code == 200, "GET /meetings works without an Origin header")

    # The panel reloads python only through a restart; the WebSocket lives on a
    # different path, but a broken /health would make the panel show "unavailable".
    health = client.get("/health")
    check(health.status_code == 200, "/health still answers")

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print("\nserver http checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
