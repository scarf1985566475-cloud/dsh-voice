"""End-to-end check of the live WebSocket service.

Streams a synthesized sample into a real ``/live`` connection the way the GUI
panel does, then asserts the server produced committed captions, translations,
and a parseable event stream. This is the layer the browser talks to, so it is
the layer worth testing separately from the session object.

Usage::

    .venv-audio/bin/python tests/server_ws_test.py [--url ws://127.0.0.1:8768/live]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dsh_voice.audio_io import decode_to_mono  # noqa: E402
from dsh_voice.cli import _say_to_file, pick_say_voice  # noqa: E402
from dsh_voice.config import PROJECT_DIR, SAMPLE_RATE  # noqa: E402

SAMPLE_TEXT = (
    "Welcome to the platform review. The migration finished ahead of schedule. "
    "We are now monitoring error rates for the next twenty four hours."
)


async def run(url: str, audio: np.ndarray, pace: float) -> list[dict]:
    """Stream audio into the service and collect its events."""
    import websockets

    events: list[dict] = []
    async with websockets.connect(url, max_size=None) as socket:
        async def receive() -> None:
            async for message in socket:
                if isinstance(message, str):
                    event = json.loads(message)
                    events.append(event)
                    kind = event.get("type")
                    if kind == "final":
                        print(f"  COMMIT   #{event['id']:<3} [{event['start']:.1f}s] {event.get('text', '')[:76]}")
                    elif kind == "partial":
                        print(f"  preview  #{event['id']:<3} {event.get('text', '')[:76]}")
                    elif kind == "translation":
                        print(f"  zh       #{event['id']:<3} {event.get('text', '')[:76]}")
                    elif kind == "error":
                        print(f"  ERROR    {event.get('message')}")

        reader = asyncio.create_task(receive())
        # Announce the same toggle the panel sends, then stream real-time frames.
        await socket.send(json.dumps({"type": "config", "translate_partials": True}))
        frame = SAMPLE_RATE // 10
        for offset in range(0, audio.size, frame):
            await socket.send(audio[offset:offset + frame].astype("<f4").tobytes())
            await asyncio.sleep(pace)
        await socket.send(json.dumps({"type": "flush"}))
        # Wait for the ack, then for parity. The ack is only sent after the
        # service has committed and translated the tail, so checking parity
        # before it arrives would pass on a half-finished stream.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if any(event.get("type") == "flushed" for event in events):
                finals = sum(1 for event in events if event.get("type") == "final")
                translated = sum(
                    1 for event in events if event.get("type") == "translation" and event.get("final")
                )
                if finals > 0 and translated >= finals:
                    break
            await asyncio.sleep(0.25)
        reader.cancel()
    return events


def audit_meeting(http_url: str, meeting_id: str, finals: list[dict], cleanup: bool) -> list[str]:
    """Verify, over HTTP, the record the live session just produced.

    The panel and the agent both read meetings through these endpoints, so a
    record that exists on disk but is unreachable through the API is still
    broken. The record is deleted afterwards so running the test does not
    accumulate junk in the operator's own meeting list.

    Args:
        http_url: Service base URL.
        meeting_id: Meeting the session reported in its ``ready`` event.
        finals: Committed caption events from the run.
        cleanup: Delete the record once the checks have run.

    Returns:
        One message per violated invariant; empty when the record is served.
    """
    problems: list[str] = []
    with httpx.Client(base_url=http_url, timeout=20.0) as client:
        listed = client.get("/meetings", params={"limit": 50})
        if listed.status_code != 200:
            return [f"GET /meetings returned HTTP {listed.status_code}"]
        rows = listed.json().get("meetings", [])
        if not any(row.get("id") == meeting_id for row in rows):
            problems.append(f"meeting {meeting_id} is not in GET /meetings")
        else:
            row = next(row for row in rows if row.get("id") == meeting_id)
            print(f"  listed       {row['title']} · {row['segments']} 段 · {row['duration_s']:.1f}s")
            if row.get("segments") != len(finals):
                problems.append(f"listing reports {row.get('segments')} segments, events show {len(finals)}")
            if not row.get("ended_at"):
                problems.append("listing shows the meeting as never finalized")

        detail = client.get(f"/meetings/{meeting_id}")
        if detail.status_code != 200:
            problems.append(f"GET /meetings/{meeting_id} returned HTTP {detail.status_code}")
        else:
            segments = detail.json().get("meeting", {}).get("segments", [])
            print(f"  segments     {len(segments)}")
            if len(segments) != len(finals):
                problems.append(f"detail returns {len(segments)} segments, events show {len(finals)}")
            translated = sum(1 for segment in segments if segment.get("zh"))
            print(f"  translated   {translated}/{len(segments)}")
            if translated < len(finals):
                problems.append(f"only {translated} of {len(segments)} segments carry a translation")

        transcript = client.get(f"/meetings/{meeting_id}/transcript", params={"bilingual": "false"})
        if transcript.status_code != 200:
            problems.append(f"GET transcript returned HTTP {transcript.status_code}")
        elif finals and finals[0].get("text", "")[:20] not in transcript.json().get("text", ""):
            problems.append("the served transcript does not contain the first committed caption")

        if cleanup:
            removed = client.delete(f"/meetings/{meeting_id}")
            if removed.status_code != 200:
                problems.append(f"DELETE returned HTTP {removed.status_code}")
            else:
                after = client.get("/meetings", params={"limit": 50}).json().get("meetings", [])
                if any(row.get("id") == meeting_id for row in after):
                    problems.append("the deleted meeting is still listed")
                else:
                    print("  cleaned up   deleted the test record")
    return problems


def main() -> int:
    """Run the WebSocket test against a live service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8768/live")
    parser.add_argument("--audio")
    parser.add_argument("--pace", type=float, default=0.1)
    parser.add_argument("--keep-record", action="store_true",
                        help="keep the meeting record instead of deleting it after the checks")
    args = parser.parse_args()

    if args.audio:
        source = Path(args.audio).expanduser()
    else:
        source = PROJECT_DIR / "selftest" / "server.aiff"
        print(f"synthesizing sample with voice {pick_say_voice(None)}...")
        _say_to_file(SAMPLE_TEXT, source, pick_say_voice(None))

    audio = decode_to_mono(source, SAMPLE_RATE)
    print(f"streaming {audio.size / SAMPLE_RATE:.2f}s into {args.url}\n")
    events = asyncio.run(run(args.url, audio, args.pace))

    finals = [event for event in events if event.get("type") == "final"]
    translations = [event for event in events if event.get("type") == "translation" and event.get("final")]
    errors = [event for event in events if event.get("type") == "error"]
    ready = [event for event in events if event.get("type") == "ready"]
    flushed = [event for event in events if event.get("type") == "flushed"]

    print("\nsummary")
    print(f"  ready        {bool(ready)}")
    print(f"  flush ack    {bool(flushed)}")
    print(f"  commits      {len(finals)}")
    print(f"  translations {len(translations)}")
    print(f"  errors       {len(errors)}")
    print(f"  english      {' '.join(event.get('text', '') for event in finals)}")
    print(f"  chinese      {' '.join(event.get('text', '') for event in translations)}")

    problems: list[str] = []
    if not ready:
        problems.append("no 'ready' event: the service did not accept the session")
    if not finals:
        problems.append("no committed captions arrived over the socket")
    if not flushed:
        problems.append("the service never acknowledged the flush")
    if len(translations) < len(finals):
        problems.append(f"{len(finals) - len(translations)} committed spans have no translation")
    if errors:
        problems.append(f"{len(errors)} error events: {[event.get('message') for event in errors]}")

    print("\nmeeting record over HTTP")
    meeting_id = (ready[0].get("meeting") or {}).get("id") if ready else None
    if meeting_id is None:
        problems.append("the ready event carried no meeting: the session was not recorded")
    else:
        print(f"  id           {meeting_id}")
        problems.extend(audit_meeting(
            args.url.replace("ws://", "http://").replace("/live", ""),
            meeting_id,
            finals,
            cleanup=not args.keep_record,
        ))

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nWebSocket path and meeting record are clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
