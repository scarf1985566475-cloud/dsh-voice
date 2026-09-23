"""The local dsh-voice HTTP/WebSocket service.

Endpoints:

* ``GET  /`` — a self-contained microphone test page (mic -> captions ->
  translation) that exercises exactly the protocol the DSH panel uses, so the
  pipeline can be verified without the GUI.
* ``GET  /health`` — readiness of the model, ffmpeg, and the translation leg.
* ``GET  /devices`` — capture devices the bundled ffmpeg can address.
* ``POST /transcribe`` — multipart upload, or ``{"path": ...}`` for a local file.
* ``POST /translate`` — English text -> Chinese, for ad-hoc use.
* ``WS   /live`` — the live stream: binary float32 PCM in, JSON events out.

The service is strictly a loopback sidecar for the DSH web GUI.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import __version__
from .config import PROJECT_DIR, SAMPLE_RATE, InterpreterSettings, load_settings
from .live import LiveSession
from .meetings import MeetingError, MeetingStore, meeting_summary_row

LOGGER = logging.getLogger("dsh_voice.server")


class TranscribeRequest(BaseModel):
    """Request body for path-based transcription."""

    path: str
    language: str | None = None
    translate: bool = False
    summarize: bool = False


class TranslateRequest(BaseModel):
    """Request body for ad-hoc translation."""

    text: str
    context: list[str] | None = None


class SummaryRequest(BaseModel):
    """Request body for summarizing a stored meeting."""

    force: bool = False


class TitleRequest(BaseModel):
    """Request body for renaming a stored meeting."""

    title: str


class ServiceState:
    """Process-wide readiness shared by the endpoints."""

    def __init__(self) -> None:
        self.settings: InterpreterSettings = load_settings()
        self.warmup_state: str = "cold"
        self.warmup_error: str | None = None
        self.started_at = time.time()
        self.meetings = MeetingStore(self.settings.meetings_dir)


STATE = ServiceState()


def _warmup() -> None:
    """Load the Whisper weights once, off the event loop."""
    from .asr import ModelUnavailableError, warmup

    STATE.warmup_state = "loading"
    try:
        warmup()
    except ModelUnavailableError as error:
        STATE.warmup_state = "failed"
        STATE.warmup_error = str(error)
        LOGGER.error("model warmup failed: %s", error)
    except Exception as error:  # noqa: BLE001 - report any load failure to /health
        STATE.warmup_state = "failed"
        STATE.warmup_error = f"{type(error).__name__}: {error}"
        LOGGER.exception("model warmup failed")
    else:
        STATE.warmup_state = "ready"
        LOGGER.info("whisper model ready at %s", STATE.settings.model)


def create_app() -> FastAPI:
    """Build the FastAPI application (a factory so tests can bind their own state)."""

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Warm the weights in a worker thread: the server must answer /health
        # immediately, and the first live span should not pay the load cost.
        asyncio.get_running_loop().run_in_executor(None, _warmup)
        yield

    app = FastAPI(title="dsh-voice", version=__version__, lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        """Report service readiness."""
        from .asr import model_present
        from .audio_io import ffmpeg_version

        settings = STATE.settings
        return JSONResponse({
            "ok": STATE.warmup_state != "failed",
            "version": __version__,
            "uptime_s": round(time.time() - STATE.started_at, 1),
            "warmup": STATE.warmup_state,
            "warmup_error": STATE.warmup_error,
            "model_present": model_present(),
            "settings": settings.as_public_dict(),
            "ffmpeg": ffmpeg_version(),
            "sample_rate": SAMPLE_RATE,
        })

    @app.get("/devices")
    async def devices() -> JSONResponse:
        """List capture devices usable by the recording helper."""
        from .audio_io import list_input_devices

        return JSONResponse({"devices": list_input_devices()})

    @app.post("/transcribe")
    async def transcribe(payload: TranscribeRequest) -> JSONResponse:
        """Transcribe a local media file, optionally translating it to Chinese."""
        from .asr import ModelUnavailableError, transcribe_file

        loop = asyncio.get_running_loop()
        try:
            transcript = await loop.run_in_executor(
                None, lambda: transcribe_file(payload.path, language=payload.language),
            )
        except ModelUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except Exception as error:  # noqa: BLE001 - decode failures are client errors
            raise HTTPException(status_code=400, detail=f"{type(error).__name__}: {error}") from error

        body: dict[str, Any] = {"ok": True, "transcript": transcript.as_dict()}
        if payload.translate and transcript.text:
            from .translate import Translator

            translator = Translator(
                api_base=STATE.settings.api_base,
                api_model=STATE.settings.api_model,
                api_key=STATE.settings.api_key,
                target_language=STATE.settings.target_language,
            )
            try:
                result = await translator.translate_async(transcript.text)
                body["translation"] = result.as_dict()
            finally:
                await translator.aclose()
        return JSONResponse(body)

    @app.post("/transcribe-upload")
    async def transcribe_upload(file: UploadFile) -> JSONResponse:
        """Transcribe an uploaded audio payload through ffmpeg."""
        from .asr import ModelUnavailableError, transcribe_array
        from .audio_io import decode_bytes

        data = await file.read()
        loop = asyncio.get_running_loop()
        try:
            audio = await loop.run_in_executor(None, lambda: decode_bytes(data))
            transcript = await loop.run_in_executor(None, lambda: transcribe_array(audio))
        except ModelUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except Exception as error:  # noqa: BLE001 - decode failures are client errors
            raise HTTPException(status_code=400, detail=f"{type(error).__name__}: {error}") from error
        return JSONResponse({"ok": True, "filename": file.filename, "transcript": transcript.as_dict()})

    @app.post("/translate")
    async def translate(payload: TranslateRequest) -> JSONResponse:
        """Translate English text into Chinese."""
        from .translate import Translator

        translator = Translator(
            api_base=STATE.settings.api_base,
            api_model=STATE.settings.api_model,
            api_key=STATE.settings.api_key,
            target_language=STATE.settings.target_language,
        )
        try:
            result = await translator.translate_async(payload.text, payload.context)
        finally:
            await translator.aclose()
        return JSONResponse(result.as_dict())

    @app.post("/recordings/summarize")
    async def summarize_recording(payload: TranscribeRequest) -> JSONResponse:
        """Transcribe an existing recording, store it as a meeting, and summarize it.

        This is the offline counterpart of a live session: the same record shape
        and the same minutes, produced from a file instead of a microphone.
        """
        from .recordings import build_meeting_from_file

        try:
            return JSONResponse(await build_meeting_from_file(
                payload.path,
                settings=STATE.settings,
                store=STATE.meetings,
                language=payload.language,
                summarize=True,
            ))
        except MeetingError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.websocket("/live")
    async def live(websocket: WebSocket) -> None:
        """Stream audio in and caption/translation events out."""
        await websocket.accept()
        settings = STATE.settings

        async def send(event: dict[str, Any]) -> None:
            await websocket.send_json(event)

        recorder = None
        if settings.record:
            try:
                recorder = STATE.meetings.create(
                    source="live",
                    language=settings.language,
                    target_language=settings.target_language,
                    model=settings.model,
                )
            except OSError as error:
                # A read-only disk must not cost the user their live captions.
                LOGGER.warning("could not open a meeting record: %s", error)
                await send({"type": "error", "message": f"会议记录无法创建：{error}"})

        session = LiveSession(settings, send, recorder=recorder)
        await session.start()
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is not None:
                    await _handle_control(session, send, text)
                    continue
                payload = message.get("bytes")
                if not payload:
                    continue
                samples = np.frombuffer(payload, dtype="<f4")
                await session.process_audio(samples)
        except WebSocketDisconnect:
            pass
        except Exception as error:  # noqa: BLE001 - report then close cleanly
            LOGGER.exception("live session failed")
            with contextlib.suppress(Exception):
                await send({"type": "error", "message": f"{type(error).__name__}: {error}"})
        finally:
            await session.stop()
            if recorder is not None:
                LOGGER.info(
                    "meeting %s archived: %d segments, %.1fs audio",
                    recorder.meta.id, recorder.meta.segments, recorder.meta.duration_s,
                )

    # ── meeting records ──────────────────────────────────────────────────────

    @app.get("/meetings")
    async def list_meetings(limit: int = 50) -> JSONResponse:
        """List recorded meetings, newest first."""
        rows = [meeting_summary_row(meta) for meta in STATE.meetings.list(limit=max(1, min(limit, 500)))]
        return JSONResponse({
            "ok": True,
            "count": len(rows),
            "meetings": rows,
            "root": str(STATE.meetings.root),
        })

    @app.get("/meetings/{meeting_id}")
    async def read_meeting(meeting_id: str, transcript: bool = True) -> JSONResponse:
        """Read one meeting: metadata, transcript, and any saved summary."""
        try:
            meeting = STATE.meetings.read(meeting_id, include_segments=transcript)
        except MeetingError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return JSONResponse({"ok": True, "meeting": meeting.as_dict(include_segments=transcript)})

    @app.get("/meetings/{meeting_id}/transcript")
    async def meeting_transcript(meeting_id: str, bilingual: bool = True) -> JSONResponse:
        """Return one meeting's transcript as plain text."""
        try:
            meeting = STATE.meetings.read(meeting_id)
        except MeetingError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return JSONResponse({
            "ok": True,
            "meeting_id": meeting_id,
            "text": meeting.transcript_text(bilingual=bilingual),
            "markdown": meeting.transcript_markdown(bilingual=bilingual),
        })

    @app.post("/meetings/{meeting_id}/summary")
    async def meeting_summary(meeting_id: str, payload: SummaryRequest) -> JSONResponse:
        """Generate (or return the cached) minutes for a recorded meeting."""
        try:
            meeting = STATE.meetings.read(meeting_id)
        except MeetingError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        if meeting.summary and not payload.force:
            return JSONResponse({
                "ok": True,
                "cached": True,
                "summary": meeting.summary,
                "structured": meeting.summary_structured,
            })
        from .summarize import MeetingSummarizer

        summarizer = MeetingSummarizer(
            api_base=STATE.settings.api_base,
            api_model=STATE.settings.api_model,
            api_key=STATE.settings.api_key,
        )
        try:
            result = await summarizer.summarize_meeting_async(meeting)
        finally:
            await summarizer.aclose()
        if not result.ok:
            STATE.meetings.mark_summary_failed(meeting_id)
            raise HTTPException(status_code=502, detail=result.error or "summarization failed")
        STATE.meetings.save_summary(meeting_id, result.markdown, result.structured)
        return JSONResponse({
            "ok": True,
            "cached": False,
            "summary": result.markdown,
            "structured": result.structured,
            "chunks": result.chunks,
            "ms": result.ms,
        })

    @app.post("/meetings/{meeting_id}/title")
    async def rename_meeting(meeting_id: str, payload: TitleRequest) -> JSONResponse:
        """Rename a recorded meeting."""
        try:
            meta = STATE.meetings.set_title(meeting_id, payload.title)
        except MeetingError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return JSONResponse({"ok": True, "meta": meta.as_dict()})

    @app.delete("/meetings/{meeting_id}")
    async def delete_meeting(meeting_id: str) -> JSONResponse:
        """Delete a recorded meeting and its transcript."""
        try:
            STATE.meetings.delete(meeting_id)
        except MeetingError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return JSONResponse({"ok": True, "deleted": meeting_id})

    @app.get("/", response_class=HTMLResponse)
    async def test_page() -> HTMLResponse:
        """Serve the standalone microphone test page."""
        return HTMLResponse(TEST_PAGE)

    return app


async def _handle_control(
    session: LiveSession,
    send: Any,
    raw: str,
) -> None:
    """Apply one client control message."""
    try:
        message = json.loads(raw)
    except ValueError:
        await send({"type": "error", "message": "control message is not JSON"})
        return
    kind = message.get("type")
    if kind == "config":
        session.configure(translate_partials=message.get("translate_partials"))
        await send({"type": "configured", "settings": session.settings.as_public_dict()})
    elif kind == "title":
        title = str(message.get("title") or "")
        if session.recorder is None:
            await send({"type": "error", "message": "本次会话没有开启会议记录"})
            return
        session.recorder.set_title(title)
        await send({"type": "titled", "meeting": session.recorder.meta.as_dict()})
    elif kind == "flush":
        # Awaited on purpose: the ack is the client's signal that the last
        # sentence's translation has been sent, so it can close safely.
        await session.flush()
        await send({"type": "flushed"})
    elif kind == "ping":
        await send({"type": "pong", "t": message.get("t")})
    else:
        await send({"type": "error", "message": f"unknown control message {kind!r}"})


#: The standalone test page. Deliberately dependency-free and self-contained so
#: it can be opened straight from the service it exercises.
TEST_PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>dsh-voice 同声传译测试台</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 15px/1.6 -apple-system, "PingFang SC", system-ui, sans-serif;
         background: #16181d; color: #e8eaed; }
  header { padding: 16px 22px; border-bottom: 1px solid #2b2f38; display: flex;
           gap: 14px; align-items: center; flex-wrap: wrap; }
  h1 { font-size: 16px; margin: 0; font-weight: 600; }
  button { font: inherit; padding: 7px 16px; border-radius: 8px; border: 1px solid #3a3f4b;
           background: #23272f; color: #e8eaed; cursor: pointer; }
  button:hover { background: #2c313a; }
  button[data-on="1"] { background: #d9432f; border-color: #d9432f; color: #fff; }
  .pill { font-size: 12px; padding: 3px 9px; border-radius: 999px; background: #23272f; color: #9aa0aa; }
  .pill[data-state="ready"] { background: #1d3b2a; color: #6ee7a8; }
  .pill[data-state="failed"] { background: #3b1d1d; color: #f8a3a3; }
  main { padding: 18px 22px 60px; max-width: 900px; }
  .row { border-left: 3px solid #2b2f38; padding: 10px 0 10px 14px; margin-bottom: 12px; }
  .row[data-final="1"] { border-left-color: #4b8bf5; }
  .en { color: #c8cdd6; }
  .zh { font-size: 19px; margin-top: 4px; }
  .zh[data-pending="1"] { color: #8b93a1; }
  .meta { font-size: 12px; color: #6b7280; margin-top: 4px; }
  .err { color: #f8a3a3; font-size: 13px; white-space: pre-wrap; }
  #level { width: 120px; height: 8px; background: #23272f; border-radius: 4px; overflow: hidden; }
  #level > i { display: block; height: 100%; width: 0; background: #4b8bf5; }
  label { font-size: 13px; color: #9aa0aa; display: flex; gap: 6px; align-items: center; }
</style>
</head>
<body>
<header>
  <h1>dsh-voice 同声传译测试台</h1>
  <button id="toggle">开始收听</button>
  <span class="pill" id="status">连接中…</span>
  <div id="level"><i></i></div>
  <label><input type="checkbox" id="partial" checked> 预览也翻译</label>
  <button id="clear">清空</button>
</header>
<main id="log"></main>
<script>
const log = document.getElementById('log');
const statusEl = document.getElementById('status');
const toggle = document.getElementById('toggle');
const levelBar = document.querySelector('#level > i');
const partialBox = document.getElementById('partial');
let ws = null, ctx = null, node = null, stream = null, running = false;
const rows = new Map();

function ensureRow(id) {
  let row = rows.get(id);
  if (!row) {
    row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = '<div class="en"></div><div class="zh" data-pending="1"></div><div class="meta"></div>';
    log.appendChild(row);
    rows.set(id, row);
    log.scrollTop = log.scrollHeight;
    window.scrollTo(0, document.body.scrollHeight);
  }
  return row;
}
function render(ev) {
  const row = ensureRow(ev.id);
  const en = row.querySelector('.en'), zh = row.querySelector('.zh'), meta = row.querySelector('.meta');
  if (ev.type === 'partial') { en.textContent = ev.text; meta.textContent = '识别中 ' + (ev.asr_ms||0) + 'ms'; }
  if (ev.type === 'final') {
    en.textContent = ev.text; row.dataset.final = '1';
    meta.textContent = ev.start.toFixed(1) + 's · ASR ' + ev.asr_ms + 'ms';
    if (!ev.text) { zh.textContent = '（无语音）'; zh.dataset.pending = '0'; }
  }
  if (ev.type === 'translation') {
    zh.textContent = ev.text || '';
    zh.dataset.pending = ev.final ? '0' : '1';
    if (ev.ok === false) { zh.textContent = '（翻译不可用）'; meta.textContent += ' · 翻译失败: ' + (ev.error||''); }
    else if (ev.ms) meta.textContent += ' · 译 ' + ev.ms + 'ms';
  }
}
function setStatus(text, state) { statusEl.textContent = text; statusEl.dataset.state = state || ''; }

async function start() {
  stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true } });
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  const src = ctx.createMediaStreamSource(stream);
  node = ctx.createScriptProcessor(2048, 1, 1);
  const inRate = ctx.sampleRate, outRate = 16000;
  src.connect(node); node.connect(ctx.destination);
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(proto + '://' + location.host + '/live');
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'ready') setStatus('就绪 · ' + (ev.settings.model_ready ? '模型已就绪' : '模型未下载'), 'ready');
    else if (ev.type === 'error') { const d = document.createElement('div'); d.className = 'err'; d.textContent = ev.message; log.appendChild(d); }
    else render(ev);
  };
  ws.onclose = () => { if (running) setStatus('连接已断开', 'failed'); };
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let carry = new Float32Array(0);
  node.onaudioprocess = (e) => {
    if (!running) return;
    const input = e.inputBuffer.getChannelData(0);
    let peak = 0; for (let i = 0; i < input.length; i++) peak = Math.max(peak, Math.abs(input[i]));
    levelBar.style.width = Math.min(100, peak * 140) + '%';
    let buf;
    if (inRate === outRate) buf = input.slice();
    else {
      const ratio = inRate / outRate;
      const total = carry.length + input.length;
      const src = new Float32Array(total); src.set(carry, 0); src.set(input, carry.length);
      const outLen = Math.floor(total / ratio);
      buf = new Float32Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const pos = i * ratio, i0 = Math.floor(pos), frac = pos - i0;
        buf[i] = (src[i0] || 0) * (1 - frac) + (src[i0 + 1] || 0) * frac;
      }
      carry = src.slice(Math.floor(outLen * ratio));
    }
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(buf.buffer);
  };
}
function stop() {
  running = false;
  if (ws) { try { ws.send(JSON.stringify({ type: 'flush' })); } catch (_) {} ws.close(); ws = null; }
  if (node) { node.disconnect(); node.onaudioprocess = null; node = null; }
  if (ctx) { ctx.close(); ctx = null; }
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  levelBar.style.width = '0%';
  setStatus('已停止');
}
toggle.onclick = async () => {
  if (!running) {
    try { await start(); running = true; toggle.textContent = '停止'; toggle.dataset.on = '1'; setStatus('监听中…', 'ready'); }
    catch (err) { setStatus('启动失败: ' + err.message, 'failed'); stop(); }
  } else { stop(); toggle.textContent = '开始收听'; toggle.dataset.on = '0'; }
};
partialBox.onchange = () => { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'config', translate_partials: partialBox.checked })); };
document.getElementById('clear').onclick = () => { log.innerHTML = ''; rows.clear(); };
fetch('/health').then(r => r.json()).then(h => {
  if (h.warmup === 'failed') setStatus('模型加载失败', 'failed');
  else setStatus(h.warmup === 'ready' ? '服务就绪' : '模型加载中…', h.warmup === 'ready' ? 'ready' : '');
}).catch(() => setStatus('服务不可达', 'failed'));
</script>
</body>
</html>
"""


def main(reload: bool = False) -> None:
    """Run the service with uvicorn (the ``dsh-voice serve`` entry point).

    Args:
        reload: Watch the project's Python sources and restart the worker when
            they change. Development only: a restart drops any in-flight live
            session (the meeting is archived and finalized on the way down, so
            nothing is lost except the on-screen captions). The browser panel
            reconnects only when the user starts listening again.
    """
    import uvicorn

    settings = STATE.settings
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host, port = settings.host, settings.port
    if not reload:
        uvicorn.run(create_app(), host=host, port=port, log_level="info")
        return
    # The reloader needs an import string it can re-import in the worker; an app
    # object is already constructed and cannot be re-created per restart.
    uvicorn.run(
        "dsh_voice.server:create_app",
        factory=True,
        host=host,
        port=port,
        log_level="info",
        reload=True,
        reload_dirs=[str(PROJECT_DIR)],
    )


if __name__ == "__main__":
    main()
