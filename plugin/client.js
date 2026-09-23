/**
 * Browser half of the voice-interpreter plugin: a "同传" conversation view.
 *
 * The panel captures the microphone, resamples it to Whisper's 16 kHz mono
 * float32 contract, and streams it to the local dsh-voice service over a
 * WebSocket. Events come back keyed by span id, so a partial line is replaced
 * in place by its final rather than appended as a second line.
 *
 * The bundle is hand-written plain JavaScript in the client module system's
 * lazy-CJS form (`window.__ModuleLoader__.load({ id, factory })`): it requires
 * only platform seed words (`react`), so it needs no build step and no
 * declared externals beyond the baseline.
 */
window.__ModuleLoader__.load({
  id: 'dsh-voice-interpreter',
  factory: (require) => {
    var module = { exports: {} }
    var exports = module.exports

    const React = require('react')

    /** Target sample rate: Whisper's fixed input rate. */
    const SAMPLE_RATE = 16000
    /** Mic callback size; 2048 frames is a stable compromise across engines. */
    const FRAME_SIZE = 2048
    /** Rows kept in the panel; a long meeting must not grow the DOM without bound. */
    const MAX_ROWS = 400
    /** How long "停止" waits for the service to flush and translate the last sentence. */
    const FLUSH_GRACE_MS = 8000

    /** Where the local service lives. Override with localStorage for a non-default port. */
    const SERVICE_URL = (() => {
      try {
        return localStorage.getItem('dsh-voice.service-url') || 'ws://127.0.0.1:8768/live'
      } catch {
        return 'ws://127.0.0.1:8768/live'
      }
    })()
    const HTTP_URL = SERVICE_URL.replace(/^ws/, 'http').replace(/\/live$/, '')

    /** Panel styles. Token-based with fallbacks so light and dark themes both read well. */
    const CSS = `
.dsh-voice { display: flex; flex-direction: column; height: 100%; min-height: 0;
  color: var(--dsw-alias-label-primary, #1f2328); font-family: inherit; }
.dsh-voice__modes { display: flex; gap: 6px; padding: 8px 16px 0; flex: 0 0 auto; }
.dsh-voice__mode { font: inherit; font-size: 12px; padding: 4px 12px; cursor: pointer; border-radius: 6px;
  border: 1px solid transparent; background: transparent; color: var(--dsw-alias-label-secondary, #57606a); }
.dsh-voice__mode:hover { background: var(--dsw-alias-interactive-bg-hover, #f3f4f6); }
.dsh-voice__mode[data-active="1"] { background: var(--dsw-alias-bg-layer-2, #f1f3f5);
  color: var(--dsw-alias-label-primary, #1f2328); border-color: var(--dsw-alias-border-l2, #e3e6ea); }
.dsh-voice__host { display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0; }
.dsh-voice__pane { display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0; }
.dsh-voice__split { display: flex; flex: 1 1 auto; min-height: 0; }
.dsh-voice__list { width: 248px; flex: 0 0 auto; overflow-y: auto;
  border-right: 1px solid var(--dsw-alias-border-l1, #eaecef); }
.dsh-voice__listhead { display: flex; align-items: center; justify-content: space-between; gap: 8px;
  padding: 8px 12px; font-size: 12px; color: var(--dsw-alias-label-tertiary, #6e7781);
  border-bottom: 1px solid var(--dsw-alias-border-l1, #eaecef); position: sticky; top: 0;
  background: var(--dsw-alias-bg-layer-1, #fff); }
.dsh-voice__item { padding: 9px 12px; cursor: pointer; border-bottom: 1px solid var(--dsw-alias-border-l1, #eaecef); }
.dsh-voice__item:hover { background: var(--dsw-alias-interactive-bg-hover, #f3f4f6); }
.dsh-voice__item[data-active="1"] { background: var(--dsw-alias-bg-layer-2, #f1f3f5); }
.dsh-voice__item-title { font-size: 13px; margin-bottom: 3px; font-weight: 500; }
.dsh-voice__item-meta { font-size: 11px; color: var(--dsw-alias-label-tertiary, #8b949e); display: flex; gap: 6px;
  align-items: center; flex-wrap: wrap; }
.dsh-voice__detail { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; }
.dsh-voice__detailbar { display: flex; gap: 8px; align-items: center; padding: 8px 14px; flex-wrap: wrap;
  border-bottom: 1px solid var(--dsw-alias-border-l1, #eaecef); }
.dsh-voice__detailbody { flex: 1 1 auto; overflow-y: auto; padding: 14px 18px 48px; }
.dsh-voice__seg { margin-bottom: 12px; }
.dsh-voice__stamp { font-size: 11px; color: var(--dsw-alias-label-tertiary, #8b949e); margin-right: 6px;
  font-family: var(--ds-font-family-code, ui-monospace, monospace); }
.dsh-voice__btn--tiny { font-size: 12px; padding: 3px 9px; }
.dsh-voice__tag { font-size: 11px; padding: 1px 7px; border-radius: 999px;
  background: rgba(23,140,80,.14); color: #14804a; }
.dsh-voice__tag[data-state="none"] { background: var(--dsw-alias-bg-layer-2, #f1f3f5);
  color: var(--dsw-alias-label-tertiary, #8b949e); }
.dsh-voice__tag[data-state="failed"] { background: rgba(207,34,46,.14); color: #cf222e; }
.dsh-voice__tag[data-state="file"] { background: rgba(9,105,218,.14); color: #0969da; }
.dsh-voice__notice { font-size: 13px; line-height: 1.9; color: var(--dsw-alias-label-secondary, #57606a);
  padding: 14px 16px; }
.dsh-voice__titlerow { display: flex; align-items: center; gap: 8px; padding: 7px 16px; flex-wrap: wrap;
  border-bottom: 1px solid var(--dsw-alias-border-l1, #eaecef); }
.dsh-voice__titlerow-label { font-size: 12px; color: var(--dsw-alias-label-tertiary, #6e7781); }
.dsh-voice__titlerow-hint { font-size: 12px; color: var(--dsw-alias-label-tertiary, #8b949e); }
.dsh-voice__input { font: inherit; font-size: 13px; padding: 4px 8px; border-radius: 6px; min-width: 220px;
  border: 1px solid var(--dsw-alias-border-l2, #d0d7de); background: var(--dsw-alias-bg-layer-1, #fff); color: inherit; }
.dsh-voice__minutes { font-size: 13.5px; line-height: 1.8; }
.dsh-voice__h { font-size: 14px; margin: 16px 0 6px; font-weight: 600; }
.dsh-voice__h:first-child { margin-top: 0; }
.dsh-voice__p { margin: 6px 0; }
.dsh-voice__ul { margin: 6px 0; padding-left: 20px; }
.dsh-voice__ul > li { margin: 3px 0; }
.dsh-voice__hr { border: none; border-top: 1px solid var(--dsw-alias-border-l1, #eaecef); margin: 16px 0 8px; }
.dsh-voice__table { border-collapse: collapse; margin: 8px 0; font-size: 13px; }
.dsh-voice__table th, .dsh-voice__table td { text-align: left; padding: 5px 12px 5px 0;
  border-bottom: 1px solid var(--dsw-alias-border-l1, #eaecef); vertical-align: top; }
.dsh-voice__table th { color: var(--dsw-alias-label-secondary, #57606a); font-weight: 600; }
.dsh-voice__bar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 10px 16px; border-bottom: 1px solid var(--dsw-alias-border-l2, #e3e6ea); }
.dsh-voice__title { font-size: 14px; font-weight: 600; margin-right: 2px; }
.dsh-voice__spacer { flex: 1 1 auto; }
.dsh-voice__status { font-size: 12px; padding: 3px 9px; border-radius: 999px;
  background: var(--dsw-alias-bg-layer-2, #f1f3f5); color: var(--dsw-alias-label-secondary, #57606a); }
.dsh-voice__status[data-state="live"] { background: rgba(217,67,47,.14); color: #c0331f; }
.dsh-voice__status[data-state="ready"] { background: rgba(23,140,80,.14); color: #14804a; }
.dsh-voice__status[data-state="error"] { background: rgba(207,34,46,.14); color: #cf222e; }
.dsh-voice__btn { font: inherit; font-size: 13px; padding: 5px 12px; cursor: pointer;
  border-radius: 7px; border: 1px solid var(--dsw-alias-border-l2, #d0d7de);
  background: var(--dsw-alias-bg-layer-1, #fff); color: inherit; }
.dsh-voice__btn:hover { background: var(--dsw-alias-interactive-bg-hover, #f3f4f6); }
.dsh-voice__btn[data-primary="1"] { background: var(--dsw-alias-button-primary-fill, #0969da);
  border-color: transparent; color: #fff; }
.dsh-voice__btn[data-primary="1"][data-live="1"] { background: #d9432f; }
.dsh-voice__btn:disabled { opacity: .5; cursor: not-allowed; }
.dsh-voice__check { display: flex; align-items: center; gap: 5px; font-size: 12px;
  color: var(--dsw-alias-label-secondary, #57606a); cursor: pointer; user-select: none; }
.dsh-voice__meter { width: 84px; height: 6px; border-radius: 3px; overflow: hidden;
  background: var(--dsw-alias-bg-layer-2, #eaecef); }
.dsh-voice__meter > i { display: block; height: 100%; width: 0; background: #0969da; transition: width .08s linear; }
.dsh-voice__meta { font-size: 12px; color: var(--dsw-alias-label-tertiary, #6e7781);
  padding: 6px 16px; border-bottom: 1px solid var(--dsw-alias-border-l1, #eaecef);
  display: flex; gap: 14px; flex-wrap: wrap; }
.dsh-voice__log { flex: 1 1 auto; min-height: 0; overflow-y: auto; padding: 14px 16px 32px; }
.dsh-voice__empty { color: var(--dsw-alias-label-tertiary, #6e7781); font-size: 13px; line-height: 1.9; }
.dsh-voice__empty code { font-family: var(--ds-font-family-code, ui-monospace, monospace);
  background: var(--dsw-alias-bg-layer-2, #f1f3f5); padding: 1px 5px; border-radius: 4px; font-size: 12px; }
.dsh-voice__row { border-left: 3px solid var(--dsw-alias-border-l2, #e3e6ea);
  padding: 8px 0 8px 12px; margin-bottom: 10px; }
.dsh-voice__row[data-final="1"] { border-left-color: #4b8bf5; }
.dsh-voice__en { font-size: 13px; line-height: 1.55; color: var(--dsw-alias-label-secondary, #57606a); }
.dsh-voice__row[data-final="1"] .dsh-voice__en { color: var(--dsw-alias-label-primary, #1f2328); }
.dsh-voice__zh { font-size: 17px; line-height: 1.5; margin-top: 3px; }
.dsh-voice__zh[data-pending="1"] { color: var(--dsw-alias-label-tertiary, #6e7781); }
.dsh-voice__rowmeta { font-size: 11px; color: var(--dsw-alias-label-tertiary, #8b949e); margin-top: 4px; }
.dsh-voice__rowmeta[data-error="1"] { color: #cf222e; }
.dsh-voice__banner { margin: 16px; padding: 12px 14px; border-radius: 8px; font-size: 13px; line-height: 1.7;
  background: rgba(207,34,46,.08); border: 1px solid rgba(207,34,46,.3); color: #a40e26; }
.dsh-voice__banner code { font-family: var(--ds-font-family-code, ui-monospace, monospace);
  background: rgba(0,0,0,.06); padding: 1px 5px; border-radius: 4px; }
`

    const styleEl = document.createElement('style')
    styleEl.textContent = CSS
    styleEl.setAttribute('data-plugin-css', 'dsh-voice-interpreter')
    document.head.appendChild(styleEl)

    /**
     * Stream microphone audio to the local service and dispatch its events.
     *
     * Owns the AudioContext, the capture graph, and the socket; every browser
     * object it creates is released by {@link LiveClient#stop}.
     */
    class LiveClient {
      /**
       * @param {object} options - Wiring for one client instance.
       * @param {(event: object) => void} options.onEvent - Server event sink.
       * @param {(state: string, detail?: string) => void} options.onState - Connection state sink.
       * @param {(level: number) => void} options.onLevel - Input level sink, 0..1.
       */
      constructor({ onEvent, onState, onLevel }) {
        this.onEvent = onEvent
        this.onState = onState
        this.onLevel = onLevel
        this.socket = null
        this.audioContext = null
        this.processor = null
        this.source = null
        this.stream = null
        this.silentGain = null
        this.carry = new Float32Array(0)
        this.active = false
      }

      /** Open the microphone and the socket, then start streaming. */
      async start() {
        if (this.active) return
        this.onState('connecting')
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true },
        })
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)()
        if (this.audioContext.state === 'suspended') await this.audioContext.resume()
        this.source = this.audioContext.createMediaStreamSource(this.stream)
        this.processor = this.audioContext.createScriptProcessor(FRAME_SIZE, 1, 1)
        // A zero-gain leg to the destination keeps the graph pulled in browsers
        // that suspend a disconnected processor, without echoing the mic back.
        this.silentGain = this.audioContext.createGain()
        this.silentGain.gain.value = 0
        this.source.connect(this.processor)
        this.processor.connect(this.silentGain)
        this.silentGain.connect(this.audioContext.destination)

        const socket = new WebSocket(SERVICE_URL)
        socket.binaryType = 'arraybuffer'
        this.socket = socket
        socket.onmessage = (message) => {
          let event
          try {
            event = JSON.parse(message.data)
          } catch {
            return
          }
          if (event.type === 'ready') this.onState('live', { settings: event.settings, meeting: event.meeting })
          else if (event.type === 'error') this.onEvent(event)
          else this.onEvent(event)
        }
        socket.onerror = () => this.onState('error', '连接本地服务失败')
        socket.onclose = () => {
          if (this.active) this.onState('error', '与本地服务的连接已断开')
          this.active = false
        }
        await new Promise((resolve, reject) => {
          socket.onopen = resolve
          const previous = socket.onerror
          socket.onerror = (error) => {
            previous?.(error)
            reject(new Error('无法连接本地服务'))
          }
        })

        const inRate = this.audioContext.sampleRate
        this.processor.onaudioprocess = (event) => {
          if (!this.active) return
          const input = event.inputBuffer.getChannelData(0)
          let peak = 0
          for (let index = 0; index < input.length; index += 1) {
            const value = input[index] < 0 ? -input[index] : input[index]
            if (value > peak) peak = value
          }
          this.onLevel(peak)
          const block = inRate === SAMPLE_RATE ? input.slice() : this.resample(input, inRate)
          if (this.socket && this.socket.readyState === WebSocket.OPEN) this.socket.send(block.buffer)
        }
        this.active = true
      }

      /**
       * Linearly resample one block to 16 kHz, carrying the remainder between calls.
       * @param {Float32Array} input - Block at the device rate.
       * @param {number} inRate - Device sample rate.
       * @returns {Float32Array} Block at 16 kHz.
       */
      resample(input, inRate) {
        const ratio = inRate / SAMPLE_RATE
        const merged = new Float32Array(this.carry.length + input.length)
        merged.set(this.carry, 0)
        merged.set(input, this.carry.length)
        const outLength = Math.floor(merged.length / ratio)
        const out = new Float32Array(outLength)
        for (let index = 0; index < outLength; index += 1) {
          const position = index * ratio
          const lower = Math.floor(position)
          const fraction = position - lower
          out[index] = (merged[lower] || 0) * (1 - fraction) + (merged[lower + 1] || 0) * fraction
        }
        this.carry = merged.slice(Math.floor(outLength * ratio))
        return out
      }

      /**
       * Apply a runtime toggle on the service.
       * @param {boolean} translatePartials - Whether previews are translated too.
       */
      configure(translatePartials) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
          this.socket.send(JSON.stringify({ type: 'config', translate_partials: translatePartials }))
        }
      }

      /**
       * Name the meeting being recorded.
       * @param {string} title - New title.
       */
      rename(title) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
          this.socket.send(JSON.stringify({ type: 'title', title }))
        }
      }

      /**
       * Stop capture and close the socket, releasing every browser resource.
       *
       * The service is asked to flush first and given a moment to acknowledge:
       * the ack means the last sentence's translation has already been sent, so
       * closing immediately after it cannot swallow the caption the user is
       * waiting for.
       *
       * @param {object} [options] - Teardown options.
       * @param {boolean} [options.graceful] - Wait for the flush ack. Pass
       *   false when the plugin is being unloaded, where holding the microphone
       *   for up to the grace period would delay releasing it for no benefit.
       * @returns {Promise<void>} Resolves once the socket is closed.
       */
      async stop({ graceful = true } = {}) {
        this.active = false
        if (this.processor) {
          this.processor.onaudioprocess = null
        }
        const socket = this.socket
        if (graceful && socket && socket.readyState === WebSocket.OPEN) {
          const settled = new Promise((resolve) => {
            const timer = setTimeout(resolve, FLUSH_GRACE_MS)
            socket.addEventListener('message', (message) => {
              try {
                if (JSON.parse(message.data).type === 'flushed') {
                  clearTimeout(timer)
                  resolve()
                }
              } catch {
                /* not a control payload; ignore */
              }
            })
            socket.addEventListener('close', () => { clearTimeout(timer); resolve() })
          })
          try {
            socket.send(JSON.stringify({ type: 'flush' }))
          } catch {
            /* the socket went away first; nothing to flush into */
          }
          await settled
        }
        this.socket = null
        if (socket) {
          try {
            socket.close()
          } catch {
            /* closing twice is harmless */
          }
        }
        if (this.processor) {
          try {
            this.processor.disconnect()
          } catch {
            /* already detached */
          }
        }
        this.processor = null
        if (this.source) {
          try {
            this.source.disconnect()
          } catch {
            /* already detached */
          }
        }
        this.source = null
        if (this.silentGain) {
          try {
            this.silentGain.disconnect()
          } catch {
            /* already detached */
          }
        }
        this.silentGain = null
        if (this.audioContext) {
          this.audioContext.close().catch(() => {})
        }
        this.audioContext = null
        if (this.stream) {
          for (const track of this.stream.getTracks()) track.stop()
        }
        this.stream = null
        this.carry = new Float32Array(0)
        this.onLevel(0)
      }
    }

    const STATE_LABEL = {
      idle: '未开始',
      connecting: '连接中…',
      live: '监听中',
      error: '连接失败',
    }

    /**
     * The live tab: transport controls, mic meter, caption log, and the meeting
     * being recorded.
     * @param {object} props - Wiring for the tab.
     * @param {(meeting: object|null) => void} props.onArchived - Called when a
     *   recording stops, with the archived meeting (or null when recording was
     *   disabled), so the shell can refresh the records list.
     * @returns {object} The rendered element tree.
     */
    // ── the live session, scoped to the app rather than to the panel ─────────
    /**
     * A recording must survive navigating away from the panel.
     *
     * The session used to live in `LivePanel`'s own state, which tied it to the
     * component's mount: switching view tabs or conversations unmounted the
     * panel, its cleanup ran, and the microphone stopped mid-meeting — the one
     * thing a meeting recorder must not do.
     *
     * The session now lives here, at module scope, owned by the *plugin's*
     * lifetime. Mounting the panel attaches a view to a running session and
     * unmounting only detaches it. Exactly two things stop a session: the
     * 停止 button, and the plugin being unloaded (hot reload or disable), which
     * has to release the microphone.
     */
    const voiceSession = {
      client: null,
      rows: [],
      state: 'idle',
      detail: '',
      level: 0,
      stats: null,
      settings: null,
      notice: null,
      meeting: null,
      translatePartials: true,
      listeners: new Set(),
      snapshot: null,
    }

    /** Build the immutable value subscribers read. */
    function sessionSnapshot() {
      return {
        rows: voiceSession.rows,
        state: voiceSession.state,
        detail: voiceSession.detail,
        level: voiceSession.level,
        stats: voiceSession.stats,
        settings: voiceSession.settings,
        notice: voiceSession.notice,
        meeting: voiceSession.meeting,
        translatePartials: voiceSession.translatePartials,
      }
    }

    /**
     * Publish a change to every attached view.
     *
     * The snapshot is rebuilt here, not in the getter: `useSyncExternalStore`
     * requires a stable reference between changes, and returning a fresh object
     * per read re-renders forever.
     */
    function sessionPublish() {
      voiceSession.snapshot = sessionSnapshot()
      for (const listener of voiceSession.listeners) listener()
    }

    /** Attach a view. Detaching must never touch the session itself. */
    function sessionSubscribe(listener) {
      voiceSession.listeners.add(listener)
      return () => { voiceSession.listeners.delete(listener) }
    }

    /** Read the current snapshot (same object until something changes). */
    function sessionGetSnapshot() {
      return voiceSession.snapshot
    }

    /** Merge one server event into the row it belongs to. */
    function sessionApplyEvent(event) {
      if (event.type === 'error') {
        voiceSession.notice = event.message
        sessionPublish()
        return
      }
      if (event.stats) voiceSession.stats = event.stats
      if (event.id === undefined) {
        sessionPublish()
        return
      }
      const next = voiceSession.rows.slice()
      let index = -1
      for (let cursor = next.length - 1; cursor >= 0; cursor -= 1) {
        if (next[cursor].id === event.id) { index = cursor; break }
      }
      const row = index === -1
        ? { id: event.id, en: '', zh: '', enFinal: false, zhFinal: false, start: 0, asr: 0, tr: 0, error: null }
        : next[index]
      const merged = { ...row }
      if (event.type === 'partial') {
        merged.en = event.text
        merged.enFinal = false
        merged.start = event.start
        merged.asr = event.asr_ms
      } else if (event.type === 'final') {
        merged.en = event.text
        merged.enFinal = true
        merged.start = event.start
        merged.asr = event.asr_ms
        if (!event.text) { merged.zh = '（无语音）'; merged.zhFinal = true }
      } else if (event.type === 'translation') {
        merged.zh = event.text
        merged.zhFinal = Boolean(event.final)
        merged.tr = event.ms
        merged.error = event.ok === false ? (event.error || '翻译失败') : null
      } else {
        return // an event this panel does not render is not a state change
      }
      if (index === -1) next.push(merged)
      else next[index] = merged
      voiceSession.rows = next.length > MAX_ROWS ? next.slice(next.length - MAX_ROWS) : next
      sessionPublish()
    }

    /** Drop the rendered captions. The meeting on disk is unaffected. */
    function sessionClearRows() {
      voiceSession.rows = []
      sessionPublish()
    }

    /** Start listening, if not already running. */
    async function sessionStart() {
      if (voiceSession.client) return
      voiceSession.notice = null
      sessionPublish()
      const client = new LiveClient({
        onEvent: sessionApplyEvent,
        onState: (next, payload) => {
          voiceSession.state = next
          if (next === 'live' && payload) {
            voiceSession.settings = payload.settings || null
            if (payload.meeting) voiceSession.meeting = payload.meeting
          }
          if (next === 'error' && typeof payload === 'string') voiceSession.detail = payload
          sessionPublish()
        },
        onLevel: (value) => {
          // Throttle: a level meter does not need 40 notifications a second.
          const rounded = Math.round(value * 50) / 50
          if (rounded === voiceSession.level) return
          voiceSession.level = rounded
          sessionPublish()
        },
      })
      voiceSession.client = client
      try {
        await client.start()
        client.configure(voiceSession.translatePartials)
        voiceSession.state = 'live'
        sessionPublish()
      } catch (error) {
        client.stop()
        voiceSession.client = null
        voiceSession.state = 'error'
        voiceSession.detail = error && error.message ? error.message : String(error)
        voiceSession.notice = '无法连接本地转写服务。'
        sessionPublish()
      }
    }

    /**
     * Stop listening and report the archived meeting.
     * @returns {Promise<object|null>} The meeting that was finalized, if any.
     */
    async function sessionStop() {
      const client = voiceSession.client
      voiceSession.client = null
      voiceSession.state = 'idle'
      voiceSession.level = 0
      sessionPublish()
      if (!client) return voiceSession.meeting
      // Awaited: the flushes and translations in flight belong to the last thing
      // the speaker said, and the record is finalized once they have landed.
      await client.stop()
      voiceSession.notice = null
      sessionPublish()
      return voiceSession.meeting
    }

    /**
     * Release the session because the plugin is going away.
     *
     * Not graceful on purpose: a hot reload or a disable must hand the
     * microphone back immediately rather than hold it for the flush handshake.
     * The server archives the meeting when the socket drops.
     */
    function sessionDispose() {
      const client = voiceSession.client
      voiceSession.client = null
      voiceSession.state = 'idle'
      voiceSession.level = 0
      sessionPublish()
      if (client) client.stop({ graceful: false })
    }

    voiceSession.snapshot = sessionSnapshot()

    /**
     * The live tab: transport controls, mic meter, caption log, and the meeting
     * being recorded. A pure view over {@link voiceSession}, so it can be
     * unmounted and remounted without touching the recording.
     * @param {object} props - Wiring for the tab.
     * @param {(meeting: object|null) => void} props.onArchived - Called when a
     *   recording stops, with the archived meeting (or null when recording was
     *   disabled), so the shell can refresh the records list.
     * @returns {object} The rendered element tree.
     */
    function LivePanel({ onArchived }) {
      const session = React.useSyncExternalStore(
        sessionSubscribe, sessionGetSnapshot, sessionGetSnapshot,
      )
      const { rows, state, detail, level, settings, stats, notice, meeting, translatePartials } = session
      const [titleDraft, setTitleDraft] = React.useState(meeting ? meeting.title || '' : '')

      const logRef = React.useRef(null)
      const stickRef = React.useRef(true)

      // Follow the store's meeting: a rename, or a session started while this
      // view was detached, must not leave a stale name in the box.
      const meetingId = meeting ? meeting.id : null
      React.useEffect(() => {
        setTitleDraft(meeting ? meeting.title || '' : '')
      }, [meetingId]) // eslint-disable-line react-hooks/exhaustive-deps -- draft tracks the meeting, not its title

      React.useEffect(() => {
        if (!stickRef.current || !logRef.current) return
        const element = logRef.current
        element.scrollTop = element.scrollHeight
      }, [rows])

      const onScroll = React.useCallback(() => {
        const element = logRef.current
        if (!element) return
        stickRef.current = element.scrollHeight - element.scrollTop - element.clientHeight < 80
      }, [])

      const toggle = React.useCallback(async () => {
        if (voiceSession.client) {
          const archived = await sessionStop()
          onArchived(archived)
          return
        }
        await sessionStart()
      }, [onArchived])

      const onPartialToggle = React.useCallback((event) => {
        const next = event.target.checked
        voiceSession.translatePartials = next
        sessionPublish()
        if (voiceSession.client) voiceSession.client.configure(next)
      }, [])

      const rename = React.useCallback(() => {
        const title = titleDraft.trim()
        if (!title || !voiceSession.client) return
        voiceSession.client.rename(title)
        voiceSession.meeting = voiceSession.meeting ? { ...voiceSession.meeting, title } : voiceSession.meeting
        sessionPublish()
      }, [titleDraft])

      const copyAll = React.useCallback(() => {
        const text = rows
          .map((row) => [row.en, row.zh].filter(Boolean).join('\n'))
          .filter(Boolean)
          .join('\n\n')
        if (!text) return
        navigator.clipboard?.writeText(text).catch(() => {})
      }, [rows])

      const live = state === 'live' || state === 'connecting'
      const header = React.createElement('div', { className: 'dsh-voice__bar' },
        React.createElement('span', { className: 'dsh-voice__title' }, '同声传译'),
        React.createElement('span', {
          className: 'dsh-voice__status',
          'data-state': state === 'live' ? 'live' : state === 'error' ? 'error' : state === 'idle' ? '' : 'ready',
        }, `${STATE_LABEL[state]}${state === 'error' && detail ? ` · ${detail}` : ''}`),
        React.createElement('div', { className: 'dsh-voice__meter' },
          React.createElement('i', { style: { width: `${Math.min(100, level * 140)}%` } })),
        React.createElement('div', { className: 'dsh-voice__spacer' }),
        React.createElement('label', { className: 'dsh-voice__check' },
          React.createElement('input', { type: 'checkbox', checked: translatePartials, onChange: onPartialToggle }),
          '预览也翻译'),
        React.createElement('button', { className: 'dsh-voice__btn', onClick: copyAll, disabled: rows.length === 0 }, '复制'),
        React.createElement('button', { className: 'dsh-voice__btn', onClick: sessionClearRows, disabled: rows.length === 0 }, '清空'),
        React.createElement('button', {
          className: 'dsh-voice__btn',
          'data-primary': '1',
          'data-live': live ? '1' : '0',
          onClick: toggle,
        }, live ? '停止' : '开始收听'),
      )

      const meta = React.createElement('div', { className: 'dsh-voice__meta' },
        React.createElement('span', null, `服务 ${HTTP_URL}`),
        React.createElement('span', null, settings
          ? (settings.model_ready ? '模型已就绪' : '模型未下载')
          : '未连接'),
        React.createElement('span', null, settings && settings.translation_available ? `翻译 ${settings.api_model}` : '翻译不可用'),
        stats ? React.createElement('span', null, `已识别 ${stats.segments} 段`) : null,
        stats ? React.createElement('span', null, `识别 ${stats.last_asr_ms}ms`) : null,
        stats ? React.createElement('span', null, `翻译 ${stats.last_translate_ms}ms`) : null,
        settings ? React.createElement('span', null, settings.record ? '正在记录会议' : '未开启记录') : null,
        meeting ? React.createElement('span', null, `记录 #${meeting.id}`) : null,
      )

      const titleRow = meeting
        ? React.createElement('div', { className: 'dsh-voice__titlerow' },
            React.createElement('span', { className: 'dsh-voice__titlerow-label' }, '会议名称'),
            React.createElement('input', {
              className: 'dsh-voice__input',
              value: titleDraft,
              placeholder: '给这次会议起个名字',
              onChange: (event) => setTitleDraft(event.target.value),
              onKeyDown: (event) => { if (event.key === 'Enter') rename() },
            }),
            React.createElement('button', {
              className: 'dsh-voice__btn',
              onClick: rename,
              disabled: !titleDraft.trim() || titleDraft === meeting.title,
            }, '保存名称'),
            React.createElement('span', { className: 'dsh-voice__titlerow-hint' },
              live ? '停止后会自动存档，可在「会议记录」里生成纪要' : '已存档，可在「会议记录」里生成纪要'),
          )
        : null

      const body = rows.length === 0
        ? React.createElement('div', { className: 'dsh-voice__empty' },
            React.createElement('div', null, '点击「开始收听」，对着麦克风说英语，这里会实时显示英文字幕与中文译文。'),
            React.createElement('div', null, '音频与识别全部在本机完成；只有英文句子会发送给 DeepSeek 做翻译。'),
            settings && settings.model_ready === false
              ? React.createElement('div', null, '模型尚未下载：运行 ',
                  React.createElement('code', null, 'dsh-voice download-model'), '（约 1.6GB）。')
              : null,
          )
        : rows.map((row) => React.createElement('div', {
            key: row.id,
            className: 'dsh-voice__row',
            'data-final': row.enFinal ? '1' : '0',
          },
          React.createElement('div', { className: 'dsh-voice__en' }, row.en || '…'),
          React.createElement('div', {
            className: 'dsh-voice__zh',
            'data-pending': row.zhFinal ? '0' : '1',
          }, row.zh || (row.enFinal ? '' : '')),
          React.createElement('div', {
            className: 'dsh-voice__rowmeta',
            'data-error': row.error ? '1' : '0',
          }, [
            row.enFinal ? `${Number(row.start).toFixed(1)}s` : '识别中',
            row.asr ? `ASR ${row.asr}ms` : null,
            row.tr ? `译 ${row.tr}ms` : null,
            row.error,
          ].filter(Boolean).join(' · ')),
        ))

      const banner = notice
        ? React.createElement('div', { className: 'dsh-voice__banner' },
            React.createElement('div', null, notice),
            React.createElement('div', null, '启动本地服务：',
              React.createElement('code', null, 'dsh-voice serve'),
              '（或在对话中让我启动它）。'),
            detail ? React.createElement('div', null, detail) : null,
          )
        : null

      return React.createElement('div', { className: 'dsh-voice' },
        header,
        meta,
        titleRow,
        banner,
        React.createElement('div', { className: 'dsh-voice__log', ref: logRef, onScroll }, body),
      )
    }

    // ── meeting records ──────────────────────────────────────────────────────

    /**
     * Fetch JSON from the local service, surfacing backend errors as thrown ones.
     * @param {string} path - Path below the service root.
     * @param {object} [options] - Fetch options.
     * @returns {Promise<object>} The parsed payload.
     */
    async function fetchJson(path, options) {
      const response = await fetch(`${HTTP_URL}${path}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
      })
      let payload = null
      try {
        payload = await response.json()
      } catch {
        payload = null
      }
      if (!response.ok) {
        const detail = payload && (payload.detail || payload.error)
        throw new Error(detail || `HTTP ${response.status}`)
      }
      return payload || {}
    }

    /**
     * Format a duration in seconds the way a listing reads best.
     * @param {number} seconds - Duration in seconds.
     * @returns {string} A short human duration.
     */
    function formatDuration(seconds) {
      const total = Math.max(0, Math.round(Number(seconds) || 0))
      if (total >= 3600) return `${Math.floor(total / 3600)}小时${String(Math.floor((total % 3600) / 60)).padStart(2, '0')}分`
      if (total >= 60) return `${Math.floor(total / 60)}分${String(total % 60).padStart(2, '0')}秒`
      return `${total}秒`
    }

    /**
     * Format an ISO timestamp for a listing row.
     * @param {string} iso - ISO-8601 timestamp.
     * @returns {string} Local `MM-DD HH:MM`, or the raw value when unparseable.
     */
    function formatStamp(iso) {
      const date = new Date(iso)
      if (Number.isNaN(date.getTime())) return iso || ''
      const pad = (value) => String(value).padStart(2, '0')
      return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`
    }

    /** Split inline Markdown into text/bold/code spans. */
    function renderInline(text, keyPrefix) {
      const parts = []
      const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g
      let cursor = 0
      let match = pattern.exec(text)
      let index = 0
      while (match !== null) {
        if (match.index > cursor) parts.push(text.slice(cursor, match.index))
        const token = match[0]
        if (token.startsWith('**')) {
          parts.push(React.createElement('strong', { key: `${keyPrefix}-b${index}` }, token.slice(2, -2)))
        } else {
          parts.push(React.createElement('code', { key: `${keyPrefix}-c${index}` }, token.slice(1, -1)))
        }
        cursor = match.index + token.length
        index += 1
        match = pattern.exec(text)
      }
      if (cursor < text.length) parts.push(text.slice(cursor))
      return parts
    }

    /**
     * Render the minutes as elements.
     *
     * Deliberately not a general Markdown engine — it handles exactly what the
     * summarizer emits (headings, bullets, the action-item table, rules, bold
     * and code) and treats everything else as text, so nothing is ever injected
     * as HTML.
     *
     * @param {string} markdown - Rendered minutes.
     * @returns {Array} React children.
     */
    function renderMinutes(markdown) {
      const lines = String(markdown || '').split('\n')
      const blocks = []
      let bullets = []
      let table = []

      const flushBullets = () => {
        if (bullets.length === 0) return
        blocks.push(React.createElement('ul', { key: `ul${blocks.length}`, className: 'dsh-voice__ul' },
          bullets.map((item, index) => React.createElement('li', { key: index }, renderInline(item, `li${index}`)))))
        bullets = []
      }
      const flushTable = () => {
        if (table.length === 0) return
        const [head, ...body] = table
        blocks.push(React.createElement('table', { key: `tb${blocks.length}`, className: 'dsh-voice__table' },
          React.createElement('thead', null, React.createElement('tr', null,
            head.map((cell, index) => React.createElement('th', { key: index }, cell)))),
          React.createElement('tbody', null, body.map((row, rowIndex) => React.createElement('tr', { key: rowIndex },
            row.map((cell, cellIndex) => React.createElement('td', { key: cellIndex }, renderInline(cell, `td${rowIndex}-${cellIndex}`))))))))
        table = []
      }

      for (const raw of lines) {
        const line = raw.trimEnd()
        const trimmed = line.trim()
        if (trimmed.startsWith('|')) {
          flushBullets()
          const cells = trimmed.replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim())
          if (cells.every((cell) => /^:?-{2,}:?$/.test(cell))) continue // separator row
          table.push(cells)
          continue
        }
        flushTable()
        if (trimmed === '') {
          flushBullets()
          continue
        }
        if (/^#{1,3}\s/.test(trimmed)) {
          flushBullets()
          const level = trimmed.match(/^#+/)[0].length
          blocks.push(React.createElement(`h${Math.min(level, 3)}`, {
            key: `h${blocks.length}`,
            className: 'dsh-voice__h',
          }, renderInline(trimmed.replace(/^#+\s*/, ''), `h${blocks.length}`)))
          continue
        }
        if (/^([-*]|\d+\.)\s/.test(trimmed)) {
          bullets.push(trimmed.replace(/^([-*]|\d+\.)\s*/, ''))
          continue
        }
        if (/^-{3,}$/.test(trimmed)) {
          flushBullets()
          blocks.push(React.createElement('hr', { key: `hr${blocks.length}`, className: 'dsh-voice__hr' }))
          continue
        }
        flushBullets()
        blocks.push(React.createElement('p', { key: `p${blocks.length}`, className: 'dsh-voice__p' },
          renderInline(trimmed, `p${blocks.length}`)))
      }
      flushBullets()
      flushTable()
      return blocks
    }

    /**
     * The records tab: past meetings, their transcripts, and their minutes.
     * @param {object} props - Wiring for the tab.
     * @param {number} props.refreshToken - Bumped by the shell to force a reload
     *   (for example when a recording just stopped).
     * @returns {object} The rendered element tree.
     */
    function MeetingsPanel({ refreshToken }) {
      const [meetings, setMeetings] = React.useState([])
      const [selectedId, setSelectedId] = React.useState(null)
      const [detail, setDetail] = React.useState(null)
      const [error, setError] = React.useState(null)
      const [busy, setBusy] = React.useState('')
      const [showTranscript, setShowTranscript] = React.useState(false)

      const loadList = React.useCallback(async (preferredId) => {
        try {
          const payload = await fetchJson('/meetings?limit=100')
          const rows = payload.meetings || []
          setMeetings(rows)
          setError(null)
          setSelectedId((current) => {
            const wanted = preferredId || current
            if (wanted && rows.some((row) => row.id === wanted)) return wanted
            return rows.length > 0 ? rows[0].id : null
          })
        } catch (reason) {
          setError(reason && reason.message ? reason.message : String(reason))
        }
      }, [])

      React.useEffect(() => { loadList() }, [loadList, refreshToken])

      React.useEffect(() => {
        if (!selectedId) {
          setDetail(null)
          return
        }
        let cancelled = false
        fetchJson(`/meetings/${encodeURIComponent(selectedId)}`)
          .then((payload) => { if (!cancelled) setDetail(payload.meeting || null) })
          .catch((reason) => { if (!cancelled) setError(reason && reason.message ? reason.message : String(reason)) })
        return () => { cancelled = true }
      }, [selectedId, refreshToken])

      const summarize = React.useCallback(async (force) => {
        if (!selectedId) return
        setBusy('summary')
        setError(null)
        try {
          const payload = await fetchJson(`/meetings/${encodeURIComponent(selectedId)}/summary`, {
            method: 'POST',
            body: JSON.stringify({ force: Boolean(force) }),
          })
          setDetail((previous) => (previous
            ? { ...previous, summary: payload.summary || '', summary_structured: payload.structured || null }
            : previous))
          await loadList(selectedId)
        } catch (reason) {
          setError(reason && reason.message ? reason.message : String(reason))
        } finally {
          setBusy('')
        }
      }, [selectedId, loadList])

      const remove = React.useCallback(async () => {
        if (!selectedId) return
        // Deleting a record is irreversible, so it asks first.
        if (!window.confirm(`删除会议记录 ${selectedId}？该操作不可撤销。`)) return
        setBusy('delete')
        try {
          await fetchJson(`/meetings/${encodeURIComponent(selectedId)}`, { method: 'DELETE' })
          setSelectedId(null)
          setDetail(null)
          await loadList()
        } catch (reason) {
          setError(reason && reason.message ? reason.message : String(reason))
        } finally {
          setBusy('')
        }
      }, [selectedId, loadList])

      const copy = React.useCallback((text) => {
        if (!text) return
        navigator.clipboard?.writeText(text).catch(() => {})
      }, [])

      const list = React.createElement('div', { className: 'dsh-voice__list' },
        React.createElement('div', { className: 'dsh-voice__listhead' },
          React.createElement('span', null, `会议记录 · ${meetings.length}`),
          React.createElement('button', {
            className: 'dsh-voice__btn dsh-voice__btn--tiny',
            onClick: () => loadList(),
          }, '刷新'),
        ),
        meetings.length === 0
          ? React.createElement('div', { className: 'dsh-voice__notice' },
              '还没有会议记录。在「实时」里点「开始收听」，或让我用 summarize_recording 处理一个录音文件。')
          : meetings.map((row) => React.createElement('div', {
              key: row.id,
              className: 'dsh-voice__item',
              'data-active': row.id === selectedId ? '1' : '0',
              onClick: () => setSelectedId(row.id),
            },
            React.createElement('div', { className: 'dsh-voice__item-title' }, row.title || row.id),
            React.createElement('div', { className: 'dsh-voice__item-meta' },
              `${formatStamp(row.started_at)} · ${formatDuration(row.wall_seconds || row.duration_s)} · ${row.segments} 段`),
            React.createElement('div', { className: 'dsh-voice__item-meta' },
              React.createElement('span', {
                className: 'dsh-voice__tag',
                'data-state': row.summary_status === 'ready' ? 'ready' : (row.summary_status === 'failed' ? 'failed' : 'none'),
              }, row.summary_status === 'ready' ? '有纪要' : (row.summary_status === 'failed' ? '纪要失败' : '无纪要')),
              row.source === 'file' ? React.createElement('span', { className: 'dsh-voice__tag', 'data-state': 'file' }, '录音文件') : null),
          )),
      )

      const detailBar = detail
        ? React.createElement('div', { className: 'dsh-voice__detailbar' },
            React.createElement('span', { className: 'dsh-voice__title' }, detail.meta.title || detail.meta.id),
            React.createElement('span', { className: 'dsh-voice__item-meta' },
              `${formatStamp(detail.meta.started_at)} · ${formatDuration(detail.meta.wall_seconds || detail.meta.duration_s)} · ${detail.meta.segments} 段`),
            React.createElement('div', { className: 'dsh-voice__spacer' }),
            React.createElement('button', {
              className: 'dsh-voice__btn dsh-voice__btn--tiny',
              onClick: () => setShowTranscript((value) => !value),
            }, showTranscript ? '看纪要' : '看全文'),
            React.createElement('button', {
              className: 'dsh-voice__btn dsh-voice__btn--tiny',
              onClick: () => copy(detail.summary),
              disabled: !detail.summary,
            }, '复制纪要'),
            React.createElement('button', {
              className: 'dsh-voice__btn dsh-voice__btn--tiny',
              disabled: busy === 'summary',
              onClick: () => summarize(Boolean(detail.summary)),
            }, busy === 'summary' ? '生成中…' : (detail.summary ? '重新生成' : '生成纪要')),
            React.createElement('button', {
              className: 'dsh-voice__btn dsh-voice__btn--tiny',
              disabled: busy === 'delete',
              onClick: remove,
            }, '删除'),
          )
        : null

      const detailBody = !detail
        ? React.createElement('div', { className: 'dsh-voice__notice' }, '从左侧选择一次会议。')
        : showTranscript
          ? React.createElement('div', { className: 'dsh-voice__detailbody' },
              (detail.segments || []).map((segment) => React.createElement('div', {
                key: segment.id,
                className: 'dsh-voice__seg',
              },
              React.createElement('div', null,
                React.createElement('span', { className: 'dsh-voice__stamp' }, `[${formatDuration(segment.start)}]`),
                React.createElement('span', { className: 'dsh-voice__en' }, segment.en)),
              segment.zh ? React.createElement('div', { className: 'dsh-voice__zh', 'data-pending': '0' }, segment.zh) : null,
              )))
          : React.createElement('div', { className: 'dsh-voice__detailbody' },
              detail.summary
                ? React.createElement('div', { className: 'dsh-voice__minutes' }, renderMinutes(detail.summary))
                : React.createElement('div', { className: 'dsh-voice__notice' },
                    '还没有纪要。点「生成纪要」，我会把这份逐句记录整理成中文会议纪要',
                    '（概览、关键要点、决定事项、待办事项、风险）。'))

      return React.createElement('div', { className: 'dsh-voice__pane' },
        error ? React.createElement('div', { className: 'dsh-voice__banner' },
          React.createElement('div', null, error),
          React.createElement('div', null, '若服务未运行：',
            React.createElement('code', null, 'dsh-voice serve'))) : null,
        React.createElement('div', { className: 'dsh-voice__split' },
          list,
          React.createElement('div', { className: 'dsh-voice__detail' }, detailBar, detailBody),
        ),
      )
    }

    /**
     * The "同传" tab: a live interpretation pane and a meeting-record pane.
     *
     * The live pane stays mounted while the records pane is shown, so browsing
     * past meetings does not stop a recording in progress.
     * @returns {object} The rendered element tree.
     */
    function InterpreterView() {
      const [mode, setMode] = React.useState('live')
      const [refreshToken, setRefreshToken] = React.useState(0)

      const onArchived = React.useCallback((meeting) => {
        setRefreshToken((value) => value + 1)
        if (meeting) setMode('meetings')
      }, [])

      const modeButton = (id, label) => React.createElement('button', {
        key: id,
        className: 'dsh-voice__mode',
        'data-active': mode === id ? '1' : '0',
        onClick: () => setMode(id),
      }, label)

      return React.createElement('div', { className: 'dsh-voice' },
        React.createElement('div', { className: 'dsh-voice__modes' },
          modeButton('live', '实时'),
          modeButton('meetings', '会议记录'),
        ),
        React.createElement('div', {
          className: 'dsh-voice__host',
          style: { display: mode === 'live' ? 'flex' : 'none' },
        }, React.createElement(LivePanel, { onArchived })),
        React.createElement('div', {
          className: 'dsh-voice__host',
          style: { display: mode === 'meetings' ? 'flex' : 'none' },
        }, React.createElement(MeetingsPanel, { refreshToken })),
      )
    }

    /**
     * Register the "同传" tab into the conversation view ring.
     *
     * The session is deliberately *not* owned by this registration: the slot
     * unmounts whenever the user looks at another tab or conversation, and a
     * meeting recorder that stops when you glance elsewhere is useless. The
     * plugin's own lifetime owns it instead, so unmounting detaches a view and
     * only unloading the plugin releases the microphone.
     *
     * @param {object} ctx - Client root context.
     */
    function apply(ctx) {
      ctx.slots.inject('conversation.view', () => ctx.slots.register({
        name: 'conversation.view',
        id: 'interpreter',
        order: 20,
        label: () => '同传',
      }, InterpreterView))
      ctx.effect(() => () => { sessionDispose() }, 'voice-interpreter: release the microphone on unload')
    }

    exports.apply = apply
    exports.inject = ['slots']
    // Test seam: the pure helpers the render harness exercises directly. The
    // module loader only reads `apply`/`inject`, so this costs nothing at boot
    // and keeps the minutes renderer verifiable without a browser.
    exports.__internal = {
      renderMinutes, renderInline, formatDuration, formatStamp, fetchJson,
      // The live session is app-scoped state, so the render harness can drive it
      // without mounting a component — which is exactly the property that keeps
      // a recording alive across navigation.
      voiceSession, sessionStart, sessionStop, sessionDispose, sessionApplyEvent,
      sessionSubscribe, sessionGetSnapshot, sessionPublish, sessionClearRows,
      LiveClient,
    }
    return module.exports
  },
})

// Hot reload: client-hmr stat-polls this file every 500ms and swaps the module
// in an open page without a refresh. See the README section on hot reload.
