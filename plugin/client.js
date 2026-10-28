/**
 * Browser half of the voice-interpreter plugin: a "同传" conversation view.
 *
 * The panel captures the microphone, resamples it to Qwen3-ASR's 16 kHz mono
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

    /** Target sample rate: Qwen3-ASR's fixed input rate. */
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

    /** Panel preferences survive reloads; the service contract does not change. */
    const PREFS_KEY = 'dsh-voice.preferences'
    const DEFAULT_PREFS = {
      translatePartials: true,
      viewMode: 'bilingual',
      fontSize: 'md',
      follow: true,
    }
    const VIEW_LABELS = { bilingual: '双语', zh: '仅中文', en: '仅英文' }
    const FONT_LABELS = { sm: '小', md: '中', lg: '大' }

    /** Read stored preferences, falling back to defaults on any doubt. */
    function loadPrefs() {
      try {
        const stored = JSON.parse(localStorage.getItem(PREFS_KEY) || 'null')
        if (stored && typeof stored === 'object') return { ...DEFAULT_PREFS, ...stored }
      } catch {
        /* a corrupt entry is not worth failing the panel over */
      }
      return { ...DEFAULT_PREFS }
    }

    /** Persist preferences; a locked-down storage backend is ignored. */
    function savePrefs(prefs) {
      try {
        localStorage.setItem(PREFS_KEY, JSON.stringify(prefs))
      } catch {
        /* private mode: preferences live for the session only */
      }
    }

    /** ``mm:ss`` for a duration in seconds. */
    function formatClock(seconds) {
      const total = Math.max(0, Math.floor(Number(seconds) || 0))
      return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`
    }

    /** Split text into ``{ text, hit }`` parts so a search term can be marked. */
    function splitMatch(text, term) {
      const raw = String(text || '')
      const needle = String(term || '').trim()
      if (!needle) return [{ text: raw, hit: false }]
      const lower = raw.toLowerCase()
      const target = needle.toLowerCase()
      const parts = []
      let cursor = 0
      for (;;) {
        const found = lower.indexOf(target, cursor)
        if (found === -1) {
          if (cursor < raw.length) parts.push({ text: raw.slice(cursor), hit: false })
          break
        }
        if (found > cursor) parts.push({ text: raw.slice(cursor, found), hit: false })
        parts.push({ text: raw.slice(found, found + needle.length), hit: true })
        cursor = found + needle.length
      }
      return parts.length ? parts : [{ text: raw, hit: false }]
    }

    /** Caption rows to Markdown, with one timestamped block per row. */
    function rowsToMarkdown(rows, title) {
      const head = `# ${title || '同传记录'}\n\n`
      const body = rows.map((row) => {
        const lines = [`**[${formatClock(row.start)}]**`]
        if (row.en) lines.push(row.en)
        if (row.zh) lines.push(`> ${row.zh}`)
        return lines.join('\n')
      }).join('\n\n')
      return `${head}${body}\n`
    }

    /** Caption rows to SRT (English then Chinese per cue). */
    function rowsToSrt(rows) {
      const stamp = (seconds) => {
        const total = Math.max(0, Number(seconds) || 0)
        const hours = String(Math.floor(total / 3600)).padStart(2, '0')
        const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, '0')
        const secs = String(Math.floor(total % 60)).padStart(2, '0')
        const millis = String(Math.round((total - Math.floor(total)) * 1000)).padStart(3, '0')
        return `${hours}:${minutes}:${secs},${millis}`
      }
      return rows.map((row, index) => {
        const end = Number(row.end) > Number(row.start) ? row.end : Number(row.start) + 2.5
        const text = [row.en, row.zh].filter(Boolean).join('\n')
        return `${index + 1}\n${stamp(row.start)} --> ${stamp(end)}\n${text}\n`
      }).join('\n')
    }

    /** Download generated caption text without touching the service. */
    function downloadText(filename, text, mime) {
      const blob = new Blob([text], { type: mime || 'text/plain;charset=utf-8' })
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = filename
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    }

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
.dsh-voice__head { position: sticky; top: 0; z-index: 3;
  background: var(--dsw-alias-bg-layer-1, #fff); }
.dsh-voice__log { flex: 1 1 auto; min-height: 0; overflow-y: auto; padding: 14px 16px 32px;
  max-height: max(240px, calc(100dvh - 230px)); overscroll-behavior: contain; }
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
/* ── richer live panel ─────────────────────────────────────────────── */
.dsh-voice { position: relative; }
.dsh-voice__dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  margin-right: 6px; background: currentColor; vertical-align: 1px; opacity: .7; }
.dsh-voice__status[data-state="live"] .dsh-voice__dot { opacity: 1; animation: dsh-voice-pulse 1.6s ease-in-out infinite; }
@keyframes dsh-voice-pulse { 0%, 100% { opacity: 1 } 50% { opacity: .35 } }
.dsh-voice__clock { font-size: 12px; color: var(--dsw-alias-label-secondary, #57606a);
  font-family: var(--ds-font-family-code, ui-monospace, monospace); }
.dsh-voice__chip { font-size: 11px; padding: 2px 8px; border-radius: 999px; white-space: nowrap;
  background: var(--dsw-alias-bg-layer-2, #f1f3f5); color: var(--dsw-alias-label-secondary, #57606a);
  border: 1px solid transparent; }
.dsh-voice__chip[data-state="ok"] { background: rgba(23,140,80,.12); color: #14804a; }
.dsh-voice__chip[data-state="warn"] { background: rgba(212,134,10,.14); color: #9a6700; }
.dsh-voice__chip[data-state="rec"] { background: rgba(207,34,46,.12); color: #b3261e; }
.dsh-voice__iconbtn { font: inherit; font-size: 12px; padding: 4px 9px; cursor: pointer; border-radius: 7px;
  border: 1px solid var(--dsw-alias-border-l2, #d0d7de); background: transparent;
  color: var(--dsw-alias-label-secondary, #57606a); }
.dsh-voice__iconbtn:hover, .dsh-voice__iconbtn[data-active="1"] {
  background: var(--dsw-alias-interactive-bg-hover, #f3f4f6); color: var(--dsw-alias-label-primary, #1f2328); }
.dsh-voice__btn--tiny[data-active="1"] { background: rgba(212,134,10,.16); border-color: rgba(212,134,10,.4); color: #9a6700; }
.dsh-voice__settings { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center;
  padding: 10px 16px; border-bottom: 1px solid var(--dsw-alias-border-l1, #eaecef);
  background: var(--dsw-alias-bg-layer-2, #fafbfc); }
.dsh-voice__setgroup { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.dsh-voice__setlabel { font-size: 12px; color: var(--dsw-alias-label-tertiary, #6e7781); }
.dsh-voice__segbtn { font: inherit; font-size: 12px; padding: 3px 9px; cursor: pointer; border-radius: 999px;
  border: 1px solid var(--dsw-alias-border-l2, #d0d7de); background: var(--dsw-alias-bg-layer-1, #fff); color: inherit; }
.dsh-voice__segbtn[data-active="1"] { background: rgba(9,105,218,.12); border-color: rgba(9,105,218,.45); color: #0969da; }
.dsh-voice__sethint { font-size: 11px; color: var(--dsw-alias-label-tertiary, #8b949e); }
.dsh-voice__searchbar { display: flex; align-items: center; gap: 8px; padding: 7px 16px;
  border-bottom: 1px solid var(--dsw-alias-border-l1, #eaecef); }
.dsh-voice__searchicon { font-size: 13px; color: var(--dsw-alias-label-tertiary, #8b949e); }
.dsh-voice__input--search { min-width: 0; flex: 1 1 auto; }
.dsh-voice__searchcount { font-size: 12px; color: var(--dsw-alias-label-tertiary, #6e7781); white-space: nowrap; }
.dsh-voice__meta-warn { color: #b3261e; }
.dsh-voice__rowhead { display: flex; align-items: baseline; gap: 6px; margin-bottom: 2px; }
.dsh-voice__rowtime { font-size: 11px; color: var(--dsw-alias-label-tertiary, #8b949e);
  font-family: var(--ds-font-family-code, ui-monospace, monospace); }
.dsh-voice__lang { font-size: 10px; padding: 0 5px; border-radius: 4px; letter-spacing: .02em;
  background: var(--dsw-alias-bg-layer-2, #f1f3f5); color: var(--dsw-alias-label-secondary, #57606a); }
.dsh-voice__lang[data-lang="zh"] { background: rgba(207,34,46,.10); color: #b3261e; }
.dsh-voice__badge { font-size: 10px; padding: 0 5px; border-radius: 4px; margin-left: 4px;
  background: rgba(9,105,218,.10); color: #0969da; }
.dsh-voice__badge--tr { background: rgba(212,134,10,.14); color: #9a6700; }
.dsh-voice__row[data-final="1"] .dsh-voice__zh { font-size: 17px; }
.dsh-voice[data-font="sm"] .dsh-voice__zh { font-size: 15px; }
.dsh-voice[data-font="lg"] .dsh-voice__zh { font-size: 20px; }
.dsh-voice[data-font="lg"] .dsh-voice__en { font-size: 14px; }
.dsh-voice__zh[data-pending="1"] { opacity: .68; }
.dsh-voice__row[data-final="0"] .dsh-voice__zh[data-pending="1"]::after { content: ""; display: inline-block;
  width: 7px; height: 7px; margin-left: 4px; border-radius: 50%; background: currentColor;
  animation: dsh-voice-blink 1.1s ease-in-out infinite; vertical-align: 1px; }
@keyframes dsh-voice-blink { 0%, 100% { opacity: .2 } 50% { opacity: .9 } }
.dsh-voice__zh-error { color: #b3261e; font-size: 14px; }
.dsh-voice__zh { min-height: 1.5em; }
.dsh-voice__row[data-partial="1"] { border-left-style: dashed; }
.dsh-voice__row[data-partial="1"] .dsh-voice__zh { opacity: .6; }
.dsh-voice__row[data-error="1"] { border-left: 3px dashed #cf222e; }
.dsh-voice__truncnote { display: flex; align-items: center; gap: 8px; font-size: 12px;
  padding: 8px 16px; margin: 0 0 8px; color: #9a6700; background: rgba(212,134,10,.10);
  border-bottom: 1px solid rgba(212,134,10,.25); }
.dsh-voice__sr { position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip-path: inset(50%); white-space: nowrap; border: 0; padding: 0; margin: -1px; }
.dsh-voice__btn:focus-visible, .dsh-voice__iconbtn:focus-visible, .dsh-voice__segbtn:focus-visible {
  outline: 2px solid #0969da; outline-offset: 1px; }
.dsh-voice__mark { background: rgba(255,196,0,.35); color: inherit; padding: 0 1px; border-radius: 2px; }
.dsh-voice__followpill { position: absolute; right: 22px; bottom: 66px; z-index: 4;
  font: inherit; font-size: 12px; padding: 5px 12px; cursor: pointer; border-radius: 999px;
  border: 1px solid var(--dsw-alias-border-l2, #d0d7de); background: var(--dsw-alias-bg-layer-1, #fff);
  color: var(--dsw-alias-label-primary, #1f2328); box-shadow: 0 2px 10px rgba(0,0,0,.12); }
.dsh-voice__footer { display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  padding: 7px 16px 9px; font-size: 11px; color: var(--dsw-alias-label-tertiary, #6e7781);
  border-top: 1px solid var(--dsw-alias-border-l1, #eaecef); background: var(--dsw-alias-bg-layer-1, #fff); }
.dsh-voice__footer-warn { color: #b3261e; }
.dsh-voice__empty-card { max-width: 560px; padding: 18px 20px; border-radius: 10px;
  border: 1px dashed var(--dsw-alias-border-l2, #d0d7de); background: var(--dsw-alias-bg-layer-1, #fff); }
.dsh-voice__empty-title { font-size: 14px; font-weight: 600; margin-bottom: 8px; color: var(--dsw-alias-label-primary, #1f2328); }
.dsh-voice__empty-list { margin: 0; padding-left: 18px; line-height: 1.9; }
.dsh-voice__empty-warn { margin-top: 10px; color: #9a6700; font-size: 12px; }
.dsh-voice__empty-hint { margin-top: 10px; font-size: 11px; color: var(--dsw-alias-label-tertiary, #8b949e); }
@media (max-width: 720px) {
  .dsh-voice__bar { gap: 6px; padding: 8px 10px; }
  .dsh-voice__meta { padding: 5px 10px; gap: 8px; }
  .dsh-voice__settings { padding: 8px 10px; }
  .dsh-voice__searchbar { padding: 6px 10px; }
  .dsh-voice__log { padding: 10px 10px 24px; }
  .dsh-voice__footer { padding: 6px 10px; }
  .dsh-voice__followpill { right: 12px; bottom: 60px; }
}
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
        this.muted = false
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
          if (!this.active) return
          this.active = false
          // The service went away mid-meeting: release the microphone now
          // instead of leaving the capture graph running against a dead socket.
          this.stop({ graceful: false })
          this.onState('error', '与本地服务的连接已断开')
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
          if (this.muted) {
            this.onLevel(0)
            return
          }
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
       * Pause or resume the microphone without ending the session.
       * @param {boolean} muted - True to stop sending audio upstream.
       */
      setMuted(muted) {
        this.muted = Boolean(muted)
        if (this.muted) this.onLevel(0)
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
      prefs: loadPrefs(),
      muted: false,
      startedAt: null,
      endedAt: null,
      listeners: new Set(),
      snapshot: null,
    }
    voiceSession.translatePartials = voiceSession.prefs.translatePartials

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
        prefs: voiceSession.prefs,
        muted: voiceSession.muted,
        startedAt: voiceSession.startedAt,
        endedAt: voiceSession.endedAt,
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
        ? { id: event.id, en: '', zh: '', enFinal: false, zhFinal: false, start: 0, end: 0, asr: 0, tr: 0, error: null, language: null }
        : next[index]
      const merged = { ...row }
      if (event.type === 'partial') {
        merged.en = event.text
        merged.enFinal = false
        merged.start = event.start
        merged.asr = event.asr_ms
        if (event.language) merged.language = event.language
      } else if (event.type === 'final') {
        merged.en = event.text
        merged.enFinal = true
        merged.start = event.start
        merged.end = Number(event.end) || merged.end
        merged.asr = event.asr_ms
        if (event.language) merged.language = event.language
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
      voiceSession.muted = false
      voiceSession.startedAt = Date.now()
      voiceSession.endedAt = null
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
        voiceSession.startedAt = Date.now()
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
      voiceSession.muted = false
      voiceSession.endedAt = Date.now()
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
    function LivePanel({ onArchived, onOpenRecords }) {
      const session = React.useSyncExternalStore(
        sessionSubscribe, sessionGetSnapshot, sessionGetSnapshot,
      )
      const {
        rows, state, detail, level, settings, stats, notice, meeting,
        muted, startedAt, endedAt, prefs,
      } = session
      const [titleDraft, setTitleDraft] = React.useState(meeting ? meeting.title || '' : '')
      const [settingsOpen, setSettingsOpen] = React.useState(false)
      const [searchOpen, setSearchOpen] = React.useState(false)
      const [searchTerm, setSearchTerm] = React.useState('')
      const [now, setNow] = React.useState(Date.now())

      const logRef = React.useRef(null)
      const stickRef = React.useRef(true)
      const scrollerRef = React.useRef(null)
      const searchRef = React.useRef(null)

      // Follow the store's meeting: a rename, or a session started while this
      // view was detached, must not leave a stale name in the box.
      const meetingId = meeting ? meeting.id : null
      React.useEffect(() => {
        setTitleDraft(meeting ? meeting.title || '' : '')
      }, [meetingId]) // eslint-disable-line react-hooks/exhaustive-deps -- draft tracks the meeting, not its title

      const stateLive = state === 'live'
      const busy = state === 'live' || state === 'connecting'

      // One clock tick per second is enough for a duration readout.
      React.useEffect(() => {
        if (!startedAt) return undefined
        setNow(Date.now())
        const timer = setInterval(() => setNow(Date.now()), 1000)
        return () => clearInterval(timer)
      }, [startedAt, endedAt])
      const elapsed = startedAt ? Math.max(0, ((endedAt || now) - startedAt) / 1000) : 0

      // A live session whose input stays at zero means the microphone is muted
      // or blocked; surface it instead of looking stalled.
      const quietSinceRef = React.useRef(0)
      const [silent, setSilent] = React.useState(false)
      React.useEffect(() => {
        if (!stateLive) {
          quietSinceRef.current = 0
          setSilent(false)
          return undefined
        }
        const timer = setInterval(() => {
          if (level > 0.001) {
            quietSinceRef.current = 0
            setSilent(false)
            return
          }
          if (!quietSinceRef.current) quietSinceRef.current = Date.now()
          setSilent(Date.now() - quietSinceRef.current > 3000)
        }, 1000)
        return () => clearInterval(timer)
      }, [stateLive, level])

      const updatePrefs = React.useCallback((patch) => {
        voiceSession.prefs = { ...voiceSession.prefs, ...patch }
        savePrefs(voiceSession.prefs)
        if (patch.translatePartials !== undefined) {
          voiceSession.translatePartials = patch.translatePartials
          if (voiceSession.client) voiceSession.client.configure(patch.translatePartials)
        }
        sessionPublish()
      }, [])

      /** Resolve the element whose scrollbar moves the captions. */
      const resolveScroller = React.useCallback(() => {
        let node = logRef.current
        while (node && node !== document.body && node !== document.documentElement) {
          if (node.scrollHeight > node.clientHeight + 2) return node
          node = node.parentElement
        }
        return document.scrollingElement || document.documentElement
      }, [])

      const onScroll = React.useCallback(() => {
        const node = scrollerRef.current
        if (!node) return
        const atBottom = node.scrollHeight - node.scrollTop - node.clientHeight < 80
        stickRef.current = atBottom
        if (atBottom !== voiceSession.prefs.follow) updatePrefs({ follow: atBottom })
      }, [updatePrefs])

      // Follow the newest caption, unless the viewer scrolled away.
      React.useEffect(() => {
        if (!rows.length) {
          stickRef.current = voiceSession.prefs.follow
          return
        }
        if (!voiceSession.prefs.follow || !stickRef.current) return
        const node = scrollerRef.current || resolveScroller()
        if (!node) return
        node.scrollTop = node.scrollHeight
      }, [rows, resolveScroller])

      // Bind the stickiness listener to whichever element owns the scrollbar.
      React.useEffect(() => {
        const node = resolveScroller()
        if (!node || node === scrollerRef.current) return
        if (scrollerRef.current) scrollerRef.current.removeEventListener('scroll', onScroll)
        scrollerRef.current = node
        node.addEventListener('scroll', onScroll, { passive: true })
      }, [rows, resolveScroller, onScroll]) // eslint-disable-line react-hooks/exhaustive-deps -- node identity can change without rows

      React.useEffect(() => {
        if (searchOpen) searchRef.current?.focus()
      }, [searchOpen])

      const gotoLatest = React.useCallback(() => {
        const node = scrollerRef.current || resolveScroller()
        if (node) node.scrollTop = node.scrollHeight
        stickRef.current = true
        updatePrefs({ follow: true })
      }, [resolveScroller, updatePrefs])

      const toggle = React.useCallback(async () => {
        if (voiceSession.client) {
          const archived = await sessionStop()
          onArchived(archived)
          return
        }
        await sessionStart()
      }, [onArchived])

      const toggleMute = React.useCallback(() => {
        voiceSession.muted = !voiceSession.muted
        if (voiceSession.client) voiceSession.client.setMuted(voiceSession.muted)
        sessionPublish()
      }, [])

      const onPartialToggle = React.useCallback((event) => {
        updatePrefs({ translatePartials: event.target.checked })
      }, [updatePrefs])

      const rename = React.useCallback(() => {
        const title = titleDraft.trim()
        if (!title || !voiceSession.client) return
        voiceSession.client.rename(title)
        voiceSession.meeting = voiceSession.meeting ? { ...voiceSession.meeting, title } : voiceSession.meeting
        sessionPublish()
      }, [titleDraft])

      const copyAll = React.useCallback(() => {
        if (!rows.length) return
        navigator.clipboard?.writeText(rowsToMarkdown(rows, meeting && meeting.title)).catch(() => {})
      }, [rows, meeting])

      const exportMarkdown = React.useCallback(() => {
        const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 16)
        downloadText(`dsh-voice-${stamp}.md`, rowsToMarkdown(rows, meeting && meeting.title), 'text/markdown;charset=utf-8')
      }, [rows, meeting])

      const exportSrt = React.useCallback(() => {
        const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 16)
        downloadText(`dsh-voice-${stamp}.srt`, rowsToSrt(rows), 'text/plain;charset=utf-8')
      }, [rows])

      // Shortcuts: space starts/stops, M mutes, / searches, F jumps to latest.
      React.useEffect(() => {
        const onKey = (event) => {
          const target = event.target
          const typing = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
          if (event.key === 'Escape' && searchOpen) {
            setSearchOpen(false)
            setSearchTerm('')
            return
          }
          if (typing) return
          if (event.key === ' ') {
            event.preventDefault()
            toggle()
          } else if (event.key.toLowerCase() === 'm' && stateLive) {
            toggleMute()
          } else if (event.key.toLowerCase() === 'f') {
            gotoLatest()
          } else if (event.key === '/') {
            event.preventDefault()
            setSearchOpen(true)
          }
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
      }, [toggle, toggleMute, gotoLatest, searchOpen, stateLive])

      const term = searchTerm.trim()
      const visibleRows = term
        ? rows.filter((row) => `${row.en} ${row.zh}`.toLowerCase().includes(term.toLowerCase()))
        : rows
      const failedTranslations = rows.filter((row) => row.error).length
      const average = (key) => {
        const values = rows.map((row) => Number(row[key]) || 0).filter((value) => value > 0)
        return values.length ? Math.round(values.reduce((sum, value) => sum + value, 0) / values.length) : 0
      }
      const LANG_BADGE = { en: 'EN', zh: '中', yue: '粤', ja: '日', ko: '韩', fr: '法', de: '德', es: '西', ru: '俄', it: '意', pt: '葡' }
      const langLabel = (code) => LANG_BADGE[code] || (code ? String(code).toUpperCase() : '')
      const withMatch = (text) => splitMatch(text, searchTerm).map((part, index) => (
        part.hit
          ? React.createElement('mark', { key: index, className: 'dsh-voice__mark' }, part.text)
          : part.text
      ))

      const statusText = state === 'live'
        ? (muted ? '已暂停收音' : silent ? '未检测到声音 · 检查麦克风' : '监听中')
        : STATE_LABEL[state]
      const chips = []
      if (settings) {
        chips.push(React.createElement('span', {
          key: 'model', className: 'dsh-voice__chip', title: settings.model_repo || '',
          'data-state': settings.model_ready ? 'ok' : 'warn',
        }, settings.model_ready ? 'Qwen3 · 本机' : '模型未下载'))
        chips.push(React.createElement('span', {
          key: 'translate', className: 'dsh-voice__chip', title: settings.api_model || '',
          'data-state': settings.translation_available ? 'ok' : 'warn',
        }, settings.translation_available ? '英译中 · DeepSeek' : '翻译未配置'))
        if (settings.record) {
          chips.push(React.createElement('span', {
            key: 'record', className: 'dsh-voice__chip', 'data-state': 'rec',
          }, meeting ? `记录 #${meeting.id}` : '记录中'))
        }
      }

      const header = React.createElement('div', { className: 'dsh-voice__bar' },
        React.createElement('span', { className: 'dsh-voice__title' }, '同声传译'),
        React.createElement('span', {
          className: 'dsh-voice__status',
          role: 'status',
          'data-state': state === 'live' ? 'live' : state === 'error' ? 'error' : state === 'idle' ? '' : 'ready',
        },
        React.createElement('i', { className: 'dsh-voice__dot' }),
        statusText + (state === 'error' && detail ? ` · ${detail}` : '')),
        startedAt ? React.createElement('span', { className: 'dsh-voice__clock', title: '本次会话时长' }, formatClock(elapsed)) : null,
        stateLive ? React.createElement('div', { className: 'dsh-voice__meter', title: '麦克风电平', 'aria-hidden': 'true' },
          React.createElement('i', { style: { width: `${Math.min(100, level * 140)}%` } })) : null,
        React.createElement('div', { className: 'dsh-voice__spacer' }),
        ...chips,
        React.createElement('button', {
          className: 'dsh-voice__iconbtn', title: '搜索字幕（/）',
          'data-active': searchOpen ? '1' : '0',
          onClick: () => setSearchOpen((open) => !open),
        }, '搜索'),
        React.createElement('button', {
          className: 'dsh-voice__iconbtn', title: '面板设置',
          'data-active': settingsOpen ? '1' : '0',
          onClick: () => setSettingsOpen((open) => !open),
        }, '设置'),
        React.createElement('button', { className: 'dsh-voice__btn dsh-voice__btn--tiny', onClick: copyAll, disabled: rows.length === 0 }, '复制'),
        React.createElement('button', { className: 'dsh-voice__btn dsh-voice__btn--tiny', onClick: sessionClearRows, disabled: rows.length === 0 }, '清空'),
        stateLive ? React.createElement('button', {
          className: 'dsh-voice__btn dsh-voice__btn--tiny',
          'data-active': muted ? '1' : '0',
          onClick: toggleMute,
          title: '暂停/恢复收音（M）',
        }, muted ? '恢复收音' : '暂停收音') : null,
        React.createElement('button', {
          className: 'dsh-voice__btn', 'data-primary': '1', 'data-live': busy ? '1' : '0',
          disabled: state === 'connecting',
          onClick: toggle,
        }, state === 'connecting' ? '连接中…' : busy ? '停止' : '开始收听'),
      )

      const settingsRow = settingsOpen ? React.createElement('div', { className: 'dsh-voice__settings' },
        React.createElement('div', { className: 'dsh-voice__setgroup' },
          React.createElement('span', { className: 'dsh-voice__setlabel' }, '显示'),
          ['bilingual', 'zh', 'en'].map((id) => React.createElement('button', {
            key: id, className: 'dsh-voice__segbtn',
            'data-active': prefs.viewMode === id ? '1' : '0',
            onClick: () => updatePrefs({ viewMode: id }),
          }, VIEW_LABELS[id]))),
        React.createElement('div', { className: 'dsh-voice__setgroup' },
          React.createElement('span', { className: 'dsh-voice__setlabel' }, '字号'),
          ['sm', 'md', 'lg'].map((id) => React.createElement('button', {
            key: id, className: 'dsh-voice__segbtn',
            'data-active': prefs.fontSize === id ? '1' : '0',
            onClick: () => updatePrefs({ fontSize: id }),
          }, FONT_LABELS[id]))),
        React.createElement('div', { className: 'dsh-voice__setgroup' },
          React.createElement('label', { className: 'dsh-voice__check' },
            React.createElement('input', {
              type: 'checkbox', checked: prefs.follow,
              onChange: (event) => updatePrefs({ follow: event.target.checked }),
            }), '自动跟随最新字幕'),
          React.createElement('label', { className: 'dsh-voice__check' },
            React.createElement('input', {
              type: 'checkbox', checked: prefs.translatePartials,
              onChange: onPartialToggle,
            }), '预览（未定稿）也翻译')),
        React.createElement('div', { className: 'dsh-voice__setgroup' },
          React.createElement('span', { className: 'dsh-voice__setlabel' }, '导出'),
          React.createElement('button', { className: 'dsh-voice__btn dsh-voice__btn--tiny', onClick: exportMarkdown, disabled: rows.length === 0 }, 'Markdown'),
          React.createElement('button', { className: 'dsh-voice__btn dsh-voice__btn--tiny', onClick: exportSrt, disabled: rows.length === 0 }, 'SRT')),
        React.createElement('div', { className: 'dsh-voice__setgroup dsh-voice__sethint' },
          '快捷键：空格 开始/停止 · M 暂停收音 · / 搜索 · F 回到最新'),
      ) : null

      const searchRow = (searchOpen || term) ? React.createElement('div', { className: 'dsh-voice__searchbar' },
        React.createElement('span', { className: 'dsh-voice__searchicon' }, '⌕'),
        React.createElement('input', {
          ref: searchRef,
          className: 'dsh-voice__input dsh-voice__input--search',
          value: searchTerm,
          placeholder: '搜索英文字幕或中文译文（Esc 清除）',
          onChange: (event) => setSearchTerm(event.target.value),
        }),
        term ? React.createElement('span', { className: 'dsh-voice__searchcount' }, `${visibleRows.length}/${rows.length}`) : null,
        React.createElement('button', {
          className: 'dsh-voice__btn dsh-voice__btn--tiny',
          onClick: () => { setSearchTerm(''); setSearchOpen(false) },
        }, '关闭'),
      ) : null

      const meta = React.createElement('div', { className: 'dsh-voice__meta' },
        React.createElement('span', null, `服务 ${HTTP_URL}`),
        React.createElement('span', null, settings && settings.language ? `语言 ${settings.language}` : '语言 自动识别'),
        React.createElement('span', null, settings ? (settings.model_ready ? '模型已就绪' : '模型未下载') : '未连接'),
        stats ? React.createElement('span', null, `已识别 ${stats.segments} 段`) : null,
        stats ? React.createElement('span', null, `ASR 平均 ${average('asr')}ms · 最近 ${stats.last_asr_ms}ms`) : null,
        stats ? React.createElement('span', null, `翻译 平均 ${average('tr')}ms · 最近 ${stats.last_translate_ms}ms`) : null,
        stats && stats.dropped_hallucinations
          ? React.createElement('span', null, `丢弃幻觉 ${stats.dropped_hallucinations}`) : null,
        stats && stats.failed_translations
          ? React.createElement('span', { className: 'dsh-voice__meta-warn' }, `翻译失败 ${stats.failed_translations}`) : null,
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
              busy ? '停止后会自动存档，可在「会议记录」里生成纪要' : '已存档，可在「会议记录」里生成纪要'),
          )
        : null

      const empty = React.createElement('div', { className: 'dsh-voice__empty' },
        React.createElement('div', { className: 'dsh-voice__empty-card' },
          React.createElement('div', { className: 'dsh-voice__empty-title' }, '开始一场同声传译'),
          React.createElement('ul', { className: 'dsh-voice__empty-list' },
            React.createElement('li', null, '点击「开始收听」并授权麦克风，即可实时看到字幕。'),
            React.createElement('li', null, '本机 Qwen3-ASR 自动识别中英文口音；DeepSeek 负责英译中。'),
            React.createElement('li', null, '停止后自动存档，可在「会议记录」里生成纪要或导出。'),
          ),
          settings && settings.model_ready === false
            ? React.createElement('div', { className: 'dsh-voice__empty-warn' },
                '模型尚未下载：运行 ', React.createElement('code', null, 'dsh-voice download-model'), '（约 0.7GB）。')
            : null,
          React.createElement('div', { className: 'dsh-voice__empty-hint' },
            '快捷键：空格 开始/停止 · M 暂停收音 · / 搜索 · F 回到最新'),
        ),
      )

      const body = rows.length === 0
        ? empty
        : (visibleRows.length === 0
          ? React.createElement('div', { className: 'dsh-voice__empty' }, '没有匹配的字幕。')
          : visibleRows.map((row) => {
              const showEn = prefs.viewMode !== 'zh'
              const showZh = prefs.viewMode !== 'en'
              return React.createElement('div', {
                key: row.id,
                className: 'dsh-voice__row',
                'data-final': row.enFinal ? '1' : '0',
                'data-partial': row.enFinal ? '0' : '1',
                'data-error': row.error ? '1' : '0',
              },
              React.createElement('div', { className: 'dsh-voice__rowhead' },
                React.createElement('span', { className: 'dsh-voice__rowtime', title: '句首时间' }, formatClock(row.start)),
                row.language
                  ? React.createElement('span', { className: 'dsh-voice__lang', 'data-lang': row.language }, langLabel(row.language))
                  : null,
                React.createElement('span', { className: 'dsh-voice__spacer' }),
                row.enFinal ? null : React.createElement('span', { className: 'dsh-voice__badge' }, '识别中'),
                row.enFinal && !row.zhFinal && showZh
                  ? React.createElement('span', { className: 'dsh-voice__badge dsh-voice__badge--tr' }, '翻译中')
                  : null,
              ),
              showEn ? React.createElement('div', { className: 'dsh-voice__en' }, withMatch(row.en || '…')) : null,
              showZh ? React.createElement('div', {
                className: 'dsh-voice__zh',
                'data-pending': row.zhFinal ? '0' : '1',
              },
              row.error && !row.zh
                ? React.createElement('span', { className: 'dsh-voice__zh-error' }, `⚠ 翻译失败：${row.error}`)
                : (row.zh || (row.enFinal ? (row.zhFinal ? '（无语音）' : '译中…') : '翻译中…'))) : null,
              React.createElement('div', {
                className: 'dsh-voice__rowmeta',
                'data-error': row.error ? '1' : '0',
              }, [
                row.enFinal ? '最终' : '实时',
                row.asr ? `ASR ${row.asr}ms` : null,
                row.tr ? `译 ${row.tr}ms` : null,
                row.error,
              ].filter(Boolean).join(' · ')),
              )
            }))

      const banner = notice
        ? React.createElement('div', { className: 'dsh-voice__banner' },
            React.createElement('div', null, notice),
            React.createElement('div', null, '启动本地服务：',
              React.createElement('code', null, 'dsh-voice serve'),
              '（或在对话中让我启动它）。'),
            detail ? React.createElement('div', null, detail) : null,
            state === 'error'
              ? React.createElement('button', {
                  className: 'dsh-voice__btn dsh-voice__btn--tiny',
                  style: { marginTop: '8px' },
                  onClick: toggle,
                }, '重试')
              : null,
          )
        : null

      const footer = rows.length
        ? React.createElement('div', { className: 'dsh-voice__footer' },
            React.createElement('span', { className: 'dsh-voice__privacy' },
              '音频与识别在本机完成；仅英文片段会发送给 DeepSeek 翻译。'),
            React.createElement('span', { className: 'dsh-voice__spacer' }),
            React.createElement('span', null, `${rows.length} 段`),
            startedAt ? React.createElement('span', null, `时长 ${formatClock(elapsed)}`) : null,
            failedTranslations ? React.createElement('span', { className: 'dsh-voice__footer-warn' }, `翻译失败 ${failedTranslations}`) : null,
            !busy && meeting && onOpenRecords
              ? React.createElement('button', {
                  className: 'dsh-voice__btn dsh-voice__btn--tiny',
                  onClick: onOpenRecords,
                }, '查看会议记录')
              : null,
          )
        : null

      const followPill = rows.length && !prefs.follow
        ? React.createElement('button', { className: 'dsh-voice__followpill', onClick: gotoLatest }, '↓ 回到最新')
        : null

      const truncated = rows.length >= MAX_ROWS
      const truncNote = truncated
        ? React.createElement('div', { className: 'dsh-voice__truncnote' },
            React.createElement('span', null, `仅显示最近 ${MAX_ROWS} 句，更早的句子在会议存档里。`),
            onOpenRecords
              ? React.createElement('button', { className: 'dsh-voice__btn dsh-voice__btn--tiny', onClick: onOpenRecords }, '查看会议记录')
              : null)
        : null

      const lastFinalZh = rows.length
        ? [...rows].reverse().find((row) => row.enFinal && row.zh)
        : null

      return React.createElement('div', { className: 'dsh-voice', 'data-font': prefs.fontSize },
        React.createElement('div', { className: 'dsh-voice__head' },
          header, settingsRow, searchRow, meta, titleRow),
        banner,
        React.createElement('div', {
          className: 'dsh-voice__log', ref: logRef, onScroll,
          role: 'log', 'aria-live': 'off', 'aria-label': '实时字幕',
        }, truncated ? [truncNote, body] : body),
        React.createElement('div', {
          className: 'dsh-voice__sr', 'aria-live': 'polite', 'aria-atomic': 'true',
        }, lastFinalZh ? lastFinalZh.zh : ''),
        followPill,
        footer,
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
        }, React.createElement(LivePanel, { onArchived, onOpenRecords: () => setMode('meetings') })),
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
      loadPrefs, savePrefs, formatClock, splitMatch, rowsToMarkdown, rowsToSrt,
      DEFAULT_PREFS,
      // The live session is app-scoped state, so the render harness can drive it
      // without mounting a component — which is exactly the property that keeps
      // a recording alive across navigation.
      voiceSession, sessionStart, sessionStop, sessionDispose, sessionApplyEvent,
      sessionSubscribe, sessionGetSnapshot, sessionPublish, sessionClearRows,
      LiveClient, LivePanel,
    }
    return module.exports
  },
})

// Hot reload: client-hmr stat-polls this file every 500ms and swaps the module
// in an open page without a refresh. See the README section on hot reload.
