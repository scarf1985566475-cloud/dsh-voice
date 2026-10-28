/**
 * Headless checks for the browser half of the voice-interpreter plugin.
 *
 * The panel half only runs inside the DSH web client, so a typo there is
 * invisible until a human opens the tab. This harness loads the real bundle
 * with a stub `react` (no profile dependency needed), then exercises the pure
 * helpers and one full render of `LivePanel` with caption rows.
 *
 * Usage:
 *   DSH_DESKTOP_NODE_EXECUTABLE='/Applications/DeepSeek Harness.app/Contents/MacOS/DeepSeek Harness' \
 *     "/Applications/DeepSeek Harness.app/Contents/Resources/runtime/bin/node" \
 *     tests/panel_smoke_test.mjs
 */

import { readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const BUNDLE = resolve(HERE, '..', 'plugin', 'client.js')

const failures = []
const check = (condition, message) => {
  if (condition) console.log(`  ok   ${message}`)
  else { console.log(`  FAIL ${message}`); failures.push(message) }
}

// ── browser stubs ──────────────────────────────────────────────────────────
const styleSink = []
const documentStub = {
  head: { appendChild: (element) => styleSink.push(element) },
  body: { appendChild: () => {}, removeChild: () => {} },
  createElement: () => ({ setAttribute() {}, textContent: '', click() {}, remove() {} }),
  querySelectorAll: () => [],
  documentElement: {},
  scrollingElement: null,
}
const localStorageStub = {
  store: new Map(),
  getItem(key) { return this.store.has(key) ? this.store.get(key) : null },
  setItem(key, value) { this.store.set(key, String(value)) },
}
let entry = null
const windowStub = {
  __ModuleLoader__: { load: (value) => { entry = value } },
  addEventListener: () => {},
  removeEventListener: () => {},
}
const globals = {
  Blob: class Blob {},
  URL: { createObjectURL: () => 'blob:stub', revokeObjectURL: () => {} },
  fetch: async () => ({ ok: true, json: async () => ({}) }),
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  console,
}
new Function('window', 'document', 'localStorage', 'navigator', 'Blob', 'URL', 'fetch', 'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'console', readFileSync(BUNDLE, 'utf8'))(
  windowStub, documentStub, localStorageStub, {}, globals.Blob, globals.URL, globals.fetch,
  globals.setTimeout, globals.clearTimeout, globals.setInterval, globals.clearInterval, console,
)

// Minimal React: enough to server-render the component tree once. Hooks that
// schedule effects are no-ops, so the render is deterministic and side-effect
// free; handlers are never invoked.
const ReactStub = {
  createElement: (type, props, ...children) => ({ type, props: props || {}, children }),
  useState: (initial) => [typeof initial === 'function' ? initial() : initial, () => {}],
  useRef: (initial) => ({ current: initial === undefined ? null : initial }),
  useEffect: () => {},
  useCallback: (callback) => callback,
  useSyncExternalStore: (_subscribe, getSnapshot) => getSnapshot(),
}

const module = entry.factory((id) => {
  if (id === 'react') return ReactStub
  throw new Error(`unexpected require: ${id}`)
})
const internals = module.__internal

// ── pure helpers ───────────────────────────────────────────────────────────
check(internals.formatClock(0) === '00:00', 'formatClock pads zero')
check(internals.formatClock(65) === '01:05', 'formatClock rolls minutes')
check(internals.formatClock(3599) === '59:59', 'formatClock stays under an hour')

const parts = internals.splitMatch('Hello World', 'world')
check(parts.length === 2 && parts[1].hit && parts[1].text === 'World', 'splitMatch marks the hit')

const markdown = internals.rowsToMarkdown([
  { id: 1, start: 3.2, end: 6.7, en: 'Hello', zh: '你好' },
], '会议')
check(markdown.includes('**[00:03]**'), 'markdown carries a clock stamp')
check(markdown.includes('> 你好'), 'markdown keeps the translation as a quote')

const srt = internals.rowsToSrt([{ id: 1, start: 3.2, end: 6.7, en: 'Hello', zh: '你好' }])
check(srt.includes('00:00:03,200 --> 00:00:06,700'), 'srt renders cue timings in order')
check(srt.includes('Hello\n你好'), 'srt keeps both lines in one cue')

localStorageStub.setItem('dsh-voice.preferences', JSON.stringify({ viewMode: 'zh' }))
const prefs = internals.loadPrefs()
check(prefs.viewMode === 'zh' && prefs.follow === true, 'loadPrefs merges stored values over defaults')
internals.savePrefs({ ...prefs, fontSize: 'lg' })
check(JSON.parse(localStorageStub.getItem('dsh-voice.preferences')).fontSize === 'lg', 'savePrefs round-trips')

// ── session event merge ────────────────────────────────────────────────────
const stats = {
  segments: 1, previews: 1, translations: 1, failed_translations: 0,
  dropped_stale: 0, dropped_hallucinations: 0, audio_seconds: 6.7,
  last_asr_ms: 420, last_translate_ms: 900,
}
internals.sessionApplyEvent({ type: 'final', id: 1, start: 0, end: 3.2, text: 'Hello world', language: 'en', asr_ms: 420, stats })
internals.sessionApplyEvent({ type: 'partial', id: 2, start: 3.2, text: 'next one', language: 'zh', asr_ms: 260, stats })
internals.sessionApplyEvent({ type: 'translation', id: 1, final: true, text: '你好世界', ms: 900, ok: true, stats })
internals.voiceSession.settings = {
  model_ready: true, model_repo: 'mlx-community/Qwen3-ASR-0.6B-4bit',
  translation_available: true, api_model: 'deepseek-chat', language: null, record: false,
}
internals.voiceSession.startedAt = Date.now() - 65000
internals.sessionPublish()

const session = internals.sessionGetSnapshot()
check(session.rows.length === 2, 'rows accumulate per span id')
check(session.rows[0].zh === '你好世界' && session.rows[0].language === 'en', 'translation merges into its final row')
check(session.rows[0].end === 3.2, 'final rows keep their end time')
check(session.rows[1].language === 'zh', 'partial rows carry the detected language')

// ── one render of the panel ────────────────────────────────────────────────
const tree = internals.LivePanel({ onArchived() {}, onOpenRecords() {} })
const text = []
const walk = (node) => {
  if (node === null || node === undefined || typeof node === 'boolean') return
  if (typeof node === 'string' || typeof node === 'number') { text.push(String(node)); return }
  if (Array.isArray(node)) { node.forEach(walk); return }
  walk(node.children)
}
walk(tree)
const flat = text.join(' ')
check(flat.includes('同声传译'), 'render shows the panel title')
check(flat.includes('Hello world'), 'render shows the English caption')
check(flat.includes('你好世界'), 'render shows the Chinese translation')
check(flat.includes('EN'), 'render shows the detected-language badge')
check(flat.includes('Qwen3 · 本机'), 'render shows the model chip')
check(flat.includes('开始收听'), 'idle panel offers the start button')

console.log('')
if (failures.length) {
  console.error(`${failures.length} check(s) failed`)
  process.exit(1)
}
console.log('all panel checks passed')
