/**
 * Regression test for the bug this refactor exists for: a recording used to die
 * whenever the user looked at another tab or conversation.
 *
 * The session state used to live inside the panel component, so unmounting the
 * panel ran its cleanup and stopped the microphone. It now lives at module
 * scope, owned by the plugin's lifetime. The assertions below are written
 * against that property directly:
 *
 *   * events land in the session with **no component mounted at all**;
 *   * attaching and detaching a view (what mount/unmount does) leaves the
 *     session untouched — this is the regression;
 *   * the panel renders what is already in the session, so returning to the tab
 *     shows the meeting already in progress;
 *   * only an explicit stop, or unloading the plugin, releases the microphone.
 *
 * Everything here is stubbed at the browser boundary (getUserMedia, the audio
 * graph, the WebSocket), so it runs in Node with no browser and no service.
 *
 * Usage:
 *   node tests/plugin_session_test.mjs
 */

import { createRequire } from 'node:module'
import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const PROJECT = resolve(HERE, '..')
const BUNDLE = join(PROJECT, 'plugin', 'client.js')

/** Same React instance the shell seeds; the profile anchor is searched, not hardcoded. */
function profileRequireFor() {
  const candidates = [
    process.env.DSH_PROFILE_PACKAGE,
    process.env.DSH_HOME && join(process.env.DSH_HOME, 'profiles', 'web', 'package.json'),
    join(homedir(), '.dsh-home', 'profiles', 'web', 'package.json'),
    join(PROJECT, 'node_modules', 'package.json'),
  ].filter(Boolean)
  for (const anchor of candidates) {
    const require = createRequire(anchor)
    try {
      require.resolve('react')
    } catch {
      continue
    }
    return require
  }
  throw new Error(
    'cannot resolve react for the browser-side tests. Set DSH_HOME to your DSH home directory '
    + '(the one holding profiles/), or DSH_PROFILE_PACKAGE to a package.json inside a profile. '
    + `Tried:\n  ${candidates.join('\n  ')}`,
  )
}

const profileRequire = profileRequireFor()

const failures = []
const check = (condition, message) => {
  if (condition) console.log(`  ok   ${message}`)
  else { console.log(`  FAIL ${message}`); failures.push(message) }
}

// ── browser stubs ───────────────────────────────────────────────────────────

/** WebSocket stub: connects on the next microtask and records what was sent. */
class FakeWebSocket {
  static OPEN = 1
  static CLOSED = 3
  static instances = []
  constructor(url) {
    this.url = url
    this.readyState = 0
    this.sent = []
    this.listeners = new Map()
    FakeWebSocket.instances.push(this)
    queueMicrotask(() => {
      this.readyState = 1
      if (this.onopen) this.onopen()
    })
  }
  send(data) { this.sent.push(data) }
  close() {
    this.readyState = 3
    if (this.onclose) this.onclose()
  }
  addEventListener(type, handler) {
    const list = this.listeners.get(type) || []
    list.push(handler)
    this.listeners.set(type, list)
  }
  /** Deliver a server frame to both the `onmessage` slot and any listeners. */
  emit(data) {
    const event = { data: JSON.stringify(data) }
    if (this.onmessage) this.onmessage(event)
    for (const handler of this.listeners.get('message') || []) handler(event)
  }
  /** Deliver a control ack such as `flushed`. */
  static last() { return FakeWebSocket.instances[FakeWebSocket.instances.length - 1] }
}

const stoppedTracks = []
const builtGraphs = []

function makeAudioContext() {
  const track = { stop: () => stoppedTracks.push(true) }
  const stream = { getTracks: () => [track] }
  const context = {
    sampleRate: 48000,
    state: 'running',
    destination: {},
    resume: async () => {},
    close: async () => {},
    createMediaStreamSource: () => ({ connect: () => {}, disconnect: () => {} }),
    createScriptProcessor: () => ({ connect: () => {}, disconnect: () => {}, onaudioprocess: null }),
    createGain: () => ({ gain: { value: 1 }, connect: () => {}, disconnect: () => {} }),
  }
  builtGraphs.push(context)
  return context
}

const navigatorStub = {
  mediaDevices: {
    getUserMedia: async () => {
      const track = { stop: () => stoppedTracks.push(true) }
      return { getTracks: () => [track] }
    },
  },
  clipboard: { writeText: async () => {} },
}

const documentStub = {
  head: { appendChild: () => {} },
  createElement: () => ({ setAttribute: () => {}, textContent: '' }),
  querySelectorAll: () => [],
}

let registration = null
const windowStub = {
  __ModuleLoader__: { load: (entry) => { registration = entry } },
  AudioContext: function AudioContextStub() { return makeAudioContext() },
}

const source = readFileSync(BUNDLE, 'utf8')
const load = new Function('window', 'document', 'localStorage', 'navigator', 'WebSocket', 'AudioContext', source)
load(windowStub, documentStub, { getItem: () => null, setItem: () => {} }, navigatorStub, FakeWebSocket, windowStub.AudioContext)

const exports = registration.factory((specifier) => {
  if (specifier === 'react') return profileRequire('react')
  if (specifier === 'react/jsx-runtime') return profileRequire('react/jsx-runtime')
  throw new Error(`undeclared external: ${specifier}`)
})

const {
  voiceSession, sessionStart, sessionStop, sessionDispose, sessionApplyEvent,
  sessionSubscribe, sessionGetSnapshot, sessionClearRows,
} = exports.__internal
const React = profileRequire('react')
const { renderToStaticMarkup } = profileRequire('react-dom/server')

// ── the session exists with nothing mounted ─────────────────────────────────

console.log('session is app-scoped')
check(voiceSession !== undefined, 'the bundle exposes the session store')
check(sessionGetSnapshot().state === 'idle', 'starts idle')

await sessionStart()
check(sessionGetSnapshot().state === 'live', 'starting works with no component mounted')
check(FakeWebSocket.instances.length === 1, 'opened one socket')
check(voiceSession.client !== null, 'the session holds the client')

const socket = FakeWebSocket.last()
socket.emit({ type: 'ready', sample_rate: 16000, settings: { model_ready: true, record: true, api_model: 'deepseek-chat' }, meeting: { id: '20260101-000000-000-abcd', title: '测试会议' } })
check(sessionGetSnapshot().settings?.model_ready === true, 'captures the ready settings')
check(sessionGetSnapshot().meeting?.id === '20260101-000000-000-abcd', 'captures the meeting being recorded')

console.log('\nevents land without a view')
socket.emit({ type: 'partial', id: 1, start: 0, text: 'Good morning every', asr_ms: 200 })
check(sessionGetSnapshot().rows.length === 1, 'a preview creates its row')
check(sessionGetSnapshot().rows[0].en === 'Good morning every', 'preview text is stored')

socket.emit({ type: 'final', id: 1, start: 0, end: 3.2, text: 'Good morning everyone.', asr_ms: 300, stats: { segments: 1, last_asr_ms: 300 } })
socket.emit({ type: 'translation', id: 1, text: '大家早上好。', final: true, ok: true, ms: 400, stats: { segments: 1, last_asr_ms: 300, last_translate_ms: 400 } })
const row = sessionGetSnapshot().rows[0]
check(sessionGetSnapshot().rows.length === 1, 'the commit replaces the preview rather than appending')
check(row.enFinal === true && row.en === 'Good morning everyone.', 'the row is finalized')
check(row.zh === '大家早上好。' && row.zhFinal === true, 'the translation lands on the same row')
check(sessionGetSnapshot().stats?.last_translate_ms === 400, 'stats are tracked')

// ── the regression ──────────────────────────────────────────────────────────

console.log('\nattaching and detaching a view must not stop the recording')
const seen = []
const detach = sessionSubscribe(() => seen.push(sessionGetSnapshot().rows.length))
check(seen.length === 0, 'subscribing alone changes nothing')
sessionApplyEvent({ type: 'partial', id: 2, start: 4, text: 'Revenue grew', asr_ms: 150 })
check(seen.length === 1, 'an attached view is notified of changes')
detach()
sessionApplyEvent({ type: 'partial', id: 2, start: 4, text: 'Revenue grew twelve percent', asr_ms: 180 })
check(seen.length === 1, 'a detached view stops being notified')
check(sessionGetSnapshot().state === 'live', 'the session is still live after the view detached')
check(voiceSession.client !== null, 'the client is still held')
check(socket.readyState === FakeWebSocket.OPEN, 'the socket is still open')
check(sessionGetSnapshot().rows.length === 2, 'captions kept accumulating while nothing was mounted')
check(stoppedTracks.length === 0, 'the microphone was never released')

// ── returning to the tab shows the meeting in progress ──────────────────────

console.log('\nreturning to the tab')
let component = null
exports.apply({
  slots: { inject: (key, cb) => cb(), register: (options, comp) => { component = comp } },
  effect: (callback) => callback() && undefined,
})
check(typeof component === 'function', 'the tab component is registered')
const markup = renderToStaticMarkup(React.createElement(component, { onArchived: () => {} }))
check(markup.includes('Good morning everyone.'), 'the fresh view renders captions recorded before it mounted')
check(markup.includes('大家早上好。'), 'including their translations')
check(markup.includes('测试会议'), 'and the name of the meeting in progress')
check(markup.includes('停止'), 'and offers stop, not start')

// ── explicit stop releases the microphone ───────────────────────────────────

console.log('\nexplicit stop')
const stopPromise = sessionStop()
socket.emit({ type: 'flushed' })
const archived = await stopPromise
check(sessionGetSnapshot().state === 'idle', 'stopping returns the session to idle')
check(voiceSession.client === null, 'the client is released')
check(stoppedTracks.length === 1, 'the microphone track was stopped')
check(archived?.id === '20260101-000000-000-abcd', 'the archived meeting is reported back')
check(sessionGetSnapshot().rows.length === 2, 'the captions remain readable after stopping')

// ── plugin unload releases the microphone too ───────────────────────────────

console.log('\nplugin unload')
await sessionStart()
check(sessionGetSnapshot().state === 'live', 'started again for the unload case')
const cleanups = []
exports.apply({
  slots: { inject: (key, cb) => cb(), register: () => {} },
  effect: (callback, label) => { cleanups.push({ dispose: callback(), label }) },
})
check(cleanups.length === 1, 'apply registers exactly one lifetime effect')
check(typeof cleanups[0].dispose === 'function', 'the effect returns a disposer')
cleanups[0].dispose()
check(voiceSession.client === null, 'unloading the plugin releases the client')
check(sessionGetSnapshot().state === 'idle', 'and leaves the session idle')
check(stoppedTracks.length === 2, 'and hands the microphone back')

console.log('')
if (failures.length > 0) {
  console.log(`FAILED (${failures.length}):`)
  for (const failure of failures) console.log(`  - ${failure}`)
  process.exit(1)
}
console.log('the recording survives navigation')
