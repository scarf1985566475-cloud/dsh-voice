/**
 * Host-half check of the plugin: exports, config handling, and the reuse rule.
 *
 * The Node half decides whether to spawn a Python service, so the behaviour
 * worth pinning is that it *never* fights a server that already answers, that
 * `autoStart: false` disables it entirely, and that a missing interpreter is a
 * warning rather than a crash.
 *
 * Usage:
 *   node tests/plugin_host_test.mjs
 */

import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const PROJECT = resolve(HERE, '..')
const ENTRY = join(PROJECT, 'plugin', 'index.js')

const failures = []
const check = (condition, message) => {
  if (condition) console.log(`  ok   ${message}`)
  else { console.log(`  FAIL ${message}`); failures.push(message) }
}

/** Build a fake client context that records the supervised effect. */
function makeContext() {
  const logs = []
  const effects = []
  return {
    logs,
    effects,
    ctx: {
      logger: {
        info: message => logs.push(`info:${message}`),
        warn: message => logs.push(`warn:${message}`),
      },
      effect: (callback, label) => { effects.push({ callback, label, dispose: callback() }) },
    },
  }
}

const module_ = await import(pathToFileURL(ENTRY).href)

console.log('exports')
check(module_.name === 'voice-interpreter', `declares a plugin name (got ${module_.name})`)
check(typeof module_.apply === 'function', 'exports apply()')

// ── a healthy service must be reused, never replaced ────────────────────────

console.log('\nreuse of a running service')
const realFetch = globalThis.fetch
let probeCount = 0
globalThis.fetch = async () => {
  probeCount += 1
  return { ok: true, json: async () => ({ ok: true, warmup: 'ready' }) }
}

const running = makeContext()
module_.apply(running.ctx, {
  host: '127.0.0.1',
  port: 8768,
  projectDir: PROJECT,
  python: '/nonexistent/python',
})
await new Promise(resolve => setTimeout(resolve, 60))
check(running.effects.length === 1, 'registers exactly one supervised effect')
check(probeCount >= 1, 'probes /health before doing anything else')
check(
  running.logs.some(line => line.includes('reusing')),
  'reuses the listening service instead of spawning a second one',
)
check(
  !running.logs.some(line => line.includes('starting')),
  'does not spawn when the service already answers',
)
check(typeof running.effects[0].dispose === 'function', 'effect returns a disposer')

// ── autoStart: false is a complete opt-out ──────────────────────────────────

console.log('\nautoStart disabled')
const disabled = makeContext()
const probesBefore = probeCount
module_.apply(disabled.ctx, { autoStart: false })
await new Promise(resolve => setTimeout(resolve, 30))
check(disabled.effects.length === 0, 'registers no effect')
check(probeCount === probesBefore, 'does not probe')
check(disabled.logs.some(line => line.includes('autoStart disabled')), 'says so in the log')

// ── a missing interpreter warns instead of throwing ─────────────────────────

console.log('\nmissing interpreter')
globalThis.fetch = async () => { throw new Error('ECONNREFUSED') }
const missing = makeContext()
module_.apply(missing.ctx, { projectDir: PROJECT, python: '/nonexistent/python' })
await new Promise(resolve => setTimeout(resolve, 60))
check(
  missing.logs.some(line => line.includes('interpreter not found')),
  'warns that the interpreter is missing',
)
check(
  !missing.logs.some(line => line.startsWith('warn:voice-interpreter: start failed')),
  'does not throw out of the effect body',
)

// ── absent config warns rather than guessing ────────────────────────────────

console.log('\nunconfigured')
const unconfigured = makeContext()
module_.apply(unconfigured.ctx, {})
await new Promise(resolve => setTimeout(resolve, 60))
check(
  unconfigured.logs.some(line => line.includes('set projectDir and python')),
  'asks for the paths it needs instead of guessing',
)

for (const context of [running, disabled, missing, unconfigured]) {
  for (const effect of context.effects) effect.dispose()
}
globalThis.fetch = realFetch

console.log('')
if (failures.length > 0) {
  console.log(`FAILED (${failures.length}):`)
  for (const failure of failures) console.log(`  - ${failure}`)
  process.exit(1)
}
console.log('host half behaves')
