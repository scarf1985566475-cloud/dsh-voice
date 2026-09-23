/**
 * Render-level check of the browser bundle, without a browser.
 *
 * The panel half of the plugin only ever runs inside the DSH web client, which
 * makes a typo in it invisible until a human opens the tab. This harness loads
 * the real bundle in a Node sandbox with the real React from the DSH profile,
 * captures the module-loader registration, drives the slot registration with a
 * fake client context, and server-renders the component — so a broken bundle,
 * a bad slot option, or a render-time throw fails here instead of in the GUI.
 *
 * Usage:
 *   node tests/plugin_render_test.mjs
 */

import { createRequire } from 'node:module'
import { existsSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const PROJECT = resolve(HERE, '..')
const BUNDLE = join(PROJECT, 'plugin', 'client.js')

/**
 * Resolve the same React instance the web shell seeds into its module table.
 * Anchored at the DSH profile so module identity matches the running app; the
 * profile path follows DSH_HOME, so the harness is not tied to one machine.
 */
function profileRequireFor() {
  const home = process.env.DSH_HOME || join(homedir(), '.dsh-home')
  const anchor = process.env.DSH_PROFILE_PACKAGE || join(home, 'profiles', 'web', 'package.json')
  const require = createRequire(anchor)
  try {
    require.resolve('react')
  } catch {
    throw new Error(
      `cannot resolve react from ${anchor}. Set DSH_HOME to your DSH home directory ` +
      '(the one holding profiles/), or DSH_PROFILE_PACKAGE to a package.json inside the profile.',
    )
  }
  return require
}

const profileRequire = profileRequireFor()

const failures = []
const check = (condition, message) => {
  if (condition) console.log(`  ok   ${message}`)
  else { console.log(`  FAIL ${message}`); failures.push(message) }
}

// ── sandbox ─────────────────────────────────────────────────────────────────

const styleSink = []
const documentStub = {
  head: { appendChild: (element) => styleSink.push(element) },
  createElement: (tag) => ({ tag, setAttribute: () => {}, textContent: '' }),
  querySelectorAll: () => [],
}

let registration = null
const windowStub = {
  __ModuleLoader__: { load: (entry) => { registration = entry } },
  addEventListener: () => {},
}

const storageStub = {
  getItem: () => null,
  setItem: () => {},
}

const source = readFileSync(BUNDLE, 'utf8')
// The bundle is a classic script that touches `window`/`document` at load time;
// evaluating it with the stubs injected is exactly what the shell does.
const load = new Function('window', 'document', 'localStorage', 'navigator', 'WebSocket', 'AudioContext', source)
load(windowStub, documentStub, storageStub, { mediaDevices: {} }, class {}, class {})

console.log('bundle')
check(registration !== null, 'registers itself through window.__ModuleLoader__.load')
check(registration?.id === 'dsh-voice-interpreter', `registers under the package name (got ${registration?.id})`)
check(typeof registration?.factory === 'function', 'exposes a factory')
check(styleSink.length === 0, 'loading the script has no side effects beyond registration')

const requireShim = (specifier) => {
  if (specifier === 'react') return profileRequire('react')
  if (specifier === 'react/jsx-runtime') return profileRequire('react/jsx-runtime')
  throw new Error(`bundle required an undeclared external: ${specifier}`)
}

const exports = registration.factory(requireShim)
check(typeof exports.apply === 'function', 'factory returns apply()')
check(Array.isArray(exports.inject) && exports.inject.includes('slots'), 'declares the slots service')
// Styles are injected by the factory body, so they must appear at materialization.
check(styleSink.length === 1, `injects exactly one stylesheet at materialization (got ${styleSink.length})`)
check(String(styleSink[0]?.textContent ?? '').includes('.dsh-voice'), 'the stylesheet carries the panel rules')

// ── slot registration ───────────────────────────────────────────────────────

console.log('\nslot registration')
let registrationOptions = null
let registeredComponent = null
let injectedKey = null
const registeredEffects = []
const ctx = {
  slots: {
    inject: (key, callback) => { injectedKey = key; callback() },
    register: (options, component) => { registrationOptions = options; registeredComponent = component },
  },
  effect: (callback, label) => { registeredEffects.push({ dispose: callback(), label }) },
}
exports.apply(ctx)
check(injectedKey === 'conversation.view', `injects into conversation.view (got ${injectedKey})`)
check(registrationOptions?.id === 'interpreter', 'registers a stable entry id')
check(typeof registrationOptions?.label === 'function' && registrationOptions.label().length > 0,
  'supplies a tab label')
check(typeof registrationOptions?.order === 'number', 'supplies a tab order')
check(typeof registeredComponent === 'function', 'registers a component')
// The session outlives the panel, so its teardown must be tied to the plugin's
// lifetime rather than to any component's.
check(registeredEffects.length === 1 && typeof registeredEffects[0].dispose === 'function',
  'registers one plugin-lifetime effect to release the microphone')

// ── render ──────────────────────────────────────────────────────────────────

console.log('\nrender')
const React = profileRequire('react')
const { renderToStaticMarkup } = profileRequire('react-dom/server')
let markup = ''
try {
  markup = renderToStaticMarkup(React.createElement(registeredComponent, {}))
} catch (error) {
  failures.push(`render threw: ${error.message}`)
}
check(markup.length > 0, 'renders without throwing')
for (const expected of ['同声传译', '开始收听', '预览也翻译', 'dsh-voice', '连接']) {
  check(markup.includes(expected), `renders "${expected}"`)
}
check(markup.includes('dsh-voice__log'), 'renders the caption log container')

// The idle state must tell the user what to do, not sit empty.
check(markup.includes('点击'), 'idle state explains how to start')

console.log('\ntabs')
for (const expected of ['实时', '会议记录', '还没有会议记录']) {
  check(markup.includes(expected), `renders the "${expected}" pane`)
}

// ── minutes rendering ───────────────────────────────────────────────────────

console.log('\nminutes renderer')
const { renderMinutes, formatDuration, formatStamp } = exports.__internal
const SAMPLE_MINUTES = [
  '# Q3 复盘会',
  '',
  '## 概览',
  '',
  '本次会议回顾第三季度业绩。',
  '',
  '## 决定事项',
  '',
  '- 把迁移推迟到 11 月。',
  '',
  '## 待办事项',
  '',
  '| 事项 | 负责人 | 时间点 |',
  '| --- | --- | --- |',
  '| 准备供应商对比 | Sarah | 下周五之前 |',
  '',
  '---',
  '',
  '*记录：4 段*',
].join('\n')

let minutesMarkup = ''
try {
  minutesMarkup = renderToStaticMarkup(React.createElement('div', null, renderMinutes(SAMPLE_MINUTES)))
} catch (error) {
  failures.push(`minutes render threw: ${error.message}`)
}
check(minutesMarkup.includes('<h1'), 'renders the title as a heading')
check(minutesMarkup.includes('<h2'), 'renders sections as headings')
check(minutesMarkup.includes('<ul') && minutesMarkup.includes('<li'), 'renders bullet lists')
check(minutesMarkup.includes('<table') && minutesMarkup.includes('<th'), 'renders the action table')
check(minutesMarkup.includes('<hr'), 'renders the horizontal rule')
check(!minutesMarkup.includes('| --- |'), 'swallows the table separator row')
check(minutesMarkup.includes('准备供应商对比') && minutesMarkup.includes('Sarah'), 'renders action-item cells')
check(minutesMarkup.includes('把迁移推迟到 11 月。'), 'renders decision bullets verbatim')

// Markdown must be rendered as elements, never injected as HTML.
const HOSTILE = '# 标题\n\n<script>alert(1)</script>\n\n<img src=x onerror=alert(1)>\n\n**加粗**'
const hostileMarkup = renderToStaticMarkup(React.createElement('div', null, renderMinutes(HOSTILE)))
check(!hostileMarkup.includes('<script'), 'never emits a script element from summary text')
check(!hostileMarkup.includes('<img'), 'never emits an img element from summary text')
check(hostileMarkup.includes('&lt;script&gt;'), 'escapes markup in summary text')
check(hostileMarkup.includes('<strong>加粗</strong>'), 'renders paired bold markers')

console.log('\nformatters')
check(formatDuration(45) === '45秒', `formats sub-minute durations (${formatDuration(45)})`)
check(formatDuration(125) === '2分05秒', `formats minute durations (${formatDuration(125)})`)
check(formatDuration(3725) === '1小时02分', `formats hour durations (${formatDuration(3725)})`)
check(formatStamp('2026-09-23T07:00:00Z').includes('09-2'), 'formats an ISO timestamp')
check(formatStamp('not-a-date') === 'not-a-date', 'passes an unparseable stamp through')

console.log('')
if (failures.length > 0) {
  console.log(`FAILED (${failures.length}):`)
  for (const failure of failures) console.log(`  - ${failure}`)
  process.exit(1)
}
console.log(`browser bundle is sound (${markup.length} markup chars)`)
