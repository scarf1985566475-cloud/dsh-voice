/**
 * Node half of the voice-interpreter client plugin.
 *
 * A `dsh.client` row gives this package a host-side entity as well as a browser
 * bundle. The host side earns its place by supervising the local Python service
 * the panel talks to: it probes `/health` at activation, starts `dsh-voice
 * serve` when nothing answers, re-checks on a watchdog interval so a crashed
 * service heals without a DSH restart, and kills only a child it started
 * itself — never an operator's own server.
 *
 * @module dsh-voice-interpreter
 */

import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'

/** Cordis plugin name, used by loader diagnostics. */
export const name = 'voice-interpreter'

/** Health probe budget: the service answers instantly, so a slow reply means "not ours". */
const PROBE_TIMEOUT_MS = 1500

/** How long to wait for a spawned service to become healthy before reporting failure. */
const START_TIMEOUT_MS = 90_000

/** How often the watchdog re-checks a service that is expected to be up. */
const WATCHDOG_INTERVAL_MS = 30_000

/**
 * Resolve one config value with an environment fallback.
 * @param {unknown} value - Explicit config value.
 * @param {string} envName - Environment variable consulted when config is unset.
 * @param {string} fallback - Value used when neither is present.
 * @returns {string} The resolved string.
 */
function pick(value, envName, fallback) {
  if (typeof value === 'string' && value.length > 0) return value
  const fromEnv = process.env[envName]
  if (typeof fromEnv === 'string' && fromEnv.length > 0) return fromEnv
  return fallback
}

/**
 * Probe the service's health endpoint.
 * @param {string} url - Base URL of the service.
 * @param {number} timeoutMs - Per-attempt timeout.
 * @returns {Promise<object|null>} The health payload, or null when unreachable.
 */
async function probe(url, timeoutMs = PROBE_TIMEOUT_MS) {
  try {
    const response = await fetch(`${url}/health`, { signal: AbortSignal.timeout(timeoutMs) })
    if (!response.ok) return null
    const payload = await response.json()
    return payload !== null && typeof payload === 'object' ? payload : null
  } catch {
    return null
  }
}

/**
 * Wait for a freshly spawned service to answer.
 * @param {string} url - Base URL of the service.
 * @param {() => import('node:child_process').ChildProcess | null} current - Reads the live child.
 * @param {number} timeoutMs - Overall budget.
 * @returns {Promise<boolean>} Whether the service became healthy.
 */
async function waitUntilHealthy(url, current, timeoutMs) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const child = current()
    if (child === null || child.exitCode !== null) return false
    if (await probe(url, 1000) !== null) return true
    await new Promise(resolve => setTimeout(resolve, 500))
  }
  return false
}

/**
 * Build the supervisor this fiber owns.
 *
 * @param {object} options - Resolved wiring.
 * @param {object} options.logger - Host logger (may be undefined on a bare context).
 * @param {string} options.url - Base URL of the service.
 * @param {string} options.host - Bind host passed to a spawned service.
 * @param {number} options.port - Bind port passed to a spawned service.
 * @param {string} options.projectDir - `dsh-voice` checkout directory.
 * @param {string} options.python - Interpreter with `dsh_voice` installed.
 * @returns {{ start: () => Promise<boolean>, stop: () => void }} Supervisor handle.
 */
function createSupervisor({ logger, url, host, port, projectDir, python }) {
  /** @type {import('node:child_process').ChildProcess | null} */
  let child = null
  let stopped = false
  let running = false

  const info = message => logger?.info?.(`voice-interpreter: ${message}`)
  const warn = message => logger?.warn?.(`voice-interpreter: ${message}`)

  /**
   * Spawn the service and wait for it to answer.
   * @returns {Promise<boolean>} Whether it became healthy.
   */
  const spawnService = async () => {
    if (!projectDir || !python) {
      warn(
        'set projectDir and python in the plugin config (or DSH_VOICE_PROJECT / DSH_VOICE_PYTHON '
        + 'in the host environment) to let this plugin start the service automatically',
      )
      return false
    }
    if (!existsSync(python)) {
      warn(`interpreter not found at ${python}; start the service manually`)
      return false
    }
    info(`starting ${python} -m dsh_voice.server in ${projectDir}`)
    child = spawn(python, ['-m', 'dsh_voice.server'], {
      cwd: projectDir,
      env: { ...process.env, DSH_VOICE_HOST: host, DSH_VOICE_PORT: String(port) },
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false,
    })
    child.stdout?.on('data', chunk => info(`[out] ${String(chunk).trimEnd()}`))
    child.stderr?.on('data', chunk => info(`[err] ${String(chunk).trimEnd()}`))
    child.on('exit', (code, signal) => {
      if (!stopped) warn(`service exited (code ${code}, signal ${signal})`)
      child = null
    })
    const healthy = await waitUntilHealthy(url, () => child, START_TIMEOUT_MS)
    if (healthy) info(`service ready at ${url}`)
    else warn(`service did not become healthy at ${url} within ${START_TIMEOUT_MS}ms`)
    return healthy
  }

  /**
   * Ensure the service is up. Reuses an already-listening server, never fights one.
   * @returns {Promise<boolean>} Whether the service is healthy after the call.
   */
  const start = async () => {
    if (stopped) return false
    if (running) return true
    running = true
    try {
      if (await probe(url) !== null) {
        info(`reusing the service already listening on ${url}`)
        return true
      }
      if (child !== null && child.exitCode === null) {
        // A previous start attempt is still coming up; the watchdog will re-check.
        return false
      }
      return await spawnService()
    } finally {
      running = false
    }
  }

  return {
    start,
    stop: () => {
      stopped = true
      if (child !== null && child.exitCode === null) child.kill('SIGTERM')
      child = null
    },
  }
}

/**
 * Host-side plugin body: keep the local interpretation service available.
 *
 * @param {import('@deepseek-ai/cordis').Context} ctx - Host context.
 * @param {object} [config] - Row config from `cordis.patch.yml`.
 * @param {string} [config.host] - Service bind host (default `127.0.0.1`).
 * @param {number} [config.port] - Service port (default `8768`).
 * @param {boolean} [config.autoStart] - Supervise the service (default `true`).
 * @param {string} [config.projectDir] - `dsh-voice` checkout directory.
 * @param {string} [config.python] - Interpreter that has `dsh_voice` installed.
 */
export function apply(ctx, config = {}) {
  const host = pick(config.host, 'DSH_VOICE_HOST', '127.0.0.1')
  const port = Number(config.port ?? process.env.DSH_VOICE_PORT ?? 8768)
  const url = `http://${host}:${port}`

  if (config.autoStart === false) {
    ctx.logger?.info?.(`voice-interpreter: autoStart disabled; expecting the service at ${url}`)
    return
  }

  ctx.effect(() => {
    const supervisor = createSupervisor({
      logger: ctx.logger,
      url,
      host,
      port,
      projectDir: pick(config.projectDir, 'DSH_VOICE_PROJECT', ''),
      python: pick(config.python, 'DSH_VOICE_PYTHON', ''),
    })

    supervisor.start().catch(error => ctx.logger?.warn?.(`voice-interpreter: start failed: ${error}`))
    // The watchdog is what makes a crashed service self-heal: without it, the
    // panel would stay dead until DSH restarted.
    const timer = setInterval(() => {
      supervisor.start().catch(error => ctx.logger?.warn?.(`voice-interpreter: watchdog failed: ${error}`))
    }, WATCHDOG_INTERVAL_MS)
    timer.unref?.()

    return () => {
      clearInterval(timer)
      supervisor.stop()
    }
  }, 'voice-interpreter: local service supervisor')
}
