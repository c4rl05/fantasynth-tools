// Media-element seek accuracy per container/codec (collector). Capture method: an AudioWorklet
// records the element's output with its context frame, and (ctx.currentTime, audio.currentTime)
// pairs are logged every ~1 ms. It runs over the files in <workspace>\seektest\ made by
// seek_formats_prep.py, as a format x seek x rep loop in one browser. Worth re-running when Chrome updates.
//
// The workspace follows the same rule as music-events/workspace.py (MUSIC_EVENTS_WORKSPACE, else
// <main checkout>/../../music-events if it exists, else <main checkout>/../music-events). playwright resolves from this folder's own package.json
// (music-events/measure/package.json): run `npm install` in music-events/measure once, then
// `npx playwright install chromium` there unless you use --channel chrome.
//
//   node music-events\measure\seek_formats.mjs --slug track-a [--formats mp3,wav,flac,opus,m4a,w48.wav]
//        [--seeks none,0,0.5,13.7,61.3,120,150,200] [--reps 2] [--secs 4] [--channel chrome] [--out DIR]
//        [--seek-mode paused|playing]
//
// "none" = no seek at all (play from the start). Every other value is a real seek (currentTime = S),
// done while paused after canplaythrough (default) or 2 s into playback (--seek-mode playing).
// Analysis: seek_formats_analyze.py (reference = ffmpeg's decode of the SAME served file).
import http from 'node:http'
import fs from 'node:fs'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const MEASURE_DIR = path.dirname(fileURLToPath(import.meta.url))  // music-events/measure
const TOOL_DIR = path.resolve(MEASURE_DIR, '..')                      // music-events
const TOOLS_REPO = path.resolve(TOOL_DIR, '..')                       // the repo root

// workspace.py's rule: MUSIC_EVENTS_WORKSPACE, else <main checkout>/../../music-events if that
// folder already exists, else <main checkout>/../music-events (a sibling of the clone). The main
// checkout is found through git's common dir, so a worktree resolves to the same place; without
// git it is this checkout. path.dirname of a root is the root, so a root checkout never throws.
function workspaceDir() {
  if (process.env.MUSIC_EVENTS_WORKSPACE) return path.resolve(process.env.MUSIC_EVENTS_WORKSPACE)
  let main = TOOLS_REPO
  try {
    const common = execFileSync('git', ['rev-parse', '--path-format=absolute', '--git-common-dir'],
      { cwd: TOOL_DIR, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim()
    if (common) main = path.dirname(common)
  } catch { /* no git: fall back to this checkout */ }
  const up1 = path.dirname(main)
  const far = path.join(path.dirname(up1), 'music-events')
  if (path.dirname(up1) !== up1 && fs.existsSync(far) && fs.statSync(far).isDirectory()) return far
  return path.join(up1, 'music-events')
}
const SEEKDIR = path.join(workspaceDir(), 'seektest')
const TYPES = { mp3: 'audio/mpeg', wav: 'audio/wav', flac: 'audio/flac', opus: 'audio/ogg', m4a: 'audio/mp4', webm: 'audio/webm' }
const CANPLAY = { mp3: 'audio/mpeg', wav: 'audio/wav', flac: 'audio/flac', opus: 'audio/ogg; codecs="opus"',
                  m4a: 'audio/mp4; codecs="mp4a.40.2"', 'w48.wav': 'audio/wav', webm: 'audio/webm; codecs="opus"' }

const argv = process.argv.slice(2)
const opt = (name, dflt) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : dflt }
const SLUG = opt('--slug', undefined)
if (!SLUG) {
  console.error('pass --slug <slug> (a slug prepared by seek_formats_prep.py in <workspace>/seektest/)')
  process.exit(2)
}
const FORMATS = opt('--formats', 'mp3,wav,flac,opus,m4a,w48.wav').split(',')
const SEEKS = opt('--seeks', 'none,0,0.5,13.7,61.3,120,150,200').split(',').map((s) => (s === 'none' ? null : Number(s)))
const REPS = Number(opt('--reps', '2'))
const SECS = Number(opt('--secs', '4'))
const CHANNEL = opt('--channel', undefined)
const SEEK_MODE = opt('--seek-mode', 'paused')
const OUT = path.resolve(opt('--out', path.join(SEEKDIR, 'runs', `${SLUG}_${CHANNEL || 'chromium'}_${SEEK_MODE}`)))

let chromium
try {
  ({ chromium } = createRequire(path.join(MEASURE_DIR, 'package.json'))('playwright'))
} catch (e) {
  console.error(`playwright not found: run \`npm install\` in ${MEASURE_DIR} (${e.message.split('\n')[0]})`)
  process.exit(2)
}
fs.mkdirSync(OUT, { recursive: true })  // only once playwright is there

const WORKLET = `
class Cap extends AudioWorkletProcessor {
  constructor() {
    super(); this.buf = new Float32Array(128 * 256); this.n = 0; this.f0 = -1; this.on = true
    this.port.onmessage = (e) => { if (e.data === 'stop') { this.on = false; this.flush(); this.port.postMessage('done') } }
  }
  flush() {
    if (this.n) { const d = this.buf.slice(0, this.n); this.port.postMessage({ f0: this.f0, d }, [d.buffer]) }
    this.n = 0; this.f0 = -1
  }
  process(inputs) {
    if (!this.on) return true
    const ch = inputs[0] && inputs[0][0]
    const len = ch ? ch.length : 128
    if (this.f0 >= 0 && currentFrame !== this.f0 + this.n) this.flush()
    if (this.f0 < 0) this.f0 = currentFrame
    if (this.n + len > this.buf.length) { this.flush(); this.f0 = currentFrame }
    if (ch) this.buf.set(ch, this.n); else this.buf.fill(0, this.n, this.n + len)
    this.n += len
    return true
  }
}
registerProcessor('cap', Cap)
`

function serveFile(req, res, file, type) {
  const size = fs.statSync(file).size
  res.setHeader('Accept-Ranges', 'bytes')
  res.setHeader('Content-Type', type)
  res.setHeader('Cache-Control', 'no-store')
  const m = /bytes=(\d*)-(\d*)/.exec(req.headers.range || '')
  if (m) {
    const start = m[1] ? Number(m[1]) : Math.max(0, size - Number(m[2]))
    const end = m[1] && m[2] ? Math.min(Number(m[2]), size - 1) : size - 1
    res.statusCode = 206
    res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`)
    res.setHeader('Content-Length', end - start + 1)
    fs.createReadStream(file, { start, end }).pipe(res)
  } else {
    res.setHeader('Content-Length', size)
    fs.createReadStream(file).pipe(res)
  }
}

const server = http.createServer((req, res) => {
  const u = decodeURIComponent(req.url.split('?')[0])
  let m
  if (u === '/') { res.setHeader('Content-Type', 'text/html'); return res.end('<!doctype html><title>seek</title><body>seek</body>') }
  if (u === '/worklet.js') { res.setHeader('Content-Type', 'text/javascript'); return res.end(WORKLET) }
  if ((m = /^\/f\/([\w.]+)$/.exec(u))) {
    const file = path.join(SEEKDIR, m[1])
    const ext = m[1].split('.').pop()
    if (fs.existsSync(file) && TYPES[ext]) return serveFile(req, res, file, TYPES[ext])
  }
  res.statusCode = 404; res.end('not found')
})
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const BASE = `http://127.0.0.1:${server.address().port}`

const browser = await chromium.launch({
  headless: true,
  channel: CHANNEL,
  args: ['--autoplay-policy=no-user-gesture-required'],
  ignoreDefaultArgs: ['--mute-audio'],
})
const page = await browser.newPage()
page.on('console', (msg) => console.log('  [page]', msg.text()))
await page.goto(BASE + '/')
const version = browser.version()
const canPlay = await page.evaluate((types) => Object.fromEntries(Object.entries(types).map(([k, t]) => [k, new Audio().canPlayType(t)])), CANPLAY)
console.log(`browser ${version} (${CHANNEL || 'bundled chromium'}) headless; out ${OUT}; canPlayType ${JSON.stringify(canPlay)}`)

await page.evaluate(() => {
  window.f32b64 = (f) => {
    const u8 = new Uint8Array(f.buffer, f.byteOffset, f.byteLength)
    let s = ''
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000))
    return btoa(s)
  }
})

const meta = { browser: version, channel: CHANNEL || 'bundled-chromium', slug: SLUG, secs: SECS, seekMode: SEEK_MODE,
               canPlay, runs: {} }
const writeMeta = () => fs.writeFileSync(path.join(OUT, 'meta.json'), JSON.stringify(meta, null, 1))

for (let rep = 0; rep < REPS; rep++) {
  for (const seek of SEEKS) {
    for (const fmt of FORMATS) {
      const file = `${SLUG}.${fmt}`
      const key = `${fmt}__${seek === null ? 'none' : seek}__${rep}`
      let r
      try {
        r = await page.evaluate(async ({ url, secs, seek, seekMode }) => {
          const ctx = new AudioContext()
          await ctx.audioWorklet.addModule('/worklet.js')
          const a = new Audio()
          a.preload = 'auto'
          a.src = url
          await new Promise((res, rej) => {
            const t = setTimeout(() => rej(new Error('load timeout')), 20000)
            a.addEventListener('canplaythrough', () => { clearTimeout(t); res() }, { once: true })
            a.addEventListener('error', () => { clearTimeout(t); rej(new Error('audio error ' + a.error?.code + ' ' + a.error?.message)) }, { once: true })
            a.load()
          })
          const duration = a.duration
          const doSeek = () => new Promise((res) => { a.addEventListener('seeked', res, { once: true }); a.currentTime = seek })
          if (seek !== null && seekMode === 'paused') await doSeek()
          const src = ctx.createMediaElementSource(a)
          const node = new AudioWorkletNode(ctx, 'cap', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] })
          src.connect(node)
          node.connect(ctx.destination)
          const chunks = []
          let done
          const finished = new Promise((r) => { done = r })
          node.port.onmessage = (e) => { if (e.data === 'done') done(); else chunks.push(e.data) }
          await ctx.resume()
          const pairs = []
          const tick = () => pairs.push([ctx.currentTime, a.currentTime, performance.now()])
          const timer = setInterval(tick, 1)
          await a.play()
          let seekedAtCtx = null
          if (seek !== null && seekMode === 'playing') {
            await new Promise((r) => setTimeout(r, 2000))
            seekedAtCtx = ctx.currentTime
            await doSeek()
          }
          await new Promise((r) => setTimeout(r, secs * 1000))
          clearInterval(timer)
          a.pause()
          node.port.postMessage('stop')
          await finished
          chunks.sort((x, y) => x.f0 - y.f0)
          const f0 = chunks[0].f0
          const last = chunks[chunks.length - 1]
          const total = last.f0 + last.d.length - f0
          const all = new Float32Array(total)
          let gaps = 0
          let expect = f0
          for (const c of chunks) { if (c.f0 !== expect) gaps++; all.set(c.d, c.f0 - f0); expect = c.f0 + c.d.length }
          const info = { sampleRate: ctx.sampleRate, baseLatency: ctx.baseLatency, outputLatency: ctx.outputLatency,
                         duration, seekedAtCtx, f0, frames: total, gaps, pairs, capture: window.f32b64(all) }
          src.disconnect(); node.disconnect(); a.src = ''; await ctx.close()
          return info
        }, { url: `${BASE}/f/${file}`, secs: SECS, seek, seekMode: SEEK_MODE })
      } catch (e) {
        console.log(`${key}: FAILED ${e.message.split('\n')[0]}`)
        meta.runs[key] = { error: e.message.split('\n')[0], fmt, seek, rep, file }
        writeMeta()
        continue
      }
      fs.writeFileSync(path.join(OUT, `${key}.f32`), Buffer.from(r.capture, 'base64'))
      delete r.capture
      fs.writeFileSync(path.join(OUT, `${key}.pairs.json`), JSON.stringify(r.pairs))
      r.nPairs = r.pairs.length
      delete r.pairs
      Object.assign(r, { fmt, seek, rep, file })
      meta.runs[key] = r
      writeMeta()
      console.log(`${key}: ctx ${r.sampleRate} Hz, dur ${r.duration}, ${r.frames} frames, ${r.nPairs} pairs, gaps ${r.gaps}`)
    }
  }
}

writeMeta()
await browser.close()
server.close()
console.log('done')
