/**
 * risk.spec —— 风险声明**双向一致**：`src/index.mjs` 的 TOOL_RISK ⇄ `manifest.yaml` 的 capabilities。
 * 两处写不一致是"声明面与实现面分叉"，必须由测试锁死。
 */
import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { TOOL_RISK, TOOL_NAMES } from '../../src/index.mjs'

export const NAME = 'unit/risk'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')

/** 极简 YAML 解析（只认本包 manifest 的形状，避免引入依赖）。 */
export function parseCapabilities(text) {
  const out = []
  let cur = null
  let inCaps = false
  for (const raw of text.split('\n')) {
    const line = raw.replace(/\s+$/, '')
    if (/^capabilities:\s*$/.test(line)) { inCaps = true; continue }
    if (/^[a-z_]+:\s*$/.test(line) && !/^\s/.test(line)) { if (inCaps && !/^capabilities/.test(line)) inCaps = false }
    if (!inCaps) continue
    const m = /^\s*-\s*id:\s*(\S+)/.exec(line)
    if (m) { cur = { id: m[1] }; out.push(cur); continue }
    if (!cur) continue
    const kv = /^\s+(\w+):\s*(.+?)\s*$/.exec(line)
    if (!kv) continue
    let [, k, v] = kv
    if (k === 'resources') v = v.replace(/^\[|\]$/g, '').split(',').map((s) => s.trim().replace(/^'|'$/g, '')).filter(Boolean)
    cur[k] = v
  }
  return out
}

export async function run(t) {
  const manifest = readFileSync(join(PKG, 'manifest.yaml'), 'utf8')
  const caps = parseCapabilities(manifest)
  t.assert(caps.length === 6, `manifest 声明了 6 个 capability（实际 ${caps.length}）`)

  const byTool = new Map(caps.map((c) => [c.tool, c]))
  for (const name of TOOL_NAMES) {
    const c = byTool.get(name)
    t.assert(Boolean(c), `manifest 里有 ${name}`)
    if (!c) continue
    const r = TOOL_RISK[name]
    t.assert(c.risk === r.risk, `${name} risk 一致（manifest=${c.risk} 代码=${r.risk}）`)
    t.assert(c.sandbox === r.sandbox, `${name} sandbox 一致（manifest=${c.sandbox} 代码=${r.sandbox}）`)
    t.assert(String(c.idempotent) === String(r.idempotent), `${name} idempotent 一致`)
    t.assert(c.confirmation === r.confirmation, `${name} confirmation 一致`)
    const mr = (c.resources || []).slice().sort().join(',')
    const cr = (r.resources || []).slice().sort().join(',')
    t.assert(mr === cr, `${name} resources 一致（manifest=[${mr}] 代码=[${cr}]）`)
  }
  t.assert(byTool.size === 6, 'manifest 的 tool 名不重复')

  // not_provided 必须显式声明（白名单语义：未声明 = 拒绝）
  for (const must of ['rl', 'qat', 'quantization', 'distributed_training', 'modify_base_model', 'sanitize_data', 'synthesize_data']) {
    t.assert(new RegExp(`-\\s*id:\\s*${must}\\b`).test(manifest), `not_provided 显式声明了 ${must}`)
  }
}
