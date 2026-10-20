/**
 * corpus-audit —— 无标注语料的 **license/PII/去泄漏** 单一实现。
 *
 * ## 为什么单独成模块（而不是写在插件 core 里、也不是写在 step0 脚本里）
 * Step 0 的探测脚本与插件的 `prepare` 必须用**同一套规则**：
 * 两份实现一旦分叉，"探测时合格"就不保证"生产时合格"——
 * 这正是本项目反复吃亏的一类缺陷（生产者与校验器各写一套，互相都没验过真东西）。
 * ⇒ 规则只写在这里；`step0/step0-corpus-audit.mjs` 与 `prepare` 都调它。
 *
 * ## 两条口径（同时是 `training-pretrain-contract` §5 未决项的答案）
 * **PII**：扫 100% 的待训练文档（**不抽样**）；命中分两层（沿用 training-sft 的 `classifyHit`），
 *         只有"非占位(other)"才算真 PII；处置 = **剔除整个文档**，只记哈希，**不改写正文**。
 * **去泄漏双轨**：轨 1 整文档哈希；轨 2 文档正文**包含**评估项全文（len≥24）。
 *         剔完**再复核一遍**（不是"我剔过了所以没有"）。
 */
import { createReadStream, readFileSync, existsSync, readdirSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { createGunzip } from 'node:zlib'
import { createInterface } from 'node:readline'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { contentHash, assertNoLeakage, piiScan } from '@xiciyang/data-core'

export const CHARS_PER_TOKEN = 4

/* ───────────────────────────────────────────── license（以实际卡片为准） ── */
export const DOLMA_LICENSE = Object.freeze({
  spdx: 'odc-by',
  class: 'permissive-with-attribution',
  requiresAttribution: true,
  commercialUse: true,
  source: 'https://hf-mirror.com/datasets/allenai/dolma',
  evidence: [
    'HF API tags 含 "license:odc-by"（https://hf-mirror.com/api/datasets/allenai/dolma）',
    '卡片正文 frontmatter `license: odc-by` + 正文 "We have updated the license of Dolma to ODC-BY"（/raw/main/README.md）',
    '仓库 LICENSE.md 首行 "# ODC Attribution License (ODC-By)"（/raw/main/LICENSE.md）',
  ],
  attributionText: 'Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research (Soldaini et al., 2024), Allen Institute for AI. Licensed under ODC-BY 1.0.',
})

/* ────────────────────────────────────────────── 评估集（只读，固定） ── */
export function loadJsonl(p) {
  if (!existsSync(p)) return null
  return readFileSync(p, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l))
}

export function loadEvalSets({ datasetsRoot = join(homedir(), '.ai-infra/state/datasets') } = {}) {
  const sets = []
  for (const d of (existsSync(datasetsRoot) ? readdirSync(datasetsRoot) : [])) {
    const p = join(datasetsRoot, d, 'test.jsonl')
    if (existsSync(p)) sets.push({ name: `wikitext-test(${d.includes('c512') ? 'c512' : 'c128'})`, kind: 'lm', file: p, rows: loadJsonl(p) ?? [], keyOf: (r) => r.t })
  }
  const mmlu = process.env.DSH_MMLU_PATH
  if (existsSync(mmlu)) sets.push({ name: 'mmlu-test', kind: 'qa', file: mmlu, rows: loadJsonl(mmlu) ?? [], keyOf: (r) => r.question })
  const held = join(datasetsRoot, 'yahma-alpaca-cleaned__sft__s20260929__n1000/heldout.jsonl')
  if (existsSync(held)) sets.push({ name: 'alpaca-heldout', kind: 'sft', file: held, rows: loadJsonl(held) ?? [], keyOf: (r) => `${String(r.instruction ?? '').trim()}\u0001${String(r.input ?? '').trim()}\u0001${String(r.output ?? '').trim()}` })
  return sets
}

/* ───────────────────────────────────────── PII 分层（沿用 SFT 口径） ── */
const RESERVED_DOMAIN = /@(example\.(com|org|net|edu)|test\.(com|org)|localhost|domain\.(com|org)|email\.com|yourdomain\.com|company\.com|mycompany\.com|site\.com|x\.com|foo\.com|bar\.com|abc\.com)$/i
const RESERVED_IP = /^(10\.|127\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|0\.|255\.|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)/
const PLACEHOLDER_LOCAL = /^(user|test|example|your|name|email|someone|john\.?doe|jane\.?doe|abc|foo|bar|admin|info|contact|hello|me|you|my)@/i

/** 命中分层：返回 `other` 即真 PII（其余为保留域/占位，不算）。 */
export function classifyHit(kind, text) {
  if (kind === 'email') {
    if (RESERVED_DOMAIN.test(text)) return 'reserved-domain'
    if (PLACEHOLDER_LOCAL.test(text)) return 'placeholder-localpart'
    return 'other'
  }
  if (kind === 'ipv4') return RESERVED_IP.test(text) ? 'reserved-ip' : 'other'
  return 'other'
}

/* ─────────────────── 轨 2：行内包含（滚动哈希候选 + 精确复核，无假阴性） ── */
const W = 24
const MUL = 1000003
const POW = (() => { let p = 1; for (let i = 0; i < W - 1; i += 1) p = Math.imul(p, MUL); return p })()

/**
 * 朴素 `for(doc) for(item) doc.includes(item)` 是 O(文档×评估项)：
 * 本机实测 22,968 × 14,870 ≈ 3.4 亿次 includes ⇒ 会跑几小时。故两段式：
 *   ① 评估项取**非重叠 24-gram** 建倒排索引（项被完整包含 ⇒ 其自身每个窗口必在文档中，**无假阴性**）
 *   ② 文档用 int32 滚动哈希扫 24-gram **滑动窗口**（步长 1）得候选
 *   ③ 候选才精确复核（碰撞只多花一点复核，不会漏判）
 */
export function buildProbeIndex(items) {
  const idx = new Map()
  items.forEach((it, i) => {
    const t = it.text
    for (let p = 0; p + W <= t.length; p += W) {
      let h = 0
      for (let k = 0; k < W; k += 1) h = (Math.imul(h, MUL) + t.charCodeAt(p + k)) | 0
      let s = idx.get(h)
      if (!s) { s = new Set(); idx.set(h, s) }
      s.add(i)
    }
  })
  return idx
}

function windowHashes(text) {
  const n = text.length
  if (n < W) return []
  const out = []
  let h = 0
  for (let i = 0; i < W; i += 1) h = (Math.imul(h, MUL) + text.charCodeAt(i)) | 0
  out.push(h)
  for (let i = W; i < n; i += 1) {
    h = (h - Math.imul(text.charCodeAt(i - W), POW)) | 0
    h = (Math.imul(h, MUL) + text.charCodeAt(i)) | 0
    out.push(h)
  }
  return out
}

/** 返回该文档**包含**的评估项下标数组，无则 null。 */
export function containedItems(text, probeIndex, items) {
  if (text.length < W) return null
  const cands = new Set()
  for (const h of windowHashes(text)) {
    const s = probeIndex.get(h)
    if (s) for (const i of s) cands.add(i)
  }
  if (cands.size === 0) return null
  const hits = []
  for (const i of cands) if (text.includes(items[i].text)) hits.push(i)
  return hits.length ? hits : null
}

/* ────────────────────────────────────────────────────── 主流程 ── */
export async function sha256File(p) {
  return new Promise((res, rej) => {
    const h = createHash('sha256'); const s = createReadStream(p)
    s.on('data', (c) => h.update(c)); s.on('error', rej); s.on('end', () => res(`sha256:${h.digest('hex')}`))
  })
}

/**
 * 读分片 → 确定性选择 → PII → 双轨去泄漏 → 复核。
 * **只读**：不写任何东西；返回选定与剔除明细（调用方决定落盘什么）。
 *
 * @param {{shardPath:string, targetTokens:number, evalSets?:Array, log?:Function}} o
 */
export async function auditCorpus({ shardPath, targetTokens = 15000000, evalSets = null, log = () => {} }) {
  const t0 = Date.now()
  const shardSha256 = await sha256File(shardPath)

  const all = []
  let totalDocs = 0; let totalChars = 0
  const rl = createInterface({ input: createReadStream(shardPath).pipe(createGunzip()), crlfDelay: Infinity })
  for await (const line of rl) {
    const s = line.trim(); if (!s) continue
    let r; try { r = JSON.parse(s) } catch { continue }
    totalDocs += 1
    const text = String(r.text ?? '')
    totalChars += text.length
    all.push({ id: r.id ?? null, source: r.source ?? null, text })
  }
  log(`分片共 ${totalDocs} 文档 / ${totalChars} 字符（≈${Math.round(totalChars / CHARS_PER_TOKEN)} token）`)

  // 确定性选择：contentHash 升序取前 N（不用文件顺序 —— Dolma 分片按 source 成块，前缀选择有位置偏斜）
  for (const d of all) d.h = contentHash(d.text)
  all.sort((a, b) => (a.h < b.h ? -1 : a.h > b.h ? 1 : 0))
  const targetChars = targetTokens * CHARS_PER_TOKEN
  const selected = []
  let selChars = 0
  for (const d of all) { if (selChars >= targetChars) break; selected.push(d); selChars += d.text.length }
  log(`选中 ${selected.length} 文档 / ${selChars} 字符（≈${Math.round(selChars / CHARS_PER_TOKEN)} token）`)

  const sets = evalSets ?? loadEvalSets()
  const evalIndex = new Map()
  const evalTextIndex = []
  for (const es of sets) {
    for (const r of es.rows) {
      const t = es.keyOf(r)
      if (typeof t !== 'string' || !t.trim()) continue
      // 文档侧用 contentHash(text)（不 trim）⇒ 两种形态都登记，避免"仅尾随空白不同"漏判
      evalIndex.set(contentHash(t), es.name)
      evalIndex.set(contentHash(t.trim()), es.name)
      const tt = t.trim()
      if (tt.length >= 24) evalTextIndex.push({ set: es.name, text: tt, hash: contentHash(tt) })
    }
  }
  const probeIndex = buildProbeIndex(evalTextIndex)
  log(`评估项 ${evalTextIndex.length} 条（len≥24）· 轨2 倒排 ${probeIndex.size} 桶`)

  const hashHits = []; const piiDocs = []; const containHits = []; const kept = []
  const piiPatternCount = new Map()

  for (let i = 0; i < selected.length; i += 1) {
    const d = selected[i]
    const hitSet = evalIndex.get(d.h)
    if (hitSet) { hashHits.push({ docHash: d.h, collidesWith: hitSet, source: d.source }); continue }

    const scan = piiScan(d.text)
    const nonReserved = []
    for (const f of scan.findings) {
      const raw = d.text.slice(f.index, f.index + f.length)
      const cls = classifyHit(f.pattern, raw)
      piiPatternCount.set(f.pattern, (piiPatternCount.get(f.pattern) ?? 0) + 1)
      if (cls === 'other') nonReserved.push({ pattern: f.pattern, hash: f.sampleHash })
    }
    if (nonReserved.length > 0) { piiDocs.push({ docHash: d.h, source: d.source, findings: nonReserved }); continue }

    const ch = containedItems(d.text, probeIndex, evalTextIndex)
    if (ch) {
      containHits.push({ docHash: d.h, source: d.source, hits: ch.length, sets: [...new Set(ch.map((x) => evalTextIndex[x].set))], itemHashes: ch.slice(0, 3).map((x) => evalTextIndex[x].hash) })
      continue
    }
    kept.push(d)
    if ((i + 1) % 5000 === 0) log(`  扫描 ${i + 1}/${selected.length}（保留 ${kept.length}）`)
  }

  // 复核（对保留集再跑一遍双轨 —— 不是"我剔过了所以没有"）
  const recheckHash = assertNoLeakage(
    kept.map((d) => ({ t: d.text })),
    sets.flatMap((s) => s.rows.map((r) => ({ t: s.keyOf(r) }))),
    { keyOf: (x) => x.t },
  )
  let recheckContain = 0
  for (const d of kept) if (containedItems(d.text, probeIndex, evalTextIndex)) recheckContain += 1

  return {
    shardSha256, totalDocs, totalChars,
    selected, kept,
    exclusions: { hashHits, piiDocs, containHits },
    piiPatternCount: Object.fromEntries(piiPatternCount),
    recheck: { keptDocs: kept.length, hashIntersection: recheckHash.intersection.length, containmentHits: recheckContain, ok: recheckHash.intersection.length === 0 && recheckContain === 0 },
    evalSets: sets.map((s) => ({ name: s.name, kind: s.kind, rows: s.rows.length, file: s.file ?? null })),
    elapsedMs: Date.now() - t0,
  }
}
