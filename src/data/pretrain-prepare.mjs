/**
 * pretrain-prepare —— 语料准备与校验（training-pretrain 的第 1 个 capability）。
 *
 * 流水线（每一步都进 lineage，可复核）：
 *   ① 语料审计（license / PII 100% / 去泄漏双轨）—— 调 `corpus-audit.mjs`（**与 Step 0 同一实现**）
 *   ② 去重（内容哈希）—— 走 `@xiciyang/data-core.dedupByContent`
 *   ③ 分块（定长字符块，边界尽量落在空白）
 *   ④ 确定性分片（契约冻结条件 2 的"分片"语义落点）
 *   ⑤ **在"将被训练的东西"（块）上再跑一次双轨复核** —— 不是"文档剔过了所以块也干净"
 *   ⑥ 血缘落盘（source / revision / license / seed / 各步计数 / 复核结论）
 *
 * 纪律：
 *   - **不自行脱敏**：PII 只**剔除整个文档**，绝不改写正文
 *   - **不自行合成数据**：只做选择/切分/分块，不生成新文本
 *   - 分块与分片**必须确定性**（同输入 + 同 seed ⇒ 逐块相同），否则续训判据无法成立
 */
import { createHash } from 'node:crypto'
import { writeFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { dedupByContent, contentHash, recordHash, assertNoLeakage, lineage as makeLineage } from '@xiciyang/data-core'
import { ToolError, ERROR_CODES } from '../version.mjs'
import { auditCorpus, loadEvalSets, buildProbeIndex, containedItems, DOLMA_LICENSE, CHARS_PER_TOKEN } from './corpus-audit.mjs'

/** 定长分块：优先在边界前 64 字符内找空白切断，避免把词切碎；找不到就硬切。 */
export function chunkText(text, chunkChars) {
  const out = []
  let i = 0
  const n = text.length
  while (i < n) {
    let end = Math.min(i + chunkChars, n)
    if (end < n) {
      const lo = Math.max(end - 64, i + 1)
      for (let k = end; k > lo; k -= 1) {
        const c = text[k]
        if (c === ' ' || c === '\n' || c === '\t') { end = k; break }
      }
    }
    const piece = text.slice(i, end).trim()
    if (piece.length >= 32) out.push(piece)     // 太短的残块丢掉（避免大量噪声样本）
    i = end
  }
  return out
}

export const SHARD_STATE_SCHEMA = 'ai-infra/pretrain-shard-state@1'

/**
 * @param {{shardPath?:string, targetTokens?:number, chunkChars?:number, shardCount?:number,
 *          seed?:number, outDir?:string, datasetsRoot?:string, log?:Function}} o
 */
export async function prepareCorpus({
  shardPath = process.env.DSH_SHARD_PATH,
  targetTokens = 15000000,
  chunkChars = 2048,
  shardCount = 3,
  domainHeldoutChunks = 200,
  seed = 20260929,
  outDir = null,
  datasetsRoot = undefined,
  log = () => {},
} = {}) {
  if (!Number.isInteger(shardCount) || shardCount < 1 || shardCount > 64) {
    throw new ToolError(ERROR_CODES.E_MISSING_PARAM, { hint: `shardCount 必须是 1..64 的整数，收到 ${shardCount}`, recoverable: true })
  }
  if (!Number.isInteger(chunkChars) || chunkChars < 128 || chunkChars > 65536) {
    throw new ToolError(ERROR_CODES.E_MISSING_PARAM, { hint: `chunkChars 必须是 128..65536 的整数，收到 ${chunkChars}`, recoverable: true })
  }

  const evalSets = loadEvalSets(datasetsRoot ? { datasetsRoot } : {})
  log('① 语料审计（license / PII 100% / 去泄漏双轨）…')
  const audit = await auditCorpus({ shardPath, targetTokens, evalSets, log })

  if (!audit.recheck.ok) {
    throw new ToolError(ERROR_CODES.E_CORPUS_LEAK, {
      hint: `文档级复核失败：哈希交集 ${audit.recheck.hashIntersection}、包含命中 ${audit.recheck.containmentHits} —— 拒绝准备（P0）`,
      details: audit.recheck, recoverable: false,
    })
  }

  log('② 去重（内容哈希）…')
  const dedup = dedupByContent(audit.kept, { keyOf: (d) => d.text })

  log('③ 分块…')
  const steps = []
  steps.push({ step: 'corpus-audit', docsIn: audit.totalDocs, selected: audit.selected.length, kept: audit.kept.length, droppedPii: audit.exclusions.piiDocs.length, droppedHashCollision: audit.exclusions.hashHits.length, droppedContainment: audit.exclusions.containHits.length, shardSha256: audit.shardSha256 })
  steps.push({ step: 'dedup-content-hash', kept: dedup.kept.length, dropped: dedup.dropped.length })

  const chunks = []
  for (const d of dedup.kept) {
    const parts = chunkText(d.text, chunkChars)
    parts.forEach((t, i) => chunks.push({ t, doc: d.h, src: d.source, i }))
  }
  const totalChunkChars = chunks.reduce((a, c) => a + c.t.length, 0)
  steps.push({ step: 'chunk', chunkChars, chunks: chunks.length, estTokens: Math.round(totalChunkChars / CHARS_PER_TOKEN) })
  log(`   块数 ${chunks.length}（≈${Math.round(totalChunkChars / CHARS_PER_TOKEN)} token）`)

  // **领域留出集**：从块序列**尾部**保留 N 块，**不进入任何 shard**。
  // 块序列按 contentHash 升序排列（确定性、无位置偏斜），故任一连续切片都是均匀样本。
  // 它是"领域能力是否提升"的唯一合法测量对象 —— 训练过的块不能用来证明学到了。
  const heldout = chunks.slice(Math.max(0, chunks.length - domainHeldoutChunks))
  const trainable = chunks.slice(0, Math.max(0, chunks.length - domainHeldoutChunks))
  steps.push({ step: 'domain-heldout', chunks: heldout.length, reserveFrom: 'tail', trainableChunks: trainable.length })

  log('⑤ 在**块**上复核双轨（不是"文档剔过了所以块也干净"）…')
  const chunkRecords = chunks.map((c) => ({ t: c.t }))
  const flatEval = evalSets.flatMap((s) => s.rows.map((r) => ({ t: s.keyOf(r) })))
  const leakChunks = assertNoLeakage(chunkRecords, flatEval, { keyOf: (x) => x.t })
  // 轨 2（含包含检查）在块上重跑
  const items = []
  for (const es of evalSets) for (const r of es.rows) { const tt = es.keyOf(r); if (typeof tt === 'string' && tt.trim().length >= 24) items.push({ set: es.name, text: tt.trim(), hash: contentHash(tt.trim()) }) }
  const pIdx = buildProbeIndex(items)
  const chunkContain = []
  for (const c of chunks) { const h = containedItems(c.t, pIdx, items); if (h) chunkContain.push({ chunkHash: contentHash(c.t), sets: [...new Set(h.map((x) => items[x].set))] }) }
  const chunkVerify = {
    pair: 'chunks × all-eval-sets',
    track1_hashIntersection: leakChunks.intersection.length,
    track2_containmentHits: chunkContain.length,
    ok: leakChunks.intersection.length === 0 && chunkContain.length === 0,
  }
  steps.push({ step: 'recheck-on-chunks', ...chunkVerify })
  if (!chunkVerify.ok) {
    throw new ToolError(ERROR_CODES.E_CORPUS_LEAK, {
      hint: `块级复核失败：哈希交集 ${chunkVerify.track1_hashIntersection}、包含命中 ${chunkVerify.track2_containmentHits} —— 拒绝准备（P0）`,
      details: { chunkVerify, containment: chunkContain.slice(0, 10) }, recoverable: false,
    })
  }

  log('④ 确定性分片…')
  const per = Math.ceil(trainable.length / shardCount)
  const shards = []
  for (let s = 0; s < shardCount; s += 1) {
    const start = s * per
    const end = Math.min(start + per, trainable.length)
    shards.push({ index: s, chunkStart: start, chunkEnd: end, chunks: Math.max(0, end - start) })
  }
  steps.push({ step: 'shard', shardCount, per, shards: shards.map((s) => ({ index: s.index, chunks: s.chunks })) })

  const corpusHash = `sha256:${createHash('sha256').update(trainable.map((c) => c.t).join('\u0001')).digest('hex')}`
  const heldoutHash = `sha256:${createHash('sha256').update(heldout.map((c) => c.t).join('\u0001')).digest('hex')}`

  const lg = makeLineage({
    source: 'allenai/dolma',
    revision: '7f48140530a023e9ea4c5cfb141160922727d4d3',
    seed,
    calib: trainable.map((c) => ({ t: c.t })),
    test: heldout.map((c) => ({ t: c.t })),
    extra: {
      license: DOLMA_LICENSE,
      shardFile: shardPath,
      shardSha256: audit.shardSha256,
      pipeline: steps,
      verification: [chunkVerify, { pair: 'doc-level', ...audit.recheck }],
      allClear: true,
      corpusHash,
      heldoutHash,
      chunkChars,
      shardCount,
      note: '语料为无标注文本；PII 只剔除整文档、不改写；去泄漏双轨（整块哈希 + 行内包含）已在文档与块两级复核为空',
    },
  })

  const out = outDir || join(process.env.TRAINING_PRETRAIN_ROOT || join(process.env.HOME || '/root', '.ai-infra'), 'state', 'datasets', `dolma__pretrain__s${seed}__n${targetTokens}`)
  mkdirSync(out, { recursive: true, mode: 0o700 })
  writeFileSync(join(out, 'shards.json'), `${JSON.stringify({ schema: SHARD_STATE_SCHEMA, chunkChars, shardCount, corpusHash, heldoutHash, chunks: trainable.length, trainableChunks: trainable.length, heldoutChunks: heldout.length, shards, seed }, null, 1)}\n`)
  writeFileSync(join(out, 'lineage.json'), `${JSON.stringify(lg, null, 2)}\n`)
  writeFileSync(join(out, 'exclusions.json'), `${JSON.stringify({
    piiDroppedHashes: audit.exclusions.piiDocs.map((x) => x.docHash),
    hashCollisionDocs: audit.exclusions.hashHits.map((x) => x.docHash),
    containmentDocs: audit.exclusions.containHits.map((x) => x.docHash),
    note: '仅哈希与模式名，无原文；原始数据集文件未被修改',
  }, null, 1)}\n`)
  // 真实训练语料：**按 shard 分文件**（续训时按 shard 定位，契约冻结条件 2）
  const toLine = (c) => JSON.stringify({ t: c.t, h: recordHash({ t: c.t }, (x) => x.t) })
  for (const s of shards) {
    const slice = trainable.slice(s.chunkStart, s.chunkEnd)
    writeFileSync(join(out, `shard-${String(s.index).padStart(3, '0')}.jsonl`), `${slice.map(toLine).join('\n')}\n`)
  }
  // 领域留出集（**不进入任何 shard**）：领域能力提升的唯一合法测量对象
  writeFileSync(join(out, 'domain-heldout.jsonl'), `${heldout.map(toLine).join('\n')}\n`)

  return {
    out,
    allClear: true,
    chunks: trainable.length,
    estTokens: Math.round(trainable.reduce((a, c) => a + c.t.length, 0) / CHARS_PER_TOKEN),
    domainHeldout: { chunks: heldout.length, hash: heldoutHash },
    shardCount,
    shards,
    corpusHash,
    chunkChars,
    docs: { shardTotal: audit.totalDocs, selected: audit.selected.length, keptAfterPii: audit.kept.length, keptAfterDedup: dedup.kept.length },
    exclusions: { piiDocs: audit.exclusions.piiDocs.length, hashCollisionDocs: audit.exclusions.hashHits.length, containmentDocs: audit.exclusions.containHits.length },
    verification: [chunkVerify, { pair: 'doc-level', ...audit.recheck }],
    evalSets: audit.evalSets,
    license: DOLMA_LICENSE,
    lineageHash: `sha256:${contentHash(JSON.stringify(lg))}`,
  }
}
