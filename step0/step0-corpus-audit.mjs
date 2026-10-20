/**
 * step0-corpus-audit —— 长任务 3 · 阶段 2 Step 0（**只读**）的 CLI 壳。
 *
 * 规则实现在 `../src/data/corpus-audit.mjs`（**与插件的 prepare 同一份**）——
 * 探测与生产用同一套规则，避免"探测时合格、生产时不合格"的分叉。
 *
 * 用法：node step0/step0-corpus-audit.mjs [--target-tokens 15000000] [--out DIR]
 */
import { writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { auditCorpus, DOLMA_LICENSE, CHARS_PER_TOKEN } from '../src/data/corpus-audit.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const argv = process.argv.slice(2)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] ? argv[i + 1] : d }

const SHARD = argOf('--shard', process.env.DSH_SHARD_PATH || '')
const TARGET_TOKENS = Number(argOf('--target-tokens', '15000000'))
const OUT = argOf('--out', join(HERE, '..', 'out'))
const log = (m) => process.stderr.write(`[step0] ${m}\n`)

const REJECTED = [
  { repo: 'cerebras/SlimPajama-627B', reason: 'gated：HF API 返回 "Invalid username or password." ⇒ 未授权不可下载', evidence: 'https://hf-mirror.com/api/datasets/cerebras/SlimPajama-627B' },
  { repo: 'togethercomputer/RedPajama-Data-1T', reason: '① 卡片 frontmatter 未声明 license（只有 task_categories/language/pretty_name）⇒ 按"以实际卡片为准"不能凭印象认定 Apache-2.0；② 数据本体不在 HF，只有 urls/*.txt 指向 data.together.xyz，该域实测 403', evidence: 'https://hf-mirror.com/api/datasets/togethercomputer/RedPajama-Data-1T ; HEAD https://data.together.xyz/ → 403' },
  { repo: 'HuggingFaceFW/fineweb', reason: '备选（未被否）：gated=false、tags 含 license:odc-by、HF 上直接有 parquet；本轮选 Dolma 因其同源自带 LICENSE.md 全文与显式 URL 清单', evidence: 'https://hf-mirror.com/api/datasets/HuggingFaceFW/fineweb' },
]

const r = await auditCorpus({ shardPath: SHARD, targetTokens: TARGET_TOKENS, log })

const report = {
  schema: 'ai-infra/pretrain-corpus-audit@1',
  generatedAt: new Date().toISOString(),
  source: {
    repo: 'allenai/dolma', revision: '7f48140530a023e9ea4c5cfb141160922727d4d3',
    sample: 'dolma-v1_6-8B-sample (olmo-data.org)', shard: SHARD, shardSha256: r.shardSha256,
    license: DOLMA_LICENSE,
  },
  rejected: REJECTED,
  shardTotals: { docs: r.totalDocs, chars: r.totalChars, estTokens: Math.round(r.totalChars / CHARS_PER_TOKEN) },
  selection: {
    rule: 'contentHash(text) 升序取前 N 至 token 目标（确定性、无位置偏斜）',
    targetTokens: TARGET_TOKENS, docs: r.selected.length,
    chars: r.selected.reduce((a, d) => a + d.text.length, 0),
    estTokens: Math.round(r.selected.reduce((a, d) => a + d.text.length, 0) / CHARS_PER_TOKEN),
    sources: [...new Set(r.selected.map((d) => d.source))],
  },
  pii: {
    口径: '扫 100% 的待训练文档（不抽样）；命中仅"非占位(other)"才处置；处置=剔除整个文档，仅记哈希',
    scannedDocs: r.selected.length,
    patternHits: r.piiPatternCount,
    droppedDocs: r.exclusions.piiDocs.length,
    droppedRate: `${(100 * r.exclusions.piiDocs.length / r.selected.length).toFixed(3)}%`,
    droppedHashes: r.exclusions.piiDocs.map((x) => x.docHash),
    sampleFindings: r.exclusions.piiDocs.slice(0, 10),
  },
  leakage: {
    track1_docHashVsEvalItems: { dropped: r.exclusions.hashHits.length, detail: r.exclusions.hashHits.slice(0, 10) },
    track2_docContainsEvalItem: { dropped: r.exclusions.containHits.length, checkedItems: undefined, detail: r.exclusions.containHits.slice(0, 10) },
    recheck: r.recheck,
  },
  evalSets: r.evalSets,
  kept: { docs: r.kept.length, chars: r.kept.reduce((a, d) => a + d.text.length, 0), estTokens: Math.round(r.kept.reduce((a, d) => a + d.text.length, 0) / CHARS_PER_TOKEN) },
  verdict: null,
  elapsedMs: r.elapsedMs,
}
// 注意：`checkedItems` 不能留 undefined —— 本包在长任务 2 缺陷 3 上设了闸门，自己先守规矩
report.leakage.track2_docContainsEvalItem.checkedItems = r.evalSets.reduce((a, s) => a + s.rows, 0)
report.verdict = r.recheck.ok && r.kept.length > 1000 ? 'CORPUS_ACCEPTED' : 'CORPUS_REJECTED'

mkdirSync(OUT, { recursive: true })
writeFileSync(join(OUT, 'pretrain-corpus-audit.json'), `${JSON.stringify(report, null, 1)}\n`)
writeFileSync(join(OUT, 'pretrain-corpus-exclusions.json'), `${JSON.stringify({
  piiDroppedHashes: r.exclusions.piiDocs.map((x) => x.docHash),
  hashCollisionDocs: r.exclusions.hashHits.map((x) => x.docHash),
  containmentDocs: r.exclusions.containHits.map((x) => x.docHash),
  note: '仅哈希与模式名，无原文；原始数据集文件未被修改',
}, null, 1)}\n`)

console.log('── 语料审计 ──')
console.log(`源      : ${report.source.repo} @ ${report.source.revision}  (${report.source.sample})`)
console.log(`license : ${DOLMA_LICENSE.spdx} (${DOLMA_LICENSE.class}) 需署名=${DOLMA_LICENSE.requiresAttribution}`)
console.log(`分片    : ${r.totalDocs} 文档 / ≈${report.shardTotals.estTokens} token  sha=${r.shardSha256.slice(0, 24)}…`)
console.log(`选中    : ${report.selection.docs} 文档 / ≈${report.selection.estTokens} token  来源=${report.selection.sources.join(',')}`)
console.log(`PII     : ${JSON.stringify(r.piiPatternCount)} ⇒ 剔除 ${report.pii.droppedDocs}（${report.pii.droppedRate}）`)
console.log(`去泄漏  : 轨1 整文档哈希 ${report.leakage.track1_docHashVsEvalItems.dropped} · 轨2 行内包含 ${report.leakage.track2_docContainsEvalItem.dropped}`)
console.log(`复核    : 保留 ${r.recheck.keptDocs}；哈希交集 ${r.recheck.hashIntersection}；包含 ${r.recheck.containmentHits} ⇒ ok=${r.recheck.ok}`)
console.log(`verdict : ${report.verdict}   (${(r.elapsedMs / 1000).toFixed(1)}s)`)
console.log(`report  -> ${join(OUT, 'pretrain-corpus-audit.json')}`)
