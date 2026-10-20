/**
 * vertical-loop —— 长任务 3 · 阶段 2 的**真机垂直闭环**（training-pretrain）。
 *
 *   prepare（语料：license/PII/双轨去泄漏/分块/分片）→ train（分片 0，绝对量不变量）
 *   → export（全参数导出为 HF 目录）→ validate（serving 部署 + 领域门禁 + 遗忘门禁）
 *
 * 全程只用**本包已交付的 capability 实现**（src/core.mjs），不自造循环、不改判据。
 * 用法：node tests/realmachine/vertical-loop.mjs [--steps 500] [--shards 0]
 */
import { writeFileSync, mkdirSync, readFileSync, existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { prepare, train, exportRun, validate, getReport } from '../../src/core.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const HOME = homedir()
const ROOT = process.env.TRAINING_PRETRAIN_ROOT || join(HOME, '.ai-infra')
const OUT = join(HERE, '..', '..', 'out', 'pretrain-vertical-loop.json')

const argv = process.argv.slice(2)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] ? argv[i + 1] : d }
const STEPS = Number(argOf('--steps', '500'))
const SHARD = Number(argOf('--shard', '0'))
const TRAIN_TOKENS = Number(argOf('--target-tokens', '15000000'))
const log = (m) => process.stderr.write(`[loop] ${m}\n`)

const t0 = Date.now()
const report = { schema: 'ai-infra/pretrain-vertical-loop@1', startedAt: new Date().toISOString(), steps: STEPS, shard: SHARD }
mkdirSync(dirname(OUT), { recursive: true })
const flush = () => writeFileSync(OUT, `${JSON.stringify(report, null, 1)}\n`)
flush()

let gpuTrain = 0; let gpuValidate = 0
try {
  /* ── ① prepare ── */
  log('① prepare（语料审计 + 分块 + 分片）…')
  const pr = await prepare({ root: ROOT, targetTokens: TRAIN_TOKENS, chunkChars: 2048, shardCount: 3, domainHeldoutChunks: 200, seed: 20260929, log })
  report.prepare = {
    datasetDir: pr.datasetDir, chunks: pr.chunks, estTokens: pr.estTokens, shardCount: pr.shardCount,
    shards: pr.shards, domainHeldout: pr.domainHeldout, corpusHash: pr.corpusHash, chunkChars: pr.chunkChars,
    docs: pr.docs, exclusions: pr.exclusions, verification: pr.verification,
    license: { spdx: pr.license.spdx, class: pr.license.class, requiresAttribution: pr.license.requiresAttribution },
    audit: pr.audit,
  }
  flush()

  /* ── ② train（分片 0） ── */
  log(`② train（shard ${SHARD}，${STEPS} 步）…`)
  const tr = await train({
    root: ROOT, datasetDir: pr.datasetDir, shardIndex: SHARD,
    lr: 2e-5, batchSize: 4, gradAccum: 4, maxSeqLen: 512, maxSteps: STEPS, warmupSteps: 50,
    seed: 20260929, checkpointEvery: 50, checkpointKeep: 3, log,
  })
  gpuTrain += Number(tr.gpuHours ?? 0)
  report.train = tr
  flush()

  /* ── ③ export ── */
  log('③ export（全参数导出）…')
  const reg = await getReport({ root: ROOT, runId: tr.runId })
  const ckptName = reg.run?.checkpoint?.name
  if (!ckptName) throw new Error('注册表里没有 checkpoint.name，无法导出')
  const ex = await exportRun({ root: ROOT, runId: tr.runId, checkpointName: ckptName, log })
  report.export = { checkpointName: ckptName, ...ex }
  flush()

  /* ── ④ validate（阈值由调用方显式给出，不内置默认） ── */
  const thresholds = { minDomainGainPct: Number(argOf('--min-domain-gain', '1.0')), zForgetting: 1.96 }
  log(`④ validate（阈值 ${JSON.stringify(thresholds)}）…`)
  const va = await validate({ root: ROOT, runId: tr.runId, deploy: true, nDomain: 200, nLm: 200, nMmlu: 300, thresholds, teardown: true, log })
  gpuValidate += Number(((va.ms ?? 0) / 3_600_000).toFixed(4))
  report.validate = {
    thresholds, pass: va.pass,
    domainGate: va.domainGate, forgettingGate: va.forgettingGate,
    base: va.base, pt: va.pt, ms: va.ms, audit: va.audit,
  }
  report.status = 'ok'
} catch (e) {
  report.status = 'failed'
  report.error = String(e?.message ?? e).slice(0, 800)
  log(`失败：${report.error}`)
}
report.budget = { gpuHoursTrain: Number(gpuTrain.toFixed(4)), gpuHoursValidate: Number(gpuValidate.toFixed(4)), gpuHoursTotal: Number((gpuTrain + gpuValidate).toFixed(4)) }
report.wallMs = Date.now() - t0
report.finishedAt = new Date().toISOString()
flush()

console.log('── 真机垂直闭环 ──')
console.log(`status: ${report.status}${report.error ? ` (${report.error})` : ''}`)
if (report.prepare) console.log(`prepare: ${report.prepare.chunks} 块 / ${report.prepare.shardCount} 分片 / 领域留出 ${report.prepare.domainHeldout.chunks}；PII 剔 ${report.prepare.exclusions.piiDocs}；轨1=${report.prepare.verification[0].track1_hashIntersection} 轨2=${report.prepare.verification[0].track2_containmentHits}`)
if (report.train) console.log(`train  : run=${report.train.runId} steps=${report.train.steps} globalStep=${report.train.globalStep} gpuHours=${report.train.gpuHours} 连续性=${report.train.continuity?.ok}`)
if (report.export) console.log(`export : params=${report.export.params} weights=${String(report.export.weightsSha256).slice(0, 24)}…`)
if (report.validate) {
  const v = report.validate
  console.log(`domain : base=${v.base.domain.meanNll?.toFixed(6)} pt=${v.pt.domain.meanNll?.toFixed(6)} gate=${v.domainGate.pass}`)
  console.log(`forget : mmlu ${v.base.mmlu.accuracy?.toFixed(4)}→${v.pt.mmlu.accuracy?.toFixed(4)} (${v.forgettingGate.mmlu?.pass}) · wikiNLL Δ=${v.forgettingGate.lm?.deltaNll?.toFixed(6)} bound=${v.forgettingGate.lm?.bound?.toFixed(6)} (${v.forgettingGate.lm?.pass})`)
  console.log(`pass   : ${v.pass}`)
}
console.log(`预算   : train ${report.budget.gpuHoursTrain} + validate ${report.budget.gpuHoursValidate} = ${report.budget.gpuHoursTotal} GPU-h  wall=${(report.wallMs / 60000).toFixed(1)}min`)
console.log(`report -> ${OUT}`)
void existsSync; void readFileSync
