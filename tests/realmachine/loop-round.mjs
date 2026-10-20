/**
 * loop-round —— 有限闭环验证（长任务 3 · 阶段 6）的**单轮驱动**。
 *
 * ## 定义（题面原文）
 *   候选模型 → 训练（sft/pretrain/rl/qat 按需）→ 部署（serving）→ 推理 → 评估（testing-core）→ 晋级/回退
 *
 * ## 红线（本脚本的硬约束）
 *   - **每轮人工确认晋级** ⇒ 本脚本**永不**把候选标成"已晋级"。它只产出
 *     `promotion: { recommended, reason, evidence, status: 'PENDING_HUMAN_CONFIRMATION' }`。
 *   - **测试集只读**：只读评估集；产物一律写 run 目录。
 *   - **模型血缘可追溯**：记录 baseModel / corpusHash / shardIndex / 续训源 / weightsSha256。
 *   - **≤3 轮**：轮号由调用方给定，脚本拒绝 round > 3。
 *
 * 用法：node tests/realmachine/loop-round.mjs --round 2 --shard 1 --resume-run <RUN> --resume-ckpt <CKPT> [--steps 500]
 */
import { writeFileSync, mkdirSync, readFileSync, existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { train, exportRun, validate, getReport } from '../../src/core.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = process.env.TRAINING_PRETRAIN_ROOT || join(homedir(), '.ai-infra')
const OUT = join(HERE, '..', '..', 'out', 'pretrain-loop-report.json')
const DATASET = join(ROOT, 'state', 'datasets', 'dolma__pretrain__s20260929__n15000000')

const argv = process.argv.slice(2)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] ? argv[i + 1] : d }
const ROUND = Number(argOf('--round', '1'))
const SHARD = Number(argOf('--shard', '0'))
const STEPS = Number(argOf('--steps', '500'))
const RESUME_RUN = argOf('--resume-run', null)
const RESUME_CKPT = argOf('--resume-ckpt', null)
const MIN_DOMAIN_GAIN = Number(argOf('--min-domain-gain', '1.0'))
const MAX_ROUNDS = 3
const log = (m) => process.stderr.write(`[round${ROUND}] ${m}\n`)

if (ROUND > MAX_ROUNDS) { console.error(`拒绝：轮号 ${ROUND} 超过红线上限 ${MAX_ROUNDS}`); process.exit(2) }

const report = (() => { try { return JSON.parse(readFileSync(OUT, 'utf8')) } catch { return { schema: 'ai-infra/pretrain-loop@1', maxRounds: MAX_ROUNDS, rounds: [] } } })()
mkdirSync(dirname(OUT), { recursive: true })
const flush = () => writeFileSync(OUT, `${JSON.stringify(report, null, 1)}\n`)

const round = {
  round: ROUND, startedAt: new Date().toISOString(), shardIndex: SHARD, maxSteps: STEPS,
  resumedFrom: RESUME_RUN ? { runId: RESUME_RUN, checkpoint: RESUME_CKPT } : null,
  thresholds: { minDomainGainPct: MIN_DOMAIN_GAIN, zForgetting: 1.96 },
  lineage: {}, decision: null, error: null,
}
const idx = report.rounds.findIndex((r) => r.round === ROUND)
if (idx >= 0) report.rounds[idx] = round; else report.rounds.push(round)
flush()

let gpu = 0
try {
  /* ── ① 训练（候选） ── */
  log(`① 训练候选（shard ${SHARD}，${STEPS} 步${RESUME_RUN ? `，续训自 ${RESUME_RUN}/${RESUME_CKPT}` : ''}）`)
  const tr = await train({
    root: ROOT, datasetDir: DATASET, shardIndex: SHARD, lr: 2e-5, batchSize: 4, gradAccum: 4,
    maxSeqLen: 512, maxSteps: STEPS, warmupSteps: 50, seed: 20260929 + ROUND,
    checkpointEvery: 50, checkpointKeep: 3, resumeRunId: RESUME_RUN, resumeCheckpoint: RESUME_CKPT, log,
  })
  gpu += Number(tr.gpuHours ?? 0)
  round.train = tr
  round.lineage.mutation = { runId: tr.runId, shardIndex: SHARD, globalStep: tr.globalStep, steps: STEPS, continuityOk: tr.continuity?.ok }
  flush()

  /* ── ② 导出 ── */
  log('② 导出')
  const reg = await getReport({ root: ROOT, runId: tr.runId })
  const ckpt = reg.run?.checkpoint?.name
  const ex = await exportRun({ root: ROOT, runId: tr.runId, checkpointName: ckpt, log })
  round.export = { checkpointName: ckpt, weightsSha256: ex.weightsSha256, params: ex.params, globalStep: ex.globalStep }
  const ds = JSON.parse(readFileSync(join(DATASET, 'shards.json'), 'utf8'))
  round.lineage.frozen = { baseModel: '/data/models/Qwen2.5-0.5B-Instruct', corpusHash: ds.corpusHash, heldoutHash: ds.heldoutHash, chunkChars: ds.chunkChars, shardCount: ds.shardCount }
  flush()

  /* ── ③ 部署 + 推理 + 评估（serving + testing-core） ── */
  log('③ 部署 + 评估（领域门禁 + 遗忘门禁）')
  const va = await validate({
    root: ROOT, runId: tr.runId, deploy: true, nDomain: 200, nLm: 200, nMmlu: 300,
    thresholds: { minDomainGainPct: MIN_DOMAIN_GAIN, zForgetting: 1.96 }, teardown: true, log,
  })
  gpu += Number(((va.ms ?? 0) / 3_600_000).toFixed(4))
  round.evaluation = {
    pass: va.pass,
    domain: { base: va.base.domain.meanNll, cand: va.pt.domain.meanNll, rawPct: va.domainGate.deltas?.domainNll?.rawPct, gatePass: va.domainGate.pass },
    forgetting: { mmluBase: va.base.mmlu.accuracy, mmluCand: va.pt.mmlu.accuracy, mmluPass: va.forgettingGate.mmlu?.pass, lmDelta: va.forgettingGate.lm?.deltaNll, lmBound: va.forgettingGate.lm?.bound, lmPass: va.forgettingGate.lm?.pass },
    ms: va.ms,
  }
  flush()

  /* ── ④ 决策（**只给建议，不晋级**） ── */
  const pass = va.pass === true
  round.decision = {
    recommended: pass ? 'PROMOTE' : 'ROLLBACK',
    reason: pass
      ? `领域门禁通过（NLL ${va.base.domain.meanNll.toFixed(6)} → ${va.pt.domain.meanNll.toFixed(6)}，相对 ${va.domainGate.deltas?.domainNll?.rawPct?.toFixed(2)}% ≥ ${MIN_DOMAIN_GAIN}%）且遗忘门禁通过`
      : `门禁未过：domain=${va.domainGate.pass} forgetting=${va.forgettingGate.pass}`,
    // ⚠️ **绝不自动晋级**：状态恒为待人工确认
    status: 'PENDING_HUMAN_CONFIRMATION',
    decidedBy: null,
    note: '本脚本永不置为已晋级 —— 题面红线：每轮人工确认晋级，不自动晋级',
  }
  // 回退是**安全动作**，可以自动执行：门禁不过则把该 run 标记 rolled-back（不删除产物）
  if (!pass) {
    const { rollback } = await import('../../src/core.mjs')
    const rb = await rollback({ root: ROOT, runId: tr.runId, purge: false, log })
    round.decision.autoRollback = rb
  }
} catch (e) {
  round.error = String(e?.message ?? e).slice(0, 800)
  log(`失败：${round.error}`)
}
round.gpuHours = Number(gpu.toFixed(4))
round.finishedAt = new Date().toISOString()
flush()

console.log(`── 第 ${ROUND} 轮 ──`)
if (round.error) console.log('ERROR:', round.error)
if (round.train) console.log(`train: run=${round.train.runId} steps=${round.train.steps} globalStep=${round.train.globalStep} 连续性=${round.train.continuity?.ok}`)
if (round.export) console.log(`export: weights=${String(round.export.weightsSha256).slice(0, 24)}…`)
if (round.evaluation) {
  const e = round.evaluation
  console.log(`domain: ${e.domain.base.toFixed(6)} → ${e.domain.cand.toFixed(6)} (${e.domain.rawPct?.toFixed(2)}%) pass=${e.domain.gatePass}`)
  console.log(`forget: mmlu ${e.forgetting.mmluBase}→${e.forgetting.mmluCand} (${e.forgetting.mmluPass}) · wikiΔ=${e.forgetting.lmDelta?.toFixed(6)} bound=${e.forgetting.lmBound?.toFixed(6)} (${e.forgetting.lmPass})`)
}
if (round.decision) console.log(`decision: ${round.decision.recommended} · status=${round.decision.status}`)
console.log(`gpuHours: ${round.gpuHours}`)
console.log(`report -> ${OUT}`)
void existsSync
