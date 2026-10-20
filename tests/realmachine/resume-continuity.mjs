/**
 * resume-continuity —— 真机验证**分片续训的连续性**（契约冻结条件 2；长任务 2 缺陷 2 的正面回答）。
 *
 * 契约原文要求：「从第 N 个 shard 的 checkpoint 恢复后，**步号与数据位置必须连续**（不重不漏），
 * 且该连续性**可机器验证**（不是"看起来接上了"）」。
 *
 * 本脚本设计成**三段真机实验**，每段都断言绝对量：
 *
 *   段 1  train 20 步（shard 0，checkpointEvery=10）
 *         ⇒ 期望 globalStep=20、consumedChunks=20×16=320
 *   段 2  **同分片续训**：从 step-000010 恢复，再跑 15 步
 *         ⇒ 期望 globalStep=25（10+15）、consumedChunks=400（25×16）
 *         ⇒ 且容器 init 时必须报回与 checkpoint **逐字段相同**的 globalStep/consumedChunks/cursor/epoch
 *   段 3  **推进分片**：从 shard 0 的末 checkpoint 恢复，训练 shard 1
 *         ⇒ 权重继承、但本分片数据位置从 0 起、baseGlobalStep 重置为上一段的 globalStep
 *
 * 关键：**只看"步号递增"是不够的** —— 缺陷 2 的两侧步号断言当时全对。
 * 这里的每段都同时核 `consumedChunks == (globalStep − baseGlobalStep) × chunksPerStep`（绝对量）。
 */
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { train, getReport } from '../../src/core.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = process.env.TRAINING_PRETRAIN_ROOT || join(homedir(), '.ai-infra')
const DATASET = join(ROOT, 'state', 'datasets', 'dolma__pretrain__s20260929__n15000000')
const OUT = join(HERE, '..', '..', 'out', 'pretrain-resume-continuity.json')
const log = (m) => process.stderr.write(`[resume] ${m}\n`)

const CP = 16   // batchSize 4 × gradAccum 4
const report = { schema: 'ai-infra/pretrain-resume-continuity@1', startedAt: new Date().toISOString(), chunksPerStep: CP, segments: [] }
mkdirSync(dirname(OUT), { recursive: true })
const flush = () => writeFileSync(OUT, `${JSON.stringify(report, null, 1)}\n`)
flush()

const COMMON = { root: ROOT, datasetDir: DATASET, lr: 2e-5, batchSize: 4, gradAccum: 4, maxSeqLen: 512, warmupSteps: 10, seed: 20260929, checkpointEvery: 10, checkpointKeep: 5 }
let gpu = 0
const t = []

try {
  /* ── 段 1：shard 0，20 步 ── */
  log('段1 train shard0 20 步 …')
  const s1 = await train({ ...COMMON, shardIndex: 0, maxSteps: 20, log })
  gpu += Number(s1.gpuHours ?? 0)
  const exp1 = { globalStep: 20, consumedChunks: 20 * CP }
  const ok1 = s1.globalStep === exp1.globalStep && s1.continuity?.observedConsumedChunks === exp1.consumedChunks
  report.segments.push({ seg: 1, kind: 'fresh-shard0', runId: s1.runId, expect: exp1, got: { globalStep: s1.globalStep, consumedChunks: s1.continuity?.observedConsumedChunks }, continuity: s1.continuity, ok: ok1 })
  t.push(['段1 绝对量（20 步 = 320 块）', ok1])
  flush()

  /* ── 段 2：同分片续训（从 step-000010 恢复，15 步） ── */
  log('段2 同分片续训（step-000010 + 15 步）…')
  const s2 = await train({ ...COMMON, shardIndex: 0, maxSteps: 15, resumeRunId: s1.runId, resumeCheckpoint: 'step-000010', log })
  gpu += Number(s2.gpuHours ?? 0)
  const exp2 = { globalStep: 25, consumedChunks: 25 * CP }
  const ok2 = s2.globalStep === exp2.globalStep && s2.continuity?.observedConsumedChunks === exp2.consumedChunks
  const reg2 = await getReport({ root: ROOT, runId: s2.runId })
  const prev2 = reg2.run?.resumedFrom?.prevState
  const fieldEq = prev2 ? ['globalStep', 'consumedChunks', 'cursor', 'epoch', 'shardIndex'].every((k) => Number(prev2[k]) === Number({ globalStep: 10, consumedChunks: 10 * CP, cursor: (10 * CP) % 13658, epoch: 0, shardIndex: 0 }[k])) : false
  report.segments.push({
    seg: 2, kind: 'same-shard-resume', runId: s2.runId, checkpoint: 'step-000010',
    expect: exp2, got: { globalStep: s2.globalStep, consumedChunks: s2.continuity?.observedConsumedChunks },
    checkpointState: prev2, prevStateMatchesArithmetic: fieldEq, continuity: s2.continuity, ok: ok2,
  })
  t.push(['段2 绝对量（10+15=25 步 ⇒ 400 块，不是"从 0 重新数"）', ok2])
  t.push(['段2 checkpoint 里的步号/数据位置与算术一致（10 步 = 160 块）', fieldEq])
  flush()

  /* ── 段 3：推进分片（shard 0 末 checkpoint → shard 1） ── */
  log('段3 推进分片（shard1，10 步）…')
  // ⚠️ **坑（本脚本第一版就踩了）**：checkpoint 目录名用的是 **Node 侧「本次 run 内」的 state.step**，
  // 不是绝对 globalStep。所以 seg2（15 步、checkpointEvery=10）产出的目录是 `step-000010`，
  // **不是** `step-000015`（也不是全局的 step-000025）。
  // 绝对步号只存在 **`trainer_state.json` 的 `globalStep`** 里 —— 名字是相对量、状态是绝对量。
  // 故这里**从注册表读回真实名字**（不猜），并在报告里显式记录这个命名约定。
  const reg2b = await getReport({ root: ROOT, runId: s2.runId })
  const ckptName = reg2b.run?.checkpoint?.name
  if (!ckptName) throw new Error('seg2 的注册表里没有 checkpoint.name，无法推进分片')
  const lastCkpt = ckptName
  // ⚠️ **第二个坑（本脚本第二版才想清）**：checkpoint 按 `checkpointEvery` 落盘，
  // 所以**最后一个 checkpoint 不一定在 run 的终点**。seg2 跑了 15 步、每 10 步存一次
  // ⇒ 最后一个 checkpoint 在 **本地第 10 步**（globalStep=20），而不是终点 25。
  // 于是从它续训的正确期望是 `20 + 10 = 30`，**不是** `25 + 10 = 35`。
  // 这不是缺陷，而是 checkpoint 语义的必然结果，且必须写进契约：
  //   **「不重不漏」是相对"被恢复的那个 checkpoint"而言的**；
  //   checkpoint 之后、run 结束之前的那段工作**没有落盘**，续训时会**重做**（上界 = checkpointEvery 步）。
  const srcStatePath = join(ROOT, 'run', 'training-pretrain', s2.runId, 'checkpoints', lastCkpt, 'trainer_state.json')
  const srcState = JSON.parse(readFileSync(srcStatePath, 'utf8'))
  const s3 = await train({ ...COMMON, shardIndex: 1, maxSteps: 10, resumeRunId: s2.runId, resumeCheckpoint: lastCkpt, log })
  gpu += Number(s3.gpuHours ?? 0)
  // 推进分片后：globalStep 从**源 checkpoint 的 globalStep** 继续累加；consumedChunks 按新分片从 0 计
  const exp3 = { globalStep: Number(srcState.globalStep) + 10, consumedChunks: 10 * CP }
  const ok3 = s3.globalStep === exp3.globalStep && s3.continuity?.observedConsumedChunks === exp3.consumedChunks
  report.checkpointNamingConvention = {
    note: 'checkpoint 目录名 = Node 侧「本次 run 内」state.step（相对量）；绝对步号在 trainer_state.json 的 globalStep',
    seg2RunId: s2.runId, seg2StepsTaken: s2.steps, seg2RunFinalGlobalStep: s2.globalStep, observedCheckpointName: lastCkpt,
    seg2LastCheckpointGlobalStep: Number(srcState.globalStep),
    naiveGuessWouldBe: `step-${String(s2.steps).padStart(6, '0')}`,
    uncheckpointedTail: {
      note: '「不重不漏」是相对**被恢复的那个 checkpoint** 而言的：checkpoint 之后到 run 结束之间的工作未落盘，续训会重做',
      seg2StepsAfterLastCheckpoint: s2.steps - 10,
      replayUpperBoundSteps: COMMON.checkpointEvery,
    },
  }
  report.segments.push({ seg: 3, kind: 'shard-advance', runId: s3.runId, checkpoint: lastCkpt, shardIndex: 1, expect: exp3, got: { globalStep: s3.globalStep, consumedChunks: s3.continuity?.observedConsumedChunks }, continuity: s3.continuity, ok: ok3 })
  t.push(['段3 推分片后 globalStep 继续累加、consumedChunks 按新分片重置', ok3])
  flush()
} catch (e) {
  report.error = String(e?.message ?? e).slice(0, 800)
  log(`失败：${report.error}`)
}

report.checks = t.map(([name, ok]) => ({ name, ok }))
report.passed = t.every(([, ok]) => ok) && !report.error
report.budget = { gpuHours: Number(gpu.toFixed(4)) }
report.finishedAt = new Date().toISOString()
flush()

console.log('── 分片续训连续性（真机） ──')
for (const s of report.segments) console.log(`seg${s.seg} ${s.kind}: expect globalStep=${s.expect.globalStep} chunks=${s.expect.consumedChunks} | got ${s.got.globalStep}/${s.got.consumedChunks} ⇒ ${s.ok}`)
for (const c of report.checks) console.log(`${c.ok ? 'PASS' : 'FAIL'} ${c.name}`)
if (report.error) console.log('ERROR:', report.error)
console.log(`passed: ${report.passed}   gpuHours: ${report.budget.gpuHours}`)
console.log(`report -> ${OUT}`)
