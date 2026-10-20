/**
 * train —— 预训练执行（**公共循环走 training-core**，本文件只做适配与编排）。
 *
 * 边界与 training-sft 同形：
 *   training-core 负责：循环、步数/墙钟/GPU 小时上限、checkpoint 编排、审计钩子接口
 *   training-pretrain 负责：语料整理与校验、预训练配置、框架适配、指标抽取、门禁判定
 *
 * ## 本文件的核心：**绝对量断言**（长任务 2 缺陷 2 的正面回答）
 *
 * 缺陷 2 是「续训 20 步实跑 220 步，而两侧步号断言**全对**」——
 * 说明**相对量断言**（"这一步比上一步大 1"）无法证明断言的语义正确。故这里同时断言三组：
 *
 *   A. **相对**：`r.globalStep === globalStepOffset + state.step + 1`（步号不重不跳）
 *   B. **绝对（步↔数据）**：`r.consumedChunks === baseConsumedChunks + (r.globalStep - baseGlobalStep) * chunksPerStep`
 *      —— 这条把"步数"与"数据位置"**锁死**：漏步/重放/游标重置/换语料，四种错误都会破等式
 *   C. **增量**：`r.chunksThisStep === chunksPerStep`（每步恰好消费这么多块）
 *
 * 恢复时另有一组**逐字段**断言（见 `assertResumeContinuity`）：容器报回的
 * `globalStep/consumedChunks/cursor/epoch/shardIndex` 必须与 checkpoint **完全相同**，
 * 否则说明"权重加载了但游标重置了"——那正是缺陷 2 的形态。
 */
import { createTrainingLoop, saveCheckpoint, listCheckpoints, BUDGET_DEFAULTS } from '@xiciyang/training-core'
import { ToolError, ERROR_CODES } from './version.mjs'
import { ContainerWorker, trainingMounts } from './worker.mjs'

/** 恢复连续性：容器报回的状态必须与 checkpoint **逐字段相同**。 */
export function assertResumeContinuity(containerInit, prevState, { corpusHash, shardIndex }) {
  const problems = []
  if (!prevState) return { ok: true, checks: [], note: '全新 run，无 checkpoint 可对' }
  const f = ['globalStep', 'consumedChunks', 'cursor', 'epoch', 'shardIndex']
  const checks = []
  for (const k of f) {
    const want = prevState[k]
    const got = containerInit[k]
    const ok = Number(want) === Number(got)
    checks.push({ field: k, checkpoint: want, container: got, ok })
    if (!ok) problems.push(`${k}: checkpoint=${want} 容器=${got}`)
  }
  // 语料一致性：不允许对着另一份语料续训
  if (prevState.corpusHash && corpusHash && prevState.corpusHash !== corpusHash) {
    problems.push(`corpusHash: checkpoint=${prevState.corpusHash} 当前=${corpusHash}`)
  }
  // 分片推进是**显式**的：要么同分片、要么正好推进一格（不允许跳分片）
  const advancing = Number(shardIndex) === Number(prevState.shardIndex) + 1
  const same = Number(shardIndex) === Number(prevState.shardIndex)
  if (!advancing && !same) problems.push(`shardIndex 非法跳变：checkpoint=${prevState.shardIndex} → 目标=${shardIndex}（只允许同分片或 +1）`)
  checks.push({ field: 'shardTransition', checkpoint: prevState.shardIndex, target: shardIndex, ok: advancing || same })
  return { ok: problems.length === 0, checks, problems, shardAdvance: advancing }
}

/**
 * 构造 training-core 的插件对象（其 `step(batch, state)` 转发给容器 worker）。
 * `batchSource` 只承担**步预算/调度**角色；数据游标在容器内（见 train 顶层注释）。
 */
export function makePretrainPlugin({ worker, runDir, cfg, checkpointDir = null }) {
  const chunksPerStep = cfg.batchSize * cfg.gradAccum
  return {
    id: 'pretrain-fs',
    algorithm: 'pretrain',
    async prepare() {
      const r = await worker.call('init', {
        config: {
          baseModel: `/models/${cfg.baseModelDirName}`,
          shardFile: `/data/shard-${String(cfg.shardIndex).padStart(3, '0')}.jsonl`,
          shardIndex: cfg.shardIndex,
          batchSize: cfg.batchSize,
          gradAccum: cfg.gradAccum,
          maxSeqLen: cfg.maxSeqLen,
          maxSteps: cfg.maxSteps,
          warmupSteps: cfg.warmupSteps,
          lr: cfg.lr,
          weightDecay: cfg.weightDecay ?? 0.0,
          seed: cfg.seed,
          corpusHash: cfg.corpusHash,
          // ⚠️ `prepare()` 是**逐个字段显式转发**的，漏一个就静默丢一个。
          // 实测踩到：漏了 `lrTotalSteps` ⇒ 容器拿不到 ⇒ 回落到本 run 的 maxSteps ⇒ 续训 lr=0 ⇒ 空操作训练。
          // （是容器内新加的"空操作训练"闸门把它抓出来的；否则会再次静默产出原封 checkpoint。）
          lrTotalSteps: cfg.lrTotalSteps,
          globalStepOffset: cfg.globalStepOffset ?? 0,
          resumeFrom: cfg.resumeFrom || undefined,
        },
      }, { timeoutMs: 30 * 60 * 1000 })
      return r
    },
    async step(_batch, state) {
      const r = await worker.call('step')
      // A. 相对：步号不重不跳
      const expected = (cfg.globalStepOffset ?? 0) + state.step + 1
      if (r.globalStep !== expected) {
        throw new ToolError(ERROR_CODES.E_RESUME_CONTINUITY, {
          hint: `步号失步：容器 globalStep=${r.globalStep}，期望 ${expected}（Node step=${state.step} + offset=${cfg.globalStepOffset ?? 0}）`,
          details: { containerGlobalStep: r.globalStep, nodeStep: state.step, offset: cfg.globalStepOffset ?? 0, invariant: r.invariant },
          recoverable: false,
        })
      }
      // B. 绝对（步↔数据）：把步数与数据位置锁死
      const lhs = r.consumedChunks - cfg.baseConsumedChunks
      const rhs = (r.globalStep - cfg.baseGlobalStep) * chunksPerStep
      if (lhs !== rhs) {
        throw new ToolError(ERROR_CODES.E_RESUME_CONTINUITY, {
          hint: `绝对量不变量被破坏：consumedChunksΔ=${lhs} 而 步数Δ×每步块数=${rhs}（步=${r.globalStep}）—— 说明数据位置与步数脱钩（漏步/重放/游标重置）`,
          details: { lhs, rhs, invariant: r.invariant, chunksPerStep },
          recoverable: false,
        })
      }
      // C. 增量：每步恰好消费 chunksPerStep 块
      if (r.chunksThisStep !== chunksPerStep) {
        throw new ToolError(ERROR_CODES.E_RESUME_CONTINUITY, {
          hint: `本步消费 ${r.chunksThisStep} 块，应为 ${chunksPerStep}`,
          details: { chunksThisStep: r.chunksThisStep, chunksPerStep }, recoverable: false,
        })
      }
      return {
        loss: r.loss,
        metrics: { gradNorm: r.metrics?.gradNorm, lr: r.lr, tokens: r.tokens, ms: r.ms, gpuMemGiB: r.gpuMemGiB, epoch: r.epoch, consumedChunks: r.consumedChunks, shardIndex: r.shardIndex },
      }
    },
    async finalize() { return { done: true } },
    async checkpointPayload({ state }) {
      const name = `step-${String(state.step).padStart(6, '0')}`
      const dir = checkpointDir ? `${checkpointDir}/${name}` : `${runDir}/ckpt-${name}`
      const saved = await worker.call('save_adapter', { dir: `/out/${relOut(dir, runDir)}` }, { timeoutMs: 20 * 60 * 1000 })
      // checkpointPayload 里**带上续训状态 + 产物级校验**：
      //   trainerState  → 恢复时逐字段核对数据位置（契约冻结条件 2）
      //   weightsChanged→ 断言"权重真的动了"（阶段 6 抓到的"空操作训练"防线）
      return {
        dir: saved.dir, files: saved.files, step: state.step, epoch: state.epoch, bytes: saved.bytes,
        trainerState: saved.state, invariant: saved.invariant,
        weightsChanged: saved.weightsChanged, lrMaxSeen: saved.lrMaxSeen, lrZeroSteps: saved.lrZeroSteps,
        fingerprint: saved.fingerprint, initFingerprint: saved.initFingerprint,
      }
    },
  }
}

function relOut(dir, runDir) {
  const marker = `${runDir}/`
  return dir.startsWith(marker) ? dir.slice(marker.length) : dir
}

/** 跑一次预训练。返回 training-core 的 run 结果 + checkpoint 清单 + 连续性证据。 */
export async function runPretrain({
  runDir, image, modelDir, datasetDir, cfg, limits, checkpoint,
  onEvent = () => {}, log = () => {}, workerFactory = null, auditHooks = null, extraMounts = [],
}) {
  const worker = workerFactory
    ? workerFactory({ runDir, cfg })
    : new ContainerWorker({
      image,
      mounts: trainingMounts({ modelDir, datasetDir, outDir: runDir, pythonDir: null, extra: extraMounts }),
      entry: '/work/pretrain_trainer.py',
      log,
    }).start()

  const plugin = makePretrainPlugin({ worker, runDir, cfg, checkpointDir: checkpoint?.dir ?? null })
  const gpuClock = { t0: Date.now(), acc: 0 }
  const events = []
  const continuity = []

  const hooks = {
    beforeStep: async (c) => {
      if (c.state.step === 0) gpuClock.t0 = Date.now()
      if (auditHooks?.beforeStep) await auditHooks.beforeStep(c)
    },
    afterStep: async ({ state, out }) => {
      const dt = Date.now() - gpuClock.t0
      gpuClock.t0 = Date.now()
      gpuClock.acc += dt
      state.gpuHours = gpuClock.acc / 3_600_000
      continuity.push({ step: state.step, consumedChunks: out?.metrics?.consumedChunks, epoch: out?.metrics?.epoch, shardIndex: out?.metrics?.shardIndex, ok: true })
      events.push({ type: 'step', step: state.step, loss: out?.loss ?? null, ms: dt, gpuHours: Number(state.gpuHours.toFixed(4)) })
      onEvent({ type: 'step', step: state.step, loss: out?.loss ?? null, gpuHours: state.gpuHours })
      if (state.step % 25 === 0) log(`  step ${state.step} loss=${Number(out?.loss ?? NaN).toFixed(4)} chunks=${out?.metrics?.consumedChunks} gpuHours=${state.gpuHours.toFixed(3)}`)
      if (auditHooks?.afterStep) await auditHooks.afterStep({ state, out })
    },
  }

  let loop
  try {
    loop = createTrainingLoop({
      plugin,
      batchSource: (state) => (state.step < limits.maxSteps ? { index: state.step } : null),
      limits, checkpoint, onEvent: (e) => { events.push(e); onEvent(e) }, hooks, epochs: 1,
    })
    const result = await loop.run()
    if (auditHooks?.onStop && result?.stop) await auditHooks.onStop({ state: result.state, stop: result.stop })
    const ckpts = await listCheckpoints({ dir: `${runDir}/checkpoints` })
    return { result, events, checkpoints: ckpts, continuity, worker }
  } finally {
    try { await worker.stop?.() } catch (e) { log(`worker 停止异常（忽略）：${e.message}`) }
  }
}

export { BUDGET_DEFAULTS, saveCheckpoint }
