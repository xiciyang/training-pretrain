/**
 * core —— training-pretrain 的 6 个 capability 实现（薄适配层）。
 *
 * 能力→源：
 *   prepare   → @xiciyang/data-core（去重/分块/去泄漏双轨/血缘/PII）+ 本包 corpus-audit
 *   train     → @xiciyang/training-core（公共循环/上限/checkpoint）+ 容器内 transformers（全参数）
 *   validate  → inference-serving（部署）+ @xiciyang/testing-core（判定）
 *   rollback / get_report / health → 状态与审计（state-core / audit-core / sandbox-core）
 */
import { readFileSync, writeFileSync, existsSync, promises as fs } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFileSync } from 'node:child_process'

import { dualGate, relativeChange, CONTRACT_VERSION as TESTING_CORE_VERSION } from '@xiciyang/testing-core'
import { BUDGET_DEFAULTS, enforceLimits, instrumentFromDeployReport, assertInstrumentUsable } from '@xiciyang/training-core'

import { ToolError, ERROR_CODES, PACKAGE_VERSION, SCHEMA_VERSION } from './version.mjs'
import { paths, audit, auditVerify, readRegistry, updateRegistry, sandboxProbe, requireContainerSandbox, digestArgs, mkdirp, nowId, contentHash } from './runtime.mjs'
import { runPretrain } from './train.mjs'
import { evalLM, evalMMLU, evalUniform, forgettingGate, measurementCompleteness, attachIntegrity, EVAL_PROTOCOL } from './evaluate.mjs'
import { resolveServingPath, ServingMcpClient, callWithConfirm } from './serving-client.mjs'
import { prepareCorpus } from './data/pretrain-prepare.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG_ROOT = join(HERE, '..')
const WORKSPACE = process.env.DSH_WORKSPACE || join(PKG_ROOT, '..', '..')
const SCHEMA_DIR = join(process.env.INFERENCE_CONTRACTS_DIR || join(WORKSPACE, 'inference-contracts'), 'state-schema')

export const TRAINING_IMAGE = 'training-sft/toolchain:0.1.0'
export const DEFAULT_LIMITS = Object.freeze({ ...BUDGET_DEFAULTS, maxSteps: 500, maxGpuHours: 2, concurrency: 1, maxParams: 0.5e9 })
const SCHEMA_FILE = join(SCHEMA_DIR, 'pretrain-registry.schema.json')
function schemaPath() { return existsSync(SCHEMA_FILE) ? SCHEMA_FILE : null }
const P = (root) => paths(root)

async function withRun(root, runId, mutator) {
  return updateRegistry(root, (doc) => {
    const i = (doc.runs || []).findIndex((r) => r.id === runId)
    if (i < 0) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
    const runs = [...(doc.runs || [])]
    runs[i] = mutator({ ...runs[i] })
    return { ...doc, runs }
  }, { schemaPath: schemaPath() })
}

function readJsonl(file) {
  return readFileSync(file, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l))
}

/* ══════════════════════════════════════════════════════════ 1. prepare */

export async function prepare({ root = paths().root, shardPath = process.env.DSH_SHARD_PATH, targetTokens = 15000000, chunkChars = 2048, shardCount = 3, domainHeldoutChunks = 200, seed = 20260929, outDir = null, log = () => {} }) {
  const t0 = Date.now()
  await audit(root, { capability: 'prepare', action: 'prepare', phase: 'begin', ...digestArgs({ shardPath, targetTokens, chunkChars, shardCount, domainHeldoutChunks, seed }) })
  const out = await prepareCorpus({ shardPath, targetTokens, chunkChars, shardCount, domainHeldoutChunks, seed, outDir, log })
  if (!out.allClear) {
    await audit(root, { capability: 'prepare', action: 'prepare', phase: 'end', outcome: 'leak-detected', ok: false })
    throw new ToolError(ERROR_CODES.E_CORPUS_LEAK, { hint: `语料校验未通过（allClear=false）—— 拒绝训练`, details: { verification: out.verification }, recoverable: false })
  }
  const r = await audit(root, { capability: 'prepare', action: 'prepare', phase: 'end', outcome: 'ok', chunks: out.chunks, shardCount, ms: Date.now() - t0 })
  log(`语料就绪：${out.chunks} 块 / ${shardCount} 分片 / 领域留出 ${out.domainHeldout.chunks} → ${out.out}`)
  return {
    datasetDir: out.out, chunks: out.chunks, estTokens: out.estTokens, shardCount, shards: out.shards,
    domainHeldout: out.domainHeldout, corpusHash: out.corpusHash, chunkChars,
    docs: out.docs, exclusions: out.exclusions, verification: out.verification,
    evalSets: out.evalSets, license: out.license, lineageHash: out.lineageHash,
    audit: { seq: r.seq, hash: r.hash },
  }
}

/* ══════════════════════════════════════════════════════════ 2. train */

export async function train({
  root = paths().root, baseModel = process.env.DSH_BASE_MODEL, datasetDir, shardIndex = 0,
  lr = 2e-5, weightDecay = 0.0, batchSize = 4, gradAccum = 4, maxSeqLen = 512, maxSteps = 500, warmupSteps = 50,
  seed = 20260929, limits = {}, checkpointEvery = 50, checkpointKeep = 3,
  confirm = null, resumeRunId = null, resumeCheckpoint = null, log = () => {},
}) {
  const p = P(root)
  await mkdirp(p.runsDir)
  const limitsResolved = { ...DEFAULT_LIMITS, ...limits, maxSteps }
  if (limitsResolved.maxSteps > 500) throw new ToolError(ERROR_CODES.E_BUDGET_EXCEEDED, { hint: `maxSteps=${limitsResolved.maxSteps} 超过单次上限 500`, recoverable: true })
  if (limitsResolved.maxGpuHours > 2) throw new ToolError(ERROR_CODES.E_BUDGET_EXCEEDED, { hint: `maxGpuHours=${limitsResolved.maxGpuHours} 超过单次上限 2`, recoverable: true })

  if (!datasetDir) throw new ToolError(ERROR_CODES.E_MISSING_PARAM, { hint: 'datasetDir 必填（先调 training_pretrain_prepare）', recoverable: true })
  const lineageFile = join(datasetDir, 'lineage.json')
  const shardsFile = join(datasetDir, 'shards.json')
  if (!existsSync(lineageFile) || !existsSync(shardsFile)) throw new ToolError(ERROR_CODES.E_CORPUS_INVALID, { hint: `语料目录不完整（缺 lineage.json 或 shards.json）：${datasetDir}`, recoverable: true })
  const lineage = JSON.parse(readFileSync(lineageFile, 'utf8'))
  if (lineage.allClear !== true) throw new ToolError(ERROR_CODES.E_CORPUS_LEAK, { hint: '语料血缘里 allClear !== true —— 拒绝训练', recoverable: false })
  const shardsMeta = JSON.parse(readFileSync(shardsFile, 'utf8'))
  if (shardsMeta.shardCount !== undefined && (shardIndex < 0 || shardIndex >= shardsMeta.shardCount)) {
    throw new ToolError(ERROR_CODES.E_MISSING_PARAM, { hint: `shardIndex=${shardIndex} 越界（共 ${shardsMeta.shardCount} 个分片）`, recoverable: true })
  }
  if (!existsSync(join(baseModel, 'config.json'))) throw new ToolError(ERROR_CODES.E_BASE_MODEL_NOT_FOUND, { hint: `基座模型目录不合法：${baseModel}`, recoverable: true })
  if (!existsSync(join(datasetDir, `shard-${String(shardIndex).padStart(3, '0')}.jsonl`))) throw new ToolError(ERROR_CODES.E_CORPUS_INVALID, { hint: `缺少分片文件 shard-${String(shardIndex).padStart(3, '0')}.jsonl`, recoverable: true })

  const probe = await sandboxProbe()
  requireContainerSandbox(probe)

  // ── 续训准备：从 checkpoint 的 payload 读回**真实**状态（不猜）
  let resumeFromContainer = null
  let prevState = null
  let globalStepOffset = 0
  let baseGlobalStep = 0
  let baseConsumedChunks = 0
  if (resumeRunId && resumeCheckpoint) {
    const srcDir = join(p.runsDir, resumeRunId, 'checkpoints', resumeCheckpoint)
    if (!existsSync(join(srcDir, 'trainer_state.json'))) {
      throw new ToolError(ERROR_CODES.E_RESUME_STATE_MISSING, { hint: `续训源缺 trainer_state.json（无它无法证明数据位置连续）：${srcDir}`, recoverable: false })
    }
    prevState = JSON.parse(readFileSync(join(srcDir, 'trainer_state.json'), 'utf8'))
    resumeFromContainer = `/out-src/checkpoints/${resumeCheckpoint}`
    globalStepOffset = Number(prevState.globalStep ?? 0)
    // 同分片 ⇒ 延续原基线；推进分片 ⇒ 本分片基线重置（与 trainer 的语义逐字对齐）
    if (Number(prevState.shardIndex) === Number(shardIndex)) {
      baseGlobalStep = Number(prevState.baseGlobalStep ?? globalStepOffset)
      baseConsumedChunks = Number(prevState.baseConsumedChunks ?? 0)
    } else {
      baseGlobalStep = globalStepOffset
      baseConsumedChunks = 0
    }
    log(`续训：源=${resumeRunId}/${resumeCheckpoint}（globalStep=${globalStepOffset}，shard ${prevState.shardIndex}→${shardIndex}，基线 baseGlobalStep=${baseGlobalStep} baseConsumedChunks=${baseConsumedChunks}）`)
  }

  const runId = nowId('pt')
  const runDir = join(p.runsDir, runId)
  await mkdirp(runDir)
  const started = new Date().toISOString()
  await audit(root, { capability: 'train', action: 'train', phase: 'begin', runId, baseModel, datasetDir, shardIndex, lr, batchSize, gradAccum, maxSeqLen, maxSteps, limits: limitsResolved, sandboxTier: 'container', image: TRAINING_IMAGE, resumedFrom: resumeFromContainer })

  await updateRegistry(root, (doc) => ({
    ...doc,
    runs: [...(doc.runs || []).filter((r) => r.id !== runId), {
      id: runId, status: 'running', startedAt: started,
      baseModel: { path: baseModel, revision: readRevision(baseModel) },
      dataset: { dir: datasetDir, corpusHash: shardsMeta.corpusHash ?? null, shardCount: shardsMeta.shardCount ?? null, chunkChars: shardsMeta.chunkChars ?? null, lineageHash: `sha256:${contentHash(readFileSync(lineageFile, 'utf8'))}` },
      shardIndex,
      config: { lr, weightDecay, batchSize, gradAccum, maxSeqLen, maxSteps, warmupSteps, seed },
      resumedFrom: resumeFromContainer ? { runId: resumeRunId, checkpoint: resumeCheckpoint, prevState } : null,
      budget: limitsResolved,
      auditSeqs: [],
    }],
  }), { schemaPath: schemaPath() })

  try {
    const { result, events, checkpoints, continuity } = await runPretrain({
      runDir, image: TRAINING_IMAGE, modelDir: baseModel, datasetDir,
      extraMounts: resumeRunId ? [`${join(p.runsDir, resumeRunId)}:/out-src:ro`] : [],
      cfg: {
        baseModelDirName: baseModel.split('/').filter(Boolean).pop(),
        shardIndex, batchSize, gradAccum, maxSeqLen, maxSteps, warmupSteps, lr, weightDecay, seed,
        corpusHash: shardsMeta.corpusHash ?? null,
        // **LR 调度的 total 必须是"绝对终点"**，不是本 run 的 maxSteps。
        // 否则续训时（step 已是 500、total 却是 500）prog 直接到 1、lr=0 ⇒ **500 步全是空操作**
        // 而 globalStep/不变量照过（长任务 3 · 阶段 6 真机抓到的"仪表恒真"第三个实例）。
        // 与 training-sft 的 `lrTotalSteps = offset + 新增步数` 对齐。
        lrTotalSteps: baseGlobalStep + maxSteps,
        globalStepOffset, baseGlobalStep, baseConsumedChunks, resumeFrom: resumeFromContainer,
      },
      limits: limitsResolved,
      checkpoint: { dir: join(runDir, 'checkpoints'), every: checkpointEvery, keep: checkpointKeep },
      onEvent: () => {}, log,
    })

    const last = checkpoints[checkpoints.length - 1] ?? null
    // **产物级绝对量断言**（长任务 3 · 阶段 6 抓到的缺陷）：
    // 计数类断言（globalStep / consumedChunks / 不变量）**全部为真**，也可能**权重一步没动**
    // （实测：lr 恒 0 ⇒ 500 步空操作，三个 checkpoint 与续训源逐字节相同）。
    // ⇒ 必须断言**产物**：本 run 跑了 >0 步时，终点权重指纹必须与起点不同。
    const wc = last?.payload?.weightsChanged
    const lrMax = last?.payload?.lrMaxSeen
    if (result.state.step > 0 && wc === false) {
      throw new ToolError(ERROR_CODES.E_INTERNAL, {
        hint: `空操作训练：本 run 跑了 ${result.state.step} 步，但**权重指纹与起点相同**（lrMaxSeen=${lrMax}）—— 训练没有真正发生。拒绝把它当成"已训练"。`,
        details: { steps: result.state.step, lrMaxSeen: lrMax, fingerprint: last?.payload?.fingerprint, initFingerprint: last?.payload?.initFingerprint },
        recoverable: false,
      })
    }
    // **绝对量结账**：最终 globalStep 必须等于 offset + 本次新增步数（不是"看起来跑完了"）
    const finalGlobalStep = globalStepOffset + result.state.step
    const expectedChunks = baseConsumedChunks + (finalGlobalStep - baseGlobalStep) * (batchSize * gradAccum)
    const lastContinuity = continuity[continuity.length - 1] ?? null
    const absoluteCheck = {
      stepsTaken: result.state.step,
      globalStepOffset,
      finalGlobalStepExpected: finalGlobalStep,
      finalGlobalStepObserved: lastContinuity?.consumedChunks !== undefined ? null : null,
      expectedConsumedChunks: expectedChunks,
      observedConsumedChunks: lastContinuity?.consumedChunks ?? null,
      ok: lastContinuity ? lastContinuity.consumedChunks === expectedChunks : false,
    }
    const ar = await audit(root, { capability: 'train', action: 'train', phase: 'end', runId, outcome: result.stop?.reason ?? 'ok', steps: result.state.step, globalStep: finalGlobalStep, wallMs: result.state.wallMs, gpuHours: Number(result.state.gpuHours.toFixed(4)), finalLoss: result.state.losses.slice(-1)[0] ?? null, checkpoints: checkpoints.length, continuityOk: absoluteCheck.ok })
    await withRun(root, runId, (r) => ({
      ...r, status: 'trained', finishedAt: new Date().toISOString(),
      result: {
        steps: result.state.step, globalStep: finalGlobalStep, wallMs: result.state.wallMs, gpuHours: Number(result.state.gpuHours.toFixed(4)),
        stopReason: result.stop?.reason ?? null, stopChecks: result.stop?.checks ?? [], lossesTail: result.state.losses.slice(-10),
        checkpoints: checkpoints.map((c) => ({ name: c.name, hash: c.hash, step: c.step })),
      },
      continuity: { absoluteCheck, samples: continuity.slice(-5), totalChecked: continuity.length },
      checkpoint: last ? { dir: join(runDir, 'checkpoints'), name: last.name, payloadHash: last.hash, step: last.step, trainerState: last.payload?.trainerState ?? null } : null,
      runDir,
      auditSeqs: [...(r.auditSeqs || []), ar.seq],
    }))
    if (!absoluteCheck.ok) {
      throw new ToolError(ERROR_CODES.E_RESUME_CONTINUITY, { hint: `训练结束的绝对量结账失败：期望 consumedChunks=${expectedChunks}，实测=${lastContinuity?.consumedChunks}`, details: absoluteCheck, recoverable: false })
    }
    return {
      runId, runDir, status: 'trained', shardIndex, steps: result.state.step, globalStep: finalGlobalStep,
      gpuHours: Number(result.state.gpuHours.toFixed(4)), stopReason: result.stop?.reason ?? null,
      checkpoints: checkpoints.length, finalLoss: result.state.losses.slice(-1)[0] ?? null,
      continuity: absoluteCheck, audit: { seq: ar.seq, hash: ar.hash },
    }
  } catch (e) {
    const ar = await audit(root, { capability: 'train', action: 'train', phase: 'failed', runId, error: String(e.message).slice(0, 300) })
    await withRun(root, runId, (r) => ({ ...r, status: 'failed', error: String(e.message).slice(0, 500), auditSeqs: [...(r.auditSeqs || []), ar.seq] })).catch(() => {})
    throw e
  }
}

function readRevision(modelDir) {
  try {
    const m = join(modelDir, '.cache', 'huggingface', 'download', 'config.json.metadata')
    return existsSync(m) ? readFileSync(m, 'utf8').split('\n')[0].trim() : null
  } catch { return null }
}

/* ══════════════════════════════════════════════════════════ 2b. export */

export async function exportRun({ root = paths().root, runId, checkpointName = null, log = () => {} }) {
  const reg = await readRegistry(root, schemaPath())
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  const outDir = join(run.runDir, 'exported')
  const srcDir = checkpointName ? join(run.runDir, 'checkpoints', checkpointName) : null
  if (!srcDir || !existsSync(join(srcDir, 'trainer_state.json'))) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `需要 checkpointName 且该 checkpoint 含 trainer_state.json`, recoverable: true })
  const { ContainerWorker, trainingMounts } = await import('./worker.mjs')
  const worker = new ContainerWorker({
    image: TRAINING_IMAGE,
    mounts: trainingMounts({ modelDir: run.baseModel.path, datasetDir: run.dataset.dir, outDir: run.runDir }),
    entry: '/work/pretrain_trainer.py', log,
  }).start()
  try {
    await worker.call('init', {
      config: {
        baseModel: `/models/${run.baseModel.path.split('/').filter(Boolean).pop()}`,
        shardFile: `/data/shard-${String(run.shardIndex ?? 0).padStart(3, '0')}.jsonl`,
        shardIndex: run.shardIndex ?? 0, batchSize: run.config.batchSize, gradAccum: run.config.gradAccum,
        maxSeqLen: run.config.maxSeqLen, lr: run.config.lr, maxSteps: run.config.maxSteps, seed: run.config.seed,
        corpusHash: run.dataset.corpusHash ?? null,
        resumeFrom: `/out/checkpoints/${checkpointName}`,
      },
    }, { timeoutMs: 30 * 60 * 1000 })
    // **license 必须从血缘里读出来，读不到就响亮失败**（不许给默认值）。
    // 实测踩到：`data-core.lineage()` 把 `extra` **平铺到顶层**（`...extra`），
    // 所以 license 在 `lineage.license`，**不在** `lineage.extra.license`。
    // 第一版读了 `extra.license` ⇒ 拿到 undefined ⇒ 落到 `?? {spdx:'apache-2.0'}` 兜底 ⇒
    // 导出的模型 README 上写着 **apache-2.0**，而语料其实是 **ODC-BY**、署名文本为空。
    // **给 license 兜底默认值是危险动作**：它把"我不知道"变成"我断言它是 apache-2.0"（静默错误），
    // 与「数据集 license 必须以实际卡片为准」这条纪律直接冲突。故这里改为 fail-closed。
    const lineageDoc = JSON.parse(readFileSync(join(run.dataset.dir, 'lineage.json'), 'utf8'))
    const license = lineageDoc?.license
    if (!license || typeof license !== 'object' || !license.spdx) {
      throw new ToolError(ERROR_CODES.E_CORPUS_INVALID, {
        hint: `血缘里没有可用的 license（读的是 lineage.license）：${join(run.dataset.dir, 'lineage.json')}。拒绝导出 —— **不给 license 兜底默认值**，否则会把"不知道"写成"apache-2.0"`,
        details: { lineageKeys: Object.keys(lineageDoc ?? {}), hasTopLevelLicense: Boolean(lineageDoc?.license) },
        recoverable: false,
      })
    }
    const r = await worker.call('export_model', {
      outDir: `/out/exported`,
      license,
      readmeNote: `Continued pretraining on Dolma (ODC-BY) shard ${run.shardIndex}, base Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0).`,
      provenance: { runId, shardIndex: run.shardIndex, baseModel: run.baseModel, dataset: run.dataset, config: run.config, steps: run.result?.steps ?? null, globalStep: run.result?.globalStep ?? null, corpusHash: run.dataset.corpusHash },
    }, { timeoutMs: 30 * 60 * 1000 })
    const ar = await audit(root, { capability: 'export', action: 'export', phase: 'end', runId, outDir, weightsSha256: r.weightsSha256, params: r.params, bytes: r.bytes })
    await withRun(root, runId, (x) => ({ ...x, status: 'exported', exported: { dir: outDir, weightsSha256: r.weightsSha256, params: r.params, bytes: r.bytes, files: r.files.length, globalStep: r.globalStep }, auditSeqs: [...(x.auditSeqs || []), ar.seq] }))
    return { runId, outDir, weightsSha256: r.weightsSha256, params: r.params, bytes: r.bytes, globalStep: r.globalStep }
  } finally { await worker.stop() }
}

/* ══════════════════════════════════════════════════════════ 3. validate */

async function measureAt({ endpoint, model, domainRows, wikitext, mmlu, nDomain, nLm, nMmlu, vocabSize }) {
  const uniform = await evalUniform({ endpoint, model, prompt: 'The quick brown fox jumps over the lazy dog. In a distant country, the population of', vocabSize })
  const domain = await evalLM({ endpoint, model, rows: domainRows, limit: nDomain })
  const lm = await evalLM({ endpoint, model, rows: wikitext, limit: nLm })
  const mmluRes = await evalMMLU({ endpoint, model, rows: mmlu, limit: nMmlu })
  return { uniform, domain, lm, mmlu: mmluRes }
}

/**
 * fail-closed：**有损仪器不得用于验收**（`E_INSTRUMENT_LOSSY`）。
 * 判据在 `@xiciyang/training-core`（单一实现，四个插件共用）；这里只做错误码转译，
 * 否则 `toEnvelope` 会把非 ToolError 一律归成 `E_INTERNAL`，把可执行的补救埋进 detail。
 */
function guardInstrument(instrument, runId) {
  try {
    assertInstrumentUsable(instrument, { runId })
  } catch (e) {
    if (e?.code !== ERROR_CODES.E_INSTRUMENT_LOSSY) throw e
    throw new ToolError(ERROR_CODES.E_INSTRUMENT_LOSSY, { hint: e.message, details: e.details ?? null, recoverable: false })
  }
}

/**
 * `validate` 的状态准入规则 —— **纯函数**，便于无 GPU / 无容器回归测试（见 training-sft 同名声明的注释）。
 * 缺陷 13（同构）：`train` 留下 `trained`，`validate` 需要 `exported`，而公开工具面没有 export 入口
 * ⇒ 旧实现直接拒绝。现在 `trained` ⇒ validate 自己先 `exportRun`。
 * `failed` / `running` / `rolled-back` 仍然**必须拒绝**。
 */
export function validateStatusPlan(status) {
  if (status === 'trained') return 'auto-export'
  if (['exported', 'validated', 'gate-failed'].includes(status)) return 'proceed'
  return 'reject'
}

export async function validate({
  root = paths().root, runId, deploy = true, nDomain = 200, nLm = 200, nMmlu = 300, thresholds = null, teardown = true, log = () => {},
}) {
  const t0 = Date.now()
  const reg = await readRegistry(root, schemaPath())
  let run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  // 缺陷 13（与 training-sft 同构）：`trained` ⇒ validate 自己先 export（与 rl/qat 对齐）。
  if (validateStatusPlan(run.status) === 'auto-export') {
    log('  → run 状态为 trained，validate 内部先完成 export（与 rl/qat 对齐）')
    // 同 training-sft：必须指定 checkpoint，否则 exportRun 会到 runDir 根目录找权重/适配器而失败。
    const ckpt = run.checkpoint?.name ?? null
    if (!ckpt) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run ${runId} 状态为 trained 但没有 checkpoint 记录，无法自动 export；请显式指定 checkpointName`, recoverable: true })
    await exportRun({ root, runId, checkpointName: ckpt, log })
    const reg2 = await readRegistry(root, schemaPath())
    run = (reg2.runs || []).find((r) => r.id === runId)
    if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `export 后 run 记录消失：${runId}`, recoverable: false })
  }
  if (validateStatusPlan(run.status) === 'reject') throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run ${runId} 状态为 ${run.status}，不能 validate`, recoverable: true })
  // 阈值**必须由调用方显式给出**（eval-contract 既有纪律：不得内置默认）
  if (!thresholds || typeof thresholds.minDomainGainPct !== 'number') {
    throw new ToolError(ERROR_CODES.E_MISSING_PARAM, {
      hint: 'thresholds.minDomainGainPct 必须显式给出（领域能力至少要提升多少 %）。本插件不内置默认值 —— 不能自己定阈值、自己签收。',
      details: { expected: { minDomainGainPct: 'number', zForgetting: 'number(可选，默认 1.96)' } }, recoverable: true,
    })
  }
  const z = typeof thresholds.zForgetting === 'number' ? thresholds.zForgetting : 1.96

  await audit(root, { capability: 'validate', action: 'validate', phase: 'begin', runId, outDir: run.exported.dir, nDomain, nLm, nMmlu, thresholds })

  const domainRows = readJsonl(join(run.dataset.dir, 'domain-heldout.jsonl'))
  const sets = (await import('./data/corpus-audit.mjs')).loadEvalSets()
  const wikitext = (sets.find((s) => s.name.startsWith('wikitext-test(c128)')) || sets.find((s) => s.kind === 'lm'))?.rows ?? []
  const mmlu = (sets.find((s) => s.kind === 'qa'))?.rows ?? []
  const vocabSize = JSON.parse(readFileSync(join(run.exported.dir, 'config.json'), 'utf8')).vocab_size

  const sp = await resolveServingPath({ root })
  const client = await new ServingMcpClient(sp.serverPath).start()
  let ownerToken = null; let ptEndpoint = null; let baseEndpoint = null
  // ── 仪器来源：每次用于测量的部署各留一份归一化记录（见 training-sft 同处的说明）
  let candidateInstrument = null; let baseInstrument = null
  const results = {}
  try {
    if (!deploy) throw new ToolError(ERROR_CODES.E_MISSING_PARAM, { hint: 'validate 需要 deploy=true（闭环要求经 serving 加载器验证）', recoverable: true })

    log(`部署预训练模型 ${run.exported.dir} …`)
    const d = await callWithConfirm(client, 'inference_serving_deploy', {
      path: run.exported.dir, owner: 'training-pretrain', purpose: `长任务3 pretrain validate ${runId}`,
      servedModelName: `pt-${runId}`, gpuMemoryUtilization: 0.18, maxNumSeqs: 4,
    }, { log })
    if (d.ok === false) throw new ToolError(ERROR_CODES.E_SERVING_UNAVAILABLE, { hint: `预训练模型部署失败：${d.code} ${d.message ?? ''}`, details: d })
    ownerToken = d.ownerToken
    ptEndpoint = { url: d.endpoint.endpoint, model: d.endpoint.servedModelName, id: d.endpoint.id, container: d.endpoint.containerName, usability: d.report?.attempts?.[0]?.modelUsability ?? null }
    candidateInstrument = instrumentFromDeployReport(d.report)
    guardInstrument(candidateInstrument, runId) // 有损 ⇒ 立刻停，不做无意义的测量
    log(`  → ${ptEndpoint.url} (usability.ok=${ptEndpoint.usability?.ok}, lossy=${candidateInstrument?.lossy ?? null})`)

    log('测量预训练模型 …')
    results.pt = await measureAt({ endpoint: ptEndpoint.url, model: ptEndpoint.model, domainRows, wikitext, mmlu, nDomain, nLm, nMmlu, vocabSize })

    // 基线缓存键含：基座 + revision + 协议 + 各评估集规模 + **领域留出集哈希**（换语料必须失效）
    const domainHash = `sha256:${contentHash(readFileSync(join(run.dataset.dir, 'domain-heldout.jsonl'), 'utf8'))}`
    const cacheKey = `${run.baseModel.path}@${run.baseModel.revision ?? 'norev'}@${EVAL_PROTOCOL}@dom${nDomain}:${domainHash}/lm${nLm}/mmlu${nMmlu}`
    const cachedRaw = reg.baselineCache?.[cacheKey]
    // **缺陷 15**：只有**能证明完整**的缓存才可复用；"未知"不等于"完整"（fail-closed）。
    const cachedCheck = cachedRaw ? measurementCompleteness(cachedRaw) : null
    if (cachedRaw && !cachedCheck.complete) log(`  基线缓存**不可复用**（${cachedCheck.reason}）⇒ 本次重新测量基线`)
    const cached = (cachedRaw && cachedCheck.complete) ? cachedRaw : null
    if (cached) {
      // 基线用缓存 ⇒ 本次没有部署、没有 deploy-report ⇒ `instrument.base = null`（**不编造**）。
      results.base = cached
      baseInstrument = null
      log('基线用缓存（本次无基线部署 ⇒ instrument.base = null，不编造）')
    } else {
      log(`部署基座 ${run.baseModel.path} …`)
      const b = await callWithConfirm(client, 'inference_serving_deploy', {
        path: run.baseModel.path, owner: 'training-pretrain', purpose: `长任务3 pretrain 基线 ${runId}`,
        servedModelName: `base-pt-${runId}`, gpuMemoryUtilization: 0.18, maxNumSeqs: 4,
      }, { log })
      if (b.ok === false) throw new ToolError(ERROR_CODES.E_SERVING_UNAVAILABLE, { hint: `基座部署失败：${b.code}`, details: b })
      baseEndpoint = { url: b.endpoint.endpoint, model: b.endpoint.servedModelName, id: b.endpoint.id, ownerToken: b.ownerToken, container: b.endpoint.containerName }
      baseInstrument = instrumentFromDeployReport(b.report)
      guardInstrument(baseInstrument, runId) // 基线仪器同样不得有损（基线坏了，比较就不成立）
      log(`测量基座 … (lossy=${baseInstrument?.lossy ?? null})`)
      results.base = await measureAt({ endpoint: baseEndpoint.url, model: baseEndpoint.model, domainRows, wikitext, mmlu, nDomain, nLm, nMmlu, vocabSize })
    }

    // ── 门禁①：**领域能力**必须显著提升（否则"预训练没学到领域"）
    const validity = [
      { name: 'pt-forward', ok: results.pt.uniform.n > 0, detail: `预训练侧拿到 ${results.pt.uniform.n} 个 logprob` },
      { name: 'pt-not-uniform', ok: results.pt.uniform.ok === true, detail: results.pt.uniform.criterion ?? '' },
      { name: 'domain-samples', ok: results.pt.domain.texts >= Math.min(50, nDomain), detail: `领域留出有效文本 ${results.pt.domain.texts}/${nDomain}` },
    ]
    const domGain = relativeChange({ baseline: results.base.domain.meanNll, candidate: results.pt.domain.meanNll, direction: 'lower-is-better' })
    const domainGate = dualGate({
      validity,
      accuracy: { dropPct: 0 },                       // 本门禁只管"提升"，不设精度上限
      perf: { gainPct: -domGain.worseningPct },
      maxAccuracyDropPct: 0,
      minPerfGainPct: thresholds.minDomainGainPct,
    })
    domainGate.deltas = { domainNll: { base: results.base.domain.meanNll, pt: results.pt.domain.meanNll, rawPct: domGain.rawPct } }

    // ── 门禁②：**灾难性遗忘**（MMLU + wikitext，配对 z·SE）
    const fg = forgettingGate({ baseMmlu: results.base.mmlu, ptMmlu: results.pt.mmlu, baseLm: results.base.lm, ptLm: results.pt.lm, z })

    const pass = domainGate.pass && fg.pass

    const evalResult = {
      schema: 'ai-infra/eval-result@1',
      subject: { kind: 'checkpoint', ref: run.exported.dir, revision: null },
      measurement: {
        n: results.pt.domain.texts + results.pt.lm.texts + results.pt.mmlu.n,
        unit: 'samples',
        datasetRef: `${run.dataset.dir}#domain-heldout+wikitext-test+mmlu-test`,
        datasetHash: run.dataset.lineageHash ? String(run.dataset.lineageHash).replace(/^(sha256:)+/, 'sha256:') : null,
      },
      metrics: [
        { name: 'domain_mean_nll', value: results.pt.domain.meanNll, unit: 'nats', direction: 'lower-is-better', baseline: results.base.domain.meanNll, relativePct: domGain.rawPct },
        { name: 'mmlu_accuracy', value: results.pt.mmlu.accuracy, unit: 'ratio', direction: 'higher-is-better', baseline: results.base.mmlu.accuracy, relativePct: null },
        { name: 'wikitext_mean_nll', value: results.pt.lm.meanNll, unit: 'nats', direction: 'lower-is-better', baseline: results.base.lm.meanNll, relativePct: null },
      ],
      validity: validity.map((v) => ({ name: v.name, ok: v.ok, detail: v.detail ?? null })),
      gate: { pass, validityOk: domainGate.validityOk, reason: `${domainGate.reason} | ${fg.reasons.join('; ') || 'no-forgetting-issue'}`, checks: domainGate.checks },
      versions: { testingCore: TESTING_CORE_VERSION, assets: EVAL_PROTOCOL },
      // 仪器来源（**additive**，schema 里是可选属性）：判决是在哪两次部署上测出来的、有没有有损降级
      instrument: { base: baseInstrument, candidate: candidateInstrument },
    }
    writeFileSync(join(run.runDir, 'eval-result.json'), `${JSON.stringify(evalResult, null, 2)}\n`)

    const ar = await audit(root, {
      capability: 'validate', action: 'validate', phase: 'end', runId, outcome: pass ? 'pass' : 'fail',
      domainGate: domainGate.pass, forgettingGate: fg.pass,
      domainNllBase: results.base.domain.meanNll, domainNllPt: results.pt.domain.meanNll,
      mmluBase: results.base.mmlu.accuracy, mmluPt: results.pt.mmlu.accuracy,
      lmPplBase: results.base.lm.ppl, lmPplPt: results.pt.lm.ppl, ms: Date.now() - t0,
    })
    // **缺陷 15**：部分测量**不得**入缓存（fail-closed）。只有能证明完整的测量才写入。
    const baseCacheNext = (() => {
      if (cached) return null                       // 本次复用了完整缓存 ⇒ 保持原样
      const chk = measurementCompleteness(results.base)
      if (!chk.complete) { log(`  基线测量不完整（${chk.reason}）⇒ **不写入缓存**（fail-closed）`); return null }
      return attachIntegrity(results.base)
    })()
    await updateRegistry(root, (doc) => ({
      ...doc,
      baselineCache: baseCacheNext ? { ...(doc.baselineCache ?? {}), [cacheKey]: baseCacheNext } : (doc.baselineCache ?? {}),
      runs: (doc.runs || []).map((r) => (r.id === runId ? {
        ...r, status: pass ? 'validated' : 'gate-failed', validatedAt: new Date().toISOString(),
        endpoint: ptEndpoint,
        evaluation: { thresholds, z, domainGate, forgettingGate: fg, base: summarize(results.base), pt: summarize(results.pt) },
        decidedBudget: { gpuHoursTrain: r.result?.gpuHours ?? null, gpuHoursValidate: Number(((Date.now() - t0) / 3_600_000).toFixed(4)) },
        evalResult: { schema: evalResult.schema, path: join(run.runDir, 'eval-result.json') },
        auditSeqs: [...(r.auditSeqs || []), ar.seq],
      } : r)),
    }), { schemaPath: schemaPath() })

    return { runId, pass, domainGate, forgettingGate: fg, endpoint: ptEndpoint, base: summarize(results.base), pt: summarize(results.pt), audit: { seq: ar.seq, hash: ar.hash }, ms: Date.now() - t0 }
  } finally {
    if (teardown) {
      for (const [label, ep, tok] of [['pt', ptEndpoint, ownerToken], ['base', baseEndpoint, baseEndpoint?.ownerToken]]) {
        if (!ep) continue
        try { const r = await callWithConfirm(client, 'inference_serving_teardown', { selector: ep.id, ownerToken: tok }, { log }); log(`  teardown ${label}: ok=${r.ok} ${r.code ?? ''}`) } catch (e) { log(`  teardown ${label} 异常：${e.message}`) }
      }
    }
    await client?.close?.()
  }
}

/** 持久化测量摘要里"是谁产出了这份测量"——供下游（如 `effectJudgment`）判定**不自证**。 */
export const MEASUREMENT_PRODUCER = 'training-pretrain.validate'

/**
 * 从仪器测量里**原样搬运**完整性证据（缺陷 14/15 的持久化面）。
 *
 * ⚠️ 这三个字段是**载荷**，不是装饰：
 *   `attempted` / `failed` 一旦被摘要丢掉就**不可恢复**（`succeeded` 还能从 `perText.length` 倒推，
 *   `attempted`/`failed` 不能）—— 于是"端点中途失效"与"评估集本来就小"在**落盘后**重新变得不可区分，
 *   这正是缺陷 14 被判定为缺陷的那个理由。
 * 实测（修复前，`~/.ai-infra/state/pretrain-registry.yaml`，15 个 run）：字符串 `attempted` 出现 **0 次**，
 * `measurementCompleteness(summarize(m))` 恒为 `false`（`<part>:no-integrity-evidence`）。
 *
 * **只在源字段存在时搬运**：源里没有就**不写**（而不是补 `null`）——
 * `null` 会被读成"我们知道它是 null"，而事实是"我们不知道"，两者在 fail-closed 判据下**都必须不可复用**，
 * 但语义不同，产物不得替真相表态。
 */
const INTEGRITY_FIELDS = ['attempted', 'succeeded', 'failed', 'skipped', 'partial']
const integrityOf = (x) => {
  const out = {}
  if (!x || typeof x !== 'object') return out
  for (const k of INTEGRITY_FIELDS) if (x[k] !== undefined) out[k] = x[k]
  return out
}

/**
 * 持久化用的测量摘要。
 *
 * **保留每个仪器的完整性证据**（`attempted/succeeded/failed/partial`）—— 于是
 * `measurementCompleteness(summarize(m))` 能**自证**这份测量是否完整，
 * 而不是靠"看起来像完整"。回归测试：`tests/unit/persisted-integrity.spec.mjs`。
 */
export const summarize = (m) => ({
  uniform: { ok: m.uniform.ok, n: m.uniform.n, epsilon: m.uniform.epsilon, delta: m.uniform.delta, bitIdenticalAcrossRuns: m.uniform.bitIdenticalAcrossRuns },
  domain: { ...integrityOf(m.domain), texts: m.domain.texts, tokens: m.domain.tokens, meanNll: m.domain.meanNll, ppl: m.domain.ppl, perText: m.domain.perText },
  lm: { ...integrityOf(m.lm), texts: m.lm.texts, tokens: m.lm.tokens, meanNll: m.lm.meanNll, ppl: m.lm.ppl, perText: m.lm.perText },
  mmlu: { ...integrityOf(m.mmlu), n: m.mmlu.n, correct: m.mmlu.correct, accuracy: m.mmlu.accuracy, asked: m.mmlu.asked },
  integrity: { producedBy: MEASUREMENT_PRODUCER },
})

/* ══════════════════════════════════════════════════════════ 4. rollback */

export async function rollback({ root = paths().root, runId, purge = false, log = () => {} }) {
  const reg = await readRegistry(root, schemaPath())
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  await audit(root, { capability: 'rollback', action: 'rollback', phase: 'begin', runId, purge })
  const purged = []
  if (purge) {
    for (const d of [run.runDir, join(paths(root).runsDir, runId, 'exported')]) {
      if (d && existsSync(d) && d.startsWith(paths(root).runsDir)) { await fs.rm(d, { recursive: true, force: true }); purged.push(d) }
    }
  }
  const ar = await audit(root, { capability: 'rollback', action: 'rollback', phase: 'end', runId, purged: purged.length })
  await withRun(root, runId, (r) => ({ ...r, status: 'rolled-back', rolledBackAt: new Date().toISOString(), purged, auditSeqs: [...(r.auditSeqs || []), ar.seq] }))
  log(`run ${runId} → rolled-back（purge=${purge}, 删除 ${purged.length} 个目录）`)
  return { runId, status: 'rolled-back', purged, audit: { seq: ar.seq, hash: ar.hash } }
}

/* ══════════════════════════════════════════════════════════ 5. get_report */

export async function getReport({ root = paths().root, runId = null, limit = 10 }) {
  const reg = await readRegistry(root, schemaPath())
  const chain = await auditVerify(root)
  if (!runId) {
    return {
      runs: (reg.runs || []).slice(-limit).map((r) => ({
        id: r.id, status: r.status, shardIndex: r.shardIndex ?? null, startedAt: r.startedAt,
        steps: r.result?.steps ?? null, globalStep: r.result?.globalStep ?? null, gpuHours: r.result?.gpuHours ?? null,
        domainNll: r.evaluation?.pt?.domain?.meanNll ?? null, baseDomainNll: r.evaluation?.base?.domain?.meanNll ?? null,
        pass: r.evaluation ? (r.evaluation.domainGate?.pass && r.evaluation.forgettingGate?.pass) : null,
      })),
      audit: { ok: chain.ok, shards: chain.shards },
    }
  }
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  return { run, audit: { ok: chain.ok, shards: chain.shards, results: chain.results.map((x) => ({ day: x.day, checked: x.checked, ok: x.ok })) } }
}

/* ══════════════════════════════════════════════════════════ 6. health */

export async function health({ root = paths().root, deep = false }) {
  const p = P(root)
  const probe = await sandboxProbe()
  const chain = await auditVerify(root)
  let gpu = null
  try { gpu = execFileSync('nvidia-smi', ['--query-gpu=name,driver_version', '--format=csv,noheader'], { encoding: 'utf8', timeout: 20000 }).trim() } catch { gpu = null }
  let imageOk = false; let imageId = null
  try { imageId = execFileSync('docker', ['images', '-q', TRAINING_IMAGE], { encoding: 'utf8', timeout: 20000 }).trim() || null; imageOk = Boolean(imageId) } catch { imageOk = false }
  const baseOk = process.env.DSH_BASE_MODEL ? existsSync(join(process.env.DSH_BASE_MODEL, 'config.json')) : false
  // **serving 是本插件 validate 阶段的硬依赖**（部署 → 测量 → 拆解）。
  // 此前 health() 不报 serving ⇒ health 会"报健康"却在实际依赖缺失时仍然通过，
  // 即**健康检查没覆盖它真正依赖的组件**（LONGTASK5 阶段 2 修）。
  let servingOk = false; let servingPath = null
  try { const sp = await resolveServingPath({ root }); servingPath = sp.serverPath; servingOk = true } catch { servingOk = false }
  // **关键：不留 undefined 属性**（长任务 2 缺陷 3 的正面做法）。
  // 注意：`perf.bitsPerWeight` 这类"阶段无关"的字段，本阶段**不适用** ⇒ 整个字段省略，而不是写 undefined。
  return {
    ok: probe.container.available && chain.ok && imageOk && baseOk && servingOk,
    sandboxTiers: { process: probe.process.available, container: probe.container.available, vm: probe.vm.available },
    sandboxEvidence: probe.container.evidence,
    image: { name: TRAINING_IMAGE, available: imageOk, id: imageId },
    gpu,
    baseModel: { path: process.env.DSH_BASE_MODEL ?? null, available: baseOk },
    serving: { available: servingOk, serverPath: servingPath },
    audit: { ok: chain.ok, shards: chain.shards, detail: chain.results.map((x) => ({ day: x.day, checked: x.checked, ok: x.ok })) },
    registry: { runs: (await readRegistry(root, schemaPath())).runs?.length ?? 0, file: p.registryFile },
    limits: DEFAULT_LIMITS,
    version: { pkg: PACKAGE_VERSION, schema: SCHEMA_VERSION },
    ...(deep ? { deep: { paths: p } } : {}),
  }
}

export { enforceLimits, dualGate, PKG_ROOT }
