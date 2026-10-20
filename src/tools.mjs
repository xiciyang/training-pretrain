/**
 * tools —— 6 个工具的规格（**MCP 与 DSH 两条通道共用同一份**，不存在行为分叉）。
 *
 * 命名空间纪律：`training_pretrain_*`，与既有训练/推理插件（`training_sft_*` /
 * `inference_serving_*` / `inference_quantization_*` / `inference_tuning_*`）**均无交集**
 * —— 由 `tests/unit/namespace.spec.mjs` 去兄弟包源码实时抓名并断言交集为空。
 *
 * DSH 工具契约（**踩过的坑**）：`ctx.tools.register` 要求 `output { schema, render }`
 * **同时**给出；缺 render 会让**整条 entry 被拒**。故每个工具都带 `output`。
 */
import { CONFIRM_PARAM, ROOT_PARAM, passthroughOutput } from './output.mjs'
import * as api from './core.mjs'

export const ALL_TOOL_SPECS = [
  {
    name: 'training_pretrain_prepare',
    description: 'Prepare and validate a continued-pretraining corpus from a raw shard: read the actual dataset card license, scan 100% of the documents for non-reserved PII (documents are dropped whole, never rewritten), run the double-track de-leak check (whole-document content hash AND in-document substring containment against every evaluation set), dedup, chunk, deterministically split into shards, and reserve a domain held-out slice. Fails closed (E_CORPUS_LEAK) if either track is non-empty at document or chunk level.',
    parameters: {
      type: 'object', additionalProperties: false, required: [],
      properties: {
        shardPath: { type: 'string', description: '原始语料分片（.json.gz，每行 {id,source,text}）。默认 <SHARD>（默认取环境变量 DSH_SHARD_PATH）。' },
        targetTokens: { type: 'integer', description: '语料规模目标（约 token 数）。默认 15000000。' },
        chunkChars: { type: 'integer', description: '每块字符数（128..65536）。默认 2048。' },
        shardCount: { type: 'integer', description: '分片数（1..64）。默认 3。' },
        domainHeldoutChunks: { type: 'integer', description: '领域留出块数（不进任何 shard，用于测"领域能力是否提升"）。默认 200。' },
        seed: { type: 'integer', description: '切分/分片种子。默认 20260929。' },
        outDir: { type: 'string', description: '输出目录。默认 ~/.ai-infra/state/datasets/dolma__pretrain__s<seed>__n<targetTokens>。' },
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.prepare({ root: a?.root, shardPath: a?.shardPath, targetTokens: a?.targetTokens, chunkChars: a?.chunkChars, shardCount: a?.shardCount, domainHeldoutChunks: a?.domainHeldoutChunks, seed: a?.seed, outDir: a?.outDir ?? null }),
  },
  {
    name: 'training_pretrain_train',
    description: 'Run continued pretraining (full-parameter, NOT LoRA) through the shared training loop (@xiciyang/training-core) inside a container sandbox, with hard budget enforcement (maxSteps ≤500, maxGpuHours ≤2, concurrency 1, params ≤0.5B). Enforces the absolute step↔data-position invariant consumedChunks−base === (globalStep−baseGlobalStep)×chunksPerStep on EVERY step, so a silent step loss, replay, or cursor reset fails loudly instead of producing a checkpoint that merely looks finished. Over-budget pauses and reports — never silently degrades.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['datasetDir'],
      properties: {
        datasetDir: { type: 'string', description: 'training_pretrain_prepare 产出的语料目录（需含 lineage.json 与 shards.json，且 allClear=true）。' },
        shardIndex: { type: 'integer', description: '训练哪个分片（契约冻结条件 2 的分片语义）。默认 0。' },
        baseModel: { type: 'string', description: '基座模型本地目录（只读）。默认 <BASE_MODEL>（默认取环境变量 DSH_BASE_MODEL）。' },
        lr: { type: 'number', description: '学习率。默认 2e-5（预训练惯例：比 SFT 更小）。' },
        batchSize: { type: 'integer', description: '每微批块数。默认 4。' },
        gradAccum: { type: 'integer', description: '梯度累积步数（有效批 = batchSize×gradAccum）。默认 4。' },
        maxSeqLen: { type: 'integer', description: '最大序列长度。默认 512。' },
        maxSteps: { type: 'integer', description: '优化器步数上限（≤500）。默认 500。' },
        warmupSteps: { type: 'integer', description: 'warmup 步数。默认 50。' },
        seed: { type: 'integer', description: '训练随机种子。默认 20260929。' },
        checkpointEvery: { type: 'integer', description: '每 N 步存一个可恢复 checkpoint。默认 50。' },
        resumeRunId: { type: 'string', description: '续训：源 runId（需与 resumeCheckpoint 同时给）。' },
        resumeCheckpoint: { type: 'string', description: '续训：源 checkpoint 名（如 step-000050）。' },
        confirm: CONFIRM_PARAM,
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.train({ ...a, root: a?.root }),
  },
  {
    name: 'training_pretrain_validate',
    description: 'Close the vertical loop: if the run is still in state trained, this exports the full-parameter checkpoint as a standard HF directory first (validate performs the export internally — there is no separate export tool to call, and calling train then validate is sufficient), then deploys it through inference-serving (its loader + usability probe), measures BOTH the trained model and the frozen baseline on the same protocol, then judge with @xiciyang/testing-core: (1) a DOMAIN gate requiring the domain held-out NLL to improve by at least minDomainGainPct (caller-supplied — this plugin ships NO default threshold), and (2) a derivable FORGETTING gate bounding MMLU accuracy delta and paired wikitext NLL delta by z·SE. Deploys only its own leased endpoints and tears them down.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['runId', 'thresholds'],
      properties: {
        runId: { type: 'string', description: 'training_pretrain_train 返回的 run id。' },
        nDomain: { type: 'integer', description: '领域留出评估块数。默认 200。' },
        nLm: { type: 'integer', description: 'wikitext 评估文本数。默认 200。' },
        nMmlu: { type: 'integer', description: 'MMLU 题目数。默认 300。' },
        thresholds: { type: 'object', description: '{minDomainGainPct: number(必填), zForgetting?: number}。**minDomainGainPct 必须显式给出**，本插件不内置默认值。' },
        confirm: CONFIRM_PARAM,
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.validate({ ...a, root: a?.root }),
  },
  {
    name: 'training_pretrain_rollback',
    description: 'Roll a run back to a non-promoted state. Marks it rolled-back in the shared registry and (with purge=true) deletes ONLY that run directory under the training-pretrain run root — never a corpus, never the base model, never another run.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['runId'],
      properties: { runId: { type: 'string' }, purge: { type: 'boolean', description: '是否删除该 run 的产出目录。默认 false。' }, confirm: CONFIRM_PARAM, root: ROOT_PARAM },
    },
    output: passthroughOutput(),
    handler: (a) => api.rollback({ root: a?.root, runId: a?.runId, purge: a?.purge === true }),
  },
  {
    name: 'training_pretrain_get_report',
    description: 'Read the pretraining run registry and the audit hash chain. Without runId it lists recent runs (status, shard index, global step, GPU hours, domain NLL vs baseline, gate result); with runId it returns the full run record including the resume-continuity evidence (absolute step↔data check) plus chain verification.',
    parameters: { type: 'object', additionalProperties: false, required: [], properties: { runId: { type: 'string' }, limit: { type: 'integer' }, root: ROOT_PARAM } },
    output: passthroughOutput(),
    handler: (a) => api.getReport({ root: a?.root, runId: a?.runId ?? null, limit: a?.limit ?? 10 }),
  },
  {
    name: 'training_pretrain_health',
    description: 'Self-check for the pretraining side: behavioural sandbox tier probe (process/container/vm), training toolchain image presence, GPU, base model availability, audit chain integrity, and the shared registry path. Read-only. Every field is present and lossless-JSON clean (no undefined keys).',
    parameters: { type: 'object', additionalProperties: false, required: [], properties: { deep: { type: 'boolean' }, root: ROOT_PARAM } },
    output: passthroughOutput(),
    handler: (a) => api.health({ root: a?.root, deep: a?.deep === true }),
  },
]

export function findToolSpec(name) {
  return ALL_TOOL_SPECS.find((t) => t.name === name) || null
}
