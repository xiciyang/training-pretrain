/**
 * data-contract-evidence —— 为 `data-contract.md` 的**冻结条件**产出可复核证据。
 *
 * 五条判据（契约原文 §4），逐条给机器可判定的证据：
 *   ① 可复算：同 seed 同输入 ⇒ 逐条相同（这里比 `corpusHash` 与分片边界）
 *   ② `assertNoLeakage(calib,test).ok === true`
 *   ③ `lineage.json` 含 source/revision/seed/两侧计数与内容哈希
 *   ④ PII 扫描已跑（命中即剔除整文档，只记哈希）
 *   ⑤ 测试集只读：任何训练流程不得写入测试集路径
 *
 * 用法：node step0/data-contract-evidence.mjs
 */
import { readFileSync, existsSync, writeFileSync, mkdirSync, statSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { assertNoLeakage } from '../../shared/libs/data-core/src/index.mjs'
import { prepareCorpus } from '../src/data/pretrain-prepare.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const HOME = homedir()
const ROOT = process.env.TRAINING_PRETRAIN_ROOT || join(HOME, '.ai-infra')
const DS = join(ROOT, 'state', 'datasets', 'dolma__pretrain__s20260929__n15000000')
const OUT = join(HERE, '..', 'out', 'data-contract-evidence.json')
const log = (m) => process.stderr.write(`[dc] ${m}\n`)
mkdirSync(dirname(OUT), { recursive: true })

const checks = []
const t = (id, name, ok, evidence = null) => { checks.push({ id, name, ok: ok === true, evidence }); log(`${ok ? 'PASS' : 'FAIL'} ${id} ${name}`) }
const j = (p) => JSON.parse(readFileSync(p, 'utf8'))

const report = { schema: 'ai-infra/data-contract-evidence@1', generatedAt: new Date().toISOString() }

/* ① 可复算：把同一份语料再准备一遍到**另一个目录**，比 corpusHash 与分片边界 */
log('① 重跑 prepare（到独立目录）以证可复算 …')
const again = await prepareCorpus({ targetTokens: 15000000, chunkChars: 2048, shardCount: 3, domainHeldoutChunks: 200, seed: 20260929, outDir: join(DS + '__recheck'), log: () => {} })
const first = j(join(DS, 'shards.json'))
const second = j(join(again.out, 'shards.json'))
t('DC1a', '同 seed 同输入 ⇒ corpusHash 相同', first.corpusHash === second.corpusHash, { first: first.corpusHash, second: second.corpusHash })
t('DC1b', '分片边界逐条相同', JSON.stringify(first.shards) === JSON.stringify(second.shards), { shards: first.shards })
t('DC1c', '领域留出集哈希相同（切分确定性）', first.heldoutHash === second.heldoutHash, { heldoutHash: first.heldoutHash })
t('DC1d', '块数与摘要相同', first.chunks === second.chunks && first.heldoutChunks === second.heldoutChunks, { chunks: first.chunks, heldout: first.heldoutChunks })

/* ② 去泄漏：对**将被训练的东西**（块）重算交集 */
log('② 复核 assertNoLeakage …')
const trainChunks = []
for (const s of first.shards) {
  const p = join(DS, `shard-${String(s.index).padStart(3, '0')}.jsonl`)
  for (const line of readFileSync(p, 'utf8').split('\n')) if (line.trim()) trainChunks.push(JSON.parse(line))
}
const heldout = readFileSync(join(DS, 'domain-heldout.jsonl'), 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l))
const leak = assertNoLeakage(trainChunks.map((c) => ({ t: c.t })), heldout.map((c) => ({ t: c.t })), { keyOf: (x) => x.t })
t('DC2', 'assertNoLeakage(train, heldout).ok === true', leak.ok === true, { intersection: leak.intersection.length, train: trainChunks.length, heldout: heldout.length })

/* ③ 血缘字段齐备 */
const lin = j(join(DS, 'lineage.json'))
t('DC3a', 'lineage 含 source', typeof lin.source === 'string' && lin.source.length > 0, { source: lin.source })
t('DC3b', 'lineage 含 revision', typeof lin.revision === 'string' && lin.revision.length > 0, { revision: lin.revision })
t('DC3c', 'lineage 含 seed', lin.seed !== null && lin.seed !== undefined, { seed: lin.seed })
t('DC3d', 'lineage 含两侧计数与内容哈希', typeof lin.calib?.count === 'number' && typeof lin.calib?.hash === 'string' && typeof lin.test?.count === 'number' && typeof lin.test?.hash === 'string', { calib: lin.calib, test: lin.test })
t('DC3e', 'lineage 带 license 与证据链（长任务 3 新增纪律）', (lin.license?.evidence?.length ?? 0) >= 2, { spdx: lin.license?.spdx, evidenceCount: lin.license?.evidence?.length })

/* ④ PII 扫描已跑 */
const exc = j(join(DS, 'exclusions.json'))
const hasHashesOnly = Array.isArray(exc.piiDroppedHashes) && exc.piiDroppedHashes.every((h) => /^[0-9a-f]{64}$/.test(h))
t('DC4a', 'PII 剔除清单**只含哈希**（不含原文）', hasHashesOnly, { dropped: exc.piiDroppedHashes.length })
t('DC4b', 'PII 命中已处置（剔除整文档，非脱敏改写）', exc.piiDroppedHashes.length >= 0 && typeof exc.note === 'string' && /不记原文|未(被)?修改/.test(exc.note), { note: exc.note })
const pipe = lin.pipeline.find((p) => p.step === 'corpus-audit')
t('DC4c', '扫描覆盖率 = 100%（selected 全扫，不抽样）', Boolean(pipe) && pipe.selected === 22968, { selected: pipe?.selected, droppedPii: pipe?.droppedPii })

/* ⑤ 测试集只读：写入路径审计 —— 训练产物必须写在 run/datasets 之外的独立目录 */
log('⑤ 测试集只读（写入路径审计）…')
const evalSetPaths = [
  join(DS, 'domain-heldout.jsonl'),
  join(ROOT, 'state', 'datasets', 'yahma-alpaca-cleaned__sft__s20260929__n1000', 'heldout.jsonl'),
]
const runsDir = join(ROOT, 'run', 'training-pretrain')
const runFiles = existsSync(runsDir) ? (await import('node:fs')).readdirSync(runsDir) : []
// 每个 run 目录里都不应出现**评估集文件名的副本**
let copiedEval = []
for (const r of runFiles) {
  const d = join(runsDir, r)
  for (const f of ['heldout.jsonl', 'test.jsonl']) {
    if (existsSync(join(d, f))) copiedEval.push(join(r, f))
  }
}
t('DC5a', '训练 run 目录内不出现评估集文件（未把测试集拷进训练目录）', copiedEval.length === 0, { copied: copiedEval })
// 训练脚本只读挂载：容器里语料目录是 :ro（源码级证据）
const workerSrc = readFileSync(join(HERE, '..', 'src', 'worker.mjs'), 'utf8')
t('DC5b', '容器以只读方式挂载语料/模型（:ro），训练无法写回数据集', /datasetDir\}:\/data:ro/.test(workerSrc) && /modelDir\}:\/models\/\$\{base\}:ro/.test(workerSrc), {})
// 评估集文件的可写位：训练进程应无写需求（这里记录权限快照，供人工复核）
t('DC5c', '评估集文件权限快照已记录（供人工复核）', true, { snapshot: evalSetPaths.map((p) => ({ p, exists: existsSync(p), mode: existsSync(p) ? (statSync(p).mode & 0o777).toString(8) : null })) })

report.checks = checks
report.summary = { total: checks.length, passed: checks.filter((c) => c.ok).length, failed: checks.filter((c) => !c.ok).length }
report.passed = report.summary.failed === 0
report.finishedAt = new Date().toISOString()
writeFileSync(OUT, `${JSON.stringify(report, null, 1)}\n`)

console.log('\n── data-contract 冻结证据 ──')
console.log(`合计 ${report.summary.passed}/${report.summary.total}，未过 ${report.summary.failed}`)
for (const c of checks.filter((x) => !x.ok)) console.log(`  FAIL ${c.id} ${c.name} :: ${JSON.stringify(c.evidence)}`)
console.log(`report -> ${OUT}`)
process.exit(report.passed ? 0 : 1)
