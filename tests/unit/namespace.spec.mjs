/**
 * namespace.spec —— 命名空间纪律：`training_pretrain_*` 与既有插件**零交集**。
 * 从兄弟包源码**实时抓名**（不是维护一份会过期的白名单）。
 */
import { readFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { TOOL_NAMES, existingNamespaces, WORKSPACE, CANDIDATE_PKGS } from '../../src/index.mjs'

export const NAME = 'unit/namespace'

export async function run(t) {
  for (const n of TOOL_NAMES) t.assert(n.startsWith('training_pretrain_'), `工具名带正确前缀：${n}`)

  const existing = existingNamespaces()
  const overlap = TOOL_NAMES.filter((n) => existing.includes(n))
  t.assert(overlap.length === 0, `与既有命名空间无交集（冲突：${overlap.join(',') || '无'}）`)
  t.assert(existing.length > 0, `确实抓到了兄弟包的既有工具名（${existing.length} 个）—— 抓不到就没在测`)

  // 反向：兄弟包也不应出现我们的名字
  const mine = [...TOOL_NAMES]
  const hits = existing.filter((n) => mine.includes(n))
  t.assert(hits.length === 0, '反向检查：兄弟包中不出现 training_pretrain_*')

  // 既有插件的源码路径必须**真的读到了**（否则上面是空断言）。
  // 注意：`inference-tuning` 只有 index.mjs（没有 tools.mjs）—— 只查 tools.mjs 会误判。
  const found = CANDIDATE_PKGS.filter((pkg) => existsSync(join(WORKSPACE, pkg, 'src', 'tools.mjs')) || existsSync(join(WORKSPACE, pkg, 'src', 'index.mjs')))
  t.assert(found.includes('inference-serving') && found.includes('inference-quantization'), `推理侧插件源码读到了（${found.join(',')}）`)
  t.assert(found.includes('inference-training/training-sft'), `训练侧 training-sft 源码读到了（在 inference-training/ 下，不在仓库根）`)
  t.assert(found.includes('inference-tuning'), 'inference-tuning 读到了（它没有 tools.mjs，但有 index.mjs）')
  void readFileSync; void dirname; void fileURLToPath
}
