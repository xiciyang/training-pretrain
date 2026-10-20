/**
 * continuity.spec —— **续训连续性的绝对量断言**（长任务 2 缺陷 2 的正面测试）。
 *
 * 缺陷 2 的形态：「续训 20 步实跑 220 步，而两侧步号断言**全对**」。
 * 所以这里不满足于"断言存在"，而是**构造缺陷 2 的每一种形态并确认会被拒绝**：
 *   - 权重加载了但游标重置（consumedChunks 归零）
 *   - 步号重置（globalStep 归零）
 *   - 换了语料续训
 *   - 分片跳变
 * 以及不变量 `consumedChunksΔ == 步数Δ × 每步块数` 的算术自检。
 */
import { assertResumeContinuity } from '../../src/train.mjs'

export const NAME = 'unit/continuity'

/** 与 python 侧 invariant_ok() 同式，用于纯算术自检。 */
function invariantOk({ globalStep, baseGlobalStep, consumedChunks, baseConsumedChunks, chunksPerStep }) {
  return (consumedChunks - baseConsumedChunks) === (globalStep - baseGlobalStep) * chunksPerStep
}

export async function run(t) {
  const chunksPerStep = 16
  const prev = { globalStep: 200, baseGlobalStep: 0, consumedChunks: 3200, baseConsumedChunks: 0, cursor: 3200, epoch: 0, shardIndex: 0, corpusHash: 'sha256:AAA' }

  /* ── 正常同分片续训：应通过 ── */
  const good = assertResumeContinuity(
    { globalStep: 200, consumedChunks: 3200, cursor: 3200, epoch: 0, shardIndex: 0 },
    prev, { corpusHash: 'sha256:AAA', shardIndex: 0 },
  )
  t.assert(good.ok === true, '字段全同 ⇒ 连续性通过')
  t.assert(good.checks.length >= 6, `产出了逐字段核对记录（${good.checks.length} 条）`)

  /* ── 缺陷 2 形态 A：权重加载了但**游标重置** ── */
  const cursorReset = assertResumeContinuity(
    { globalStep: 200, consumedChunks: 0, cursor: 0, epoch: 0, shardIndex: 0 },
    prev, { corpusHash: 'sha256:AAA', shardIndex: 0 },
  )
  t.assert(cursorReset.ok === false, '游标重置（consumedChunks=0）被拒绝 —— 这正是缺陷 2 的形态')
  t.assert(cursorReset.problems.some((p) => p.includes('consumedChunks')), '拒绝理由指名 consumedChunks')

  /* ── 缺陷 2 形态 B：**步号重置** ── */
  const stepReset = assertResumeContinuity(
    { globalStep: 0, consumedChunks: 3200, cursor: 3200, epoch: 0, shardIndex: 0 },
    prev, { corpusHash: 'sha256:AAA', shardIndex: 0 },
  )
  t.assert(stepReset.ok === false, '步号重置（globalStep=0）被拒绝')
  t.assert(stepReset.problems.some((p) => p.includes('globalStep')), '拒绝理由指名 globalStep')

  /* ── 换语料续训 ── */
  const corpusSwap = assertResumeContinuity(
    { globalStep: 200, consumedChunks: 3200, cursor: 3200, epoch: 0, shardIndex: 0 },
    prev, { corpusHash: 'sha256:BBB', shardIndex: 0 },
  )
  t.assert(corpusSwap.ok === false, '换语料续训被拒绝（corpusHash 不符）')

  /* ── 分片：允许 +1（推进），禁止跳变 ── */
  const advance = assertResumeContinuity(
    { globalStep: 200, consumedChunks: 0, cursor: 0, epoch: 0, shardIndex: 1 },
    { ...prev, shardIndex: 0, consumedChunks: 3200, cursor: 3200, globalStep: 200, baseGlobalStep: 200, baseConsumedChunks: 0 },
    { corpusHash: 'sha256:AAA', shardIndex: 1 },
  )
  t.assert(advance.shardAdvance === true, 'shardIndex +1 被识别为"推进分片"')
  const jump = assertResumeContinuity(
    { globalStep: 200, consumedChunks: 3200, cursor: 3200, epoch: 0, shardIndex: 2 },
    prev, { corpusHash: 'sha256:AAA', shardIndex: 2 },
  )
  t.assert(jump.ok === false, 'shardIndex +2 被拒绝（不允许跳分片）')

  /* ── 全新 run 无 checkpoint：不报错 ── */
  t.assert(assertResumeContinuity({ globalStep: 0 }, null, { corpusHash: 'x', shardIndex: 0 }).ok === true, '全新 run 无 checkpoint ⇒ 通过')

  /* ── 绝对量不变量的算术自检（含"漏步/重放"两种破坏） ── */
  t.assert(invariantOk({ globalStep: 10, baseGlobalStep: 0, consumedChunks: 160, baseConsumedChunks: 0, chunksPerStep }), '不变量的正常态成立')
  t.assert(!invariantOk({ globalStep: 10, baseGlobalStep: 0, consumedChunks: 144, baseConsumedChunks: 0, chunksPerStep }), '少消费 1 步的块 ⇒ 不变量破坏（漏步）')
  t.assert(!invariantOk({ globalStep: 10, baseGlobalStep: 0, consumedChunks: 176, baseConsumedChunks: 0, chunksPerStep }), '多消费 1 步的块 ⇒ 不变量破坏（重放）')
  // **关键**：这正是缺陷 2 —— 相对量（步号递增）完全正常，但绝对量对不上
  const defect2 = { globalStep: 220, baseGlobalStep: 0, consumedChunks: 320, baseConsumedChunks: 0, chunksPerStep }
  t.assert(defect2.globalStep === 220, '缺陷 2 场景：步号"看起来"是对的')
  t.assert(!invariantOk(defect2), '缺陷 2 场景：绝对量不变量把它揪出来（320 ≠ 220×16=3520）')
}
