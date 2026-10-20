/**
 * persisted-integrity.spec —— **缺陷 15 的同类残留**：*测量能被重算，但不能自证*。
 *
 * ## 为什么必须有这条 spec（而不是又一条"读代码"）
 *
 * `inference-training/checks/*` 与 `tests/unit/measurement-integrity.spec.mjs` 已经守住了
 * **内存里**的完整性判定。但**落盘之后**呢？
 *
 * 实测（本机 `~/.ai-infra/state/pretrain-registry.yaml`，15 个 run）：
 *   字符串 `attempted` 在整个 registry 里出现 **0 次**；
 *   而 `summarize()` 却在 `validate()` 里被用来写 `registry.runs[].evaluation.*`、
 *   也用来做 `validate()` 的返回值。
 *   ⇒ `measurementCompleteness(summarize(m))` 恒为 `false`，
 *     理由恒为 `domain:no-integrity-evidence, lm:no-integrity-evidence, mmlu:no-integrity-evidence`。
 *
 * 也就是说：**"测量记 attempted"这条承诺在持久化层面是假的。**
 * `succeeded` 还能从 `perText.length` 倒推，但 `attempted` / `failed` **彻底丢失** ——
 * 端点中途失效（`failed` 高）与"评估集本来就小"在落盘后又变得**不可区分**，
 * 这正是缺陷 14 被判定为缺陷的那个理由，只是换了一层。
 *
 * 而 RSI 首轮的**效果判定**（`@xiciyang/rsi-loop` 的 `effectJudgment`）**必须**吃
 * `integrity.{attempted,succeeded,failed,partial}`（否则 `E_INTEGRITY_MISSING` /
 * `E_MEASUREMENT_INCOMPLETE`）—— 它只能从**产物**里取，不能靠"现场再测一次"来补
 * （那就变成"判决由判决自己产生证据"了）。所以这条不是洁癖，是**首轮的前置条件**。
 *
 * ## 本 spec 断言什么
 *   ① 完备测量的 `summarize()` 产物上 `measurementCompleteness` **必须为 complete**
 *      —— 即持久化摘要**自证**其完整性，而不是"看起来完整"；
 *   ② 每个仪器（domain/lm/mmlu）都**逐字段**带上 `attempted/succeeded/failed/partial`；
 *   ③ **反向对照（防"把失败改绿"）**：不完整的测量经 `summarize()` 后
 *      **必须仍然 complete=false**，且理由**点名** `failed=N/M` ——
 *      任何"直接返回 complete:true"的假修复都会被这条打红；
 *   ④ `producedBy` 必须在场且**不是** `effectJudgment`（不自证）。
 *
 * 不依赖 GPU / 容器 / 网络。
 */
import { summarize } from '../../src/core.mjs'
import { measurementCompleteness } from '../../src/evaluate.mjs'

export const NAME = '持久化测量的完整性证据（缺陷 15 同类残留·pretrain）'

/** 一份**完备**的测量（各仪器 attempted 都有、failed=0、partial=false）。 */
const fullMeasurement = () => ({
  uniform: { ok: true, n: 12, epsilon: 0.01, delta: 0.02, bitIdenticalAcrossRuns: true },
  domain: {
    attempted: 200, succeeded: 200, failed: 0, skipped: 0, partial: false,
    texts: 200, tokens: 900, meanNll: -2.9649, ppl: 19.2, perText: [-2.9, -3.0, -2.96],
  },
  lm: {
    attempted: 200, succeeded: 200, failed: 0, skipped: 0, partial: false,
    texts: 200, tokens: 5000, meanNll: 3.1, ppl: 22.2, perText: [3.0, 3.2],
  },
  mmlu: {
    attempted: 300, succeeded: 300, failed: 0, skipped: 0, partial: false,
    n: 300, correct: 120, accuracy: 0.4, asked: 300, okIndices: [],
  },
})

/** 一份**不完备**的测量：端点中途失效（domain 只成功 200/260）。 */
const partialMeasurement = () => {
  const m = fullMeasurement()
  m.domain = { ...m.domain, attempted: 260, succeeded: 200, failed: 60, partial: true }
  return m
}

export async function run(t) {
  // ── 前提：spec 必须**真的看到了对象**（否则下面全是空断言） ──
  t.assert(typeof summarize === 'function', 'summarize 必须可被 import 到 —— 拿不到就是在测空气')

  const s = summarize(fullMeasurement())
  t.assert(s !== null && typeof s === 'object', 'summarize 必须返回对象')

  // ── ① 持久化摘要必须**自证**完整 ──
  const chk = measurementCompleteness(s)
  t.assert(
    chk.complete === true,
    `完备测量经 summarize() 后 measurementCompleteness 必须为 complete（实得 ${JSON.stringify(chk)}）`
    + ' —— 非 complete 即意味着"落盘后无法自证完整"，缺陷 15 的同类残留仍在',
  )
  t.assert(chk.reason === null, `complete 时 reason 必须为 null（实得 ${JSON.stringify(chk.reason)}）`)

  // ── ② 逐仪器、逐字段：不是"有一个就算" ──
  for (const part of ['domain', 'lm', 'mmlu']) {
    const p = s[part]
    t.assert(p !== undefined && p !== null, `summarize 产物必须保留仪器 ${part}`)
    t.assert(typeof p.attempted === 'number', `${part}.attempted 必须是数（实得 ${typeof p.attempted}）`)
    t.assert(typeof p.succeeded === 'number', `${part}.succeeded 必须是数（实得 ${typeof p.succeeded}）`)
    t.assert(typeof p.failed === 'number', `${part}.failed 必须是数（实得 ${typeof p.failed}）`)
    t.assert(typeof p.partial === 'boolean', `${part}.partial 必须是布尔（实得 ${typeof p.partial}）`)
    t.assert(p.attempted === fullMeasurement()[part].attempted, `${part}.attempted 必须**原样保留**，不得改写`)
    t.assert(p.failed === fullMeasurement()[part].failed, `${part}.failed 必须**原样保留**，不得改写`)
    t.assert(p.succeeded === fullMeasurement()[part].succeeded, `${part}.succeeded 必须**原样保留**，不得改写`)
  }

  // ── ③ 反向对照：不完整的测量**不得**被判为完整（防"把失败改绿"） ──
  const sBad = summarize(partialMeasurement())
  const chkBad = measurementCompleteness(sBad)
  t.assert(
    chkBad.complete === false,
    `不完整测量经 summarize() 后必须仍然 complete=false（实得 ${JSON.stringify(chkBad)}）`
    + ' —— 若为 true，说明修复退化成"无条件宣称完整"，把失败改绿了',
  )
  t.assert(
    typeof chkBad.reason === 'string' && /failed=60\/260/.test(chkBad.reason),
    `理由必须**点名** failed=60/260（实得 ${JSON.stringify(chkBad.reason)}）—— 否则无法定位病因`,
  )

  // ── ④ 不自证：producedBy 在场，且不是判决者自己 ──
  t.assert(s.integrity !== undefined && s.integrity !== null, 'summarize 产物必须带 integrity 块（producedBy 的载体）')
  t.assert(
    typeof s.integrity?.producedBy === 'string' && s.integrity.producedBy !== '',
    `integrity.producedBy 必须是非空字符串（实得 ${JSON.stringify(s.integrity?.producedBy)}）`,
  )
  t.assert(
    s.integrity.producedBy !== 'effectJudgment',
    'integrity.producedBy 不得是 effectJudgment（测量不得由判决自己产生 —— 不自证）',
  )
}
