/**
 * evaluate —— training-pretrain 的**评估适配层**（任务 A：合并四份 evaluate.mjs 之后）。
 *
 * ## 边界（**不绕开共享库契约**）
 * 合并前本文件是一份 237 行的**实现**，与 training-sft / training-rl / training-qat
 * 的同名文件是**同一算法、四个所有者**。现在：
 *
 *   - 端点原语与通用评估（`post` / `get` / `evalLM` / `evalMMLU` / `evalUniform` /
 *     `measurementCompleteness` / `attachIntegrity` / `forgettingGate`）
 *     **只有一份实现**：`@xiciyang/testing-core`（`src/evaluate.mjs`）。
 *     本文件只做**再导出**（`export … from`），不含任何实现体。
 *   - 本包特有：评估协议版本 `EVAL_PROTOCOL`（进基线缓存键）。
 *
 * ## 两条判据（都可推导，不用魔数）
 *
 * **① 领域能力（本包特有，在 core.mjs 里）**：在**领域留出集**（训练没见过的块）上做配对 NLL 比较。
 *   判据不是"越高越好"的空话，而是由调用方显式给出 `minDomainGainPct`（**本插件不内置默认值**）。
 *
 * **② 灾难性遗忘（沿用 SFT 的形态，参数命名通用 base/pt）**：
 *   - MMLU 准确率：Δ = acc_base − acc_pt；SE = sqrt(p1(1−p1)/n1 + p2(1−p2)/n2)；通过 ⟺ Δ ≤ z·SE
 *   - wikitext 逐文本 NLL：**配对**比较；Δ = mean(nll_pt − nll_base)；SE = std(差)/√n；通过 ⟺ Δ ≤ z·SE
 *   `forgettingGate` 现在**同一份代码**同时服务 sft / pt / rl 三种命名（按候选侧自动识别），
 *   字段名（`ptAcc` / `pplPt` …）与合并前逐字相同。
 *
 * ## ⚠️ 阶段 1 已实测的**适用边界**（必须与结论一起读）
 * 上述 `z·SE` 只建模**逐文本测量噪声**，**不含训练随机性**。阶段 1 实测
 * `σ(Δ)_种子间 ≈ 0.6 × 判据界` ⇒ **单种子 `z·SE` 无法支撑"流程级"结论**
 * （判"这一个模型"可以，判"这套训练流程"不可以）。
 * 详见 `SFT_SEED_REPRODUCIBILITY.md` §3。
 */

/* ------------------------------------------------------- 通用评估（唯一实现在 testing-core） */

export {
  post, evalLM, evalMMLU, evalUniform,
  measurementCompleteness, attachIntegrity, forgettingGate,
} from '@xiciyang/testing-core'

// 判据本体也一并再导出（与合并前一致：pretrain 版本原本就有这一行）
export { dualGate, relativeChange, perplexity, mean, stddev, uniformSignature, meanNllFromLogprobs } from '@xiciyang/testing-core'

/** 评估协议版本 —— 进基线缓存键：协议一变，缓存自动失效。 */
export const EVAL_PROTOCOL = 'pretrain-eval@1'
