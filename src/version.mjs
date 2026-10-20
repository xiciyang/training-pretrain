/**
 * version / 错误码闭集 —— training-pretrain。
 *
 * 错误码**闭集**（与 training-sft / inference-diagnose 同一纪律：新增码 = schema 版本变更，须回文档修订）。
 * 不设"其他"桶：每个码对应一个**可独立失效、且失效语义不同**的依赖源或闸门。
 *
 * 文档修订记录（本文件即错误码的文档）：
 *   - 追加 `E_INSTRUMENT_LOSSY`：**additive**，未改任何既有码的语义、未改 SCHEMA_VERSION。
 *     语义 = "测量仪器经过 serving 的有损降级档位，故本次测量不得被签收"（fail-closed 闸门）。
 */
export const PACKAGE_VERSION = '0.1.0'
export const SCHEMA_VERSION = '1'
export const MCP_INTERFACE_VERSION = '1.0.0'
export const AUDIT_SCHEMA = 'training-pretrain/audit@1'
export const REGISTRY_SCHEMA = 'training-pretrain/registry@1'

export const ERROR_CODES = Object.freeze({
  // 参数与输入
  E_MISSING_PARAM: 'E_MISSING_PARAM',
  E_CORPUS_INVALID: 'E_CORPUS_INVALID',
  E_CORPUS_LEAK: 'E_CORPUS_LEAK',              // 语料与评估集相交（整文档哈希 或 行内包含）
  // 续训（本包特有：契约冻结条件 2）
  E_RESUME_STATE_MISSING: 'E_RESUME_STATE_MISSING',
  E_RESUME_CONTINUITY: 'E_RESUME_CONTINUITY',  // 步号/数据位置不连续（"不重不漏"被否）
  // 依赖源
  E_BASE_MODEL_NOT_FOUND: 'E_BASE_MODEL_NOT_FOUND',
  E_IMAGE_UNAVAILABLE: 'E_IMAGE_UNAVAILABLE',
  E_SANDBOX_UNAVAILABLE: 'E_SANDBOX_UNAVAILABLE',
  E_SERVING_UNAVAILABLE: 'E_SERVING_UNAVAILABLE',
  // 预算
  E_BUDGET_EXCEEDED: 'E_BUDGET_EXCEEDED',      // 超限 → 暂停并报告（不降级）
  // 状态
  E_STATE_CONFLICT: 'E_STATE_CONFLICT',
  E_RUN_NOT_FOUND: 'E_RUN_NOT_FOUND',
  E_CONFIRMATION_REQUIRED: 'E_CONFIRMATION_REQUIRED',
  // 门禁
  E_GATE_FAILED: 'E_GATE_FAILED',
  E_INSTRUMENT_LOSSY: 'E_INSTRUMENT_LOSSY',    // 测量仪器走了 serving 的有损降级档位（fail-closed）
  // **输出契约**（长任务 2 缺陷 3：`undefined` 键会让 DSH 的"无损 JSON"校验拒绝整条结果）
  E_OUTPUT_NOT_LOSSLESS: 'E_OUTPUT_NOT_LOSSLESS',
  E_INTERNAL: 'E_INTERNAL',
})

export class ToolError extends Error {
  constructor(code, body = {}) {
    super(`${code}: ${body.hint || ''}`)
    this.name = 'ToolError'
    this.code = code
    this.body = body
  }
}

export const ALL_CODES = Object.freeze(Object.values(ERROR_CODES))
