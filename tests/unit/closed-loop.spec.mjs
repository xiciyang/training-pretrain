/**
 * closed-loop.spec —— **缺陷 13 回归测试（pretrain 侧）**：`train` 的终态必须能被 `validate` 接受。
 *
 * 同 training-sft 的同名 spec：`train` 留下 `trained`，`validate` 需要 `exported`，
 * 而公开工具面没有 export 入口 ⇒ 闭环在工具面内不可达。修复后 validate 自己先 `exportRun`。
 *
 * 全部不依赖 GPU / 容器 / 网络。同时证明闸门没有被删掉。
 */
import * as api from '../../src/core.mjs'

export const NAME = '闭环状态机（缺陷 13 回归）'

const TRAIN_TERMINAL = 'trained'
const FLOW_THROUGH = ['exported', 'validated', 'gate-failed']
const MUST_REJECT = ['failed', 'running', 'rolled-back', null, undefined, '', 'weird-state']

export async function run(t) {
  t.assert(
    api.validateStatusPlan(TRAIN_TERMINAL) !== 'reject',
    `train 的终态 '${TRAIN_TERMINAL}' 被 validate 拒绝 ⇒ 缺陷 13 复发（闭环在公开工具面内不可达）`,
  )
  t.assert(
    api.validateStatusPlan(TRAIN_TERMINAL) === 'auto-export',
    `train 的终态 '${TRAIN_TERMINAL}' 应触发 auto-export，实得 '${api.validateStatusPlan(TRAIN_TERMINAL)}'`,
  )
  for (const s of FLOW_THROUGH) {
    t.assert(api.validateStatusPlan(s) === 'proceed', `状态 '${s}' 应直接放行，实得 '${api.validateStatusPlan(s)}'`)
  }
  for (const s of MUST_REJECT) {
    t.assert(api.validateStatusPlan(s) === 'reject', `非法状态 ${JSON.stringify(s)} 必须被拒，实得 '${api.validateStatusPlan(s)}'`)
  }
  for (const s of [TRAIN_TERMINAL, ...FLOW_THROUGH]) {
    t.assert(api.validateStatusPlan(s) !== 'reject', `合法状态 '${s}' 被误拒`)
  }
}
