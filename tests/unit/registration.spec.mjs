/**
 * registration.spec —— 装配期契约 + **输出契约**（长任务 2 缺陷 3 的正面测试）。
 *
 * 两条纪律：
 *  ① 注册：`ctx.tools.register` 必须收到 output{schema, render}（缺 render 会让整条 entry 被拒）。
 *  ② **注册成功 ≠ 输出契约成立**：必须**真调**，并让返回值穿过与 DSH 同义的"无损 JSON"校验。
 *     本 spec 用只读工具（health / get_report）真调 —— 它们是"每条请求都会走到"的路径。
 */
import { apply, TOOL_NAMES, TOOL_RISK, createCore } from '../../src/index.mjs'
import { ALL_TOOL_SPECS } from '../../src/tools.mjs'
import { findLosslessProblems, assertLossless } from '../../src/output.mjs'

export const NAME = 'unit/registration'

export async function run(t) {
  /* ── ① 装配：像 DSH 一样挑刺 ── */
  const registered = []
  const effects = []
  const mockCtx = {
    logger: { info() {}, warn() {}, error() {} },
    effect(fn, label) { effects.push(label); const d = fn(); return () => d },
    tools: {
      register(tool) {
        // 像 DSH 一样挑刺：缺 output / 缺 render / 缺 parameters 一律拒
        if (!tool.name) throw new Error('register 缺 name')
        if (!tool.description) throw new Error(`${tool.name} 缺 description`)
        if (!tool.parameters || tool.parameters.type !== 'object') throw new Error(`${tool.name} 缺 parameters`)
        if (!tool.output) throw new Error(`${tool.name} 缺 output（会让整条 entry 被拒）`)
        if (!tool.output.schema) throw new Error(`${tool.name} 的 output 缺 schema`)
        if (typeof tool.output.render !== 'function') throw new Error(`${tool.name} 的 output 缺 render`)
        registered.push(tool)
        return () => {}
      },
    },
    get(k) { return k === 'webServer' ? null : undefined },
  }
  apply(mockCtx)
  t.assert(registered.length === 6, `注册了 6 个工具（实际 ${registered.length}）`)
  t.assert(registered.map((r) => r.name).sort().join(',') === [...TOOL_NAMES].sort().join(','), '注册的工具名与 TOOL_NAMES 一致')
  for (const r of registered) {
    t.assert(typeof r.execute === 'function', `${r.name} 有 execute`)
    // render 必须真能产出 text content
    const out = r.output.render({}, { a: 1 })
    t.assert(Array.isArray(out) && out[0]?.type === 'text' && typeof out[0].text === 'string', `${r.name} 的 render 产出 text content`)
  }

  /* ── ② 输出契约：真调只读工具，穿过无损 JSON 校验 ── */
  const core = createCore()
  const byName = (n) => ALL_TOOL_SPECS.find((s) => s.name === n)
  for (const tool of ['training_pretrain_health', 'training_pretrain_get_report']) {
    const env = await core.run(byName(tool), {})
    const problems = findLosslessProblems(env)
    t.assert(problems.ok, `${tool} 真调返回无损 JSON（问题：${problems.problems.slice(0, 3).map((p) => p.path).join(',') || '无'}）`)
    t.assert(env.code === 'OK' || env.code === 'E_INTERNAL', `${tool} 返回可识别 code（${env.code}）`)
    t.assert(env.versions?.pkg === '0.1.0', `${tool} 带版本`)
    t.assert(env.governance?.risk, `${tool} 带治理块（risk=${env.governance?.risk}）`)
  }

  /* ── ③ 无损闸门本身：undefined 必须被抓住（这正是缺陷 3 的形态） ── */
  let threw = false
  try { assertLossless({ a: 1, b: undefined }) } catch { threw = true }
  t.assert(threw, 'assertLossless 拒绝含 undefined 的对象（长任务 2 缺陷 3 的形态）')
  threw = false
  try { assertLossless({ ok: true, nested: { deep: [1, 2, { bad: undefined }] } }) } catch { threw = true }
  t.assert(threw, 'assertLossless 递归拒绝嵌套 undefined')
  threw = false
  try { assertLossless({ n: NaN }) } catch { threw = true }
  t.assert(threw, 'assertLossless 拒绝 NaN')
  t.assert(findLosslessProblems({ a: 1, b: null, c: [1, 'x'] }).ok, 'assertLossless 接受合法值')
  t.assert(findLosslessProblems(undefined).problems[0].path === '$', 'undefined 的路径被定位到 $')

  /* ── ④ 风险声明双向一致 ── */
  t.assert(Object.keys(TOOL_RISK).length === TOOL_NAMES.length, '风险声明与工具一一对应')
}
