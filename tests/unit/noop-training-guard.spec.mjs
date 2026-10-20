/**
 * noop-training-guard.spec —— **"空操作训练"防线**（长任务 3 · 阶段 6 真机抓到）。
 *
 * ## 被测的缺陷
 * 第 2 轮续训：训练 500 步、`globalStep` 500→1000、绝对量不变量 ✅、连续性 ✅ ——
 * 但**权重一步没动**（三个 checkpoint 与续训源逐字节相同）。
 * 根因：LR 调度的 `total` 取了本 run 的 `maxSteps`，而 `step` 是**绝对** `globalStep`
 * ⇒ 续训时 prog 直接到 1、**lr=0** ⇒ 每步 opt.step() 都是空操作。
 *
 * ## 为什么这个 spec 是"源码级断言"而不是行为测试
 * LR 调度在**容器内 Python** 里，跑它要 GPU + 容器（属真机测试，不进 CI）。
 * CI 能守住的是**防线还在不在** —— 一旦有人删掉 `lrTotalSteps` 转发、
 * 拿掉容器内的空操作闸门、或去掉产物级 `weightsChanged` 断言，这里立刻红。
 * 判据的**数值行为**已由真机证据负责（`out/pretrain-loop-report.json`）。
 */
import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

export const NAME = 'unit/noop-training-guard'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const py = readFileSync(join(PKG, 'python', 'pretrain_trainer.py'), 'utf8')
const trainSrc = readFileSync(join(PKG, 'src', 'train.mjs'), 'utf8')
const coreSrc = readFileSync(join(PKG, 'src', 'core.mjs'), 'utf8')

export async function run(t) {
  /* ── ① LR 调度的 total 必须是绝对终点 ── */
  t.assert(/_lr_at/.test(py), 'trainer.py 里有 _lr_at')
  t.assert(/lrTotalSteps/.test(py), 'trainer.py 的 LR 调度读 lrTotalSteps（绝对终点），而不是只用 maxSteps')
  t.assert(
    /total = max\(int\(S\.cfg\.get\("lrTotalSteps", S\.cfg\.get\("maxSteps"/.test(py),
    'total 的取值优先级是 lrTotalSteps → maxSteps（保住向后兼容，同时修掉续训 lr=0）',
  )
  t.assert(/续训/.test(py) && /lr\s*=\s*0|lr=0|lr = 0/.test(py), 'trainer.py 里写明了"续训 lr=0"的根因（防后人删注释后再踩）')

  /* ── ② 过程级闸门：≥2 步却从未见过非零 lr ⇒ 响亮失败 ── */
  t.assert(/lr_max_seen/.test(py), 'trainer.py 跟踪 lr_max_seen')
  t.assert(/空操作训练/.test(py), 'trainer.py 有"空操作训练"闸门')
  t.assert(/ran >= 2 and S\.lr_max_seen == 0\.0/.test(py), '闸门条件是"已跑 ≥2 步且 lr 从未非零"（不是一有 lr=0 就炸 —— 调度末尾 lr 本就为 0）')

  /* ── ③ **产物级**闸门：权重指纹必须变（这一条才是正解） ── */
  t.assert(/def param_fingerprint/.test(py), 'trainer.py 有 param_fingerprint')
  t.assert(/weightsChanged/.test(py), 'trainer.py 在 save/export 里报告 weightsChanged')
  t.assert(/"weightsChanged": fp != S\.init_fingerprint/.test(py), 'weightsChanged 由"当前指纹 != 起点指纹"得出')

  /* ── ④ 转发链不能断（prepare() 是逐字段显式转发，漏一个就静默丢一个） ── */
  t.assert(/lrTotalSteps: cfg\.lrTotalSteps/.test(trainSrc), 'train.mjs 的容器 config 显式转发 lrTotalSteps')
  t.assert(/lrTotalSteps: baseGlobalStep \+ maxSteps/.test(coreSrc), 'core.mjs 计算 lrTotalSteps = baseGlobalStep + maxSteps（与 training-sft 的 offset + 新增步数 对齐）')

  /* ── ⑤ 产物级断言必须真的抛错（不是只记日志） ── */
  t.assert(/wc === false/.test(coreSrc) && /E_INTERNAL/.test(coreSrc), 'core.mjs 在 weightsChanged===false 时抛错')
  t.assert(/空操作训练：本 run 跑了/.test(coreSrc), 'core.mjs 的报错信息写明是空操作训练')

  /* ── ⑥ 反向：不得为了"让它过"而放宽 —— LR 里不允许出现"lr 下限"这类兜底 ── */
  t.assert(!/max\(lr,\s*1e-|lr\s*=\s*max\(/.test(py), 'trainer.py 未给 lr 加"下限兜底"（那会把 lr=0 掩盖成"有点效果"，属调宽容度）')
}
