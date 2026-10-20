/**
 * data.spec —— 语料侧的可判定性：分块确定性、PII 分层、**轨 2 包含检查的正确性**。
 *
 * 轨 2 是本包为了"整行哈希查不出行内包含"而引入的（用户裁决 B）。
 * 一个检查若不能证明"该抓的抓得到、不该抓的不误伤"，它就不是检查。
 * 故这里用**构造样本**双向验证：包含 ⇒ 命中；不包含 ⇒ 不命中（含"共享 24-gram 但不包含"的诱饵）。
 */
import { chunkText } from '../../src/data/pretrain-prepare.mjs'
import { classifyHit, buildProbeIndex, containedItems } from '../../src/data/corpus-audit.mjs'

export const NAME = 'unit/data'

export async function run(t) {
  /* ── 分块：确定性 + 不丢内容 + 不产碎渣 ── */
  const text = `${Array.from({ length: 400 }, (_, i) => `word${i}`).join(' ')}`
  const a = chunkText(text, 256)
  const b = chunkText(text, 256)
  t.assert(JSON.stringify(a) === JSON.stringify(b), '同输入同 chunkChars ⇒ 逐块相同（确定性）')
  t.assert(a.length > 1, `确实切成了多块（${a.length}）`)
  t.assert(a.every((c) => c.length >= 32), '所有块长度 ≥ 32（不产碎渣）')
  t.assert(a.every((c) => c.length <= 256), '所有块不超过 chunkChars')
  // 不丢内容：块按序拼接后，去空白应等于原文去空白的**前缀**。
  // 为什么是前缀而不是完全相等：`chunkText` **有意丢掉 <32 字符的尾部残块**
  // （避免把"半个词"当训练样本）。要证的是"**中间没有洞**"，而不是"尾部也必须保留"。
  const joined = a.join('').replace(/\s+/g, '')
  const orig = text.replace(/\s+/g, '')
  t.assert(orig.startsWith(joined), '块按序拼接后是原文的前缀（去空白逐字相同）—— 中间没有洞')
  t.assert(orig.length - joined.length < 32, `只可能丢尾部残块（丢了 ${orig.length - joined.length} 字符 < 32）`)
  t.assert(chunkText('short', 2048).length === 0, '过短的文本不产生块')
  const noSpace = 'x'.repeat(600)
  t.assert(chunkText(noSpace, 256).length >= 2, '无空白的长文本按硬切处理（不会卡死）')

  /* ── PII 分层：保留域/占位不算，其余算 ── */
  t.assert(classifyHit('email', 'a@example.com') === 'reserved-domain', 'example.com 是保留域')
  t.assert(classifyHit('email', 'user@foo.com') === 'reserved-domain', 'foo.com 是保留域')
  t.assert(classifyHit('email', 'test@whatever.org') === 'placeholder-localpart', 'test@ 本地部分是占位')
  t.assert(classifyHit('email', 'real.person@gmail.com') === 'other', '真实邮箱判为 other（真 PII）')
  t.assert(classifyHit('ipv4', '192.168.1.1') === 'reserved-ip', '192.168/16 是保留段')
  t.assert(classifyHit('ipv4', '8.8.8.8') === 'other', '公网 IP 判为 other')
  t.assert(classifyHit('cn-mobile', '13800138000') === 'other', '手机号判为 other')

  /* ── 轨 2：包含检查的正确性（双向） ── */
  const items = [
    { text: 'The mitochondria is the powerhouse of the cell and it produces ATP for the organism', set: 'mmlu-test', hash: 'h1' },
    { text: 'In a distant country the population of foxes declined sharply during the winter months', set: 'wikitext', hash: 'h2' },
  ]
  const idx = buildProbeIndex(items)

  // 正例：文档**包含**第 0 项全文 ⇒ 必须命中
  const docContains = 'Some leading web text. ' + items[0].text + ' And trailing prose continues here for a while.'
  const hit0 = containedItems(docContains, idx, items)
  t.assert(hit0 !== null && hit0.includes(0), '文档包含评估项全文 ⇒ 轨 2 命中（该抓的抓得到）')

  // 负例：文档与评估项**共享若干 24-gram** 但不包含全文 ⇒ 不得命中
  const shared = items[0].text.split(' ').slice(0, 8).join(' ')
  const decoy = `prefix ${shared} then it diverges completely into unrelated subject matter entirely`
  t.assert(containedItems(decoy, idx, items) === null, '共享 24-gram 但未包含全文 ⇒ 轨 2 不误伤（诱饵被正确排除）')

  // 负例：完全无关
  t.assert(containedItems('nothing to see here at all just ordinary words in a row', idx, items) === null, '无关文本 ⇒ 不命中')

  // 正例：第二项
  const doc2 = 'intro text ' + items[1].text + ' outro'
  const hit1 = containedItems(doc2, idx, items)
  t.assert(hit1 !== null && hit1.includes(1), '第二项被包含时也能命中（索引不是只对第一项有效）')

  // 短文档（< 窗口）不得崩
  t.assert(containedItems('tiny', idx, items) === null, '极短文档不崩且不命中')
}
