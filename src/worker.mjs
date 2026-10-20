/**
 * worker —— 容器内 Python 训练执行体的**持久 worker 客户端**。
 *
 * 设计（为什么不是"每步起一个容器"、也不是"Python 自己跑完"）见 python/trainer.py 顶部。
 * 这一侧负责：
 *   - 用 docker 起一个**长驻**容器，stdin/stdout 走行分隔 JSON 协议
 *   - 请求-应答配对（id 严格匹配；收到错配 id 即失败，不静默错位）
 *   - stderr 单独转发（人类可读日志），**stdout 只允许协议行**
 *   - 超时、进程退出、协议解析失败都**响亮失败**
 *
 * 安全：容器以 `--rm`、只读挂载模型与数据、可写挂载输出目录；
 * 训练工具链镜像固定（`training-sft/toolchain:0.1.0`）。
 */
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import { join, dirname } from 'node:path'
import { userInfo } from 'node:os'
import { fileURLToPath } from 'node:url'
import { ToolError, ERROR_CODES } from './version.mjs'

export class ContainerWorker {
  constructor({ image, mounts = [], gpus = true, entry = '/work/trainer.py', python = 'python3', env = {}, log = () => {}, user = null }) {
    this.image = image
    this.mounts = mounts
    this.gpus = gpus
    this.entry = entry
    this.python = python
    // **以宿主 uid:gid 运行容器**：否则容器（root）写出的文件宿主改不动 ——
    // 实测踩到：容器建了 checkpoints/step-000050/adapter/ 后，宿主写 payload.json 得 EACCES。
    // 同时把 HOME/缓存指到容器内 /tmp，避免非 root 写 /root/.cache 失败。
    this.user = user === null ? (typeof process.getuid === 'function' ? `${process.getuid()}:${process.getgid()}` : null) : user
    // **非 root uid 必须显式给 USER/LOGNAME**（长任务 3 · 阶段 3 冒烟闸实测抓到的自身缺陷）：
    //   容器内以宿主 uid 运行，而该 uid 在容器的 /etc/passwd 里**没有条目** ⇒
    //   `getpass.getuser()` 抛 `KeyError: getpwuid(): uid not found: 1000`；
    //   而 **torch 的缓存目录解析会走 getpass**（`torch._inductor.runtime.cache_dir_utils`）。
    //   两种看似无关的表现，同一个根因：
    //     ① `KeyError: getpwuid(): uid not found: 1000`（不给 TORCHINDUCTOR_CACHE_DIR 时）
    //     ② `RuntimeError: Artifact of type=inductor already registered in mega-cache artifact factory`
    //        （给了 TORCHINDUCTOR_CACHE_DIR、绕过 getpass 之后，模块半初始化再被重入）
    //   实测：补上 USER/LOGNAME 后**两条都消失**（同一镜像、同一 uid、仅此差异）。
    //   注意：这意味着**只给 TORCHINDUCTOR_CACHE_DIR 并不够** —— 它把报错换了个样子，没解决根因。
    const username = (() => {
      try { return userInfo().username || 'user' } catch { return process.env.USER || process.env.LOGNAME || 'user' }
    })()
    this.env = {
      HOME: '/tmp', USER: username, LOGNAME: username,
      XDG_CACHE_HOME: '/tmp/.cache', HF_HOME: '/tmp/.cache/huggingface',
      TRITON_CACHE_DIR: '/tmp/.triton', TORCHINDUCTOR_CACHE_DIR: '/tmp/.torchinductor',
      ...env,
    }
    this.log = log
    this.child = null
    this.seq = 0
    this.pending = new Map()
    this.stderrTail = []
  }

  args() {
    const a = ['run', '-i', '--rm']
    if (this.gpus) a.push('--gpus', 'all')
    if (this.user) a.push('--user', this.user)
    for (const m of this.mounts) a.push('-v', m)
    for (const [k, v] of Object.entries(this.env)) a.push('-e', `${k}=${v}`)
    a.push('--entrypoint', this.python, this.image, this.entry)
    return a
  }

  start() {
    if (this.child) return this
    this.child = spawn('docker', this.args(), { stdio: ['pipe', 'pipe', 'pipe'] })
    const rl = createInterface({ input: this.child.stdout })
    rl.on('line', (line) => {
      let msg
      try { msg = JSON.parse(line) } catch { this.log(`[worker] 非协议行（忽略）：${line.slice(0, 160)}`); return }
      const p = this.pending.get(msg.id)
      if (!p) { this.log(`[worker] 未知 id 的应答 ${msg.id}（忽略）`); return }
      this.pending.delete(msg.id)
      p.resolve(msg)
    })
    const rlErr = createInterface({ input: this.child.stderr })
    rlErr.on('line', (line) => {
      this.stderrTail.push(line)
      if (this.stderrTail.length > 40) this.stderrTail.shift()
      if (!line.startsWith('Loading weights') && !line.startsWith('Writing model shards') && line.trim() !== '') this.log(`[worker:err] ${line.slice(0, 200)}`)
    })
    this.child.on('exit', (code, sig) => {
      for (const [, p] of this.pending) p.reject(new Error(`训练 worker 退出（code=${code} sig=${sig}）；stderr 尾部：${this.stderrTail.slice(-4).join(' | ').slice(0, 300)}`))
      this.pending.clear()
      this.child = null
    })
    return this
  }

  call(cmd, payload = {}, { timeoutMs = 30 * 60 * 1000 } = {}) {
    if (!this.child) throw new Error('worker 未启动')
    const id = ++this.seq
    const req = JSON.stringify({ id, cmd, ...payload })
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`worker 命令 ${cmd} 超时（${timeoutMs}ms）`))
      }, timeoutMs)
      this.pending.set(id, {
        resolve: (m) => {
          clearTimeout(timer)
          if (m.ok === false) {
            const e = new ToolError(ERROR_CODES.E_INTERNAL, { hint: `训练 worker 内部错误（${cmd}）：${m.error}`, details: { cmd, trace: String(m.trace || '').slice(-600) } })
            reject(e)
          } else resolve(m)
        },
        reject: (e) => { clearTimeout(timer); reject(e) },
      })
      this.child.stdin.write(`${req}\n`)
    })
  }

  async stop() {
    if (!this.child) return
    try { await this.call('shutdown', {}, { timeoutMs: 20000 }) } catch { /* 已退出 */ }
    // **竞态（长任务 3 阶段 1 实测抓到）**：`shutdown` 应答之后 Python 才退出，
    // 容器 `exit` 事件可能已经触发并把 `this.child` 置为 null。
    // 若此处还去读 `this.child.stdin` / `c.on` 就会 TypeError（Cannot read properties of null）。
    // 长任务 2 未暴露：唯一调用方 `train.mjs` 的 finally 用 try/catch 吞掉了该异常。
    // 两处都要防：① stdin 可能随 child 一起没了；② child 可能已为 null ⇒ 进程已退，无需再等。
    try { this.child?.stdin?.end() } catch { /* ignore */ }
    const c = this.child
    if (!c) return
    await new Promise((r) => {
      if (c.exitCode !== null || c.signalCode !== null) { r(); return }
      const t = setTimeout(() => { try { c.kill('SIGKILL') } catch { /* ignore */ } ; r() }, 8000)
      c.on('exit', () => { clearTimeout(t); r() })
    })
    this.child = null
  }
}

/** 训练容器的标准挂载（模型只读 / 数据只读 / 输出可写 / 训练脚本只读）。 */
export function trainingMounts({ modelDir, datasetDir, outDir, wikitextFile = null, pythonDir = null, extra = [] }) {
  // 关键：模型目录必须挂成 `/models/<basename>`（挂成 /models 会让容器内看到的是**目录内容**，
  // transformers 找不到 config.json 就把它当成 HF repo id，报 "Repo id must be in the form ..."。
  // 这是实测踩到的第二个容器路径坑。）
  const base = String(modelDir).split('/').filter(Boolean).pop()
  const m = [`${modelDir}:/models/${base}:ro`, `${datasetDir}:/data:ro`, `${outDir}:/out`]
  // 训练脚本必须**挂进容器**（否则 /work/trainer.py 不存在）—— 这一条是实测踩到的
  const py = pythonDir || join(dirname(fileURLToPath(import.meta.url)), '..', 'python')
  m.push(`${py}:/work:ro`)
  if (wikitextFile) m.push(`${wikitextFile}:/eval/wikitext.jsonl:ro`)
  return [...m, ...extra]
}
