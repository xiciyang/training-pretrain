# DEPLOYMENT_REPORT.md — training-pretrain（长任务 3 · 阶段 2）

> 交付形态：**本地 npm 包（private）+ DSH bundle 插件**。与 `training-sft` 同形，可直接并列装配。

---

## 1. 部署物

| 项 | 值 |
|---|---|
| 包名 | `training-pretrain`（`private: true`） |
| 版本 | `0.1.0` |
| 位置 | `inference-training/training-pretrain/` |
| 入口 | `src/index.mjs`（host 半；`inject: ['tools']`） |
| CLI | `bin/training-pretrain.mjs` |
| MCP | `mcp/server.mjs`（stdio，行分隔 JSON-RPC 2.0） |
| bundle patch | `cordis.patch.yml`（`name: 'training-pretrain'` + `inject: [tools]`） |
| 能力清单 | `manifest.yaml`（6 capabilities + 7 条 `not_provided`） |
| 技能 | `skills/SKILL.md` |

**三条通道共用同一份实现**（`src/tools.mjs` 的规格 + `src/core.mjs` 的实现），**不存在行为分叉**。

---

## 2. 运行时依赖

| 类别 | 内容 |
|---|---|
| 运行时 npm 依赖 | **0**（`host.runtimeDependencies: 0`；不引入任何第三方 npm 包） |
| 共享库（file: 链接） | `@ai-infra/{training,testing,data,state,audit,sandbox}-core`（长任务 1 **冻结**的六库） |
| peerDependencies | `@deepseek-ai/dsh-tools`、`cordis` |
| optionalDependencies | `torch` / `transformers` / `accelerate`（**声明式**：实际在容器内，不在宿主装） |
| 外部服务 | `inference-serving`（MCP，validate 阶段的端点生命周期） |
| 容器镜像 | `training-sft/toolchain:0.1.0` @ `f82151dd84fb`（**复用**，不新建） |

**装配方式**（二选一）：
```
# A. 运行时热注入（本轮验证用）
dev_inject_plugin  <repo>

# B. 正式装配（重启后生效）——与 training-sft 同形
在 profile 的 cordis.patch.yml / bundles 中加入 training-pretrain（package.json 已带 dsh.bundle.patch）
```

---

## 3. 已验证的部署路径（真机）

| 步骤 | 结果 | 证据 |
|---|---|---|
| 依赖链接 | ✅ | `node_modules/@ai-infra/*` 六个 junction 指向 `inference-training/shared/libs/*` |
| 单测 | ✅ | `node tests/run.mjs` → **5 spec / 120 断言 / ALL SPECS PASSED** |
| 契约 CI | ✅ | `inference-contracts` → **73/73**（含 C17/C18） |
| 热注入 | ✅ | `dev_inject_plugin` → **host ✓**；loader 状态 `[active] … [injected]` |
| 装配期断言 | ✅ | `apply()` 未抛错 ⇒ 风险声明双向一致 / 命名空间零交集 / 6 个 `output{schema,render}` 齐备 |
| 只读工具真调 | ✅ | `health` 与 `get_report` 经 `createCore().run` 返回**无损 JSON**（`findLosslessProblems.ok=true`） |
| CLI | ✅ | `node bin/training-pretrain.mjs health` → `ok:true`；`get_report` → 9 个 run |
| 端到端闭环 | ✅ | prepare → train(500 步) → export → validate：**pass=true** |
| 续训连续性 | ✅ | 真机三段（同分片 / 推进分片）全部 PASS |
| 残留清理 | ✅ | 容器 **0**、存活端点 **0**（10 条 serving 记录全 `torn_down`）、既有服务未触碰 |

---

## 4. 运维要点（部署时容易踩的）

### 4.1 **非 root uid 必须给 `USER`/`LOGNAME`**（本轮实测抓到的自身缺陷）

容器以宿主 uid 运行（为避免 root 产出的文件宿主改不动），而该 uid 在容器 `/etc/passwd` 里没有条目：

| 配置 | 结果 |
|---|---|
| `HOME=/tmp` | ❌ `KeyError: getpwuid(): uid not found: 1000` |
| `HOME` + **`USER`/`LOGNAME`** | ✅ OK |
| `HOME` + `TORCHINDUCTOR_CACHE_DIR` | ❌ `Artifact of type=inductor already registered in mega-cache artifact factory` |

**同一根因**：`getpass.getuser()` 失败，而 **torch 的缓存目录解析会走它**；
只给 `TORCHINDUCTOR_CACHE_DIR` 只是绕开 getpass，报错换个样子。
⇒ `ContainerWorker` 已显式传 `USER`/`LOGNAME`（**两个包都修了**）。

### 4.2 沙箱层级

训练属 write 档 ⇒ **必须容器层**（`requireTier('container')`）。本机三层实测：
`process ✓ / container ✓ / vm ✗`（无 VM）。系统级动作一律拒绝。

### 4.3 状态与审计位置

| 项 | 路径 |
|---|---|
| 注册表 | `~/.ai-infra/state/pretrain-registry.yaml`（`TRAINING_PRETRAIN_ROOT` 可覆盖根） |
| 运行目录 | `~/.ai-infra/run/training-pretrain/<runId>/`（checkpoints / exported） |
| 审计链 | `~/.ai-infra/logs/audit/training-pretrain/`（实测 34 条，独立复算通过） |
| 语料 | `~/.ai-infra/state/datasets/dolma__pretrain__s<seed>__n<tokens>/` |
| 只读健康路由 | `GET /__training_pretrain/health`（webServer 存在时注册） |

### 4.4 部署时**不要**做的事

- 不要把 `training-pretrain` 与 `training-sft` 装成同一个包名 / entry id；
- 不要给领域门禁阈值加内置默认值（CI `C18d` 会红，且违反纪律）；
- 不要图省事给 license 加兜底默认值（§1.2 缺陷 B）。

---

## 5. 操作手册（最短路径）

```bash
# 1) 语料（真机 ~10s）
node bin/training-pretrain.mjs prepare --targetTokens 15000000 --shardCount 3

# 2) 训练一个分片（≤500 步 / ≤2 GPU 小时）
node bin/training-pretrain.mjs train \
  --datasetDir ~/.ai-infra/state/datasets/dolma__pretrain__s20260929__n15000000 \
  --shardIndex 0 --maxSteps 500

# 3) 续训（同分片）—— 名字请从 get_report 读，别自己拼
node bin/training-pretrain.mjs get_report --runId <RUN_ID>

# 4) 导出 + 验证（阈值必须自己给）
#    经 MCP/DSH：training_pretrain_validate { runId, thresholds:{minDomainGainPct:1.0} }
node bin/training-pretrain.mjs health
```

---

## 6. 回滚

| 场景 | 动作 |
|---|---|
| 某个 run 不要了 | `training_pretrain_rollback { runId }`（`purge:true` 才删该 run 目录；**绝不动语料/基座/其它 run**） |
| 插件本身要下线 | `dev_uninject_plugin training-pretrain`（卸 loader entry + junction，免重启） |
| 语料要重做 | 删对应 `state/datasets/dolma__pretrain__…` 目录后重跑 `prepare`（确定性：同输入+同 seed ⇒ 逐块相同） |

---

## 7. 与既有插件的关系

| 插件 | 关系 |
|---|---|
| `training-sft` | **并列**。共用六个共享库与容器 worker 模式；命名空间 `training_pretrain_*` 与 `training_sft_*` 零交集（CI 断言） |
| `inference-serving` | **调用方**（validate 阶段的 deploy/teardown）；只碰自己租用的端点 |
| `inference-quantization` | **不交互**（本包 `not_provided: quantization`） |
| `inference-tuning` | **不交互** |

---

## ✅ 交付后确证：多种子回溯结果（LONGTASK4 · 阶段 4）

本报告的 **领域能力提升** 与 **通用能力保持** 结论，经 3 个新种子
（1042/2042/3042，规则 `seed_k = 42+1000k`）以相同 serving 仪器与协议复验，**全部稳定**：

| 结论 | 单种子原值 | 3 种子 μ | **R/μ** | 判定 |
|---|---|---|---|---|
| 领域能力提升 | −2.92% | −2.96% | **0.6%** | 稳定 |
| 通用保持（MMLU） | −0.0500 | −0.0322 | 31.0% | 稳定 |
| 通用保持（wikitext ΔNLL） | −0.0645 | −0.0667 | 3.8% | 稳定 |

领域提升的种子间标准差仅 0.010 个百分点 ⇒ 该改善**由数据驱动，不由种子驱动**。
详见 `SEED_REPRODUCIBILITY_RETROSPECT.md`。
