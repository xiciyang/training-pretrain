# ACCEPTANCE_MATRIX.md — training-pretrain（长任务 3 · 阶段 2）

> **范围**：`training-pretrain` 插件（第 7 个插件）的验收矩阵。
> **证据等级**：`[verified]` 本机一手实测 / `[claimed]` 二手 / `[inferred]` 推论。
> **判据**：P0 = 必须真机；P1 = 指标门禁；P2 = 单测/集成。
> 产物：`out/pretrain-vertical-loop.json` · `out/pretrain-resume-continuity.json` · `out/pretrain-corpus-audit.json` · `STEP0_REPORT.md`

---

## 0. 汇总

```
P0（9 项，必须真机）  : 9/9 通过
P1（3 项，指标门禁）  : 3/3 通过
P2（6 项，单测/集成） : 6/6 通过（其中 1 项标注"部分"，见 P2-5）
```

**契约层**：`inference-contracts` CI **73/73**（含新增 C17 training-pretrain 单测、C18 契约与三条硬纪律）。
**训练侧单测**：`training-pretrain` 5 spec / **120 断言**；`training-sft` 4 spec / 196 断言（回归）。
**GPU 消耗**：**0.3017**（垂直闭环）+ **0.0268**（续训连续性）= **0.3285 GPU 小时**（阶段上限 8）。

---

## 1. P0 —— 必须真机（9/9）

| # | 判据 | 结果 | 证据 |
|---|---|---|---|
| **1** | **数据血缘**（source / revision / seed / license / 各步计数 / 复核结论 / `allClear`） | ✅ | `lineage.json`：`source=allenai/dolma`、`revision=7f481405…`、`seed=20260929`、`license.spdx=odc-by` 且 `evidence[]` 3 条 URL、`pipeline` 6 步、`verification` 2 组、`allClear=true` |
| **2** | **checkpoint 可被 serving 加载并真实前向** | ✅ | 导出的 HF 目录经 `inference_serving_deploy` 部署，`usability.ok=true`；随后完成领域/MMLU/wikitext 三类真实评估 |
| **3** | **契约一致**（eval-result 符合冻结 schema；判据来自 testing-core） | ✅ | CI `C18a/C18b/C18c/C18e` 全过；`testingCore` 版本随产物落盘 |
| **4** | **资源上限强制**（步数/GPU 小时/参数/并发） | ✅ | `maxSteps>500`/`maxGpuHours>2` 显式 `E_BUDGET_EXCEEDED`；`training-core.enforceLimits` 在循环内强制；CI `C18d` 断言阈值无内置默认 |
| **5** | **审计哈希链** | ✅ | `health.audit = {ok:true, shards:1, checked:34, ok:true}` —— 独立复算通过 |
| **6** | **灾难性遗忘检查**（真机、双分量） | ✅ | MMLU 0.3833 → **0.4333**（未退化）；wikitext 配对 NLL Δ=**−0.0645** bound=0.0090 → pass |
| **7** | **双轨去泄漏**（整行哈希 + 行内子串；文档级 + **块级**复核） | ✅ | 文档级：轨1 **0** / 轨2 **1**；块级复核：轨1 0 / 轨2 0；`recheck.ok=true` |
| **8** | **PII 100% 扫描**（不抽样；整文档剔除、不改写） | ✅ | 扫 22,968 → 剔 **391（1.702%）**；`exclusions.json` 仅哈希 |
| **9** | **分片续训连续性**（**绝对量**可机器验证） | ✅ | 真机三段全过：20 步→gs20/320；从 `step-000010` 续 15 步→**gs25/400**（非重置）；推分片→gs30/160（按新分片重置） |

### 1.1 P0-9 的详细证据（本包最核心的判据）

`out/pretrain-resume-continuity.json`，真机，耗 **0.0268 GPU 小时**：

| 段 | 场景 | 期望（**绝对量**） | 实测 | 结果 |
|---|---|---|---|---|
| 1 | 全新 shard 0，20 步 | globalStep 20 / consumedChunks 320 | 20 / 320 | ✅ |
| 2 | **同分片续训**：`step-000010` + 15 步 | globalStep **25** / chunks **400** | 25 / 400 | ✅ |
| 2' | checkpoint 内状态与算术一致 | gs 10 / chunks 160 / cursor 160 / epoch 0 | 逐字段一致 | ✅ |
| 3 | **推进分片**：shard 0 末 ckpt → shard 1，10 步 | globalStep **30** / chunks **160** | 30 / 160 | ✅ |

> 这正是长任务 2 **缺陷 2**（"续训 20 步实跑 220 步、而两侧步号断言**全对**"）的正面回答：
> 相对量断言当时全对却仍然错；本包用 **`consumedChunks − base == (globalStep − baseGlobalStep) × chunksPerStep`**
> 这条绝对量等式把它堵死，且**每步**都核（不是只核一次）。

### 1.2 复现时顺带修掉的两个**自身缺陷**（都在交付物里留了根因注释）

| # | 缺陷 | 根因 | 影响面 |
|---|---|---|---|
| A | `ContainerWorker.stop()` 与容器 `exit` 事件**竞态**（`this.child` 被置空后仍解引用 → `TypeError: Cannot read properties of null`） | `shutdown` 应答后进程才退出，`exit` 处理器可能已把 `child` 置 null | **`training-sft` 同样存在**（长任务 2 未暴露：`train.mjs` 的 finally 用 try/catch 吞掉了）。已在**两个包**修好 |
| B | **导出时 license 被错标为 `apache-2.0`**（实际语料是 **ODC-BY**，署名文本为空） | `data-core.lineage()` 把 `extra` **平铺到顶层**，而我读了 `lineage.extra.license` → undefined → 落到 `?? {spdx:'apache-2.0'}` **兜底默认值** | **危险**：把"我不知道"写成"我断言它是 apache-2.0"。已改为 **fail-closed**（读不到 license 就拒绝导出，不给默认值），并重新导出验证 `license: odc-by` |

> 缺陷 B 是「数据集 license 以**实际卡片**为准」这条纪律的**反面教材**：
> 真正危险的不是"读不到"，而是**读不到时给了一个看起来合理的默认值**。

---

## 2. P1 —— 指标门禁（3/3）

| # | 判据 | 阈值（**调用方显式给出**） | 实测 | 结果 |
|---|---|---|---|---|
| **10** | **领域能力提升** | `minDomainGainPct = 1.0` | 领域留出 NLL **3.248039 → 3.153146**（**−2.92%**） | ✅ |
| **11** | **通用能力保持率** | 遗忘门禁 z·SE | MMLU **0.3833 → 0.4333**（提升）；wikitext NLL **3.250258 → 3.185711**（Δ=−0.0645 ≤ bound 0.0090） | ✅ |
| **12** | **测量成立性** | `uniform.ok`（均匀签名）+ validity 三项 | `pt-forward` n>0；`pt-not-uniform` ok；`domain-samples` 200/200 | ✅ |

**门禁合成**：`pass = domainGate.pass && forgettingGate.pass` = **true**。

### 2.1 一个必须一起读的限定（沿用阶段 1 的结论）

阶段 1 实测：遗忘判据的 `z·SE` **只建模逐文本测量噪声，不含训练随机性**
（种子间 `σ(Δ) ≈ 0.6 × 判据界`，`R/μ = 1011%` ⇒ 单种子判据的"测量不成立"）。

本阶段的 wikitext Δ = **−0.0645**，是阶段 1 观测到的种子间 σ（1.3e-3）的 **约 50 倍** ——
**这个效应量远超种子噪声，故结论稳健**。但按同一条纪律，**它仍是单种子结果**：
结论只能表述为"**这个模型**在领域语料上继续预训练后，领域与通用指标都改善"，
**不能**表述为"**这套流程**一定如此"（流程级判断需多种子 + 叠加种子间方差）。

---

## 3. P2 —— 单测 / 集成（6/6，其中 1 项部分）

| # | 判据 | 结果 | 证据 |
|---|---|---|---|
| **13** | **输入处理**（参数校验、缺文件即拒、越界即拒） | ✅ | `shardIndex` 越界抛 `E_MISSING_PARAM`；`chunkChars` 128..65536、`shardCount` 1..64 校验；缺 `lineage.json`/`shards.json`/分片文件各有专门错误码 |
| **14** | **安全**（风险声明双向一致、确认闸门、白名单） | ✅ | `unit/risk` 45 断言：`TOOL_RISK` ⇄ `manifest.yaml` **逐字段一致**；`not_provided` 显式声明 7 项（rl/qat/quantization/distributed/modify_base_model/sanitize_data/synthesize_data） |
| **15** | **健壮性**（错误码闭集、fail-closed） | ✅ | 19 个错误码闭集；语料泄漏/血缘缺失/license 缺失/续训状态缺失一律 **fail-closed**（不给默认值） |
| **16** | **工程**（零运行时依赖、测试驱动、置信） | ✅ | `runtimeDependencies: 0`；`dependencies` 仅 6 个 `@ai-infra/*`（file: 链接）；`tests/run.mjs` 零依赖驱动；5 spec/120 断言全过 |
| **17** | **注册路径** | ⚠️ **部分** | `dev_inject_plugin` → **host ✓**，loader `[active] … [injected]`（`apply()` 未抛错 ⇒ 装配期三项断言全过：风险声明 / 命名空间 / `output{schema,render}`）。**输出契约**已在同一代码路径（`createCore().run`）真调并通过无损校验。**尚未**经 DSH JSON-RPC 工具层实调（需下一轮 schema 刷新），故标"部分"而非"通过" |
| **18** | **命名空间** | ✅ | `unit/namespace` 12 断言：与既有 4 个插件源码**实时抓名**比对，交集为空；且反向检查；并断言"确实抓到了兄弟包的名字"（否则是空断言） |

### 3.1 P2-17 的诚实标注

按「**注册成功 ≠ 输出契约成立**」这条纪律（长任务 2 缺陷 3），我把这一项标为**部分**而不是通过：
- ✅ 已验证：`apply()` 在真实 DSH 里跑通（loader `[active]`，fiber 未退出）；
- ✅ 已验证：6 个工具都声明了 `output{schema,render}`，且 `createCore().run` 的返回穿过**无损 JSON 闸门**；
- ❌ **未验证**：通过 DSH 的工具调用通道实调一次（本轮的 tool schema 在注入前已固定）。

---

## 4. 环境与残留（真机）

| 项 | 值 |
|---|---|
| 容器残留 | **0**（`ancestor=training-sft/toolchain:0.1.0` 实测 0；smoke 容器 0） |
| 存活端点 | **0**（serving 注册表 10 条全部 `torn_down`） |
| 既有服务 | `qwen3-embedding`（vLLM，长期在跑）**未被触碰** |
| 基座模型 | `<BASE_MODEL>` **只读**；导出产物写在独立 run 目录 |
| 训练镜像 | `training-sft/toolchain:0.1.0`（复用，不新建；"同一台机器只应有一套训练工具链"） |

---

## 5. 已知边界（如实列出）

| # | 未覆盖 | 说明 |
|---|---|---|
| 1 | **checkpoint-contract 三项判据未逐项单独跑** | 本包用同一 `training-core.saveCheckpoint`，且产出已被 serving 真实加载前向（间接覆盖）；未像长任务 1 那样单独产出一份 `b2` 式证据 |
| 2 | **跨进程/跨机续训未验证** | 续训在同一台机器的同一镜像内验证；跨机需另验（语料挂载路径与 `corpusHash` 校验已具备） |
| 3 | **领域门禁只测一个阈值（1.0%）** | 未做阈值敏感性分析；阈值本就应由调用方按业务给 |
| 4 | **单种子** | 见 §2.1 |
| 5 | **shard 数语义只验证了 3 分片下的推进** | 更大分片数未跑 |
| 6 | **DSH 工具层实调** | 见 §3.1 P2-17 |
| 7 | **PII 扫描的模式表沿用 data-core** | 未新增模式；`bearer-token`/`phone-intl` 命中极少（4/5），覆盖率未单独评估 |

---

## ✅ 多种子确证（LONGTASK4 · 阶段 4，2026-09-29）

> 依据 `SEED_REPRODUCIBILITY_RETROSPECT.md`。3 个新种子（1042/2042/3042）以**相同的 serving 仪器与协议**复验。

| 本矩阵原有结论 | 单种子原值 | 3 种子 μ | σ_seed | **R/μ** | 符号 | **状态** |
|---|---|---|---|---|---|---|
| 领域能力提升 | −2.92% | **−2.96%** | 0.010 | **0.6%** | 一致 | ✅ **已确证为流程级结论** |
| 通用保持（MMLU） | −0.0500 | −0.0322 | 0.0051 | 31.0% | 一致 | ✅ 确证（"无统计显著退化"成立） |
| 通用保持（wikitext ΔNLL） | −0.0645 | −0.0667 | 0.0013 | 3.8% | 一致 | ✅ 确证（最稳的一条） |

**结论升级**：以上三条此前只能表述为"**那一个模型**"有该效应；
现在（n=3、含种子间方差）可表述为"**这套训练流程**有该效应"。
单种子结论此后一律须标 `singleSeed: true`。

**未修改任何判据或阈值。** 证据：`out/lt4-seed-retrospect-{raw,}.json`。
