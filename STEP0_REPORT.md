# training-pretrain · Step 0 报告（**只读探测**）

> 长任务 3 · 阶段 2.1。全部结论分证据等级：`[verified]` 本机一手 / `[claimed]` 二手 / `[inferred]` 推论。
> 探测脚本：`step0/step0-corpus-audit.mjs`（产出 `out/pretrain-corpus-audit.json`、`out/pretrain-corpus-exclusions.json`）

---

## 0. 一句话

**语料可接受（`CORPUS_ACCEPTED`）**：`allenai/dolma` 8B-sample 分片 0000，license **ODC-BY**（三重实证），
PII 剔除 **391/22968 = 1.702%**，去泄漏双轨中**轨 2 抓到 1 条而轨 1 抓到 0 条** ——
**没有阻塞项，可以进入实现。**

---

## 1. 领域语料来源（候选 · license · PII · 测试集交集）

### 1.1 候选评估（题面建议的三个 + 一个备选）

| 候选 | license（**以实际卡片为准**） | 可下载性 | 结论 |
|---|---|---|---|
| `cerebras/SlimPajama-627B` | 未读到（API 即被拒） | **gated**：`https://hf-mirror.com/api/datasets/cerebras/SlimPajama-627B` 返回 `{"error":"Invalid username or password."}` `[verified]` | ❌ **拒**（未授权） |
| `togethercomputer/RedPajama-Data-1T` | **卡片 frontmatter 未声明 license**（只有 `task_categories/language/pretty_name`）`[verified]` | 数据本体不在 HF，只有 `urls/*.txt`；其指向的 `data.together.xyz` 实测 **403** `[verified]` | ❌ **拒**（license 无据 + 取不到数据） |
| `allenai/dolma` | **ODC-BY**（三重实证，见 §1.2） | 8B-sample 分片 URL 可达（`olmo-data.org`，单分片 159,779,374 B）`[verified]` | ✅ **选** |
| `HuggingFaceFW/fineweb` | 卡片 tags 含 `license:odc-by`；`gated=false`；HF 上直接有 parquet `[verified]` | 可下载 | 🔁 **备选**（本轮不用，因 Dolma 同源更完整：自带 LICENSE.md 全文 + 显式 URL 清单） |

> **原则 A 的直接收益**：题面式的"RedPajama 是 Apache-2.0"在**实际卡片里查不到**。
> 按「license 必须以实际卡片为准」，它因此**不能**被采用 —— 不是因为它不好，而是因为**无据**。

### 1.2 选定语料的 license（三重证据，逐条 `[verified]`）

```
spdx: odc-by     class: permissive-with-attribution     需署名: 是     商用: 允许
```

1. HF API tags 含 `"license:odc-by"` —— `https://hf-mirror.com/api/datasets/allenai/dolma`
2. 卡片正文 frontmatter `license: odc-by`，且正文写明
   *"2024-04-15: License Change. We have updated the license of Dolma to ODC-BY"*
   —— `https://hf-mirror.com/datasets/allenai/dolma/raw/main/README.md`
3. 仓库内 `LICENSE.md` 首行 `# ODC Attribution License (ODC-By)`
   —— `https://hf-mirror.com/datasets/allenai/dolma/raw/main/LICENSE.md`

**署名文本（将写入产物 README / provenance）**：
> Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research
> (Soldaini et al., 2024), Allen Institute for AI. Licensed under ODC-BY 1.0.

### 1.3 语料边界（**选择规则本身也要可复现**）

| 项 | 值 |
|---|---|
| 源 | `allenai/dolma`，revision `7f48140530a023e9ea4c5cfb141160922727d4d3` |
| 样本 | `dolma-v1_6-8B-sample`（`olmo-data.org`），分片 `v1_5r2_sample-0000.json.gz` |
| 分片 sha256 | `sha256:71f2ab92baf0f8a9073dc573ac61d3b28546516849ec460d1052471c5e47ee59` |
| 分片规模 | 141,838 文档 / 370,102,958 字符（≈92.5M token） |
| **选择规则** | `contentHash(text)` **升序取前 N** 至 token 目标 —— 确定性、**无位置偏斜**（不用文件顺序：Dolma 分片按 source 成块，前缀选择会引入位置偏斜） |
| 选中 | **22,968 文档 / 60,001,187 字符（≈15.0M token）**，来源 = `c4` + `gutenberg` |
| 扫描覆盖率 | **100%（22,968/22,968）**，**不抽样** |

### 1.4 PII（**这一节同时是 `training-pretrain-contract` §5 未决项的答案**）

> `training-pretrain-contract.md` §5 原文把「无标注语料的 PII 筛查口径」列为**冻结前必须先解决**的风险
> （"需要先定义扫什么、扫多少、命中阈值多少"）。以下是该口径的定稿：

**口径 v1**

| 维度 | 规定 | 理由 |
|---|---|---|
| **扫什么** | 每个文档的正文 `text` 字段 | Dolma 的载荷字段；元数据不含人可识别的正文 |
| **扫多少** | **100% 的"将被训练到"的文档**。先定语料边界，再全量扫描，**永不抽样** | 抽样必然漏长尾（项目原则「证据采样必须防止长尾偏斜」）；PII 命中率极低（本次 1.7%），抽样会把"低概率"直接变成"未观测" |
| **命中判定** | 两层：保留域邮箱 / 占位本地部分 / 保留段 IP ⇒ **非 PII**；其余（`other`）⇒ **真 PII** | 沿用 `training-sft` 已固化的 `classifyHit` 分层，避免同一台机器两套口径 |
| **处置** | **剔除整个文档**，**不改写正文**，只记哈希 | 纪律禁止"自行脱敏后继续" |
| **判据来源** | `@ai-infra/data-core` 的 `piiScan`（模式 + **校验器**） | 不绕开共享库契约。校验器可推导（身份证 ISO 7064 校验位、IPv4 八位组 ≤255），**不是宽容度调参** |

**实测结果** `[verified]`

| 项 | 值 |
|---|---|
| 扫描文档 | 22,968 |
| 模式命中 | `email` 610 · `ipv4` 27 · `phone-intl` 5 · `bearer-token` 4 · `cn-mobile` 3 |
| **剔除（非占位）** | **391 文档 = 1.702%** |
| 剔除记录 | 仅 `docHash` + 模式名 + 命中样本哈希（`out/pretrain-corpus-exclusions.json`），**无原文** |

> 注：`cn-idcard` 命中 0 —— 与 data-core 的注释一致，校验器把十六进制串里的假阳性挡住了。

### 1.5 测试集交集（**双轨**，用户裁决 B）

评估集（固定，与长任务 2 相同）：`wikitext-test(c128)` 400 · `wikitext-test(c512)` 400 ·
`mmlu-test` 14,042 · `alpaca-heldout` 200。

| 轨 | 方法 | 结果 |
|---|---|---|
| **轨 1** 整文档哈希 | `contentHash(doc.text)` ⇄ 评估集项哈希（走 `data-core.assertNoLeakage`） | **0 条** |
| **轨 2** 行内子串（包含） | 文档正文若**包含**任一评估项**全文**（len ≥ 24）⇒ 剔除 | **1 条** |
| **复核** | 对**保留集**再跑一次双轨（不是"我剔过了所以没有"） | 哈希交集 **0** · 包含 **0** ⇒ `ok=true` |

**轨 2 抓到的那 1 条**（`out/pretrain-corpus-audit.json` → `leakage.track2...detail`）：

```json
{"docHash":"13971b3a2631c1c1a9659c969d1d8c71b989db705c4270143e2781de5935ecde",
 "source":"c4","hits":1,"sets":["mmlu-test"],
 "itemHashes":["7fd7095c8f274727f5fc952a9b388e920bdafad09d67dda88d7b4d1512e8ea6a"]}
```

> **这是裁决 B 的一次干净实证**：一条 **c4 网页文档完整包含了 1 道 MMLU 题目**，
> 而**整行哈希一条都没抓到**。若只做轨 1，这道 MMLU 题会以"训练语料"的身份进入预训练 ——
> 之后用 MMLU 评估该模型就是**测自己见过的题**。用户裁决 B 在此**再次被证明必要**。

**实现注记（性能）**：朴素写法是 `for(doc) for(item) doc.includes(item)` = 22,968 × 14,870 ≈ **3.4 亿次**
includes，实测不可接受（首个实现确会跑数小时）。改为**无假阴性**的两段式：
① 评估项取**非重叠 24-gram** 建倒排索引（项被完整包含 ⇒ 其自身每个窗口必在文档中）；
② 文档用 **`Math.imul` int32 滚动哈希**扫 24-gram **滑动窗口**（步长 1，O(n) 且不分配字符串）得候选；
③ 候选才做精确 `includes` 复核。全流程 **8.9 秒**，且**不漏判**（碰撞只会多花复核）。

---

## 2. 训练框架

- **与 SFT 共用**：`torch` + `transformers` + `accelerate`，容器内运行；公共训练循环走
  `@ai-infra/training-core`（Node 侧驱动，Python 只做张量数学）——**与 training-sft 同一套接缝**。
- **工具链镜像**：复用 `training-sft/toolchain:0.1.0`（`torch 2.11.0+cu130 / transformers 5.17.0 / peft 0.21.0`）
  —— 不新建镜像的理由：预训练**不需要 peft**，该镜像已含全部所需依赖，且"同一台机器只应有一套训练工具链"。
- **不做**：分布式（DDP/FSDP/TP/PP）；并发恒为 1。

---

## 3. GPU 资源

| 项 | 实测值 |
|---|---|
| 设备 | NVIDIA GB10（统一内存），单卡 |
| 训练并发 | 恒为 1（`enforceLimits` 强制） |
| 已有占用 | `qwen3-embedding` vLLM 容器（长期在跑），**本任务不触碰** |
| 单次上限 | `maxSteps ≤ 500`、`maxGpuHours ≤ 2`、`params ≤ 0.5B` |
| 基线模型 | `/data/models/Qwen2.5-0.5B-Instruct`（0.494B） |
| 阶段总额 | **≤ 8 GPU 小时** |

---

## 4. 灾难性遗忘防护策略（**采纳阶段 1 的结论，这是本阶段最重要的一条**）

阶段 1 的复验结论是 **`测量不成立，需重设计`**（`R/μ = 1011%`、`σ(Δ)/bound = 0.595`）——
即**单种子 `z·SE` 门禁分辨不出"这套训练流程是否造成遗忘"**。因此 pretrain 的遗忘防护**不能**沿用单次 `z·SE`：

| 层 | 策略 |
|---|---|
| ① 门禁 | `MMLU 准确率 Δ ≤ z·SE` **且** `wikitext 配对 NLL Δ ≤ z·SE`（沿 use testing-core 的 `forgettingGate`） |
| ② **多种子** | 关键结论**至少 3 种子**；单种子结论只能表述为"该模型如此"，不得表述为"该流程如此" |
| ③ **噪声模型补全** | 若做"流程级"判断，判据须叠加种子间方差：`Δ ≤ z·SE_测量 ⊕ z'·σ_seed`（**不得只用单次 `z·SE`**） |
| ④ 回退 | checkpoint 保留策略 + `rollback` 能力；门禁不过即回退，不"带伤晋级" |
| ⑤ 数据侧 | 双轨去泄漏（§1.5），确保遗忘是在**未见过的**评估集上测的 |

---

## 5. 学习率预热与训练步数

预训练惯例（**长训练步数 + 小学习率**，与 SFT 的 200 步 / 1e-4 不同）：

| 超参 | 取法 | 依据 |
|---|---|---|
| `lr` | 比 SFT 更小（量级 1e-5 ~ 5e-5），warmup 后 cosine | 继续预训练的目标是"温和地适应语料分布"，大 lr 会破坏基座能力（=灾难性遗忘的直接成因） |
| `warmupSteps` | 总步数的 5~10%，**且可推导**（`warmupSteps` 是配置项，不是写死的魔数） | `_lr_at` 用配置计算，无魔数 |
| `maxSteps` | ≤ 500（硬上限） | 预算纪律 |
| 序列长度 | 比 SFT 长（SFT 用 256） | 预训练语料是无标注连续文本，长上下文才有意义 |
| 批次 | `batchSize × gradAccum`；**有效批 token 数**要记录 | 跨 run 比较时批大小必须一致 |

> **具体数值在执行时以配置落盘为准**（`run.json` 的 `hyper` 字段），本文件不预先写死。

---

## 6. 与 inference-serving 的接口

- **路径解析**：与 training-sft 同形（配置 → 兄弟目录 → 包内 → 环境变量），**从包位置推导**，不硬编码。
- **调用**：`inference_serving_deploy`（部署合并后的 HF 目录，含可用性探针）→ 评估 → `inference_serving_teardown`。
- **纪律**：只创建并拆解**自己租用**的端点；存量端点只读，绝不触碰。
- **产物必须可被 serving 加载**：这是 P0 项（checkpoint 可加载）的落点。

---

## 7. 结论与下一步

| 项 | 判定 |
|---|---|
| 语料 license | ✅ ODC-BY（三重实证，需署名） |
| PII | ✅ 口径已定稿并执行（100% 扫描，剔除 391 文档） |
| 测试集交集 | ✅ 双轨 + 复核全部为空（轨 2 抓到并剔除 1 条） |
| 无合规候选？ | ❌ 否 —— **有合规候选，不触发"暂停报告"** |
| 阻塞项 | **无** |

**⇒ 进入 2.2 写包。**
