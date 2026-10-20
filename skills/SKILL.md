---
name: training-pretrain
description: 继续预训练（continued pretraining）训练侧能力。当任务涉及"用领域语料继续训练一个基座模型"、"分片续训"、"灾难性遗忘检查"、"领域能力提升验证"时使用。覆盖 training_pretrain_prepare / train / validate / rollback / get_report / health 六个工具。
---

# training-pretrain —— 继续预训练

## 它做什么

用**领域语料**对一个已训练模型做**全参数继续预训练**，并用可推导的门禁判断"领域学到了没"与"通用能力忘了没"。

## 六个工具（按依赖顺序）

| 工具 | 作用 | 风险 |
|---|---|---|
| `training_pretrain_prepare` | 语料审计（license/PII 100%/去泄漏双轨）+ 去重 + 分块 + 确定性分片 + 领域留出 | write，无需确认 |
| `training_pretrain_train` | 分片训练（全参数），**绝对量不变量**逐步校验；支持同分片续训与推进分片 | write，需确认 |
| `training_pretrain_validate` | 导出 → serving 部署 → **领域门禁**（阈值须显式给）+ **遗忘门禁**（z·SE） | write，需确认 |
| `training_pretrain_rollback` | 回退 run（可选清产物） | write，需确认 |
| `training_pretrain_get_report` | 读注册表 + 审计链 | read |
| `training_pretrain_health` | 自检（沙箱/镜像/GPU/基座/审计链） | read |

## 两个必须记住的纪律

1. **领域门禁阈值必须由调用方显式给出**（`thresholds.minDomainGainPct`）。本插件**不内置默认值** ——
   不能自己定阈值、自己签收。"自己定阈值"和"自己给自己发合格证"是同一件事。
2. **遗忘门禁的 `z·SE` 只建模逐文本测量噪声，不含训练随机性**。
   实测（`SFT_SEED_REPRODUCIBILITY.md`）：种子间 `σ(Δ) ≈ 0.6 × 判据界`。
   ⇒ 单种子的结论只能表述为"**该模型**如此"，**不能**表述为"**这套训练流程**如此"；
   要做流程级判断，必须多种子并叠加种子间方差。

## 典型用法

```
1. training_pretrain_prepare  { targetTokens: 15000000, shardCount: 3 }
2. training_pretrain_train    { datasetDir, shardIndex: 0, maxSteps: 500 }
3. training_pretrain_validate { runId, thresholds: { minDomainGainPct: 1.0 } }
```

**续训**（长任务在 500 步上限内继续）：

```
training_pretrain_train { datasetDir, shardIndex: 0, maxSteps: 500,
                          resumeRunId, resumeCheckpoint: 'step-000500' }   # 同分片续训
training_pretrain_train { datasetDir, shardIndex: 1, ... , resumeRunId, resumeCheckpoint }  # 推进分片
```

⚠️ **checkpoint 名的坑**：目录名用的是**本次 run 内**的步号（相对量），
绝对步号只存在 `trainer_state.json` 的 `globalStep`。名字请从 `get_report` 读，不要自己拼。

## 硬边界

- 单次 ≤500 步 / ≤2 GPU 小时 / ≤0.5B 参数 / 并发 1；超限**暂停报告**，不降级。
- **不引入量化**（归 `training-qat`）；**不做 RL**（归 `training-rl`）；不做分布式。
- PII 只**剔除整个文档**，绝不改写正文；不自行合成数据；不做语料抓取。
