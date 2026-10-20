# @xiciyang/training-pretrain

继续预训练（CPT）：把原始语料切成确定性分片，用**真实 GPU** 在基座模型上继续训练，并对「权重真的变了」「格式没有崩」两件事分别取证。

DSH（DeepSeek Harness）插件包，同时可作为普通 Node 库使用。

## 它解决的问题

继续预训练最容易出现的两种假成功：

1. **空操作**——步数、loss 曲线都正常，权重其实没变。本包在训练前后取权重指纹，相同即判失败。
2. **格式崩溃**——下游指令能力被打没，而困惑度看起来还降了。本包在验收阶段跑格式门禁，并把结果与训练前的基线对比。

## 安装

```bash
dsh plugin add github:xiciyang/training-pretrain
```

要求：

- Node.js >= 20，`engines.dsh >= 0.2.0 < 0.3.0`
- 真实训练需要 NVIDIA GPU + 容器运行时 + 训练镜像；不满足时 `training_pretrain_health` 会逐项说明
- Python 框架（torch / transformers / peft / trl / accelerate）在 `optionalDependencies` 中显式声明

## 工具面（DSH）

| 工具 | 作用 |
|---|---|
| `training_pretrain_prepare` | 语料准备：分片、去重、领域留出、确定性切分、血缘与内容哈希 |
| `training_pretrain_train` | 训练：预算闸门、权重指纹（init/after 双指纹）、checkpoint 保留、审计链 |
| `training_pretrain_validate` | 验收：困惑度变化必须过有效性门禁；格式能力不得击穿底线 |
| `training_pretrain_rollback` | 回退到指定 run；purge 只允许删本插件 run 目录 |
| `training_pretrain_get_report` | 取人类可读报告 |
| `training_pretrain_health` | 环境自检：GPU / 容器 / 镜像 / 基座 / serving 依赖，逐项可用性 |

## 配置（不把本机布局写死）

| 环境变量 | 用途 |
|---|---|
| `DSH_BASE_MODEL` | 基座模型本地目录 |
| `DSH_SHARD_PATH` | 原始语料分片（`.json.gz`，每行 `{id, source, text}`） |
| `DSH_MMLU_PATH` | 评测数据路径（用于格式/能力门禁，可选） |
| `TRAINING_PRETRAIN_ROOT` | 运行目录根（默认 `~/.ai-infra`） |

## 测试

```bash
node tests/run.mjs
```

纯逻辑与契约测试在任何机器上应为绿；依赖本机历史产物的判据自检在产物缺失时显式 `SKIP` 并写明原因（汇总行打印 skipped 数），不伪装成通过。

## 安全边界

- 训练与合并需显式确认令牌；预算超限直接拒绝
- 回退只作用于本插件自己的 run 目录，越界路径被拒
- 审计为追加式哈希链，可第三方复算

## 许可

BSD-3-Clause。见 `LICENSE`。
