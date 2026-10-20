# training-sft 训练工具链镜像
#
# 为什么从 inference-quantization/toolchain:0.1.0 派生（而不是从零构建、也不是宿主机跑）：
#   ① 该镜像**已在盘**（21.7 GB，torch 2.11.0+cu130 / transformers 5.17.0 / accelerate / datasets /
#      safetensors / bitsandbytes 齐备），派生层只加 peft，构建秒级、且与既有工具链同源；
#   ② 沙箱策略（长任务 1 冻结的 `@ai-infra/sandbox-core`）：训练产出新权重属 write 档，
#      `DEFAULT_DEGRADATION.process = {policy:'upgrade', to:'container'}` —— 必须跑在容器层；
#      `vm` 层本机不可用，故系统级动作一律拒绝（本任务无此需求）。
#   ③ **版本钉死**：peft 装到固定版本，避免"谁哪天升级了 peft 导致数值变了"。
#
# 依赖声明纪律：peft 是本包 `package.json` 的 `optionalDependencies` 已声明的训练框架之一，
# 不是"引入未声明依赖"。
FROM inference-quantization/toolchain:0.1.0

# peft 0.21.0：实测与 transformers 5.17.0 / torch 2.11.0+cu130 兼容（见 PROJECT_CONTEXT §2.4 冒烟）
RUN pip install --no-cache-dir "peft==0.21.0" \
 && python3 -c "import peft, transformers, torch; print('peft', peft.__version__, '| tf', transformers.__version__, '| torch', torch.__version__)"

LABEL org.opencontainers.image.title="training-sft toolchain"
LABEL org.opencontainers.image.description="SFT/LoRA training toolchain: torch + transformers + peft (pinned)"
