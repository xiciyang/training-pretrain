#!/usr/bin/env python3
"""
rl_smoke.py —— training-rl 的 **≤0.5 GPU 小时冒烟闸**（长任务 3 · 阶段 3 的第一个动作）。

## 为什么先做冒烟而不是直接放量
RL 框架探测是**只读**的：它证明了 `trl==1.14.0` 可装、依赖零变更、API 形状存在，
但**从未真正 import 过 GRPOTrainer、更没跑过一步**。而本机是 **GB10 / sm_121**，
torch 2.11.0 的 `arch_list` 里**没有 sm_121**（探测报告 §1）——
"装得上"与"跑得起来"之间隔着真实的运行时风险。
故先把最大的未知项用最小代价打掉：**2 步 GRPO，规则 reward，不启用 vLLM**。

## 判据（预先声明，不事后调整）
  P1  `GRPOTrainer` 能实例化（import 成功不算，要真的构造出来）
  P2  `trainer.train()` 真的走完 ≥2 个优化器步，且不抛异常
  P3  产物落盘（output_dir 里有 trainer_state.json / checkpoint）
  P4  规则 reward 真的被调用过（不是"跳过了 reward 直接报成功"）
  P5  显存可测（GB10 统一内存，"显存"即进程 RSS 增量 + torch 统计）
四条全过 ⇒ 允许升级为完整交付；任一不过 ⇒ 降级为计划交付并把失败原样登记。

输出：单行 JSON（stdout 只允许这一行，便于 host 解析），人类可读信息走 stderr。
"""
import json
import os
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TENSORBOARD_LOGGING_DIR", "/tmp/tb")

RESULT = {
    "schema": "training-rl/smoke@1",
    "checks": {},
    "rewardCalls": 0,
    "rewardValues": [],
    "error": None,
    "trace": None,
}


def log(m):
    sys.stderr.write(f"[rl-smoke] {m}\n")
    sys.stderr.flush()


def main():
    t0 = time.time()
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer, GRPOConfig

    RESULT["versions"] = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
                          "deviceCapability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} cc={RESULT['versions']['deviceCapability']}")

    base = os.environ.get("BASE_MODEL", "/models/Qwen2.5-0.5B-Instruct")
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 极小的 prompt 集：只要求"能走完一步"
    prompts = ["What is 2+2? Answer with a number.", "Name the capital of France.",
               "Say hello in one word.", "What color is the sky on a clear day?"]
    ds = Dataset.from_dict({"prompt": prompts})

    reward_calls = {"n": 0, "values": []}

    def rule_reward(completions, **kwargs):
        """**规则 reward**（0 GPU 小时、逐位确定性）：答案里出现数字/关键词就给分。"""
        reward_calls["n"] += 1
        out = []
        for c in completions:
            text = c if isinstance(c, str) else " ".join(str(m.get("content", "")) for m in c)
            s = 0.0
            s += 1.0 if any(ch.isdigit() for ch in text) else 0.0
            s += 1.0 if "paris" in text.lower() else 0.0
            s += 1.0 if "hello" in text.lower() else 0.0
            s += 1.0 if "blue" in text.lower() else 0.0
            s += 0.5 if 0 < len(text.strip()) < 200 else 0.0
            out.append(s)
        reward_calls["values"].extend(out)
        return out

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model.config.use_cache = False

    cfg = GRPOConfig(
        output_dir="/out/rl-smoke",
        max_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=32,
        learning_rate=1e-5,
        beta=0.04,
        use_vllm=False,               # 冒烟不引入 vLLM colocate（那是下一个未知项）
        bf16=True,
        report_to=[],
        logging_steps=1,
        save_steps=2,
        gradient_checkpointing=False,
        temperature=1.0,
    )

    try:
        trainer = GRPOTrainer(model=model, reward_funcs=[rule_reward], args=cfg,
                              train_dataset=ds, processing_class=tok)
        RESULT["checks"]["P1_construct"] = True
        log("P1 GRPOTrainer 构造成功")
    except Exception as e:
        RESULT["checks"]["P1_construct"] = False
        RESULT["error"] = f"P1 构造失败：{type(e).__name__}: {e}"
        RESULT["trace"] = traceback.format_exc()[-1200:]
        return

    torch.cuda.reset_peak_memory_stats()
    try:
        out = trainer.train()
        RESULT["checks"]["P2_train"] = True
        RESULT["trainMetrics"] = {"steps": int(getattr(out, "global_step", -1) or -1),
                                  "trainLoss": float(getattr(out, "training_loss", float("nan")) or float("nan"))}
        log(f"P2 train() 完成 steps={RESULT['trainMetrics']['steps']} loss={RESULT['trainMetrics']['trainLoss']}")
    except Exception as e:
        RESULT["checks"]["P2_train"] = False
        RESULT["error"] = f"P2 train() 失败：{type(e).__name__}: {e}"
        RESULT["trace"] = traceback.format_exc()[-1200:]
        return

    RESULT["checks"]["P3_artifacts"] = os.path.isdir("/out/rl-smoke") and len(os.listdir("/out/rl-smoke")) > 0
    RESULT["checks"]["P4_reward_called"] = reward_calls["n"] > 0
    RESULT["rewardCalls"] = reward_calls["n"]
    RESULT["rewardValues"] = reward_calls["values"][:20]
    peak = torch.cuda.max_memory_allocated() / 2**30
    rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 2**20 if os.path.exists("/proc/self/status") else None
    RESULT["memory"] = {"torchPeakAllocatedGiB": round(peak, 3), "processRssGiB": round(rss, 3) if rss else None}
    RESULT["checks"]["P5_memory_measured"] = peak > 0
    RESULT["wallMs"] = int((time.time() - t0) * 1000)
    RESULT["gpuHoursEquivalent"] = round((time.time() - t0) / 3600, 4)


try:
    main()
except Exception as e:
    RESULT["error"] = f"未捕获：{type(e).__name__}: {e}"
    RESULT["trace"] = traceback.format_exc()[-1200:]

RESULT["passed"] = all(RESULT["checks"].get(k) is True for k in ["P1_construct", "P2_train", "P3_artifacts", "P4_reward_called", "P5_memory_measured"])
print(json.dumps(RESULT, ensure_ascii=False))
