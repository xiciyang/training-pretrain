#!/usr/bin/env python3
"""
pretrain_trainer.py —— training-pretrain 的**计算侧**执行体（容器内持久 worker）。

## 与 training-sft 的分工完全同形
  - **Node 侧（@ai-infra/training-core）**：循环、步数/墙钟/GPU 小时上限、checkpoint 编排、审计链
  - **Python 侧（本文件）**：只做张量数学（前向/反向/优化器步/保存权重/数据游标）
见 training-sft/python/trainer.py 顶部对"为什么不是 Python 自己跑完"的完整论证。

## 与 SFT 的三处实质差别
  1. **无标注语料**：样本是定长文本块，`labels = input_ids`（**不做 prompt 掩码**）—— 预训练目标是
     整段文本的语言建模，不是"只学回答"。
  2. **全参数微调（不是 LoRA）**：继续预训练的语义是让模型适应语料分布；
     LoRA 是 SFT 的参数高效技巧，用它做预训练在语义上不对（容量也不足）。
     基座模型**只读**，产物写到独立运行目录。
  3. **分片续训（契约冻结条件 2）**：本文件的**核心新增**是
     `步号 ↔ 数据位置` 的**绝对量不变量**（见下）。

## 续训不变量（本文件最重要的部分）

长任务 2 抓到的缺陷 2 是：「续训 20 步实跑 220 步，而两侧步号断言**全对**」——
**相对量断言（步号递增）全对，不等于断言的语义正确**。因此这里引入**绝对量不变量**：

    consumedChunks - baseConsumedChunks == (globalStep - baseGlobalStep) * chunksPerStep

其中：
  - `globalStep`      = 从**整个训练序列**开始算的绝对步号（跨多次续训单调递增，**不重置**）
  - `consumedChunks`  = 从**本 shard 开始**算的已消费块数（跨本 shard 内的多次续训单调递增）
  - `baseGlobalStep` / `baseConsumedChunks` = 本 shard 起点的基线（随 checkpoint 一起存，**不重算**）
  - `chunksPerStep`   = batchSize × gradAccum（一个优化器步吃掉的块数）

这条等式**同时**否掉四种错误：漏步（少算）、重放（多算）、游标重置（consumedChunks 归零）、
换了语料（hash 不符）。恢复时必须报告与 checkpoint **逐字段相同**的
`globalStep/consumedChunks/cursor/epoch/shardIndex`，Node 侧逐条核对（见 src/train.mjs）。

## 协议（行分隔 JSON，stdout 只允许出现协议行）
请求 cmd：ping | init | step | save_adapter | export_model | shutdown
"""
import json
import os
import sys
import time
import hashlib
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PROTO_OUT = sys.stdout


def log(msg: str) -> None:
    sys.stderr.write(f"[pretrain] {msg}\n")
    sys.stderr.flush()


def reply(obj: Dict[str, Any]) -> None:
    PROTO_OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    PROTO_OUT.flush()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_file(p: str) -> Optional[str]:
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_texts(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x01")
    return "sha256:" + h.hexdigest()


def param_fingerprint() -> str:
    """**权重指纹**：取前若干参数的各 1024 个元素做哈希。

    为什么需要它（长任务 3 · 阶段 6 的教训）：
    计数类断言（步号、consumedChunks、不变量）**全部为真**，也可能**权重一步没动**
    （实测：lr=0 导致 500 步全是空操作，而 checkpoint 里是续训源的原封权重）。
    ⇒ 必须有一条**落在产物上**的绝对量断言：「相对起点，权重到底变了没有」。
    这也是「断言的语义必须被独立验证」的正面做法：不要只断言**过程**，要断言**结果**。
    """
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in list(S.model.named_parameters())[:8]:
            v = p.detach().float().flatten()[:1024].cpu().numpy().tobytes()
            h.update(name.encode("utf-8"))
            h.update(v)
    return "sha256:" + h.hexdigest()


def list_files(d: str) -> List[Dict[str, Any]]:
    out = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            out.append({"name": name, "bytes": os.path.getsize(p), "sha256": sha256_file(p)})
    return out


# ─────────────────────────────────────────────────────────── 数据

class ChunkDataset(Dataset):
    """无标注定长文本块。`labels = input_ids`（不做 prompt 掩码）。"""

    def __init__(self, rows: List[Dict[str, Any]], tok, max_len: int):
        self.rows = rows
        self.tok = tok
        self.max_len = max_len
        self.truncated = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, List[int]]:
        text = str(self.rows[i].get("t", ""))
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            self.truncated += 1
        if len(ids) < 2:            # 退化样本：给一个 EOS，避免零长序列炸 collate
            ids = [self.tok.eos_token_id or 0]
        return {"input_ids": ids}


def collate(batch: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        # 预训练：整段参与损失；**padding 必须掩掉**，否则模型会学"续写 pad"
        labels.append(list(b["input_ids"]) + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device="cuda"),
        "labels": torch.tensor(labels, dtype=torch.long, device="cuda"),
        "attention_mask": torch.tensor(attn, dtype=torch.long, device="cuda"),
    }


# ─────────────────────────────────────────────────────────── 状态

class State:
    def __init__(self) -> None:
        self.model = None
        self.tok = None
        self.opt = None
        self.ds: Optional[ChunkDataset] = None
        self.rows: List[Dict[str, Any]] = []
        self.cfg: Dict[str, Any] = {}
        self.base_dir: Optional[str] = None
        # 续训不变量用的状态（全部随 checkpoint 存/取，**不重算**）
        self.global_step = 0
        self.base_global_step = 0
        self.consumed_chunks = 0
        self.base_consumed_chunks = 0
        self.cursor = 0
        self.epoch = 0
        self.shard_index = 0
        self.chunk_count = 0
        self.corpus_hash: Optional[str] = None
        self.shard_hash: Optional[str] = None
        self.chunks_per_step = 1
        self.shard_advance = False
        self.order: List[int] = []
        self.replay_violations: List[str] = []
        self.init_fingerprint: Optional[str] = None
        self.lr_max_seen = 0.0
        self.lr_zero_steps = 0


S = State()


def invariant_ok() -> bool:
    """**绝对量不变量**：consumedChunks - base == (globalStep - baseGlobalStep) * chunksPerStep"""
    lhs = S.consumed_chunks - S.base_consumed_chunks
    rhs = (S.global_step - S.base_global_step) * S.chunks_per_step
    return lhs == rhs


def invariant_detail() -> Dict[str, Any]:
    return {
        "lhs_consumedChunksDelta": S.consumed_chunks - S.base_consumed_chunks,
        "rhs_stepsTimesChunksPerStep": (S.global_step - S.base_global_step) * S.chunks_per_step,
        "globalStep": S.global_step, "baseGlobalStep": S.base_global_step,
        "consumedChunks": S.consumed_chunks, "baseConsumedChunks": S.base_consumed_chunks,
        "chunksPerStep": S.chunks_per_step, "cursor": S.cursor, "epoch": S.epoch,
        "shardIndex": S.shard_index, "chunkCount": S.chunk_count,
        "ok": invariant_ok(),
    }


def cmd_ping(_req: Dict[str, Any]) -> Dict[str, Any]:
    return {"pong": True, "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "globalStep": S.global_step, "consumedChunks": S.consumed_chunks,
            "cursor": S.cursor, "epoch": S.epoch, "shardIndex": S.shard_index}


def cmd_init(req: Dict[str, Any]) -> Dict[str, Any]:
    cfg = req["config"]
    S.cfg = cfg
    seed = int(cfg.get("seed", 20260929))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    base = cfg["baseModel"]
    S.base_dir = base
    t0 = time.time()
    S.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    if S.tok.pad_token is None:
        S.tok.pad_token = S.tok.eos_token

    resume = cfg.get("resumeFrom")            # 容器内路径（含 model.safetensors + trainer_state.json）
    model_src = resume if resume else base
    S.model = AutoModelForCausalLM.from_pretrained(
        model_src, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager"
    )
    load_s = time.time() - t0
    S.model.train()

    S.rows = load_jsonl(cfg["shardFile"])
    S.chunk_count = len(S.rows)
    S.ds = ChunkDataset(S.rows, S.tok, int(cfg.get("maxSeqLen", 512)))
    S.chunks_per_step = int(cfg.get("batchSize", 4)) * int(cfg.get("gradAccum", 4))
    S.corpus_hash = cfg.get("corpusHash")
    S.shard_hash = sha256_texts([str(r.get("t", "")) for r in S.rows])

    total = sum(p.numel() for p in S.model.parameters())
    S.opt = torch.optim.AdamW(S.model.parameters(), lr=float(cfg.get("lr", 2e-5)),
                              weight_decay=float(cfg.get("weightDecay", 0.0)))

    resumed = False
    prev = None
    if resume:
        ts_path = os.path.join(resume, "trainer_state.json")
        if not os.path.exists(ts_path):
            raise RuntimeError(f"续训源缺少 trainer_state.json：{resume}（无它无法证明数据位置连续）")
        with open(ts_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        # 语料一致性：**不允许**对着另一份语料续训
        if prev.get("corpusHash") and S.corpus_hash and prev["corpusHash"] != S.corpus_hash:
            raise RuntimeError(f"续训语料不一致：checkpoint={prev['corpusHash']} 当前={S.corpus_hash}")
        target_shard = int(cfg.get("shardIndex", prev.get("shardIndex", 0)))
        if target_shard != int(prev.get("shardIndex", 0)):
            # **推进到新分片**：权重继承，但本分片的数据位置从 0 起，
            # 并以"此刻的 globalStep"作为本分片的基线 —— 不变量按分片重置。
            S.shard_advance = True
            S.shard_index = target_shard
            S.global_step = int(prev.get("globalStep", 0))
            S.base_global_step = S.global_step
            S.consumed_chunks = 0
            S.base_consumed_chunks = 0
            S.cursor = 0
            S.epoch = 0
        else:
            # **同一分片续训**：逐字段恢复，不重算
            S.shard_index = target_shard
            S.global_step = int(prev.get("globalStep", 0))
            S.base_global_step = int(prev.get("baseGlobalStep", 0))
            S.consumed_chunks = int(prev.get("consumedChunks", 0))
            S.base_consumed_chunks = int(prev.get("baseConsumedChunks", 0))
            S.cursor = int(prev.get("cursor", 0))
            S.epoch = int(prev.get("epoch", 0))
        opt_path = os.path.join(resume, "optimizer.pt")
        if os.path.exists(opt_path):
            S.opt.load_state_dict(torch.load(opt_path, map_location="cuda", weights_only=False))
        resumed = True
    else:
        # 全新 run：globalStep 起点由 Node 给定（续"权重"但不续"步号"的场景也走这里）
        S.shard_index = int(cfg.get("shardIndex", 0))
        S.global_step = int(cfg.get("globalStepOffset", 0))
        S.base_global_step = S.global_step
        S.consumed_chunks = 0
        S.base_consumed_chunks = 0
        S.cursor = 0
        S.epoch = 0

    # 确定性数据顺序（同 seed + 同 shard ⇒ 逐块相同）
    S.order = list(range(S.chunk_count))
    random.Random(seed + 1000 * S.shard_index).shuffle(S.order)

    if not invariant_ok():
        raise RuntimeError(f"init 后不变量即不成立（恢复出的状态自相矛盾）：{invariant_detail()}")

    # **权重起点指纹**：训练结束后要据此判断"权重到底变了没有"
    S.init_fingerprint = param_fingerprint()

    return {
        "loadedInS": round(load_s, 2),
        "chunks": S.chunk_count,
        "params": total,
        "trainablePct": 100.0,
        "padTokenId": S.tok.pad_token_id,
        "vocabSize": int(S.model.config.vocab_size),
        "dtype": str(next(S.model.parameters()).dtype),
        "resumed": resumed,
        "shardAdvance": S.shard_advance,
        "shardIndex": S.shard_index,
        "globalStep": S.global_step,
        "baseGlobalStep": S.base_global_step,
        "consumedChunks": S.consumed_chunks,
        "baseConsumedChunks": S.base_consumed_chunks,
        "cursor": S.cursor,
        "epoch": S.epoch,
        "chunksPerStep": S.chunks_per_step,
        "corpusHash": S.corpus_hash,
        "shardHash": S.shard_hash,
        "prevState": prev,
        "invariant": invariant_detail(),
        "baseWeights": sha256_file(os.path.join(S.base_dir, "model.safetensors")),
    }


def _next_micro_batch(bs: int) -> List[Dict[str, List[int]]]:
    out = []
    for _ in range(bs):
        if S.cursor >= S.chunk_count:
            S.epoch += 1
            S.cursor = 0
        out.append(S.ds[S.order[S.cursor]])
        S.cursor += 1
    return out


def _lr_at(step: int) -> float:
    """warmup + cosine 衰减。

    ## ⚠️ `total` 必须用 **`lrTotalSteps`（绝对终点）**，不是本 run 的 `maxSteps`
    这是长任务 3 · 阶段 6 真机抓到的**严重缺陷**（"仪表恒真"家族的第三个实例）：
      - 本函数被调用时 `step` 是**绝对** `globalStep`；
      - 原实现把 `total` 取成本 run 的 `maxSteps`（每 run 的预算）。
    于是**续训**时（例如从 globalStep=500 起、maxSteps=500）：
      `prog = (500-50)/(500-50) = 1` ⇒ `lr = base*0.5*(1+cos(π)) = **0**`。
    结果是**每一步优化器步都是空操作**：globalStep 照涨、不变量照过、checkpoint 照存，
    但**权重一个字节都没变**（实测：三个 checkpoint 与续训源逐字节相同）。
    SFT 侧没这个问题，因为它的 `lrTotalSteps = offset + 新增步数`（见 training-sft 的 core.mjs）——
    本文件原先漏了这一步，故此处对齐。
    """
    base = float(S.cfg.get("lr", 2e-5))
    warm = int(S.cfg.get("warmupSteps", 50))
    total = max(int(S.cfg.get("lrTotalSteps", S.cfg.get("maxSteps", 500))), 1)
    if step < warm:
        return base * (step + 1) / max(warm, 1)
    prog = min(max((step - warm) / max(total - warm, 1), 0.0), 1.0)
    return base * 0.5 * (1.0 + float(np.cos(np.pi * prog)))


def cmd_step(_req: Dict[str, Any]) -> Dict[str, Any]:
    if S.model is None:
        raise RuntimeError("init 未调用")
    bs = int(S.cfg.get("batchSize", 4))
    ga = int(S.cfg.get("gradAccum", 4))
    t0 = time.time()
    lr = _lr_at(S.global_step)
    for g in S.opt.param_groups:
        g["lr"] = lr

    before_chunks = S.consumed_chunks
    losses = []
    tokens = 0
    S.opt.zero_grad(set_to_none=True)
    for _ in range(ga):
        batch = collate(_next_micro_batch(bs), S.tok.pad_token_id)
        out = S.model(**batch)
        (out.loss / ga).backward()
        losses.append(float(out.loss.detach().float().item()))
        tokens += int(batch["attention_mask"].sum().item())
        S.consumed_chunks += bs

    gnorm = 0.0
    for p in S.model.parameters():
        if p.grad is not None:
            gnorm += float(p.grad.detach().float().norm()) ** 2
    gnorm = gnorm ** 0.5
    torch.nn.utils.clip_grad_norm_(S.model.parameters(), 1.0)
    S.opt.step()
    S.opt.zero_grad(set_to_none=True)
    S.global_step += 1

    # **每步都核绝对量不变量**（不是只在最后核一次）
    delta = S.consumed_chunks - before_chunks
    if delta != S.chunks_per_step:
        S.replay_violations.append(f"step {S.global_step}: 本步消费 {delta} 块，应为 {S.chunks_per_step}")
    if not invariant_ok():
        raise RuntimeError(f"绝对量不变量被破坏：{invariant_detail()}")

    # **"空操作训练"闸门**（长任务 3 · 阶段 6 实测抓到的缺陷）：
    # 计数类断言全对，也可能因为 lr 恒为 0 而**权重一步没动**。
    # 故这里对"本 run 已跑 ≥2 步但从未见过非零 lr"**响亮失败**，而不是产出一个原封不动的 checkpoint。
    if lr > 0:
        S.lr_max_seen = max(S.lr_max_seen, lr)
    else:
        S.lr_zero_steps += 1
    ran = S.global_step - S.base_global_step
    if ran >= 2 and S.lr_max_seen == 0.0:
        raise RuntimeError(
            f"空操作训练：本 run 已跑 {ran} 步，但学习率**从未非零**（lr={lr}，"
            f"globalStep={S.global_step}，lrTotalSteps={S.cfg.get('lrTotalSteps')}）。"
            f"最常见根因：LR 调度的 total 用了本 run 的 maxSteps 而不是绝对终点 ⇒ 续训时 prog 直接到 1、lr=0。"
        )

    return {
        "loss": float(np.mean(losses)),
        "lossLast": losses[-1],
        "stepIndex": S.global_step,
        "globalStep": S.global_step,
        "consumedChunks": S.consumed_chunks,
        "chunksThisStep": delta,
        "cursor": S.cursor,
        "epoch": S.epoch,
        "shardIndex": S.shard_index,
        "chunksPerStep": S.chunks_per_step,
        "invariant": invariant_detail(),
        "lrMaxSeen": S.lr_max_seen,
        "lrZeroSteps": S.lr_zero_steps,
        "tokens": tokens,
        "lr": lr,
        "ms": int((time.time() - t0) * 1000),
        "gpuMemGiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "metrics": {"gradNorm": round(gnorm, 6), "microBatches": ga},
    }


def cmd_save_adapter(req: Dict[str, Any]) -> Dict[str, Any]:
    """保存可恢复 checkpoint：权重 + 优化器 + **续训状态**（含不变量所需全部基线字段）。"""
    d = req["dir"]
    os.makedirs(d, exist_ok=True)
    S.model.save_pretrained(d, safe_serialization=True)
    S.tok.save_pretrained(d)
    torch.save(S.opt.state_dict(), os.path.join(d, "optimizer.pt"))
    state = {
        "schema": "ai-infra/pretrain-trainer-state@1",
        "globalStep": S.global_step,
        "baseGlobalStep": S.base_global_step,
        "consumedChunks": S.consumed_chunks,
        "baseConsumedChunks": S.base_consumed_chunks,
        "cursor": S.cursor,
        "epoch": S.epoch,
        "shardIndex": S.shard_index,
        "chunkCount": S.chunk_count,
        "chunksPerStep": S.chunks_per_step,
        "corpusHash": S.corpus_hash,
        "shardHash": S.shard_hash,
        "seed": int(S.cfg.get("seed", 20260929)),
        "replayViolations": S.replay_violations,
    }
    with open(os.path.join(d, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    files = list_files(d)
    fp = param_fingerprint()
    return {"dir": d, "files": files, "bytes": sum(f["bytes"] for f in files),
            "stepIndex": S.global_step, "state": state, "invariant": invariant_detail(),
            "fingerprint": fp, "initFingerprint": S.init_fingerprint,
            "weightsChanged": fp != S.init_fingerprint,
            "lrMaxSeen": S.lr_max_seen, "lrZeroSteps": S.lr_zero_steps}


def cmd_export_model(req: Dict[str, Any]) -> Dict[str, Any]:
    """把训练后的**全参数**模型导出为标准 HF 目录（预训练无 LoRA，无需 merge）。"""
    out = req["outDir"]
    os.makedirs(out, exist_ok=True)
    S.model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    S.tok.save_pretrained(out)
    lic = req.get("license", {})
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f"license: {lic.get('spdx', 'apache-2.0')}\n")
        f.write("base_model: Qwen/Qwen2.5-0.5B-Instruct\n")
        f.write("library_name: transformers\n")
        f.write("---\n\n")
        f.write("# training-pretrain continued-pretraining checkpoint\n\n")
        f.write(str(lic.get("attributionText", "")) + "\n\n")
        f.write(req.get("readmeNote", "") + "\n")
    with open(os.path.join(out, "training-pretrain-provenance.json"), "w", encoding="utf-8") as f:
        json.dump(req.get("provenance", {}), f, ensure_ascii=False, indent=2)
    files = list_files(out)
    weights = [f for f in files if f["name"].endswith(".safetensors")]
    h = hashlib.sha256(("\n".join(f["sha256"] for f in weights)).encode()).hexdigest()
    fp = param_fingerprint()
    return {"outDir": out, "files": files, "weightsSha256": "sha256:" + h,
            "bytes": sum(f["bytes"] for f in files),
            "params": int(sum(p.numel() for p in S.model.parameters())),
            "globalStep": S.global_step, "consumedChunks": S.consumed_chunks,
            "fingerprint": fp, "initFingerprint": S.init_fingerprint,
            "weightsChanged": fp != S.init_fingerprint,
            "lrMaxSeen": S.lr_max_seen, "lrZeroSteps": S.lr_zero_steps}


def main() -> None:
    log(f"worker up tid={os.getpid()} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cmd = req.get("cmd")
        rid = req.get("id")
        t0 = time.time()
        try:
            fn = {
                "ping": cmd_ping,
                "init": cmd_init,
                "step": cmd_step,
                "save_adapter": cmd_save_adapter,
                "export_model": cmd_export_model,
            }.get(cmd)
            if cmd == "shutdown":
                reply({"id": rid, "ok": True, "bye": True})
                break
            if fn is None:
                raise ValueError(f"未知命令 {cmd}")
            res = fn(req)
            res.update({"id": rid, "ok": True, "handlerMs": int((time.time() - t0) * 1000)})
            reply(res)
        except Exception as e:
            import traceback
            reply({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})


if __name__ == "__main__":
    main()
