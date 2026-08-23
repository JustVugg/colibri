# Qwen3.6-35B-A3B in colibri — Phase 0 + Phase 1

把 Qwen3.6 加进 colibri 的分阶段实现。**Phase 0** = 权重转换 + 校验预言机;**Phase 1** = Gated Attention + 流式 MoE 的 C 引擎,DeltaNet 层先当 identity 跳过。

## 为什么这样切

Qwen3.6 不是标准 MoE,是混合架构:

```
10 x ( 3 x (Gated DeltaNet -> MoE) , 1 x (Gated Attention -> MoE) )
#   = 40 层。Gated Attention 层 = 索引 i%4==3 (3,7,11,...,39)，共 10 层 = 25%。
#   Gated DeltaNet 层 = 其余 30 层 = 75% (Phase 2 再实现)。
```

关键尺寸(以官方 `Qwen/Qwen3.6-35B-A3B` 为准):
- hidden=2048, vocab=248320, layers=40
- **Gated Attention**: q_heads=16, kv_heads=2, head_dim=256, **rope_dim=64**(只对每头前 64 维做旋转)
- MoE: n_experts=256, topk=8 (路由) + 1 共享, inter=512
- 激活参数 ~3B;整模型 ~35B

Phase 1 只跑通 25% 的注意力层 + 流式 MoE 全链路(容器/分词/流式/KV/分组路由/共享专家),**用把 DeltaNet 层替换为恒等的 HF 预言机做 token 级对齐**。这把 80% 风险(自定义线性注意力)隔离到 Phase 2。

## 本机端到端验证(无需 35B 权重)

35B bf16 预言机要 ~70GB,24GB 本子跑不了。用同布局的 **tiny 模型** 验证引擎逻辑:

```bash
cd c
# 1) 造一个 tiny Qwen3.6 形模型 + 直接吐 ref.json (attention_only)
python tools/make_qwen36_tiny.py --out ../qwen36_tiny
#     -> ../qwen36_tiny/  (模型)  + ref_qwen36.json (attention_only 参考)

# 2) 转成 colibri 容器 (专家 int8;Phase1 用 ebits=8 保真)
python tools/convert_qwen36.py --model ../qwen36_tiny --out ../qwen36_tiny_i8 --ebits 8

# 3) 编译引擎 (MinGW / gcc)
make qwen36

# 4) 跑,对 token
SNAP=../qwen36_tiny_i8 ./qwen36.exe 8 8 ref_qwen36.json
#     期望: Matching tokens: N/N
```

tiny 模型参数约几 MB,bf16 预言机仅几百 MB,24GB 本子轻松跑。对齐即证明注意力 + 流式 MoE + 分组路由 + 共享专家 + 跳过 DeltaNet 这整条链路正确。

## 真模型流程(需要能跑 HF 预言机的机器,如 GPU 服务器)

```bash
# 转换 (建议 ebits=4 给 16GB 本留余量;对齐验证用 ebits=8)
python tools/convert_qwen36.py --repo Qwen/Qwen3.6-35B-A3B --out /data/qwen36_i4 --ebits 4

# 预言机 (在能加载 35B 的机器上跑,attention_only 模式)
python tools/make_qwen36_oracle.py --repo Qwen/Qwen3.6-35B-A3B \
       --out ref_qwen36.json --mode attention_only --max-new-tokens 32

# 引擎 (在 16/24GB 目标机)
SNAP=/data/qwen36_i4 ./qwen36.exe 16 4 ref_qwen36.json
```

`--mode full` 留给 Phase 2 之后(全混合)用。

## 云端转换(推荐:笔记本只下载成品容器)

见 [`docs/qwen36-cloud-convert.md`](./qwen36-cloud-convert.md)。转换器已支持 `--repo` 拉权重 + `--upload-repo` 推回 Hub + `--low-disk` 流式省磁盘。笔记本端只 `snapshot_download` 那 18GB `.coli` 容器即可。

## 容器格式 / 张量约定

`convert_qwen36.py` 输出一个 **safetensors 分片目录**(colibri 的"容器"就是目录):
- 稠密权重(embed/attn qkv o/qk_norm/RMSNorm/router gate/shared expert/lm_head/final norm)保持原 dtype(F32 载入)。
- 每个专家三矩阵合并为 `model.layers.{l}.mlp.experts.{e}.merged_weight`(int8,拼接顺序 **g|u|d**) + `...qs`(f32 scale,顺序 gs|us|ds)——**完全复用 olmoe.c 的 `load_expert_merged`/`Slot`,零改动**。
- 附带 `qwen36_meta.json`:引擎所需的全部 Qwen3.6 专属维度(attn head_dim、rope_dim、MoE 分组、attn 层列表等)。缺失时引擎回退到 `i%4==3` + 默认值。

## 引擎要点 (`c/qwen36.c`)

- `attention()`: Qwen3 GQA + 每头 q/k_norm(RMSNorm) + **partial RoPE**(仅每头前 `rope_dim` 维旋转,其余不变)。KV cache 按 `kv_heads` 分配。
- `moe()`:HF Qwen3 MoE 路由——softmax(gate) → 可选 group-limited top-k(`n_group`/`topk_group`)→ 权重归一化 → 路由专家加权和 → **加上无门控共享专家**(SwiGLU)。可选 router bias(`e_score_correction_bias`)。
- `step()`:遍历 40 层,**`is_attn[i]==0`(DeltaNet)直接跳过 → 恒等**。pilot 预取仅对"下一层也是 attn"触发。
- 专家磁盘流式 + LRU + PILOT 预取线程 + HOT 热固定:与 olmoe 同套(`PILOT/HOT/WARMUP/WIDE/SMOOTH/CONF_LIMIT` 环境变量)。

## 必须人工核对的项

`convert_qwen36.py` 从 `config.json` + safetensors 头读出所有维度并打印。跑转换后,**核对 `qwen36_meta.json` 这些值与上方官方值一致**,尤其:
- `attn.head_dim`(应为 256)、`attn.rope_dim`(应为 64)
- `moe.n_group` / `moe.topk_group`(官方未明示,转换脚本默认 `n_group=1`;若 HF config 里有 `n_group`/`topk_group` 会被自动采用,否则无分组路由)
- `moe.has_bias`(是否含 `e_score_correction_bias`)

若 `rope_dim` 解析为 `head_dim/4` 的回退值,需确认 config 里真正的 rope 维名字(已尝试 `rope_dim`/`rotary_emb_dim`/`qk_rope_head_dim`)。

## 已知限制 / 下一步

- DeltaNet 层(75%)在 Phase 1 是恒等 —— **只验证注意力层子集**,不是完整模型。
- 未实现:27 层 ViT 视觉编码器、MTP 投机头(都属后续阶段)。
- **Phase 2**:纯 C 实现 Gated DeltaNet(递归状态 + delta rule + 门控),先 tiny 对齐再接入,届时把 `step()` 里的 `continue` 换成真正的 DeltaNet 前向,并把预言机切到 `--mode full`。

## 校验门

任意修改后:重跑 tiny 流程,**`Matching tokens: N/N`(至少前若干 token 全中)**。贪心生成对量化误差敏感,`ebits=8` 应全中;`ebits=4` 可能在若干 token 后漂移(属预期,后续用 KL/perplexity 松绑)。
