# Qwen3.6-35B-A3B in colibri — Phase 2: Gated DeltaNet in C

Phase 1 跑通了 25% 的 Gated Attention 层 + 流式 MoE,把 DeltaNet 层当恒等跳过。
**Phase 2 把 75% 的 Gated DeltaNet 层(线性回归注意力)也在 `c/qwen36.c` 里实现了**,
所以现在的引擎能跑**完整混合模型**:每层(无论 Attention 还是 DeltaNet)都带自己的 MoE/MLP。

本文件只讲 Phase 2 相对 Phase 1 的增量。Phase 1 的架构、容器格式、tiny 验证套路见
[`qwen36-phase01.md`](./qwen36-phase01.md)。

## DeltaNet 数学(已在 numpy 里 torch-free 证明)

Gated DeltaNet 子层(`HF Qwen3_5MoeGatedDeltaNet.forward`,`tools/hf_qwen3_5_moe.py`
为权威来源)逐 token:

1. 投影:`qkv = x@qkv^T` `[S, conv_dim]`;`z = x@z^T` `[S, value_dim]`;`b=x@b^T`、`a=x@a^T` `[S, vh]`。
2. **因果深度卷积**(groups=conv_dim, kernel=convk, silu):`conv_out[c] = silu(Σ_kk w[kk]·x_ring[kk] + w[convk-1]·qkv[c])`。`x_ring` 是**跨 chunk 携带的环形缓冲**(最近 `convk-1` 个原始输入),实现左填充因果卷积。
3. 切分:`q_in/k_in = conv_out[:2·key_dim_tot]`(各 `key_dim_tot`),`v_in = conv_out[2·key_dim_tot:]`(`value_dim`)。
4. `beta = sigmoid(b)`; `g = -exp(A_log)·softplus(a + dt_bias)`(都按 value head)。
5. `q/k` 沿 head 维 `repeat_interleave` 把 `vk` 头扩成 `vh` 头;**每头 l2norm**(eps 1e-6 放进 sqrt),**q 再乘 `1/sqrt(kdim)`**,k 不乘。
6. **递归 gated delta rule**(每个 value head `h`,状态 `S_h ∈ [kdim, vdim]`):
   `S_h *= exp(g)`; `kv = k·S_h`(长度 `vdim`,**向量点乘 S 的第一轴**);`delta = (v - kv)·beta`;
   `S_h += k (⊗) delta`(外积);`out = q·S_h`。
7. **Gated RMSNorm**(普通 weight,**无 +1**,rms 里 `/vdim`):`o = (o·rsqrt(mean(o²)+eps))·dn_norm·silu(z)`。
8. `out_proj`:`outr @ dn_out^T` → `[S, hidden]`。

> 关键坑(已在 Phase-2 前修掉,这里记一笔):递归里向量必须点乘 S 的**第一轴**
> (`k·S`、`q·S`),不是 `S·k`。只有当 `kdim==vdim` 时两者才巧合相等。
> Gated RMSNorm 是**普通 weight**(HF `RMSNormGated` 无 `+1`),不是通用 RMSNorm 的 `(1+weight)`。

## C 实现(`c/qwen36.c`)

- **新增 `deltanet()`**:完整实现上面 1–8,对应 `tools/_ref_dn_stream.py` 里已验证的
  `deltanet_stream`(streaming==prefill,cos≈1.0)。`matmul` 单 token 投射 + 手写递归。
- **逐层携带状态**(放在 `Model`):`DN_rec[layer]` = 递归状态 `[vh, kdim, vdim]`;
  `DN_conv[layer]` = 卷积环形缓冲 `[conv_dim, convk-1]`。两者在 `step()` 的
  prefill chunk → decode token 之间**跨调用保留**(与 HF 的 recurrent 语义一致)。
- **`Cfg`** 新增 `dn_vheads/dn_kheads/dn_kdim/dn_vdim/dn_convk/dn_conv_dim`,
  由 `qwen36_meta.json` 的 `dn_*` 字段读入(`load_meta`)。
- **`Layer`** 新增 `dn_qkv/dn_z/dn_b/dn_a/dn_conv/dn_dtbias/dn_alog/dn_norm/dn_out`。
- **`step()`**:不再 `continue` 跳过 DeltaNet 层。每层 `rmsnorm(in_ln)` 后
  `is_attn[i] ? attention() : deltanet()`,残差,`rmsnorm(post_ln)`,`moe()`。
  pilot 预取改为「下一层存在即可」(现在每层都有 MoE)。
- **KV cache 只为 attention 层分配**(DeltaNet 用递归状态,不占 KV cache,给 16GB 目标省 ~2GB)。
- **容器**:所有层都按**原始索引** `model.layers.{i}` 存放(`active_of` 现在是恒等映射),
  `linear_attn.*` 由 `convert_qwen36.py` 的通用 f16 拷贝自动导出,
  `st_read_f32` 自动转成 f32。`meta` 多出 `dn_*` 维度字段。

## 验证策略

### 本地(torch-free,已完成)
- `tools/_ref_dn_stream.py`:证明 **streaming(有状态)== prefill(零填充卷积)** 整条
  DeltaNet 数学 cos≈1.0(一次性整段、逐 token、3-then-rest 三种切分全过)。这是 C `deltanet()`
  镜像的精确算法。
- `tools/_ref_dn.py` 里的 numpy DeltaNet 与上面的 matmul 方向 / Gated RMSNorm / g 公式
  完全一致(交叉核对过),并且它本身跑 **numpy 全前向 vs HF transformers** 的 cosine 信号。

### 真机(GPU 服务器 / Spark,有 gcc + torch)
```bash
cd c
# 1) HF 预言机:numpy 全前向 vs HF,给出三层 cosine 信号(权威证明 DeltaNet 正确)
python tools/_ref_dn.py --hf ../qwen36_tiny --prompt 1,2,3,4,5
#     [1] LOGIT cos  [1] PER-LAYER cos  [2] DELTANET-ISOLATION cos —— 都应 >0.999

# 2) 编译
make qwen36

# 3) C 引擎 logits dump(与 numpy 同权重,做 torch-free 余弦比对)
SNAP=../qwen36_tiny_i8 DUMP=qwen36_logits.f32 ./qwen36.exe 8 8 ref_qwen36.json
#     -> 写 qwen36_logits.f32 (vocab 个 float32)

# 4) numpy 侧同样 dump(不需要 torch 也能跑前向;torch 只在 [1]/[2] 信号里用)
python tools/_ref_dn.py --hf ../qwen36_tiny --prompt 1,2,3,4,5 --dump ref_logits.f32

# 5) 比对余弦(应 ~0.999+;f16 容器 vs f32 源有微小误差)
python -c "import numpy as np; a=np.fromfile('qwen36_logits.f32',np.float32); b=np.fromfile('ref_logits.f32',np.float32); print('cos=', float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))"
```

### 关于 token 级对齐(重要)
tiny 模型是**随机初始化权重**(非训练权重),贪心生成的 argmax 对 f16/运算序误差极敏感,
**token 级全中在随机权重下不可能**(Phase 1 的 `ref_qwen36.json` 是 attention_only 参考,
Phase 2 引擎算的是完整模型,两者不再一致,不要再拿它比对 token)。正确判据是 **logits
余弦 ≥0.999**(上面第 5 步)。真 35B 模型用 `--mode full` 预言机时同理看余弦。

## 真模型流程(16/24GB 目标机)
```bash
# 转换(已支持全层;ebits=4 给 16GB 留余量,ebits=8 保真对齐)
python tools/convert_qwen36.py --repo Qwen/Qwen3.6-35B-A3B --out /data/qwen36_i4 --ebits 4
#   -> 含 qwen36_meta.json(dn_* 维度)
SNAP=/data/qwen36_i4 ./qwen36.exe 16 4 ref_qwen36.json   # 全混合跑通
```

## 必须人工核对的项(真模型)
转换后核对 `qwen36_meta.json`:
- `dn_vheads/dn_kheads/dn_kdim/dn_vdim/dn_convk/dn_conv_dim` 与官方 config 的
  `linear_num_value_heads / linear_num_key_heads / linear_key_head_dim /
  linear_value_head_dim / linear_conv_kernel_dim` 一致(35B 默认
  vheads=??、kdim=??、convk=4 —— 以 config 实际值为准)。
- 若 `dn_*` 缺失,引擎对 DeltaNet 层会报 "dn dims missing from meta" 并退出(正确行为)。

## 已知限制 / 下一步
- 视觉编码器(27 层 ViT)、MTP 投机头仍未实现(属后续阶段,文本可跳过)。
- 真 35B 的 `--mode full` HF 预言机应在能加载 35B 的 GPU 服务器上跑,做最终余弦校验。
- Phase 3+ :完整混合调优 + 16GB 内存工作集压测;可选视觉/MTP。
