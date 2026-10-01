# DeepSeek V4：输出头驻留 GPU

基线为 PR #1766 的 `a24d02dd`（执行代码 `371ed4e3`）。本轮处理研究中确认的
CPU 词表投影：每生成一个 token，CPU 扫描 129280 × 4096 个 BF16 权重，约 1 GiB。

## 实现边界

- 新增专用 rows32 BF16 head mirror 和 `dsv4_cuda_head_scores_exact`。一个线程拥有一个
  词表行，warp 合并读取相邻行；每行仍按列顺序独立舍入乘法与加法，与当前 AVX2 CPU
  输出头一致。129280 个输出行提供并行度，不改变归约顺序。
- 接入 `head_scores_impl` 的单行全词表结果，包括 greedy 和单行 scores/logprobs；
  argmax 仍使用原 CPU 扫描，保留首个最大值规则。多行 speculative head 尚未改动。
- 默认开启 `DSV4_CUDA_HEAD`；设置 `0` 保留原路径。仅在 AVX2 CPU 数值合同、维度
  可打包且 host head 已驻留时创建 mirror。其他架构/形状保留原路径。
- 在完整专家预载完成后才申请输出头，先检查显存余量和既有 reserve。重排阶段需要两份
  head 的临时空间，完成后释放 row-major 上传副本，仅保留 1,059,061,760 字节。
  不为了输出头逐出已经驻留的专家。mirror 计入 GPU uploaded_bytes，随 engine 释放。
- 老 Windows DLL 缺少可选接口、显存不足或执行失败时回退 CPU。已提交的复制在返回前
  排空；host head 始终保留。未接入旧的 DeepGEMM head_argmax，因为其数值合同不同。
- `DSV4_HEAD_VERIFY=1` 在真实 head_scores_impl 中逐位核对全部 GPU logits 与原
  head_bf16_dot；任何差异使当前计算失败，不能只凭选中的 token 一样就判定正确。

## 验证

- GPU 0、5，各三种形状（32×256、160×4104、129280×4096），每种三个输入，全部
  logits 逐位一致。包含全零输入、最大值并列首行、空参数、非法形状、重复上传拒绝、
  仅一份权重的驻留字节。随机 fixture 明确区分 fused FMA 和独立舍入 mul/add。
- 实际模型 64 步 head verifier：8,273,920 个 logits 零差异；文本及 token 数与上一版一致。
  此轮开启 CPU 对照，不能作为速度成绩。验证后仅合并相邻的相同编译条件，无热路径变化；
  正式性能二进制已重新构建，指纹随原始结果保存。
- CPU / CUDA sm_120 完整 engine 构建、Windows loader 交叉编译通过。
- GPU placement mock 验证默认开启、关闭、低显存拒绝、上传拒绝、重复创建、backend 拒绝、
  字节账和释放；原 placement 与 CUDA 架构标志共 8 项通过。mock 的 ASan/UBSan 通过。

远端隔离目录 `/data/test/colibri-head`，保留 baseline、release-build.log、head-test-final.log、
headverify1.err、各次 `.out` / `.err`。未启动常驻服务，vLLM 不恢复。

## 六卡正式 A/B

DeepSeek-V4-Flash-0731，提示词 `用中文简短解释为什么天空是蓝色的。`，13 prompt tokens，
64 generated tokens。基线与候选交替各三次，每次新进程；decode 为
`(generated - 1) / after_first_seconds`，不含权重加载和首 token。候选不设 HEAD 开关，
使用默认启用行为，且关闭 verifier。

| 版本 | 三次 decode token/s | 中位数 |
| --- | --- | --- |
| 基线 | 9.7013 / 9.8468 / 9.9636 | 9.8468 |
| 输出头驻留 GPU | 13.4846 / 13.3136 / 13.2436 | 13.3136 |

中位数提升 **35.21%**；六次输出文本和生成数量一致，专家均保持 43 层完整驻留，
没有 expert-read fallback。每 token 的中位 decode 耗时约从 101.6 ms 降到 75.1 ms。
完整进程 wall 中位 86.007 → 84.279 秒，预载仍占大头，不能将 decode 增益等同于冷启动收益。
这仍是单提示词、短上下文、单请求 target-only 测量，不代表并发吞吐或普遍质量验收。

环境沿用上一轮：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
DSV4_CUDA=1 DSV4_CUDA_DEVICES=0,1,2,3,4,5
OMP_NUM_THREADS=12 CTX=512
COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1
DSV4_CUDA_EXPERT_MIRRORS=2048 V4_LOADER_LANES=3
COLI_V4_SAVE_USAGE=0 V4_PREFIX_CKPT_DISK=0 COLI_V4_PREWARM=0
```

命令：`BINARY MODEL PROMPT --max-tokens 64 --memory-gb 48`。
这里沿用既有 batched Attention opt-in，并不声称所有配置均为默认值。

新增 CUDA 测试：

```sh
make -C c -f Makefile.deepseek-v4 deepseek-v4 tests/test_dsv4_head_cuda CUDA=1 CUDA_ARCH=sm_120 -j16
OMP_NUM_THREADS=12 c/tests/test_dsv4_head_cuda
OMP_NUM_THREADS=12 compute-sanitizer --tool memcheck --error-exitcode=99 c/tests/test_dsv4_head_cuda
```

完整形状的 memcheck 通过，`ERROR SUMMARY: 0 errors`。

最终性能二进制的长回答核对：上限 192，实际生成 130 tokens，跨 position=128，
与上一轮同配置候选 `371ed4e3` 的 130-token 输出及数量相同（直接复用已验证的旧版
长回答文本，不把两轮时间混算为新的长回答 A/B）。本轮 130 次 `headverify` 共
16,806,400 个 logits，零差异。长回答启用了 CPU verifier，仅用于正确性，不能作速度成绩。

制品 SHA256：

- baseline：`c5c0711ff2027466b2025009f85dd732b8415ba5bff29bbb5962a91171442995`
- candidate：`efaeb414d768b6f1a1b2cc7908354a4dbc5e131505b333a14d72d0023c272048`
- deepseek_v4.c：`604f0d53dd058f4090150f4efce5475ad6056e247f7f949e8e20cb983c0c8c3e`
- backend_cuda_dsv4.cu：`26edbb2ab032c3abcb2baeb6121b68b06f30322e7bbccaccb3ffb2882740f64c`

关闭 `DSV4_CUDA_HEAD=0` 的最终候选也生成相同的 64-token 输出，decode 9.7856 token/s。
[原始结果、环境、完整参考及指纹](dsv4-head-resident-2026-09-28.json)。
