# DeepSeek V4 六卡完整专家驻留测试

代码：`919c58cdbb0cc07258726e6caba8b36d1a7ef771`。
目标机实际编译的 `c/deepseek_v4.c` SHA-256：
`8a1265e5f4d152dc32aaa21013b0d63bf1edf9af26474151396367faa74dc219`。
原始结果：[JSON](dsv4-full-resident-2026-09-27.json)。

## 实现与验证

借鉴 Naruto 的完整设备端专家表，在引擎启动时加载全部 43 × 256 个 target
routed experts；按层归属分到六张 GPU，直接按 layer/expert 索引。只有全部设备
完成加载才发布表。显存不足、读取失败、形状不匹配、上传失败均拒绝启动，释放
部分分配。已发布表不驱逐权重，执行失败明确报错。Prefill 复用逐 token GPU MoE，
不进入 CPU expert union，也不创建重复 streaming bank。

开关 `DSV4_CUDA_RESIDENT_EXPERTS=1`；默认关闭。每卡加载预算包括已上传 dense
mirrors、完整专家权重和默认 2800 MiB 保留空间。LRU mirror capacity 不限制完整表。
MTP cache、CPU cache memory budget 保留原机制。

本地 CPU engine build、production GPU unit mock、ASan/UBSan 通过。Mock 覆盖六卡
层归属、直接索引、成功加载、第二卡预算不足、第二卡上传/读取失败后的完整清理、
lease 释放和 resident compute 失败。目标机 CUDA-enabled C 单元重新编译链接，
复用未修改源码对应的 generic CUDA backend object。

## 实测口径

- yuesheng-gpu，6 × RTX 5090；`DeepSeek-V4-Flash-0731`。
- CUDA_VISIBLE_DEVICES / DSV4_CUDA_DEVICES = 0,1,2,3,4,5；DSV4_CUDA=1。
- OMP_NUM_THREADS=12，CTX=512，memory budget=48 GiB，V4_LOADER_LANES=3。
- COLI_CUDA_ATTN_BATCH=1、COLI_CUDA_MOE_BATCH=1；DSV4_CUDA_EXPERT_MIRRORS=2048。
- COLI_V4_SAVE_USAGE=0、V4_PREFIX_CKPT_DISK=0、COLI_V4_PREWARM=0；未启用推测解码。
- 提示词：`用中文简短解释为什么天空是蓝色的。`；13 prompt tokens，64 generated tokens。
- 同一二进制，依次 full1、full2、off1、full3；每次独立进程，off1 仅关闭新开关。
- 未清理 OS page cache、未固定 CPU affinity；full 三轮、off 一轮，不是广泛性能结论。
- 解码速度 `(generated - 1) / after_first`；引擎 TTFT 不包含启动预加载。

## 结果

| 模式 | 解码速度 token/s | 引擎 TTFT 秒 | 专家预加载秒 | 进程总耗时秒 |
|---|---:|---:|---:|---:|
| full1 | 6.129 | 0.947 | 75.771 | 94.095 |
| full2 | 6.160 | 0.927 | 73.442 | 91.094 |
| off1 | 2.149 | 12.276 | - | 46.165 |
| full3 | 6.155 | 0.934 | 73.115 | 89.696 |

完整驻留三轮解码中位数 **6.155 token/s**，相对同二进制关闭开关的
2.149 token/s 为 **2.86 倍**。
三轮引擎 TTFT 中位 0.934 秒；预加载中位 73.442 秒；进程总耗时中位 91.094 秒，
高于关闭开关的 46.165 秒。因此此模式改善权重就绪后的生成，不改善一次性短请求
从零启动的总延迟。

三轮每次 expert_requests 均恰好为 11,008，与启动加载专家数一致；
`bytes=147169738752` 也完全相同，预填充和生成未增加专家存储读取。
这不等于 GPU 不再读取权重，而是之后只读已驻留的设备权重。
每卡峰值约 24,422–24,452 MiB，最后一卡 27,860 MiB；各卡峰值合计约 146.5 GiB。

四轮生成文本完全一致，也与上一版 resident-hit 优化的文本一致。
仍不宣称与早期修正 shared up/down 绑定之前的版本逐 token 等价。
没有新增 logits 误差、长上下文、并发、取消或广泛模型质量验收。
GPU 激活全程驻留、SM120 MMA、TP/EP、DSpark 加速均未包含在本次移植中。

全部进程正常退出；结束后六卡均为 2 MiB / 0%，vLLM stopped，watchdog timer inactive。
没有替换或启动常驻服务。完整日志及脚本在 `/data/test/colibri-full-resident/`。
