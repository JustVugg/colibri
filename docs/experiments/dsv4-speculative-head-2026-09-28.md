# DeepSeek V4：投机验证输出头与 GPU 候选归约

基线为 PR #1766 的 `81c6cb39`。单行 head 已驻留 GPU，但 `head_argmax_batch`
仍在 CPU 上计算所有验证行；它还漏计 `g_v4_prof_head_s`，导致投机 profile 的 head 时间不完整。
本轮将这段接入已有驻留权重，并修正计时，未更改草稿、接受判定、KV 回滚或输出提交顺序。

## 实现

- 复用 rows32 BF16 head，不增加权重副本。每个线程负责一个词表项、同时积累最多四个输入行，
  每次读取的权重供四行复用。每行仍按原列顺序执行独立舍入 mul/add，不切换到不同数值合同的 GEMM。
- 输入、输出均为 batch-major，范围 1..128。设备 scratch 随容量增长，完成后可供原设备上下文复用。
- greedy 在 GPU 按数值比较归约，每行只回传一个 int token ID 和一个 float logit，
  而不是完整词表。并列最大值选最小词表索引，`+0/-0` 按数值相等处理；NaN 被忽略，
  全部无效或不大于 `-FLT_MAX` 时返回无有效候选，与 CPU 扫描规则一致。
- 首轮 `97a47941` 以 `DSV4_CUDA_HEAD_BATCH=1` 显式开启；后续实机验收后默认开启，
  `DSV4_CUDA_HEAD_BATCH=0` 保留 CPU 回退。只接多行投机 head，
  原单行路径不变。无驻留镜像、旧 DLL 缺少可选符号、后端拒绝或容量不支持时回退原路径。
  已入队的拷贝和 kernel 在返回前排空；host head 一直保留。
- `DSV4_HEAD_VERIFY=1` 逐位核对所有批量 logits，再核对紧凑候选与 CPU 扫描。
  任何差异使当前验证失败，由原投机调用链回滚；不把候选提前作为 committed token。
- 补齐 Windows `.def` 中原单行 exact head 和新批量接口的导出。

## 已执行验证

- CPU 完整引擎构建通过，Windows loader 交叉编译通过。
- `test_v4_head_batch` 使用生产 `head_argmax_batch`：CPU 回退、紧凑输出、完整 logits 验证、
  故意破坏非最大 logit、故意破坏候选、无有效候选、profile 计时均覆盖；ASan/UBSan 通过。
- GPU placement mock 覆盖默认开启、显式关闭/开启、非法 batch、后端拒绝与关闭后的回退。
- 实机 `yuesheng-gpu`，x86_64，RTX 5090 的 GPU 0/5，SM120；CUDA 13.3、驱动 610.43.02、
  GCC 14.2。完整 CUDA 引擎与测试编译通过。隔离目录 `/data/test/colibri-head-batch-lROCPh`。
- 小规模 CUDA oracle：32×256、160×4104，batch=1/3/4/5/9/2，CPU 与 GPU logits 逐位一致，
  紧凑候选一致；覆盖不满四行、容量增长再缩小、零输入、首项并列、非法参数、NaN/Inf。
  `compute-sanitizer --tool memcheck` 同一组通过，`ERROR SUMMARY: 0 errors`。

```sh
python -m unittest c.tests.test_v4_head_batch c.tests.test_v4_gpu_placement \
  c.tests.test_deepseek_v4_cuda_arch_flags c.tests.test_deepseek_v4_dspark_source
make -C c -f Makefile.deepseek-v4 deepseek-v4 CUDA=0 -j8
x86_64-w64-mingw32-gcc -std=gnu11 -O2 -c c/backend_loader_dsv4.c -o /tmp/head-loader.o
# 以下在隔离 CUDA 主机执行：
make -C c -f Makefile.deepseek-v4 deepseek-v4 tests/test_dsv4_head_cuda CUDA=1 CUDA_ARCH=sm_120 -j12
OMP_NUM_THREADS=4 timeout 60 c/tests/test_dsv4_head_cuda --small
OMP_NUM_THREADS=4 timeout 60 /usr/local/cuda/bin/compute-sanitizer --tool memcheck \
  --error-exitcode=99 c/tests/test_dsv4_head_cuda --small
```

## 证据边界与剩余差距

首轮 `97a47941` 按编译交付约束，没有运行整模型长推理或常驻服务，也未运行完整词表 batch oracle。
随后用户要求继续实机测试并默认启用；后续验证使用同一 CUDA kernel，结果另见下方记录。

Colibri 先前 13.3136 token/s 是单流 target-only 短提示词结果；Naruto PR #30 的 77.9747
与 204.3038 分别来自投机 C1 与 C16 的另一组长文本 fixture，不能直接比较或当作本轮目标已达成。
Naruto 的设备批处理和紧凑候选思路在这里独立适配为 Colibri C/CUDA 接口，未复制其 packed
浮点排序编码，也未引入其运行时或改变 Colibri 的 AVX2 数值合同。

达到完整执行能力仍需要后续工作：

1. Attention/mHC 激活和递归压缩状态设备驻留，覆盖窗口、ratio-4/128、前缀恢复边界。
2. 可保留已接受前缀的投机状态事务；目前 Colibri 拒绝后仍恢复快照并重放 target 前缀。
3. 多请求物理批与六卡执行调度；目前六卡仍是顺序层放置，不是 Naruto 的 TP2×PP3。
4. 相同提示词、上下文、输出长度、greedy 与 cache 设置的 C1/C16/错峰正式对照。

这些项目尚未实现或验收，不能以本次输出头优化替代。vLLM 与 watchdog 保持停止。


## 六卡整模型验收与默认启用

同一 CUDA kernel 完成 GPU 0/5 的完整 `129280×4096` head oracle，batch
1/3/4/5/9/2 的全部 logits 和紧凑候选一致。完整词表 oracle 未运行 memcheck；
上面的 memcheck 结论仅覆盖小形状。

模型 `/data/models/DeepSeek-V4-Flash-0731`，六张 RTX 5090，连续层放置。
使用新增 `c/tools/bench_dsv4_head_batch.c`，同一驻留引擎、每轮新 session，
禁用内存/磁盘 prefix checkpoint。先预热一轮，再交替次序测三轮，逐 token ID
对照。计时不含引擎加载；decode token/s 为 `(generated-1)/decode_s`。

重复提示词：`请只重复下面这句话，连续写20遍，不要编号或解释：春风吹过山谷。`
25 个 prompt tokens，101 个生成 tokens。以下均为三轮中位数：

| 模式 | decode token/s | TTFT 秒 | session 总秒数 | head 秒数 |
| --- | ---: | ---: | ---: | ---: |
| 单 token target-only | 13.287 | 0.905 | 8.446 | 0.09335 |
| prompt-lookup，原 CPU batch head | 13.148 | 0.837 | 8.477 | 2.69105 |
| prompt-lookup，默认 CUDA batch head | 21.206 | 0.838 | 5.598 | 0.01923 |

相同投机策略下 decode 提升 **61.29%**；包含 session 创建和 prefill 的吞吐由
11.914 到 18.043 token/s。十二次输出逐 token 完全一致，投机两种模式每轮均为
85 drafted / 81 accepted。日志没有 CUDA 错误。这是高重复率 prompt-lookup fixture，
不能外推为普通问答、学习型 DSpark 或多请求吞吐。

额外 verifier 对照逐位比较全部 batch logits，短 fixture 为 13,186,560 项；
30 遍的长 fixture 生成 151 tokens，总位置越过 128，CPU/GPU 输出一致，
125 drafted / 123 accepted，共 19,392,000 项 batch logits 零差异。验证模式同时运行 CPU 和 GPU，耗时不纳入性能结论。

默认仅启用已有投机执行路径上的 CUDA batch head，不默认开启投机解码或 GPU drafter。
`DSV4_CUDA_HEAD_BATCH=0` 可独立回退；镜像或后端不可用时继续走 CPU。

复现短 prompt-lookup 对照（在上述 CUDA 主机、仓库根目录运行）：

```sh
make -C c -f Makefile.deepseek-v4 tools/bench_dsv4_head_batch CUDA=1 CUDA_ARCH=sm_120 -j12
printf '%s' '请只重复下面这句话，连续写20遍，不要编号或解释：春风吹过山谷。' > /tmp/head-repeat.txt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 DSV4_CUDA=1 DSV4_CUDA_DEVICES=0,1,2,3,4,5 \
OMP_NUM_THREADS=12 CTX=512 COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1 \
DSV4_CUDA_EXPERT_MIRRORS=2048 V4_LOADER_LANES=3 COLI_V4_PREWARM=0 \
V4_DRAFT=5 V4_NGRAM=1 V4_MTP=0 V4_NGRAM_PARTIAL_KEEP=1 \
c/tools/bench_dsv4_head_batch /data/models/DeepSeek-V4-Flash-0731 /tmp/head-repeat.txt 160 4 t0d
```

原始 token IDs、计时、验证统计及构建指纹见同目录
`dsv4-speculative-head-2026-09-28-results.json`。


### 普通问答与 GPU drafter 异常

提示词 `用中文简短解释为什么天空是蓝色的。`，13 prompt / 64 generated tokens。
设置 `V4_MTP=1 V4_MTP_GPU=0 V4_MTP_CONF=0 V4_NGRAM=0`，其余环境同上，
benchmark 参数为 `64 4 0vd`。每轮重置 drafter 接受率历史，保留已加载权重。
CPU drafter 是现有默认；关闭 confidence gating 仅为保证本次测试触发真实 target 验证。

三轮非 verifier 中位数：旧 head **11.721 token/s**，默认 CUDA head **12.002 token/s**，
提升 **2.39%**；head 耗时 0.16263 → 0.05582 秒。十二次输出全部一致，
四次 verifier 合计比较 2,068,480 个 batch logits，零差异，所有日志无 CUDA 错误。
每次只触发一次投机，3 drafted / 2 accepted，拒绝后恢复并重放前缀，后续主要为
单 token 解码，因此本 fixture 的整体收益有限。

另行打开 `V4_MTP_GPU=1` 时，旧 CPU head 与新 GPU head 都出现 `refill scale upload` /
`scale upload: invalid argument`，发生在首次 batch head 之前。`mtpverify` 和
`mtpformal` 分别记录 162 / 670 次 invalid argument；虽然输出一致并返回成功，
其耗时受错误回退影响，**不作为健康性能结果**。GPU drafter 仍默认关闭，本轮没有
修改其专家缓存，也没有把此异常判定为已修复。


诊断对照 `mtpnopin` 保持 `V4_MTP_GPU=1`，仅增加 `DSV4_CUDA_PIN_HOST=0`，
同一进程各运行一次旧 head / 默认 head：两次输出一致，CUDA invalid argument 降为零。
这将问题缩小到 GPU drafter 的 host pinning / 上传路径；尚不能凭此确认具体根因。
该诊断只有一轮，不作速度结论，也未改变全局 pinning 默认值。复现参数为 `64 1 0d`，
MTP 环境同普通问答部分，但 `V4_MTP_GPU=1 DSV4_CUDA_PIN_HOST=0`。

最终服务状态已核验：vLLM 容器 exited，watchdog timer disabled/inactive，
watchdog service inactive；所有实验退出，六卡均仅占用 2 MiB，无计算进程。
