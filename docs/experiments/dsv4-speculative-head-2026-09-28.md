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
- `DSV4_CUDA_HEAD_BATCH=1` 显式开启，未设置或 `0` 沿用 CPU。只接多行投机 head，
  原单行路径不变。无驻留镜像、旧 DLL 缺少可选符号、后端拒绝或容量不支持时回退原路径。
  已入队的拷贝和 kernel 在返回前排空；host head 一直保留。
- `DSV4_HEAD_VERIFY=1` 逐位核对所有批量 logits，再核对紧凑候选与 CPU 扫描。
  任何差异使当前验证失败，由原投机调用链回滚；不把候选提前作为 committed token。
- 补齐 Windows `.def` 中原单行 exact head 和新批量接口的导出。

## 已执行验证

- CPU 完整引擎构建通过，Windows loader 交叉编译通过。
- `test_v4_head_batch` 使用生产 `head_argmax_batch`：CPU 回退、紧凑输出、完整 logits 验证、
  故意破坏非最大 logit、故意破坏候选、无有效候选、profile 计时均覆盖；ASan/UBSan 通过。
- GPU placement mock 覆盖默认关闭、显式关闭/开启、非法 batch、后端拒绝与关闭后的回退。
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

按本轮编译交付约束，没有运行整模型长推理或常驻服务。新批量核的完整词表尺寸、真实 DSpark
拒绝/重放场景、六卡端到端收益尚待验证，因此不默认启用，不报告新增 token/s。
不带 `--small` 的 CUDA oracle 包含 129280×4096 形状，但本轮未运行该组。

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
