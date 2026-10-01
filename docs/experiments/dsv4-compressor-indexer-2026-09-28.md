# DeepSeek V4：继续迁移 Compressor 与 Indexer 的 CUDA 路径

## 范围与参考

机器：yuesheng-gpu，6 × RTX 5090；模型 DeepSeek-V4-Flash-0731。
比较基线为 PR #1766 的 `1c42795b`，即上一轮已启用原生量化、warp 路由、
完整专家驻留和 Indexer 查询权重驻留的版本。

本轮继续参考 Naruto `d0a00474a781a623197f24ae875b46d14d943704`：

- `core/model/deepseek_v4_compressor.cc`：把每个位置的投影交给设备，独立维护压缩状态。
- `core/model/deepseek_v4_indexer.cc`：设备计算查询变换及 head 权重。
- `backend/cuda/cuda_indexer_score.cu`：候选 key 转置暂存于 shared memory，并分解 head 工作。

移植保留 Colibri 自己的数值约定，没有直接替换为 Naruto 的归约顺序。
Naruto 文档里的推测解码接受 token 吞吐、并发总吞吐，与这里的单请求
`target_only=1` decode 口径不同，不能直接算加速倍数。

## 改动

1. **主 Compressor 与 Indexer Compressor 的 decode 投影进入 GPU。**
   原路径每层在 CPU 上计算 `wkv`、`wgate`，虽然 prefill 已有设备副本。
   新增 rows32 BF16 布局，使相邻输出行的权重读取合并；一次输入上传、一次
   kernel 同时计算两个投影，保留逐列 fused FP32 累加。上传使用默认 stream，
   重排使用 nonblocking stream，两者之间显式等待上传完成。原 row-major 副本继续
   服务 prefill，两个副本都计入驻留预算并随 layer 释放。
   APE、滑动状态、pooling、压缩边界仍由 CPU 管理。失败回退前等待已发出的下载完成。
2. **Indexer 单 token 的 Hadamard、FP4 round-trip 和 head 权重投影进入 GPU。**
   BF16 初始取整和 RoPE 仍使用原 CPU 实现，GPU 保留 butterfly 的每步运算顺序、
   BF16 舍入和 FP4 等距时选择第一个编码的规则。head 权重投影按原路径分别舍入乘法、
   加法，与 Compressor 的 FMA 约定不同。失败只影响临时空间，原查询可重新计算。
3. **Indexer 评分按 head 并行。**
   将每块 8 个候选 key 转置到 shared memory，每个 head 独立完成原来的逐维点积，
   再按原 head 顺序相加；Top-K 仍使用原 CPU 排序。旧串行 kernel 保留为回退和测试参考。
4. 补齐上一轮 CUDA 架构标志测试的 `.cuh` fixture 拷贝，修复 CI 中的 7 个依赖缺失失败。

新增路径默认启用，分别可用以下变量关闭：

```sh
DSV4_CUDA_COMPRESSOR=0
DSV4_CUDA_INDEXER_PREP=0
DSV4_CUDA_INDEXER_HEADS=0
```

Compressor 和 Indexer prepare 不要求开启整个 batched Attention 路径。
评分优化作用于原有 CUDA 评分路径；本轮没有更改 `COLI_CUDA_ATTN_BATCH` 的默认值。
旧 Windows DLL 缺少新增可选导出时保留 CPU 回退。

## 验证

- GPU 0、5：Compressor 三种真实投影宽度 256 / 512 / 1024，hidden=4096 / 4103，
  与逐列 FMA 参考逐位相同；原 prefill batch=3 输出不变。
- GPU 0、5：生产 Compressor 状态代码，ratio=4、128，以及 Indexer ratio=4，
  连续跨两个窗口，检查每步状态、输出、reset 和部分写入后的失败回退，全部逐位一致。
- GPU 0、5：Indexer query dimension=32 / 128 / 512，Hadamard/FP4 查询与 head 权重
  对照 CPU 参考逐位一致，包含全零 head；测试输入明确能区分 fused 与分别舍入的累加。
- GPU 0、5：评分 batch=1 / 3，候选数 1 / 7 / 9 / 129 / 513，含零候选及部分可见
  候选，按 head 并行结果与旧 kernel 逐位一致。
- 真实模型设置 `DSV4_IDX_VERIFY=1`，新增 packed 权重核对及 prepare 的 CPU 对照。
  64-token 请求完成 1344 次 projection、1323 次 prepare、1344 次 score 对照，
  合计 4011 次，全部零差异，输出与基线一致。
- 完整短回答生成 130 tokens，覆盖 position=128 压缩边界，输出文本及生成数量与基线
  一致，均在 192-token 上限前正常结束；三项优化全部关闭后的 64-token 输出也一致。
- Indexer 与 Compressor 状态测试通过 CUDA memcheck，均为零错误。
- CPU/CUDA 编译、Windows loader 交叉编译、原 DeepSeek V4 CPU 测试、GPU placement mock、
  CUDA 架构标志 7 项测试通过。

GPU 测试命令（需要设备 0 和 5）：

```sh
cd c
make -f Makefile.deepseek-v4 CUDA=1 CUDA_ARCH=sm_120 \
  tests/test_dsv4_compressor_cuda tests/test_dsv4_compressor_state_cuda \
  tests/test_dsv4_indexer_prepare_cuda
./tests/test_dsv4_compressor_cuda
OMP_NUM_THREADS=4 ./tests/test_dsv4_compressor_state_cuda
./tests/test_dsv4_indexer_prepare_cuda
```

## 性能测量

使用与上一轮相同的 13-token 中文科学问题、64 generated tokens、6 卡、
OMP_NUM_THREADS=12、CTX=512、memory=48 GiB。每次启动新进程，完整预载 11008 个专家；
`decode_tps=(generated-1)/after_first_seconds`，不把预载、TTFT 算入 decode。
开启既有 `COLI_CUDA_ATTN_BATCH=1` 和 `COLI_CUDA_MOE_BATCH=1`，新增开关保持默认。

早期探路版本只迁移 Compressor，之后加入 Indexer。初版三轮中有一次输出发生偏离，
其速度数据已标记为拒绝，不能作为有效加速结论。调查发现 head 投影的 FMA 约定错误，
以及权重上传与 nonblocking stream 上的重排缺少同步；修复后才重新测量。
原始失败记录、修复版本的测量与构建指纹见同名 JSON。

修复后的三轮：

| 版本 | 各轮 decode token/s | 中位数 |
| --- | --- | --- |
| 基线 `1c42795b` | 8.874、8.996、8.926 | 8.926 |
| 本轮迁移 | 9.683、9.593、9.547 | 9.593 |

提升 **7.48%**，六次短请求输出文本及生成数量一致。新版本的总进程 wall time 中位数
为 85.627 秒，基线为 86.400 秒；模型预载仍占大部分时间，因此不能把 decode 提升
当作启动耗时的同等提升。

NVIDIA 的 [Memcpy 同步约定](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
说明，可分页内存的同步 H2D 调用返回时，最终 DMA 仍可能未完成。
[nonblocking stream](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)
又不与 legacy stream 自动同步；因此立即读取新权重的重排必须显式等待上传。
本轮只在新增 packed mirror 的准备阶段增加这项依赖。

权重预读的额外展开实验没有可确认的微基准收益，未保留。

## 尚未解决的差距

这仍是顺序跨卡放置 layer，并非六卡 TP/EP。Attention 尚有主机端归一化、RoPE、
中间激活传输和同步；输出 head 仍有 CPU 工作，当前 decode 也没有接通推测解码。
本轮完成的是三组可独立回退的计算迁移，不是 Naruto 整条设备执行链的完整复刻。
短提示词与 130-token 回答的输出一致，以及 kernel 逐位测试，不等于长上下文或广泛质量验收。

## 构建与复现

测量构建、最终构建的 SHA-256 均保存在 JSON。最终构建只补齐了
`DSV4_IDX_VERIFY=0` 的关闭判断，正常执行与数值 kernel 不变；重新编译后，三个 CUDA
测试在开启或明确关闭验证的条件下均通过。远端成品：
`/data/test/colibri-indexer-correct/c/deepseek_v4`。验证结束后六张 GPU 均回到 2 MiB，
vLLM 容器和 watchdog timer 均保持停止。

同样条件的单次测量：

```sh
env -u DSV4_CUDA_COMPRESSOR -u DSV4_CUDA_INDEXER_PREP \
  -u DSV4_CUDA_INDEXER_HEADS -u DSV4_IDX_VERIFY -u DSV4_DECODE_PROF \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 DSV4_CUDA=1 \
  DSV4_CUDA_DEVICES=0,1,2,3,4,5 OMP_NUM_THREADS=12 CTX=512 \
  COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1 \
  DSV4_CUDA_EXPERT_MIRRORS=2048 V4_LOADER_LANES=3 \
  COLI_V4_SAVE_USAGE=0 V4_PREFIX_CKPT_DISK=0 COLI_V4_PREWARM=0 \
  timeout -k 10s 600s ./c/deepseek_v4 \
  /data/models/DeepSeek-V4-Flash-0731 \
  '用中文简短解释为什么天空是蓝色的。' --max-tokens 64 --memory-gb 48
```
