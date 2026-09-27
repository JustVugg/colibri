# DeepSeek V4：decode 输出投影在设备上串联

基线：`7019d0af017a33d3f69b805ac59659843215c064`，PR #1766。
参考 Naruto `d0a00474a781a623197f24ae875b46d14d943704` 的
`core/model/deepseek_v4_attention.cc::ProjectOutput`：正常执行路径将 grouped
`wo_a` 和 `wo_b` 作为一个设备执行单元，中间结果留在 workspace。

## 改动与数值约定

Colibri 原 decode 路径：上传 attended → grouped wo_a → 下载 OA → CPU BF16
舍入 → 上传 OA → wo_b → 下载 output → CPU BF16 舍入。

新增 `dsv4_cuda_wo_decode` 在同一 stream 上完成两次原有 `run_mv` 和中间、
最终的 BF16 舍入，只上传 attended、下载 output。支持 wo_a 的 fmt8 / fmt9，
wo_b 的 fmt8。没有改变矩阵乘法 kernel 或归约顺序，没有额外量化 activation。
现有 `dsv4_cuda_wo` 会对 OA 额外做 FP8 round-trip，因此不能直接复用；其原合同不变。

对于 OA=8192 的形状，每次命中减少 32 KiB D2H、32 KiB H2D 和一次 stream
同步。这是调用链上的减少量，尚不是实测端到端收益。Attended 和最终输出仍经过
CPU，输入端 Q/KV、RoPE、Compressor 状态等也没有在本轮完全迁入设备。

三个单 token Attention 实现副本均接入，batched prefill 不变。两份权重必须位于
同一设备；缺少 mirror、不支持的形状/格式、后端拒绝或旧 Windows DLL 缺少可选
导出时，继续执行原路径。任何已提交的复制都在返回前排空，避免回退与 host buffer
写入竞争。

默认启用；`DSV4_CUDA_WO_DECODE=0` 回退原路径。无需启用 `COLI_CUDA_ATTN_BATCH`。

## 本轮验证

遵循本轮编译交付范围，没有启动模型服务，也没有进行长时间端到端生成或吞吐测试。

- Linux CPU 完整 engine 构建通过。
- yuesheng-gpu 上 CUDA `sm_120` 完整 engine 和新测试构建通过。
- Windows loader 使用 MinGW 交叉编译通过。
- GPU 0、5：groups=1 / 8，wo_a fmt8 / fmt9，每种组合 3 组随机输入，合计
  24 次与旧的两次 matvec + host BF16 路径逐位相同。
  小形状 width=256、rank=128、hidden=256；实际尺寸 width=4096、rank=1024、hidden=4096。
- 测试覆盖缺失 tensor、空输出、无效 groups、维度/格式拒绝、跨设备权重。
  fmt9 测试还确认输入能区分旧 `dsv4_cuda_wo` 的额外 FP8 量化，防止错误复用。
- GPU placement mock 覆盖生产 wrapper 的默认开启、关闭、后端失败及两种 mirror 缺失。
- GPU placement mock 和 CUDA 架构标志测试共 8 项通过。

GPU 单测命令（需要 GPU 0 和 5）：

```sh
cd c
make -f Makefile.deepseek-v4 deepseek-v4 tests/test_dsv4_wo_decode_cuda CUDA=1 CUDA_ARCH=sm_120 -j16
timeout 45s ./tests/test_dsv4_wo_decode_cuda
```

主机隔离目录：`/data/test/colibri-wo-decode`。`baseline` 保留上一版本，
`c/deepseek_v4` 为本轮构建；`test.log` 保留 8 组 `exact=3/3` 输出。
CUDA binary SHA256：
`c5c0711ff2027466b2025009f85dd732b8415ba5bff29bbb5962a91171442995`。
CUDA source SHA256：
`85e05e110d9f4356e8fa763bc40da152e9209b5018777638c96ef091ad05af1d`。

上一轮的 9.593 token/s 属于 `7019d0af`，不能作为本轮速度。本轮的端到端输出
一致性和吞吐变化仍待实测。结束检查时六张 GPU 均为 2 MiB，vLLM 容器停止、
watchdog timer inactive；未恢复 vLLM。
