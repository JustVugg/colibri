# DeepSeek V4 目标引擎（CPU + 可选 CUDA 层）

[English](deepseek-v4.md)

> 本页为早期版本，尚未更新 CUDA 层（Windows DLL / Linux `CUDA=1`）、环境变量
> 参考、前缀检查点与性能数据。以英文页为准。

这是 V4 拆分后第一个 PR 中的 DeepSeek V4 Flash 目标引擎。DSpark 推测解码
不属于本 PR，将放在后续连续（stacked）PR 中。

## 当前范围

- 生产代码位于 `c/deepseek_v4.c`，实验性公共 engine/session API 位于
  `c/deepseek_v4.h`。
- 官方分片 safetensors checkpoint 通过共享 `st.h` 加载。
- 标准 MXFP4 矩阵乘法使用共享 `quant.h`。
- 统一入口 `c/coli` 会把 `run`、`chat`、`serve`、`web` 路由到 V4，
  服务模式会跨请求保留引擎和缓存。
- `--no-dspark` 只是兼容性空操作。本 PR 没有 DSpark 模型、内存层级或推测循环。
- 支持 x86-64、aarch64 Linux 和 Windows/MSYS2。

销毁 engine 前必须先销毁全部 session。

## 共享迁移状态

| checkpoint 路径 | 当前实现 | 后续工作 |
|---|---|---|
| safetensors 索引与区间读取 | 共享 `st.h` | 已完成 |
| fmt7 标准 MXFP4 matmul | 共享 `quant.h` | 已完成 |
| fmt7 常驻 rows16 专家缓存 | 临时 V4 私有布局 | **TODO：**上游提供常驻 rows16 API 后迁移 |
| fmt8 E4M3 + UE8M0 128x128 scales | 共享 `st_read_scale_f32` + `quant.h` `matmul_fp8` | 已完成 |

目前只剩 rows16 常驻缓存布局仍为 V4 私有实现。源码中的
`TODO(upstream-fmt7-rows16)` 明确标出了删除该专用布局前仍需补齐的共享 API。

## 内存策略

典型 checkpoint 有 43 层 transformer、hidden size 4096，每个稀疏层有
256 个路由专家，top-k 为 6。稠密权重大约占 6.27 GiB，常驻 BF16 输出
head 大约占 1.06 GiB；路由专家权重按 RAM 预算流式读取和缓存。

规划器先预留工作区与最小专家工作集，再在预算允许时启用 dense/head 常驻
并扩大专家缓存。Dense 常驻与 DSpark 相互独立，在当前 target-only 版本和
旧调用方传入 `--no-dspark` 时都能正常工作。

`--ram GiB` 是规划预算，不是操作系统硬上限；不传入时按当前可用内存估算。

## 下载

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /path/to/DeepSeek-V4-Flash
```

即使下载工具报告成功，个别 shard 也可能被截断。如果 `st.h` 以越界错误拒绝
某个 shard，请先把所有本地 shard 的文件大小与 Hugging Face 仓库逐一核对，
不要直接判断为引擎故障。

## 构建与使用

```bash
cd c
make deepseek-v4
python ./coli run --model /path/to/DeepSeek-V4-Flash --ram 32 \
  "What is the capital of France?"
python ./coli chat --model /path/to/DeepSeek-V4-Flash --ram 32
python ./coli serve --model /path/to/DeepSeek-V4-Flash --ram 32
python ./coli web --model /path/to/DeepSeek-V4-Flash --ram 32
```

V4 chat 使用模型原生标记。原生服务当前支持 greedy 和一个活动 KV slot。
HTTP gateway 会把 OpenAI 与 Anthropic 的工具转换为 V4 原生 prompt 协议，
再把 DSML 调用块解析回相应协议；grammar 仍不支持。参见
[各引擎 API 支持表](api.md#tool-calling-support)。请求会复用严格匹配的 prompt
前缀，只 prefill 新增后缀；prompt 分叉时则从头 prefill。进程、权重、dense、
head 与专家缓存会保持热状态。

## 验证

Tiny safetensors fixture 在本地生成、已忽略且不提交：

```bash
python -m pip install -r tools/requirements-deepseek-v4-tiny.txt
make deepseek-v4-tiny-check
```

测试覆盖加载、teacher forcing、greedy decode、长/重复 session、
`--no-dspark` 兼容，以及持久化 `SUBMIT`/`DATA`/`DONE` 协议中的两次请求。

真实 checkpoint 可运行：

```bash
make deepseek-v4-oracle MODEL=/path/to/DeepSeek-V4-Flash \
  MEMORY_GB=32 ORACLE_TEACHER_FORCING=32 ORACLE_GREEDY=20
```

这个 oracle 只验证目标引擎。DSpark 开关速度、接受率和 token 一致性证据
属于后续 stacked DSpark PR。

## 后续工作

- 增加非 greedy 采样与更多服务 slot。
- 上游提供常驻 rows16 API 后，删除剩余的 V4 私有 rows16 缓存布局。
- stacked PR 恢复 DSpark 时必须保持目标 token 不变，并提供开关性能与接受率数据。
