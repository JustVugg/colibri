# DeepSeek V4：默认驻留与设备端路由

## 改动与默认策略

完整专家驻留现在默认自动选择：`DSV4_CUDA_RESIDENT_EXPERTS=auto`（未设置等价）。
显存预算不足或 resident tier/张量几何不支持时保留原缓存模式并记录原因；
实际加载错误仍拒绝启动。`0` 禁用，`1` 要求完整驻留、预算不足也拒绝启动。
GPU 列表仍由原有 DSV4_CUDA_DEVICES / DSV4_CUDA_DEVICE 配置决定。

完整表存在且为 256 experts/top-6 时，设备端路由默认开启；
`DSV4_CUDA_RESIDENT_ROUTE=0` 可单独退回主机路由。
CUDA 内完成 router、按 expert ID 升序排列、设备指针选择和 MoE；只上传一次
激活、下载一次结果，省掉中间路由结果回传和同步。hash 路由仍从主机映射表取
固定的六个 ID，再上传这 24 字节；概率、排序和专家计算均留在设备端。
保持旧路径的 BF16 舍入、专家累加顺序，不额外量化输入。

每层 descriptor table 借用已驻留权重，由对应设备缓存拥有并释放。
新 DLL 符号是可选的，旧 Windows DLL 继续走旧路径；显式关闭 batched MoE、
禁用 BF16 routing 的构建保留原执行路径。Attention、mHC、层间激活和输出头
尚未迁移成整段设备端执行。

## 验证

- CPU engine build、CUDA SM120 build、Windows loader 交叉编译通过。
- Production unit mock 覆盖默认开启、显式关闭、table 复用/清理、非法 token/hash ID、
  后端拒绝回退；既有多卡/驻留测试和 ASan/UBSan 通过。
- `c/tests/test_dsv4_resident_route.cpp` 在 RTX 5090 GPU 0 和 5 上验证：每卡四个
  非零输出样本，普通/hash 路由交替，与 `dsv4_cuda_route + 排序 + dsv4_cuda_moe`
  的输出逐元素精确相等；拒绝越界 fixed ID。
- 真实模型：DeepSeek-V4-Flash-0731，六卡、48 GiB memory budget、CTX=512、
  OMP_NUM_THREADS=12；同上一轮提示词，13 prompt / 64 generated tokens；无推测解码。
  所有进程独立启动，未清 OS page cache、未固定 CPU affinity。

[原始 JSON](dsv4-device-route-2026-09-27.json)。

| 样本 | 路由设置 | 解码 token/s | 引擎 TTFT 秒 | 说明 |
|---|---|---:|---:|---|
| default1 | 未设置 | 6.080 | 0.934 | 64 tokens |
| single-arithmetic | 单卡自动回退 | 0.931 | 6.727 | 算术 2 tokens，不作性能比较 |
| route1 | 显式开启 | 6.210 | 0.921 | 64 tokens |
| off1 | 显式关闭 | 6.114 | 0.940 | 64 tokens |

route1 / off1 使用同一候选二进制，单轮差约 +1.6%，不足以宣称稳定吞吐提升。
default1 使用最终默认设置候选，另开启 DSV4_DECODE_PROF=1；不能与无剖析样本
直接比较微小差异。三份相同提示词的文本完全一致。

未设置两个新开关时，六卡均记录 `resident-route=on`，预加载耗时 72.574 秒；
expert_requests=11,008，全部由预加载解释，生成没有新增专家存储读取。
单卡在完整驻留预算不足时自动回退，没有启动失败，算术输出 `4` 后 EOS。
这不是长上下文/并发/全模型质量验收，也未建立与最早修正 shared 权重绑定前版本的逐 token 等价。

最终默认候选实测源码 SHA-256：
- deepseek_v4.c：`c97fe0294529319aa36a69ae2a41f105bf8ad8430f80e4c1bf886f7cdbf0a1be`
- backend_cuda_dsv4.cu：`3515521a48cbb835a645c23fd611326dcc97f42d8e2b92f671b7c96688e82fa6`

此后仅将旧 MoE 错误串清理移到快路径之前，未修改 kernel 或数值计算；最终版本
重新编译并复验 CUDA oracle。没有将带该诊断整理的版本另跑长模型性能测试。

## 剩余瓶颈

默认候选的 DSV4_DECODE_PROF 收集到 62 个完整 43 层样本，分段时间中位数：
Attention 88.5 ms，MoE 40 ms，mHC 8 ms。计时为 block 阶段，不包含完整端到端
输出头/调度等开销；不能将三项相加当成完整 token 时长。
目前应优先处理 Attention 执行路径，不能把性能差距继续归因于路由回传。

完整日志、候选构建与测试脚本保留在 `/data/test/colibri-device-route/`。
模型测试结束后六卡均为 2 MiB / 0%，vLLM 仍 stopped，watchdog inactive；未部署常驻服务。
