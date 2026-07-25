# Qwen3.6 云端转换并回传 HF Hub

目标:**在你这台 16GB 无独显笔记本上只下载最终的 `.coli` 容器(~18GB),把"下载 70GB bf16 权重 + 转换 + 暂存 88GB"的活儿丢到云端做。**

`convert_qwen36.py` 已支持:

| 参数 | 作用 |
|---|---|
| `--repo ID` | 从 HF Hub 拉权重(已支持) |
| `--upload-repo ID` | 转换完直接把 `.coli` 容器推回 HF Hub |
| `--low-disk` | 分片流式:**只删输入源 shard**(从 HF 拉的 70GB 原始权重),磁盘峰值只有"一个源 shard + 渐进输出";**生成的 18GB `.coli` 容器始终保留并上传**,适合小磁盘云实例 |
| `--hf-token` | gated 模型鉴权 + 上传鉴权 |
| `--ebits` | 专家量化位宽(默认 4;Phase-1 校验用 8) |
| `--no-readme` | 不上传 README.md |

## 推荐云环境

转换是 **CPU + 磁盘 I/O 密集型**(逐行对称量化 + safetensors 读写),不需要 GPU。

| 方案 | 磁盘 | 内存 | 备注 |
|---|---|---|---|
| RunPod / Lambda / Vast.ai CPU 实例 | 100GB+ | 32GB+ | **首选**,`--low-disk` 都不必开 |
| Colab Pro (高 RAM) | ~150GB | 高 RAM | 跑得动;`--low-disk` 更稳 |
| HF Spaces(付费 80–200GB) | 中 | 中 | 可,但免费 50GB 偏紧,务必 `--low-disk` |
| 你本机(24GB) | 88GB 峰值 | 够 | 能做,但**意义不大**——正是想避开这条 |

## 一键流程

在云实例里:

```bash
pip install torch safetensors huggingface_hub
git clone https://github.com/JustVugg/colibri && cd colibri/c

# 标准(云盘充裕):
python tools/convert_qwen36.py --repo Qwen/Qwen3.6-35B-A3B \
    --out ./qwen36_i4 --ebits 4 \
    --upload-repo YOUR_HF/qwen36-35b-a3b-colibri-i4

# 小磁盘 / HF Spaces:流式,峰值磁盘最低
python tools/convert_qwen36.py --repo Qwen/Qwen3.6-35B-A3B \
    --out ./qwen36_i4 --ebits 4 --low-disk \
    --upload-repo YOUR_HF/qwen36-35b-a3b-colibri-i4
```

`--upload-repo` 会:在 Hub 建仓 → 上传整个输出目录(safetensors 分片 + `qwen36_meta.json` + 自动生成的 `README.md`)。`README.md` 已写好使用说明,别人 `snapshot_download` 下来就能直接 `make qwen36 && ./qwen36 ...`。

## 笔记本端消费

```bash
# 只拉 18GB 容器,不碰 70GB 原始权重
SNAP=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('YOUR_HF/qwen36-35b-a3b-colibri-i4'))")
cd colibri/c && make qwen36
SNAP=$SNAP ./qwen36 16 4 ref_qwen36.json
```

## 关键尺寸核对(转换完必查)

下载前先确认 `upload-repo` 里的 `qwen36_meta.json`:
- `attn.head_dim` = 256、`attn.rope_dim` = 64
- `moe.n_group`/`moe.topk_group`(官方未明示,默认 1)、`moe.has_bias`
- 若 `rope_dim` 落到回退值 `head_dim/4`,说明 config 里真字段名没被识别(已试 `rope_dim`/`rotary_emb_dim`/`qk_rope_head_dim`)。

## 注意事项

- **Phase 1 现状**:容器里 DeltaNet 层(75%)仍是占位,引擎跳过它们。这是"能跑注意力子集"的容器,不是完整模型。完整模型要等 Phase 2(DeltaNet 纯 C 实现)。
- 上传的 repo 带 `colibri`/`qwen3.6` tag,**不是 transformers 可加载的模型**,避免别人误用。`README.md` 已明确说明。
- gated 模型(Qwen3.6 若需申请)先在 HF 网页同意协议,再传 `--hf-token`。
