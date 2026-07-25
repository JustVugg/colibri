# 在 Hugging Face Spaces 免费跑 Qwen3.6 → colibri 容器转换

HF 不提供裸 SSH 虚拟机,但 **Spaces** 给一个带公网出口 + HF Hub 访问的免费容器
(CPU 档:2 vCPU / 16GB RAM)。把"转换+上传"塞进它的启动命令即可,产物直接回传你
的 Hub 仓库。本项目已加 `--low-disk --stream-upload`,本地磁盘峰值压到约一个 shard
(~5GB),任何免费档都跑得动。

## 一次性准备:把 HF token 设成 Space 密钥

转换脚本需要写你账号下的目标仓库,所以要给 Space 一个**有 write 权限**的 token:

1. HF 右上角头像 → **Settings → Access Tokens** → 新建 token,Role 选 **Write**。
2. 进你的 Space → **Settings → Variables and Secrets → New secret**。
3. Name 填 `HF_TOKEN`,Value 贴上面那个 token。Save。
   (huggingface_hub 会自动读这个环境变量,脚本无需改。)

## 三步创建实例

1. **New Space**:https://huggingface.co/new-space
   - Owner = 你的账号(minne100)
   - Space name = 随意(如 `qwen36-colibri-convert`)
   - **SDK 选 `Docker`**,模板选 **`Blank`**(不用 Gradio/Streamlit)
   - 勾选 **`Make the Space private`**(避免别人看到中间日志;产物在你目标仓库,独立)
   - Create Space

2. **上传 `hf_space/` 里的两个文件**到 Space 根目录(用网页 "Files" 标签的 upload,
   或 git clone 后 push):
   - `Dockerfile`
   - `run.sh`
   
   (这俩文件已在本仓库 `hf_space/`。`run.sh` 里的 `--upload-repo` 改成你的目标仓库
   id;`--ebits 4` 是 int4,要更高保真改 `8`。)

3. **等它跑完**:Space 会先 build(装 torch,几分钟),然后执行 `run.sh`。
   - 点 Space 页面的 **"Logs"** 标签实时看进度(每个 shard 转完会打印
     `streamed -> .../model-xxxxx-of-xxxxx.safetensors (local freed)`)。
   - 全部完成后日志末尾出现 `==== CONVERSION + UPLOAD DONE ====`。
   - 容器会 `sleep infinity`,状态保持 Running,不影响产物(已在你 Hub 仓库)。

## 在你笔记本上取成品

```bash
SNAP=$(python -c "from huggingface_hub import snapshot_download; \
                   print(snapshot_download('minne100/qwen36-35b-a3b-colibri-i4'))")
cd colibri/c && make qwen36
SNAP=$SNAP ./qwen36 16 4 ref_qwen36.json
```

## 资源与排错

- **磁盘**:`--low-disk` 边下边删源 shard;`--stream-upload` 转完即传并删本地 → 峰值 ~5GB。
  免费 Space 够用。
- **内存**:逐 shard 处理,单 shard 峰值 ~6GB,16GB 充裕。
- **时间**:35B / int4 / CPU 约 20–60 分钟(取决于 HF 源下载带宽)。
- **失败重试**:Space 重跑即可;`--upload-repo` 用 `exist_ok=True`,已传 shard 会被覆盖。
- **Phase 1 现状**:上传的容器里 Gated DeltaNet 层(75%)仍是占位、引擎跳过——
  是"能跑注意力子集"的成品,完整模型等 Phase 2。

## 不要做的

- 别用 **ZeroGPU** 档:它是给 Spaces 推理用的共享 A100,不适合这种长时 CPU 转换任务,
  且会限流。免费 CPU 档就够。
- 别把 token 写进 `run.sh` 或提交到公开仓库:用 Space secret(`HF_TOKEN`)。
