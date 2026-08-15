_Leggi il leggimi in [Italiano](README.IT.md)._


# Colibrì Guide - Local Inference Engine

A simple guide to running **Colibrì**, a local inference engine based on GLM 5.2, without needing programming knowledge. If you already have Docker installed, you are well on your way.

---

## 📋 Table of Contents

* [What is Colibrì?](https://github.com/JustVugg/colibri/blob/main/docker/README.md#what-is-colibr%C3%AC)
* [Requirements](https://github.com/JustVugg/colibri/blob/main/docker/README.md#requirements)
    * [Hardware](https://github.com/JustVugg/colibri/blob/main/docker/README.md#hardware)
    * [Software](https://github.com/JustVugg/colibri/blob/main/docker/README.md#software)
* [How to get started](https://github.com/JustVugg/colibri/blob/main/docker/README.md#how-to-get-started)
    * [Step 1: Download the model](https://github.com/JustVugg/colibri/blob/main/docker/README.md#step-2-download-the-colibr%C3%AC-dockerfile)
    * [Step 2: Download the Colibrì Dockerfile](https://github.com/JustVugg/colibri/blob/main/docker/README.md#step-2-download-the-colibr%C3%AC-dockerfile)
    * [Step 3: Build the Docker image](https://github.com/JustVugg/colibri/blob/main/docker/README.md#step-3-build-the-docker-image)
    * [Step 4: Start Colibrì](https://github.com/JustVugg/colibri/blob/main/docker/README.md#step-4-start-colibr%C3%AC)
    * [What does that command mean?](https://github.com/JustVugg/colibri/blob/main/docker/README.md#what-does-that-command-mean)
    * [Using Colibrì](https://github.com/JustVugg/colibri/blob/main/docker/README.md#using-colibr%C3%AC)
* [Enter the container](https://github.com/JustVugg/colibri/blob/main/docker/README.md#entering-a-linux-console-inside-the-container)
* [Troubleshooting](https://github.com/JustVugg/colibri/blob/main/docker/README.md#troubleshooting)
* [Technical notes](https://github.com/JustVugg/colibri/blob/main/docker/README.md#technical-notes)
* [Frequently Asked Questions](https://github.com/JustVugg/colibri/blob/main/docker/README.md#frequently-asked-questions)
* [Support and contributions](https://github.com/JustVugg/colibri/blob/main/docker/README.md#support-and-contributions)
* [Testing on a low resource PC](https://github.com/JustVugg/colibri/blob/main/docker/README.md#testing-on-a-low-resource-pc)

---

## What is Colibrì?

Colibrì is an application that allows you to run an artificial intelligence model (GLM 5.2) directly on your computer, without connecting to external servers. It is also possible to run it in Docker, which isolates the application from the rest of the system.

> **Important note**: The model is very large. Expect to wait several minutes for an answer to a simple question, especially with low RAM. At the end of this readme, you will see the result on my PC (without a discrete graphics card), and I reach 0.01 tokens per second.

---

## Requirements

### Hardware

| RAM Memory | Works? | Notes |
| --- | --- | --- |
| < 16 GB | ❌ No | Insufficient memory |
| 24 GB | ⚠️ Maybe | Possible, needs testing |
| 32 GB | ✅ Yes | The minimum (but see memory section for Windows) |
| 48+ GB | ✅ Yes | Better |

Additionally: a **fast SSD** is essential. Colibrì uses the disk as additional memory. Having an NVidia graphics card is even better.

### Software

* **Docker Desktop** (Windows, Mac, Linux) — [download here](https://www.docker.com/products/docker-desktop/)
* **Python** (only if you want to download the model yourself)
* Windows: [python.org](https://www.python.org) or Microsoft Store
* Linux: `apt-get install python3 python3-pip`
* Mac: [python.org](https://www.python.org) or Homebrew

No build environment is needed. Everything happens inside the Docker container.

---

## How to get started

### Step 1: Download the model

The GLM 5.2 model is approximately **360 GB**. Choose one of these methods:

#### Method A: Using Python (recommended)

1. **Install the library for Hugging Face:**
```bash
python -m pip install -U huggingface_hub[cli]
```

On Linux, use `python3` instead of `python`.
2. **Download the model** (open the terminal in the folder where you want to save it):
```bash
hf_download mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp --local-dir .
```

**Example**: if you want to save it in `C:\LLM\models\glm-5.2` (Windows):
* Open PowerShell in that folder
* Copy and paste the command above
* Wait (a long time)

#### Method B: Without Python (only if necessary)

If you are on Windows and cannot get it to work with Python:

* Download manually from [Hugging Face](https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp)
* Unzip into a folder (e.g., `C:\LLM\models\glm-5.2`)

---

### Step 2: Download the Colibrì Dockerfile

1. Go to: [https://github.com/JustVugg/colibri/blob/main/docker/Dockerfile](https://github.com/JustVugg/colibri/blob/main/docker/Dockerfile)
2. Click the **Download** button (⬇️ icon) in the top right
3. Save the file in a folder (e.g., `C:\LLM\Colibrì`)

---

### Step 3: Build the Docker image

Open the terminal (PowerShell on Windows, Terminal on Mac/Linux) **in the folder where you saved the Dockerfile** and type:

**Windows:**

```bash
docker build -t colibri-i .
```

**Linux/Mac:**

```bash
sudo docker build -t colibri-i .
```

Wait for it to finish (a few minutes). If everything goes well, you will see: `Successfully tagged colibri-i:latest`

> **This builds the code on GitHub, not the code on your disk.** The Dockerfile
> clones the repository inside the image — that is what makes this path work
> with a single downloaded file and no checkout. If you are *developing* and
> want your local changes in the image, use `Dockerfile.slim` instead (see
> [Slim image](#slim-image-developers--production) below); it copies `c/` from the build
> context.

> **If you want to receive repository updates**: First delete the old image with `docker rmi colibri-i` and rebuild.

---

### Step 4: Start Colibrì

Open the terminal and type the command below (replace `C:\LLM\models\glm-5.2` with the actual path on your PC):

**Windows** (PowerShell):

```bash
$MODEL_PATH="C:\LLM\models\glm-5.2"
docker run --rm -it --name colibri-c `
  -v "$MODEL_PATH`:/app/glm-5.2" `
  -e COLI_MODEL=/app/glm-5.2 `
  colibri-i ./coli chat
```

**Mac/Linux** (Terminal/Bash):

```bash
MODEL_PATH="/path/to/glm-5.2"
docker run --rm -it --name colibri-c \
  -v "$MODEL_PATH:/app/glm-5.2" \
  -e COLI_MODEL=/app/glm-5.2 \
  colibri-i ./coli chat
```

**Example for Linux:**

```bash
MODEL_PATH="/home/user/LLM/glm-5.2"
docker run --rm -it --name colibri-c \
  -v "$MODEL_PATH:/app/glm-5.2" \
  -e COLI_MODEL=/app/glm-5.2 \
  colibri-i ./coli chat
```

---

### What does that command mean?

| Part | Explanation |
| --- | --- |
| `docker run` | Starts a container |
| `--rm` | Deletes the container when you close it |
| `-it` | Interactive mode (you can write and read) |
| `-v "PATH:/app/glm-5.2"` | Mounts your model inside the container |
| `-e COLI_MODEL=/app/glm-5.2` | Tells Colibrì where to find the model |
| `colibri-i` | Name of the Docker image |
| `./coli chat` | Starts Colibrì in chat mode |

---

### Using Colibrì

Once started, you will see a prompt like this:

```
  ──────────────────────────────────────────────────────────
  type and press Enter · Ctrl-C stops the answer · :more continues · :reset clears memory · :q exits

```

**Useful commands:**

* `Write a question + Enter` → Receive the answer
* `Ctrl + C` → Stop the answer
* `:reset` → Clear conversation memory
* `:q` → Exit

**Usage example:**

```
› How many inhabitants does China have?

China is currently the most populous country in the world. 
The population is approximately 1.41 billion people.

```

The model understands **Italian, English, Chinese, and other languages**, although it is optimized for English and Chinese.

---

## Enter the container

If you want to explore the container as if it were a normal Linux machine:

```bash
docker run --rm -it --name colibri-c \
  -v "MODEL_PATH:/app/glm-5.2" \
  -e COLI_MODEL=/app/glm-5.2 \
  colibri-i /bin/bash
```

Now you are inside Linux. Type `exit` to leave.

---

## Troubleshooting

### ❌ "Docker not found"

**Cause**: Docker is not installed or the terminal does not recognize it.

**Solution**:

1. Reinstall [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Restart your computer
3. Open a new terminal and try again

---

### ❌ "Out of memory" or container closes immediately

**Cause**: Your computer does not have enough RAM, or on Windows, WSL is using less memory than available.

**Solution for Windows (WSL):**

1. Open PowerShell and check the memory available to WSL:
```bash
wsl
cat /proc/meminfo | grep MemTotal
exit
```


Divide the number by 1,073,741,824 (which is 1024³) to get it in GB.
2. If WSL uses less than what you have, create a configuration file:
* Open a text editor (Notepad is fine)
* Copy this:
```ini
[wsl2]
memory=24GB
processors=12
swap=16GB

```

* Save the file with the name: `.wslconfig` (with the dot)
* Place it in: `C:\Users\YourUsername\`

3. Restart WSL from PowerShell:
```bash
wsl --shutdown
wsl
```

4. Check again:
```bash
# cat /proc/meminfo | grep MemTotal
# exit
```

**Solution for Mac/Linux**: Increase the RAM available to Docker from the Docker Desktop settings, or add more RAM to the computer.

---

### ❌ The answer is very slow

**Possible causes**:

1. The disk is slow
2. You have low RAM
3. Colibrì is using the disk as additional memory (normal)

**How to check disk speed:**

**Windows** (PowerShell as administrator):

```bash
winsat disk -drive C
```

Change `C` with your disk letter.

**Linux/Mac** (Terminal):

```bash
sudo hdparm -Tt /dev/sda
```

Change `/dev/sda` with your disk (check with `lsblk` on Linux).

A **modern NVMe SSD** reaches 15 GB/sec. If yours is under 2-3 GB/sec, it is slow.

---

### ❌ "Permission denied" on Linux

**Cause**: Docker requires administrator permissions.

**Solution - Option 1** (quick):

```bash
sudo docker build -t colibri-i .
sudo docker run ... (as above, with sudo in front)
```

**Solution - Option 2** (permanent):

```bash
sudo usermod -aG docker $USER
# Restart the computer
docker run ... (without sudo)
```
---

### ❌ "Image not found" or error during build

**Cause**: The Dockerfile is corrupted or not in the right folder.

**Solution**:

1. Verify that the Dockerfile is in the folder where you open the terminal:
```bash
ls Dockerfile  # Mac/Linux
dir Dockerfile # Windows
```

2. Redownload the Dockerfile from the GitHub repository
3. Delete the old image: `docker rmi colibri-i`
4. Retry the build

---

### ❌ "hf_download: command not found"

**Cause**: The Hugging Face library is not installed correctly.

**Solution**:

```bash
pip install -U huggingface_hub[cli]
# or on Linux/Mac:
pip3 install -U huggingface_hub[cli]
```

Then retry the `hf_download` command.

---

### ❌ The model does not download (timeout or network errors)

**Causes**: Slow or unstable connection, Hugging Face temporarily unavailable.

**Solution**:

1. Wait and retry the `hf_download` command
2. If it continues, download manually from [here](https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp)
3. Unzip the ZIP file into the desired folder

---

## Technical notes

### Why is the disk important?

Colibrì uses the disk as "additional virtual RAM" (paging). A **fast** disk is crucial for decent performance.

* **NVMe SSD** (recommended): 1-15 GB/sec
* **SATA SSD**: 0.5-1 GB/sec
* **Rotational hard disk**: 0.05-0.1 GB/sec ❌ (too slow)

If your disk is slow, the answers will be very slow even with a lot of RAM.

---

### Recommended default configuration for WLS on Windows

If you have **exactly 32 GB of RAM** and are using Windows, it is very likely that WLS by default is set to consume no more than 16 GB of RAM. We need to increase this limit [Troubleshooting](#troubleshooting) . In my case I adopted this configuration:

```ini
[wsl2]
memory=24GB
processors=12
swap=16GB

```

That is, in my case, I left 8 GB of RAM and 4 CPUs to Windows and gave 24 GB and 12 processors to WSL + Linux.

---

## Frequently Asked Questions

**Q: What if I have less than 32 GB of RAM?**

A: It is likely that it will not work well. You can try if you have 24 GB, but it is not guaranteed.

**Q: Can I increase the response speed?**

A: Yes, partly:

* Use a fast NVMe SSD
* Increase RAM
* Reduce the complexity of questions
* Use `:reset` to clear memory and lighten the load

**Q: Can I use Colibrì without Docker?**

A: Colibrì was born that way, but this guide assumes Docker. To build from source, see the GitHub repository.

**Q: How much internet connection do I need after downloading the model?**

A: Zero. Colibrì works completely offline.

---

## Support and contributions

If you find errors or have suggestions for improving this guide, open an issue or a pull request on the Colibrì GitHub repository.

Have fun! 🐦

---

## Testing on a low resource PC

In the first case, I asked a question in Italian; in the second, in Japanese; and in the third, I repeated the question in Japanese but requested an answer in Italian.

```
PS C:\quack\llm\colibri\docker> docker run --rm -it --name colibri-c -v "C:\quack\llm\models\glm-5.2:/app/glm-5.2" -e COLI_MODEL=/app/glm-5.2 colibri-i ./coli chat

     ▄▀▀▀▄  ▄        colibrì v1.0
  ▄▄▄▄▀▀▀▀▄▀▀        tiny engine, immense model
      ▀▀▀▀▀▀▀        GLM-5.2 · 744B MoE · int4 · streaming CPU
        ▀▀▀▀         chat · glm-5.2 · ram -GB · topp off
          ▀
  ──────────────────────────────────────────────────────────
  type and press Enter · Ctrl-C stops the answer · :more continues · :reset clears memory · :q exits

  ╭────────────────────────────────────────────────────────────────────────────────────────────────╮
  │ › Quanti abitanti ha la Cina?                                                                  │
  ╰────────────────────────────────────────────────────────────────────────────────────────────────╯

  ◆ colibrì
  La Cina è attualmente il paese più popoloso al mondo (sebbene, secondo alcune stime recenti, sia stata ormai superata dall'India).

  La popolazione totale della Repubblica Popolare Cinese è di circa 1,41 miliardi di abitanti (dati del 2020-2022 circa).
  └─ 76 tok · 0.04 tok/s · hit 3% · RSS 15.9 GB · 2012s

  ╭────────────────────────────────────────────────────────────────────────────────────────────────╮
  │ › 漫画「ワンピース」の主人公の名前を教えてください。名前だけで、それ以上のコメントはありません。                                              │
  ╰────────────────────────────────────────────────────────────────────────────────────────────────╯

  ◆ colibrì
  ルフィ
  └─ 2 tok · 0.01 tok/s · hit 1% · RSS 16.7 GB · 260s

  ╭────────────────────────────────────────────────────────────────────────────────────────────────╮
  │ › 漫画「ワンピース」の主人公の名前を教えてください。名前だけで、それ以上のコメントはありません。イタリア語で返信                                      │
  ╰────────────────────────────────────────────────────────────────────────────────────────────────╯

  ◆ colibrì
  Il nome del protagonista di One Piece è Monkey D. Luffy.
  └─ 14 tok · 0.02 tok/s · hit 2% · RSS 17.3 GB · 593s

  ╭────────────────────────────────────────────────────────────────────────────────────────────────╮
  │ ›

```

---

## Slim image (developers / production)

The guide above uses `Dockerfile`, which clones the repo inside the image — one
file to download, nothing else on your machine. For CI, servers, or anyone who
already has the repo checked out, `Dockerfile.slim` is a multi-stage build that
produces a **~115 MB** image (vs. the full toolchain the clone-in-image keeps
around), runs as a **non-root** user, and compiles the engine with a
**portable** ISA (`x86-64-v3`) so the image runs on any modern x86-64 host
instead of only the machine that built it.

Build it from the repo root (it needs the `c/` sources as context):

```bash
git clone https://github.com/JustVugg/colibri.git && cd colibri
docker build -t colibri -f docker/Dockerfile.slim .
```

Run — the model is bind-mounted at `/model`, never baked in:

```bash
docker run --rm -v /nvme/glm52_i4:/model colibri info
docker run --rm -it -v /nvme/glm52_i4:/model colibri chat --ram 24
docker run --rm -p 5000:5000 -v /nvme/glm52_i4:/model colibri \
    serve --host 0.0.0.0 --model-id glm-5.2 --ram 24
```

**Reaching it from another machine (or `coli web`).** Binding `--host 0.0.0.0`
is not enough: the server has a DNS-rebinding guard that only trusts the Host
header for loopback and its own bind address, so a browser pointed at
`http://<server-ip>:<port>` gets `403 Host header not allowed`. Behind Docker's
port mapping the container cannot know the address the browser will send, so
tell it what to accept — the exact name/IP, or `*` when the bind is already
deliberately public:

```bash
docker run --rm -p 36873:8000 -e COLI_ALLOW_INSECURE_BIND=1 \
    -v /nvme/glm52_i4:/model colibri \
    web --host 0.0.0.0 --allowed-host '*'
```

`--allowed-host '*'` disables the guard (it warns at startup); prefer an
explicit `--allowed-host my-server.lan` when you know the name. This is a
LAN-exposure decision, hence opt-in alongside `COLI_ALLOW_INSECURE_BIND=1`.

The entrypoint is `coli`, so the first argument is the subcommand (`info`,
`chat`, `serve`, `run`, `plan`, `doctor`) — no `./coli` prefix. The container
runs as UID 1000, so the model directory must be readable by that user
(a directory you own already is; a root-owned one needs `chmod -R a+rX` or a
matching `--user`).

### One-command server (docker compose)

```bash
MODEL_DIR=/nvme/glm52_i4 COLI_RAM=24 docker compose -f docker/docker-compose.yml up
```

Serves the OpenAI-compatible API on `localhost:5000`. The converter (which
needs PyTorch) is intentionally not in this image — convert from a source
checkout, then point `MODEL_DIR` at the result.

[source: 1]