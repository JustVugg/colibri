# The terminal UI (`coli tui`)

`colibri-tui` is colibri's chat in a terminal. It is one C file with no libraries
(`c/tui.c`), and it talks to `coli serve` the way the chat page does: the answer
streams from `/v1/chat/completions`, the monitor reads `/health` and `/profile`,
the expert map reads `/experts`.

<p align="center">
  <img src="media/tui/tui-chat.png" width="900" alt="the terminal UI: a conversation on the left, a question, the reasoning folded, the answer, its speed, and the hummingbird sitting on the input box">
</p>
<p align="center"><em>Qwen3.8-Flash-Next on an RTX PRO 4000 through <code>coli serve</code>, at a 64K context.</em></p>

## Start it

```bash
coli tui                       # on the coli serve at http://127.0.0.1:8000
coli tui --attach URL          # on another one
coli tui --model /nvme/qwen38  # starts a server for the model, stops it when you quit
```

With `--model`, the server's output goes to `colibri-tui-server.log` in the
system's temporary directory: on the terminal it would draw over the interface.
A server started with `--api-key` needs the same key: `coli tui --api-key KEY`,
or `COLI_API_KEY` in the environment.

The binary runs on its own as well:

```bash
colibri-tui --server http://127.0.0.1:8000
colibri-tui --demo             # no server: the same screens, made-up answers and numbers
```

Without `--server` it looks for a server on `127.0.0.1:8000` for half a second
and plays the demo when none answers. The demo says so in its first answer and
in the status line.

`make -C c colibri-tui` builds it on Linux, macOS and Windows (MSYS2 UCRT64).
The release archives carry it next to `coli`.

## What it shows

- **The conversations**, on the left: each with its model and how much of the
  context it holds. They live in the TUI's memory: closing it forgets them.
- **The answer** as it streams, its markdown drawn (bold, code, lists). The
  reasoning is folded behind its time; `ctrl+r` opens it. Under the answer, its
  tokens, tok/s and seconds. Each answer has a **copy** button on its first line,
  and each code block one on its header, which also names the language.
- **A picture**, when the server runs an image model (Qwen-Image): the message
  is the prompt, sent to `/v1/images/generations`, and the answer is the
  picture drawn in half blocks (two pixels a cell, in any 24-bit colour
  terminal) with its size, seed and steps. The PNG itself, at full size, is
  saved in `~/colibri-images` (`%USERPROFILE%\colibri-images` on Windows);
  **open** shows it in the system's viewer, **copy path** copies where it is.
- **The monitor** (`F2`): the chat page's Overview and Profiling. The model's
  state, the turn in flight (speed now, tokens written, the prompt, tokens per
  step), the engine's CPU and RAM, the last two minutes of tok/s, the last turn's
  numbers with the 40 before them, the context and the memory, the recent turns;
  and where each turn spends its time, phase by phase, when the engine reports
  phases. It adds the GPU (use, temperature, power) and its VRAM, which
  `/health` gives when `nvidia-smi` runs on the server's machine.
- **The expert map** (`ctrl+e`): a cell for each group of experts in each layer.
  Engines that publish every decode step light them as the model writes; the
  others (Qwen3.8's CUDA chain, for one) show the experts the last turn used.

<p align="center">
  <img src="media/tui/tui-monitor.png" width="900" alt="the monitor: the model's state, the live numbers, the engine's CPU and RAM, the GPU and its VRAM, the tok/s chart, the last turn's cards">
</p>
<p align="center"><em>The monitor on the same server: 20 cores, 45.5 GB of engine RAM, 21.9 of 23.9 GB of VRAM.</em></p>

<p align="center">
  <img src="media/tui/tui-map.png" width="900" alt="the chat with the expert map open: 48 layers of 512 experts, the ones the last turn used lit">
</p>
<p align="center"><em>The expert map: Qwen3.8's 48 layers of 512 experts, those the last answer used lit; 29% of them on the card, 71% in RAM.</em></p>

## The hummingbird

The chat page's mascot (`ui/index.html`), its 19 poses copied as they are by
`c/tools/make_tui_mascot.py`, with the page's moves and timings. It sits on the
input box; it reads while the model reads the prompt, leans while it reasons,
flies while it writes and cheers when it is done. Now and then it swings its
legs, looks around or flaps; after 90 seconds without a key it falls asleep.
A click is a move, a double click a dance; drag it and it flies back to its
place. `ctrl+p`, "Hummingbird: its moves", has all of them.

Where the terminal draws sixel pictures (Windows Terminal 1.22 or later, foot,
WezTerm, mlterm, xterm with `-ti 340`) the bird is a 60 x 60 picture, as on the
page; elsewhere it is drawn with half blocks, bigger. `--sixel` and `--no-sixel`
choose by hand.

## Keys

| key | does |
|---|---|
| enter | sends the message |
| ctrl+j | a new line in the message |
| ctrl+p (F1) | the commands: a new conversation, the model, the monitor, the map, the language |
| ctrl+n | a new conversation |
| tab, then the arrows | picks a conversation |
| tab, then delete | deletes it (enter confirms, esc keeps it) |
| F2 or ctrl+o | the monitor; tab switches Overview and Profiling |
| ctrl+e | the expert map |
| ctrl+r | shows or hides the reasoning |
| ctrl+b | hides the conversations |
| esc | stops the answer |
| ctrl+x | drops the files the next message would take |
| ctrl+y | copies the last answer, whole (a picture's: its file's path) |
| ctrl+k | copies the last answer's last code block |
| a click on copy | copies that answer, or that code block |
| a drag over the text | selects it; the release copies it |
| page up, page down, the wheel | scroll |
| ctrl+q | quits |

## Files and copying

Drop a file on the terminal, or paste or type its full path, and it goes with
the next message, as on the chat page: a picture as a picture, for a model that
reads them; a PDF, Word, OpenDocument, PowerPoint, HTML or text file as its
text, read by the chat page's reader (`ui/docs.py`, which `coli tui` hands to
the TUI). The files waiting are listed above the input, and the message shows
their names. Without `coli tui` (`colibri-tui` on its own) plain text files and
pictures still go; the other documents need the reader.

The copy buttons and `ctrl+y` / `ctrl+k` copy the text the model wrote: a whole
answer however long it is, a code block without its fences and with its long
lines whole. A drag over the text selects the cells on the screen and the release
copies them: what is on the screen, so a line the screen wrapped comes out as two
and an answer taller than the window comes out cut. Both go to the clipboard on
Windows; elsewhere through the terminal (OSC 52, which most terminals and tmux
take) and through `wl-copy`, `xclip` or `xsel` where one runs, `pbcopy` on
macOS. On a POSIX terminal shift and a drag still make the terminal's own
selection.

## Terminals

Windows Terminal, the Windows 10 and 11 console, and any POSIX terminal with
24-bit colour. The window needs 60 x 16 cells at least; the conversations show
from 90 columns. The mouse picks a conversation and plays with the bird.

## Not yet

System One, tools and agents, conversations saved to disk, and
languages besides English and Italian (`--lang it|en`; the default is the
system's).
