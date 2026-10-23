#!/usr/bin/env python3
"""Pictures inside a terminal, stdlib only.

Four ways to put an image on screen, best first, and the rule that picks one:

    kitty    kitty graphics protocol (kitty, Ghostty): the PNG itself, scaled
             by the terminal, alpha included
    iterm    iTerm2 inline images (iTerm2, WezTerm, VS Code with images on)
    sixel    DEC sixel (Windows Terminal 1.22+, foot, mlterm, xterm -ti vt340,
             konsole, mintty): quantized here to a 252-colour palette
    blocks   Unicode upper half blocks in 24-bit colour, two pixels per cell:
             works everywhere a colour terminal does, including tmux
    none     not a terminal, or switched off

COLI_IMAGE_PROTOCOL=kitty|iterm|sixel|blocks|none overrides the detection.
"""
import base64
import os
import re
import shutil
import sys
import time
from array import array
from operator import itemgetter

PROTOCOLS = ("kitty", "iterm", "sixel", "blocks", "none")
_ALIASES = {"iterm2": "iterm", "halfblocks": "blocks", "block": "blocks", "ansi": "blocks",
            "off": "none", "0": "none", "no": "none", "sixels": "sixel"}
_UPPER_HALF = "▀"
_DA1_CACHE = {}


# ---------------------------------------------------------------- detection
def _normalize(value):
    value = (value or "").strip().lower()
    value = _ALIASES.get(value, value)
    return value if value in PROTOCOLS else None


def da1_attributes(timeout=0.4, stdin=None, stdout=None):
    """The terminal's Primary Device Attributes, as a set of ints, or None.

    Asked with ESC [ c and read back in non-canonical mode. Attribute 4 means
    sixel graphics. Only on a POSIX terminal with both ends interactive; an
    answer that does not come within `timeout` is taken as no answer (a late
    one would otherwise surface as typed garbage, so the timeout is generous
    enough for Windows Terminal behind WSL's pseudoconsole)."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    try:
        fd_in, fd_out = stdin.fileno(), stdout.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    key = (fd_in, fd_out)
    if key in _DA1_CACHE:
        return _DA1_CACHE[key]
    result = None
    try:
        import select
        import termios
        if not (os.isatty(fd_in) and os.isatty(fd_out)):
            return None
        old = termios.tcgetattr(fd_in)
        try:
            new = termios.tcgetattr(fd_in)
            new[3] &= ~(termios.ICANON | termios.ECHO)
            new[6][termios.VMIN] = 0
            new[6][termios.VTIME] = 0
            termios.tcsetattr(fd_in, termios.TCSANOW, new)
            try:
                stdout.flush()
            except (OSError, ValueError):
                pass
            os.write(fd_out, b"\x1b[c")
            buffer = b""
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                ready, _, _ = select.select([fd_in], [], [], remaining)
                if not ready:
                    break
                chunk = os.read(fd_in, 256)
                if not chunk:
                    break
                buffer += chunk
                match = re.search(rb"\x1b\[\?([\d;]*)c", buffer)
                if match:
                    result = {int(part) for part in match.group(1).split(b";") if part}
                    break
        finally:
            termios.tcsetattr(fd_in, termios.TCSANOW, old)
    except (ImportError, OSError, ValueError, AttributeError):
        result = None
    _DA1_CACHE[key] = result
    return result


def detect(env=None, stream=None, query=True, da1=None):
    """(protocol, reason). `query` allows a DA1 round trip for the terminals
    whose sixel support depends on version or settings; `da1` replaces that
    round trip (tests)."""
    env = os.environ if env is None else env
    stream = stream or sys.stdout
    forced = _normalize(env.get("COLI_IMAGE_PROTOCOL"))
    if forced:
        return forced, "COLI_IMAGE_PROTOCOL"
    try:
        interactive = stream.isatty()
    except (AttributeError, ValueError):
        interactive = False
    if not interactive:
        return "none", "output is not a terminal"
    term = env.get("TERM", "").lower()
    program = env.get("TERM_PROGRAM", "").lower()
    if term == "dumb":
        return "none", "TERM=dumb"
    if env.get("TMUX") or term.startswith(("screen", "tmux")):
        # tmux forwards graphics only with allow-passthrough (and sixel only
        # when built for it): half blocks are the one thing that always shows.
        return "blocks", "inside tmux or screen; COLI_IMAGE_PROTOCOL forces graphics"
    if env.get("KITTY_WINDOW_ID") or term == "xterm-kitty":
        return "kitty", "kitty"
    if program == "ghostty" or term == "xterm-ghostty" or env.get("GHOSTTY_RESOURCES_DIR"):
        return "kitty", "Ghostty"
    if program == "iterm.app" or env.get("LC_TERMINAL") == "iTerm2":
        return "iterm", "iTerm2"
    if program == "wezterm" or env.get("WEZTERM_EXECUTABLE") or env.get("WEZTERM_PANE"):
        # WezTerm speaks kitty graphics only when configured to; the iTerm2
        # protocol is always on.
        return "iterm", "WezTerm"

    def sixel():
        attributes = da1 if da1 is not None else (da1_attributes() if query else None)
        return attributes is not None and 4 in attributes

    if program == "vscode":
        # xterm.js answers DA1 with sixel only when terminal images are enabled,
        # and the same setting turns on its iTerm2 decoder.
        if sixel():
            return "iterm", "VS Code terminal with images enabled"
        return "blocks", ("VS Code terminal; terminal.integrated.enableImages gives "
                          "full-resolution images")
    candidate = (env.get("WT_SESSION") or term.startswith(("foot", "mlterm", "xterm", "contour",
                                                           "mintty"))
                 or program in ("mintty", "contour") or env.get("KONSOLE_VERSION"))
    if candidate and sixel():
        return "sixel", ("Windows Terminal (sixel)" if env.get("WT_SESSION")
                         else f"{term or program} reports sixel")
    if env.get("WT_SESSION"):
        return "blocks", "Windows Terminal did not report sixel; 1.22 or newer has it"
    return "blocks", "no graphics protocol detected"


def truecolor(env=None):
    """24-bit colour is the norm; Apple's Terminal.app is the notable holdout."""
    env = os.environ if env is None else env
    if env.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return True
    return env.get("TERM_PROGRAM") != "Apple_Terminal"


def terminal_geometry(stream=None):
    """(columns, rows, cell_width_px, cell_height_px); pixels are guesses when
    the terminal does not report them (8x16 is the safe side: an image sized
    for small cells never overflows the screen)."""
    stream = stream or sys.stdout
    size = shutil.get_terminal_size((80, 24))
    columns, rows = size.columns, size.lines
    cell_w, cell_h = 8, 16
    try:
        import fcntl
        import struct
        import termios
        raw = fcntl.ioctl(stream.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
        ws_rows, ws_cols, xpix, ypix = struct.unpack("HHHH", raw)
        if ws_cols and ws_rows:
            columns, rows = ws_cols, ws_rows
            if xpix and ypix:
                cell_w, cell_h = max(1, xpix // ws_cols), max(1, ypix // ws_rows)
    except (ImportError, OSError, ValueError, AttributeError):
        pass
    return columns, rows, cell_w, cell_h


# ---------------------------------------------------------------- pixels
def _lanes(row):
    """A row of bytes as one integer with a 16-bit lane per byte, so that up to
    255 rows add up without one byte's carry reaching its neighbour."""
    wide = bytearray(len(row) * 2)
    wide[1::2] = row
    return int.from_bytes(wide, "big")


def _unlanes(value, count):
    values = array("H", value.to_bytes(count * 2, "big"))
    if sys.byteorder == "little":
        values.byteswap()
    return values


def resize(pixels, width, height, channels, new_width, new_height):
    """Box-filter resample, 8-bit pixels, any channel count.

    Rows of each vertical span are summed as big integers (one add per row,
    not per byte); columns are gathered with itemgetter, a C loop, at up to
    eight evenly spaced samples per span. Good enough for a terminal and fast
    enough to run on every preview frame."""
    if (new_width, new_height) == (width, height):
        return bytes(pixels)
    new_width, new_height = max(1, new_width), max(1, new_height)
    stride = width * channels
    count = stride
    out = bytearray(new_width * new_height * channels)
    # Column sampling plan, shared by every row.
    samples = max(1, min(8, -(-width // new_width)))
    gathers = []
    for s in range(samples):
        index = []
        for x in range(new_width):
            x0 = x * width // new_width
            x1 = max(x0 + 1, (x + 1) * width // new_width)
            source = min(width - 1, x0 + (x1 - x0) * (2 * s + 1) // (2 * samples))
            base = source * channels
            index.extend(range(base, base + channels))
        gathers.append(itemgetter(*index) if len(index) > 1 else (lambda seq, i=index[0]: (seq[i],)))
    for y in range(new_height):
        y0 = y * height // new_height
        y1 = max(y0 + 1, (y + 1) * height // new_height)
        span = list(range(y0, y1))
        if len(span) > 255:
            step = len(span) / 255
            span = [y0 + int(i * step) for i in range(255)]
        total = 0
        for source in span:
            total += _lanes(pixels[source * stride:(source + 1) * stride])
        sums = _unlanes(total, count)
        divisor = len(span) * samples
        if samples == 1:
            gathered = gathers[0](sums)
        else:
            gathered = map(sum, zip(*(gather(sums) for gather in gathers)))
        if divisor == 1:
            row = bytes(gathered)
        else:
            half = divisor // 2
            row = bytes((value + half) // divisor for value in gathered)
        out[y * new_width * channels:(y + 1) * new_width * channels] = row
    return bytes(out)


def upscale_smooth(pixels, width, height, channels, new_width, new_height):
    """Bilinear enlargement. The engine's previews carry one pixel per latent
    token (48x32 for a 768x512 picture): enlarged with resize() they come out as
    visible squares, and a preview is meant to look like a picture forming, as
    the web UI shows it (smoothly scaled and blurred), not like a mosaic."""
    new_width, new_height = max(1, new_width), max(1, new_height)
    out = bytearray(new_width * new_height * channels)
    columns = []
    for x in range(new_width):
        fx = min(max((x + 0.5) * width / new_width - 0.5, 0.0), width - 1.0)
        x0 = int(fx)
        x1 = min(width - 1, x0 + 1)
        columns.append((x0 * channels, x1 * channels, fx - x0))
    o = 0
    for y in range(new_height):
        fy = min(max((y + 0.5) * height / new_height - 0.5, 0.0), height - 1.0)
        y0 = int(fy)
        u = fy - y0
        r0 = y0 * width * channels
        r1 = min(height - 1, y0 + 1) * width * channels
        for a0, a1, t in columns:
            for c in range(channels):
                top = pixels[r0 + a0 + c] + (pixels[r0 + a1 + c] - pixels[r0 + a0 + c]) * t
                bottom = pixels[r1 + a0 + c] + (pixels[r1 + a1 + c] - pixels[r1 + a0 + c]) * t
                out[o] = int(top + (bottom - top) * u + 0.5)
                o += 1
    return bytes(out)


def flatten(pixels, width, height, channels, checker=8):
    """RGB for renderers that have no alpha: transparent areas over a grey
    checkerboard, the convention that reads as "transparent" at a glance."""
    if channels == 3:
        return bytes(pixels)
    count = width * height
    rgb = bytearray(count * 3)
    if channels == 4:
        alpha = pixels[3::4]
        rgb[0::3] = pixels[0::4]
        rgb[1::3] = pixels[1::4]
        rgb[2::3] = pixels[2::4]
        if alpha.count(255) == count:
            return bytes(rgb)
        for i, a in enumerate(alpha):
            if a == 255:
                continue
            x, y = i % width, i // width
            background = 204 if (x // checker + y // checker) % 2 else 153
            for c in range(3):
                v = rgb[i * 3 + c]
                rgb[i * 3 + c] = (v * a + background * (255 - a) + 127) // 255
        return bytes(rgb)
    if channels in (1, 2):
        grey = pixels[0::channels]
        rgb[0::3] = grey
        rgb[1::3] = grey
        rgb[2::3] = grey
        if channels == 2:
            return flatten(bytes(_interleave_alpha(rgb, pixels[1::2])), width, height, 4, checker)
        return bytes(rgb)
    raise ValueError(f"unsupported channel count {channels}")


def _interleave_alpha(rgb, alpha):
    out = bytearray(len(alpha) * 4)
    out[0::4] = rgb[0::3]
    out[1::4] = rgb[1::3]
    out[2::4] = rgb[2::3]
    out[3::4] = alpha
    return out


def fit(width, height, max_width, max_height):
    """The largest size within the box with the image's aspect ratio, never
    enlarged (a small image stays sharp rather than blocky)."""
    scale = min(1.0, max_width / width, max_height / height)
    return max(1, int(width * scale)), max(1, int(height * scale))


# ---------------------------------------------------------------- renderers
def render_blocks(pixels, width, height, channels, max_columns, max_rows=None, indent="  ",
                  colour24=True):
    """Two pixels per cell: the upper half block painted with the top pixel as
    foreground and the bottom one as background. Returns the text, one line
    per cell row, each ending in a reset."""
    max_rows = max_rows or 10**6
    new_width, new_height = fit(width, height, max_columns, max_rows * 2)
    small = resize(pixels, width, height, channels, new_width, new_height)
    rgb = flatten(small, new_width, new_height, channels)
    if new_height % 2:
        rgb += rgb[-new_width * 3:]                    # repeat the last row
        new_height += 1
    lines = []
    stride = new_width * 3
    for y in range(0, new_height, 2):
        top = rgb[y * stride:(y + 1) * stride]
        bottom = rgb[(y + 1) * stride:(y + 2) * stride]
        parts = [indent]
        previous = None
        for x in range(new_width):
            fg = top[x * 3:x * 3 + 3]
            bg = bottom[x * 3:x * 3 + 3]
            if (fg, bg) != previous:
                parts.append(_sgr(fg, bg, colour24))
                previous = (fg, bg)
            parts.append(_UPPER_HALF)
        parts.append("\x1b[0m")
        lines.append("".join(parts))
    return "\n".join(lines) + "\n"


def _sgr(fg, bg, colour24):
    if colour24:
        return f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]};48;2;{bg[0]};{bg[1]};{bg[2]}m"
    return f"\x1b[38;5;{_xterm256(fg)};48;5;{_xterm256(bg)}m"


def _xterm256(rgb):
    r, g, b = (int(round(v / 255 * 5)) for v in rgb)
    return 16 + 36 * r + 6 * g + b


def _chunks(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _tmux_wrap(sequence, env):
    """With the protocol forced inside tmux, hand the sequence through
    (needs `set -g allow-passthrough on`)."""
    if not env.get("TMUX"):
        return sequence
    return "\x1bPtmux;" + sequence.replace("\x1b", "\x1b\x1b") + "\x1b\\"


def render_kitty(png, columns, rows=None, indent="  ", env=None):
    """kitty graphics: transmit-and-display a PNG (f=100) in 4096-byte base64
    chunks, quiet (q=2) so the terminal writes no acknowledgement into stdin,
    scaled by the terminal to `columns` cells (the height follows the aspect)."""
    env = os.environ if env is None else env
    data = base64.b64encode(png).decode("ascii")
    pieces = _chunks(data, 4096)
    out = [indent]
    for i, piece in enumerate(pieces):
        more = 1 if i < len(pieces) - 1 else 0
        if i == 0:
            keys = f"a=T,f=100,q=2,c={int(columns)}" + (f",r={int(rows)}" if rows else "")
            out.append(_tmux_wrap(f"\x1b_G{keys},m={more};{piece}\x1b\\", env))
        else:
            out.append(_tmux_wrap(f"\x1b_Gm={more};{piece}\x1b\\", env))
    out.append("\n")
    return "".join(out)


def render_iterm(png, columns, indent="  ", name="colibri.png", env=None):
    """iTerm2 inline image (OSC 1337 File=), width in cells, aspect kept."""
    env = os.environ if env is None else env
    label = base64.b64encode(name.encode("utf-8")).decode("ascii")
    data = base64.b64encode(png).decode("ascii")
    sequence = (f"\x1b]1337;File=name={label};inline=1;size={len(png)};width={int(columns)};"
                f"preserveAspectRatio=1:{data}\x07")
    return indent + _tmux_wrap(sequence, env) + "\n"


_BAYER4 = (0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5)
_SIXEL_CHARS = bytes(range(63, 127))
_SIXEL_RUN = re.compile(r"(.)\1{3,}")


def _run_length(match):
    run = match.group(0)
    return f"!{len(run)}{run[0]}"


def _median_cut(histogram, colours):
    """Up to `colours` boxes over a {15-bit colour: count} histogram, cut where
    the pixels are: each time the box with the widest spread (weighted by how
    many pixels it holds) is split at the weighted median of its widest axis.
    Returns (palette of 8-bit RGB tuples, {15-bit colour: palette index})."""
    def spread(box):
        best = None
        for shift in (10, 5, 0):
            values = [(key >> shift) & 31 for key, _count in box]
            extent = max(values) - min(values)
            if best is None or extent > best[0]:
                best = (extent, shift)
        return best

    boxes = [list(histogram.items())]
    while len(boxes) < colours:
        scored = []
        for index, box in enumerate(boxes):
            if len(box) < 2:
                continue
            extent, shift = spread(box)
            if extent:
                scored.append((extent * sum(count for _key, count in box), index, shift))
        if not scored:
            break
        _score, index, shift = max(scored)
        box = sorted(boxes.pop(index), key=lambda item: (item[0] >> shift) & 31)
        half, running, cut = sum(count for _key, count in box) / 2, 0, 1
        for position, (_key, count) in enumerate(box):
            running += count
            if running >= half:
                cut = min(max(position + 1, 1), len(box) - 1)
                break
        boxes += [box[:cut], box[cut:]]
    palette, lookup = [], {}
    for index, box in enumerate(boxes):
        total = sum(count for _key, count in box) or 1
        channel = []
        for shift in (10, 5, 0):
            mean = sum((((key >> shift) & 31) * 8 + 4) * count for key, count in box) / total
            channel.append(int(mean + 0.5))
        palette.append(tuple(channel))
        for key, _count in box:
            lookup[key] = index
    return palette, lookup


def render_sixel(pixels, width, height, channels, max_width_px, max_height_px=None,
                 indent="  "):
    """DEC sixel with a palette chosen from the picture itself.

    A fixed palette (the first version: 6x7x6 colours and a 4x4 ordered dither)
    showed as grain all over a photograph. Here the 255 colours come from a
    median cut of the picture's own 15-bit histogram, and each pixel takes the
    colour of the box its 15-bit value fell in, so no nearest-colour search runs
    per pixel. A light 4x4 ordered offset before the 15-bit truncation keeps the
    smooth gradients of generated images from banding. Bands of six rows are
    encoded colour by colour with run-length compression."""
    max_height_px = max_height_px or 10**6
    new_width, new_height = fit(width, height, max_width_px, max_height_px)
    small = resize(pixels, width, height, channels, new_width, new_height)
    rgb = flatten(small, new_width, new_height, channels)
    keys = [0] * (new_width * new_height)
    histogram = {}
    for y in range(new_height):
        row_base = y * new_width
        bayer_row = (y & 3) * 4
        for x in range(new_width):
            offset = _BAYER4[bayer_row + (x & 3)] >> 1          # 0..7, under one 5-bit step
            p = (row_base + x) * 3
            key = (min(255, rgb[p] + offset) >> 3) << 10 | (min(255, rgb[p + 1] + offset) >> 3) << 5 \
                | (min(255, rgb[p + 2] + offset) >> 3)
            keys[row_base + x] = key
            histogram[key] = histogram.get(key, 0) + 1
    palette, lookup = _median_cut(histogram, 255)
    index = bytes(lookup[key] for key in keys)
    out = [indent, "\x1bPq", f'"1;1;{new_width};{new_height}']
    for colour in sorted(set(index)):
        r, g, b = palette[colour]
        out.append(f"#{colour};2;{r * 100 // 255};{g * 100 // 255};{b * 100 // 255}")
    translate = bytes.maketrans(bytes(range(64)), _SIXEL_CHARS)
    for band in range(0, new_height, 6):
        masks = {}
        for bit in range(6):
            y = band + bit
            if y >= new_height:
                break
            row = index[y * new_width:(y + 1) * new_width]
            value = 1 << bit
            for x, colour in enumerate(row):
                mask = masks.get(colour)
                if mask is None:
                    mask = masks[colour] = bytearray(new_width)
                mask[x] |= value
        first = True
        for colour, mask in masks.items():
            if not first:
                out.append("$")                      # back to the band's first column
            first = False
            text = bytes(mask).translate(translate).decode("ascii")
            out.append(f"#{colour}" + _SIXEL_RUN.sub(_run_length, text))
        out.append("-")                              # next band
    out.append("\x1b\\\n")
    return "".join(out)


def render(pixels, width, height, channels, protocol, png=None, env=None, stream=None,
           indent="  ", max_columns=None, max_rows=None):
    """The text that draws the image with `protocol`, sized to the terminal.
    `png` is the same image as PNG (kitty and iTerm2 send it as is); it is
    encoded here when not given."""
    env = os.environ if env is None else env
    columns, rows, cell_w, cell_h = terminal_geometry(stream)
    columns = max(8, (max_columns or columns) - len(indent) - 1)
    rows = max(4, (max_rows or rows) - 3)
    if protocol == "none":
        return ""
    if protocol == "blocks":
        return render_blocks(pixels, width, height, channels, columns, rows, indent,
                             truecolor(env))
    if protocol == "sixel":
        limit = int(env.get("COLI_IMAGE_MAX_PX", "0") or 0) or 1024
        return render_sixel(pixels, width, height, channels,
                            min(limit, columns * cell_w), rows * cell_h, indent)
    if png is None:
        from image_engine import encode_png
        png = encode_png(width, height, pixels, channels)
    # Cells the picture needs at its own resolution, capped by the window.
    want_cols, want_rows = fit(width, height, columns * cell_w, rows * cell_h)
    cells = max(1, min(columns, -(-want_cols // cell_w)))
    if protocol == "kitty":
        return render_kitty(png, cells, None, indent, env)
    if protocol == "iterm":
        return render_iterm(png, cells, indent, env=env)
    raise ValueError(f"unknown image protocol {protocol!r}")
