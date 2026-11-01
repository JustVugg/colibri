/* colibri-tui: colibri's terminal interface (docs/tui.md), a client of `coli serve`.
 *
 * One C file and no libraries. The screen is a grid of cells in 24-bit colour; each frame only
 * the cells that changed reach the terminal, inside one synchronized update. The keyboard is read
 * raw: VT sequences on POSIX terminals, console input records on Windows (Windows Terminal and the
 * classic console; VT output needs Windows 10 or later). It talks to the server the way the chat
 * page does: /v1/chat/completions streamed, /health and /profile for the monitor, /experts for the
 * expert map. The mascot is the chat page's, drawn as a sixel picture where the terminal draws them.
 * With no server answering it plays a demo: the same screens, with made-up answers and numbers.
 *
 *   make colibri-tui            (in c/, on Linux, macOS and Windows/MSYS2 UCRT64)
 *   coli tui [--model DIR]      attaches to a running coli serve, or starts one
 */
#ifndef _WIN32
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE
#endif
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
#ifndef DISABLE_NEWLINE_AUTO_RETURN
#define DISABLE_NEWLINE_AUTO_RETURN 0x0008
#endif
#else
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

#include "tui_mascot.h"

#ifndef COLI_VERSION
#define COLI_VERSION "dev"
#endif

/* ---------------------------------------------------------------- basics */

static int IT = 1;                       /* the interface speaks Italian or English */
#define T(it, en) (IT ? (it) : (en))

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) { fputs("colibri-tui: out of memory\n", stderr); exit(1); }
    return q;
}
static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    return memcpy(xrealloc(NULL, n), s, n);
}

static double now_s(void) {
#ifdef _WIN32
    return (double)GetTickCount64() / 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

/* formatted strings for drawing: the last 128 stay alive (a monitor card builds thirty at once) */
static char *fmt(const char *f, ...) {
    static char ring[128][256];
    static unsigned k;
    char *b = ring[k++ & 127];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof ring[0], f, ap);
    va_end(ap);
    return b;
}
/* a decimal the way the interface's language writes it: 45,7 or 45.7 */
static const char *dec(double v, int places) {
    char *s = fmt("%.*f", places, v), *p = strchr(s, '.');
    if (p && IT) *p = ',';
    return s;
}
/* a context size: 6k, 256k (binary thousands, as context windows are named) */
static const char *ksize(long v) {
    if (v < 1024) return fmt("%ld", v);
    return fmt("%ldk", (v + 512) / 1024);
}

static uint32_t rng_state = 0x9e3779b9u;
static uint32_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/* ---------------------------------------------------------------- UTF-8 */

static int u8dec(const unsigned char *b, int n, uint32_t *cp) {
    int l = b[0] < 0x80 ? 1 : (b[0] >> 5) == 6 ? 2 : (b[0] >> 4) == 14 ? 3 : (b[0] >> 3) == 30 ? 4 : 0;
    if (!l) { *cp = 0xFFFD; return 1; }
    if (n < l) return 0;
    uint32_t c = l == 1 ? b[0] : l == 2 ? b[0] & 0x1fu : l == 3 ? b[0] & 0x0fu : b[0] & 0x07u;
    for (int i = 1; i < l; i++) {
        if ((b[i] & 0xc0) != 0x80) { *cp = 0xFFFD; return 1; }
        c = (c << 6) | (b[i] & 0x3fu);
    }
    *cp = c;
    return l;
}
/* the next character of a NUL-terminated string (it never reads past the NUL) */
static uint32_t u8next(const char **s) {
    uint32_t c;
    *s += u8dec((const unsigned char *)*s, 4, &c);
    return c;
}
static int u8enc(uint32_t c, char *t) {
    if (c < 0x80) { t[0] = (char)c; return 1; }
    if (c < 0x800) { t[0] = (char)(0xc0 | c >> 6); t[1] = (char)(0x80 | (c & 0x3f)); return 2; }
    if (c < 0x10000) {
        t[0] = (char)(0xe0 | c >> 12); t[1] = (char)(0x80 | ((c >> 6) & 0x3f));
        t[2] = (char)(0x80 | (c & 0x3f));
        return 3;
    }
    t[0] = (char)(0xf0 | c >> 18); t[1] = (char)(0x80 | ((c >> 12) & 0x3f));
    t[2] = (char)(0x80 | ((c >> 6) & 0x3f)); t[3] = (char)(0x80 | (c & 0x3f));
    return 4;
}

/* columns a character takes: 0 for marks and joiners, 2 for East Asian wide
   characters and emoji, 1 for the rest */
static int wcw(uint32_t c) {
    if (c < 0x20 || (c >= 0x7f && c < 0xa0)) return 0;
    if ((c >= 0x300 && c <= 0x36f) || (c >= 0x1ab0 && c <= 0x1aff) || (c >= 0x1dc0 && c <= 0x1dff) ||
        (c >= 0x200b && c <= 0x200f) || c == 0x2060 || (c >= 0x20d0 && c <= 0x20ff) ||
        (c >= 0xfe00 && c <= 0xfe0f) || (c >= 0xfe20 && c <= 0xfe2f) || (c >= 0xe0100 && c <= 0xe01ef))
        return 0;
    if ((c >= 0x1100 && c <= 0x115f) || (c >= 0x2e80 && c <= 0x303e) || (c >= 0x3041 && c <= 0x33ff) ||
        (c >= 0x3400 && c <= 0x4dbf) || (c >= 0x4e00 && c <= 0x9fff) || (c >= 0xa000 && c <= 0xa4cf) ||
        (c >= 0xac00 && c <= 0xd7a3) || (c >= 0xf900 && c <= 0xfaff) || (c >= 0xfe30 && c <= 0xfe4f) ||
        (c >= 0xff00 && c <= 0xff60) || (c >= 0xffe0 && c <= 0xffe6) || (c >= 0x1f300 && c <= 0x1f64f) ||
        (c >= 0x1f680 && c <= 0x1f6ff) || (c >= 0x1f900 && c <= 0x1f9ff) || (c >= 0x20000 && c <= 0x3fffd))
        return 2;
    return 1;
}
static int strw(const char *s) {
    int w = 0;
    while (*s) w += wcw(u8next(&s));
    return w;
}

/* ---------------------------------------------------------------- terminal */

static int term_on, g_eof;
#ifdef _WIN32
static HANDLE hin, hout;
static DWORD in_mode0, out_mode0;
static UINT cp0;
static int win_top;
#else
static struct termios tio0;
#endif

static void raw_write(const char *s, size_t n) {
#ifdef _WIN32
    DWORD w;
    while (n) {
        if (!WriteFile(hout, s, (DWORD)n, &w, NULL) || !w) return;
        s += w;
        n -= w;
    }
#else
    while (n) {
        ssize_t w = write(1, s, n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return;
        s += w;
        n -= (size_t)w;
    }
#endif
}

static void term_restore(void) {
    static const char bye[] = "\x1b[?2026l\x1b[0m\x1b[?1006l\x1b[?1002l\x1b[?2004l\x1b[?7h"
                              "\x1b[0 q\x1b[?25h\x1b[?1049l";
    if (!term_on) return;
    term_on = 0;
    raw_write(bye, sizeof bye - 1);
#ifdef _WIN32
    SetConsoleMode(hin, in_mode0);
    SetConsoleMode(hout, out_mode0);
    SetConsoleOutputCP(cp0);
#else
    tcsetattr(0, TCSAFLUSH, &tio0);
#endif
}

#ifdef _WIN32
static BOOL WINAPI on_ctrl(DWORD t) {
    (void)t;
    term_restore();
    return FALSE;
}
#else
static void on_winch(int s) { (void)s; }
static void on_fatal(int s) {
    term_restore();
    signal(s, SIG_DFL);
    raise(s);
}
#endif

static int term_init(void) {
#ifdef _WIN32
    hin = GetStdHandle(STD_INPUT_HANDLE);
    hout = GetStdHandle(STD_OUTPUT_HANDLE);
    if (!GetConsoleMode(hin, &in_mode0) || !GetConsoleMode(hout, &out_mode0)) return -1;
    if (!SetConsoleMode(hout, out_mode0 | ENABLE_VIRTUAL_TERMINAL_PROCESSING | DISABLE_NEWLINE_AUTO_RETURN) &&
        !SetConsoleMode(hout, out_mode0 | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
        return -2;
    cp0 = GetConsoleOutputCP();
    SetConsoleOutputCP(CP_UTF8);
    /* no ENABLE_PROCESSED_INPUT: ctrl+c arrives as a key; no quick edit: the mouse reaches us */
    SetConsoleMode(hin, ENABLE_WINDOW_INPUT | ENABLE_MOUSE_INPUT | ENABLE_EXTENDED_FLAGS);
    SetConsoleCtrlHandler(on_ctrl, TRUE);
#else
    if (!isatty(0) || !isatty(1) || tcgetattr(0, &tio0)) return -1;
    struct termios t = tio0;
    t.c_iflag &= ~(tcflag_t)(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    t.c_oflag &= ~(tcflag_t)OPOST;
    t.c_cflag |= CS8;
    t.c_lflag &= ~(tcflag_t)(ECHO | ICANON | IEXTEN | ISIG);
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    if (tcsetattr(0, TCSAFLUSH, &t)) return -1;
    signal(SIGWINCH, on_winch); /* wakes poll() up: the next frame reads the new size */
    signal(SIGTERM, on_fatal);
    signal(SIGHUP, on_fatal);
    signal(SIGINT, on_fatal);
    signal(SIGQUIT, on_fatal);
#endif
    term_on = 1;
    /* alternate screen, no autowrap, bracketed paste, a blinking bar cursor, mouse (POSIX) */
    static const char hello[] = "\x1b[?1049h\x1b[?7l\x1b[?2004h\x1b[5 q\x1b[?25l"
#ifndef _WIN32
                                "\x1b[?1002h\x1b[?1006h"
#endif
        ;
    raw_write(hello, sizeof hello - 1);
    return 0;
}

static void term_size(int *w, int *h) {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO c;
    if (GetConsoleScreenBufferInfo(hout, &c)) {
        *w = c.srWindow.Right - c.srWindow.Left + 1;
        *h = c.srWindow.Bottom - c.srWindow.Top + 1;
        win_top = c.srWindow.Top;
        return;
    }
#else
    struct winsize ws;
    if (ioctl(1, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
        *w = ws.ws_col;
        *h = ws.ws_row;
        return;
    }
#endif
    *w = 100;
    *h = 30;
}

/* ---------------------------------------------------------------- JSON */

/* A small reader for the server's answers: one token for each object, array, string and
   primitive, with where it starts and ends and its parent. Values are read on demand. */
typedef struct {
    char type; /* 'o' 'a' 's' 'p' */
    int start, end, parent;
} JT;
static JT *jt;
static int njt, jtcap;
static const char *js;

static int jpush(char type, int start, int parent) {
    if (njt == jtcap) {
        jtcap = jtcap ? jtcap * 2 : 4096;
        jt = xrealloc(jt, sizeof(JT) * (size_t)jtcap);
    }
    jt[njt] = (JT){type, start, start, parent};
    return njt++;
}
/* the number of tokens, or -1 when the text is not JSON this reader follows */
static int json_parse(const char *s, int n) {
    int stack[64], sp = 0;
    js = s;
    njt = 0;
    for (int i = 0; i < n; i++) {
        char c = s[i];
        int parent = sp ? stack[sp - 1] : -1;
        if (c == '{' || c == '[') {
            if (sp == 64) return -1;
            stack[sp++] = jpush(c == '{' ? 'o' : 'a', i, parent);
        } else if (c == '}' || c == ']') {
            if (!sp) return -1;
            jt[stack[--sp]].end = i + 1;
        } else if (c == '"') {
            int start = ++i;
            while (i < n && s[i] != '"') i += s[i] == '\\' ? 2 : 1;
            if (i >= n) return -1;
            jt[jpush('s', start, parent)].end = i;
        } else if (c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != ',' && c != ':') {
            int start = i;
            while (i < n && !strchr(",]} \t\r\n", s[i])) i++;
            jt[jpush('p', start, parent)].end = i;
            i--;
        }
    }
    return sp ? -1 : njt;
}
/* the value of key in object o, or -1 */
static int jget(int o, const char *key) {
    if (o < 0 || o >= njt || jt[o].type != 'o') return -1;
    size_t kl = strlen(key);
    int expect = 0; /* 0: a key comes; 1: a value we skip; 2: the value we want */
    for (int k = o + 1; k < njt && jt[k].start < jt[o].end; k++) {
        if (jt[k].parent != o) continue;
        if (!expect) expect = jt[k].type == 's' && (size_t)(jt[k].end - jt[k].start) == kl && !memcmp(js + jt[k].start, key, kl) ? 2 : 1;
        else if (expect == 2) return k;
        else expect = 0;
    }
    return -1;
}
/* the i-th item of array a, or -1 */
static int jat(int a, int i) {
    if (a < 0 || a >= njt || jt[a].type != 'a') return -1;
    for (int k = a + 1; k < njt && jt[k].start < jt[a].end; k++)
        if (jt[k].parent == a && i-- == 0) return k;
    return -1;
}
static int jlen(int a) {
    int n = 0;
    if (a < 0 || a >= njt || jt[a].type != 'a') return 0;
    for (int k = a + 1; k < njt && jt[k].start < jt[a].end; k++) n += jt[k].parent == a;
    return n;
}
static double jnum(int k, double def) {
    if (k < 0 || jt[k].type != 'p') return def;
    char b[64];
    int n = jt[k].end - jt[k].start;
    if (n <= 0 || n >= 63 || !(js[jt[k].start] == '-' || (js[jt[k].start] >= '0' && js[jt[k].start] <= '9'))) return def;
    memcpy(b, js + jt[k].start, (size_t)n);
    b[n] = 0;
    return strtod(b, NULL);
}
/* a string's text, escapes decoded, into out (at most cap - 1 bytes): its length */
static int jstr(int k, char *out, int cap) {
    int o = 0;
    if (k < 0 || jt[k].type != 's' || cap < 1) { if (cap) out[0] = 0; return 0; }
    for (int i = jt[k].start; i < jt[k].end && o < cap - 5; i++) {
        char c = js[i];
        if (c != '\\') { out[o++] = c; continue; }
        c = js[++i];
        switch (c) {
        case 'n': out[o++] = '\n'; break;
        case 't': out[o++] = '\t'; break;
        case 'r': break;
        case 'b': case 'f': break;
        case 'u': {
            unsigned cp = 0;
            for (int d = 1; d <= 4 && i + d < jt[k].end; d++) {
                char h = js[i + d];
                cp = cp * 16 + (unsigned)(h >= '0' && h <= '9' ? h - '0' : (h | 32) >= 'a' && (h | 32) <= 'f' ? (h | 32) - 'a' + 10 : 0);
            }
            i += 4;
            if (cp >= 0xd800 && cp <= 0xdbff && i + 6 < jt[k].end && js[i + 1] == '\\' && js[i + 2] == 'u') {
                unsigned lo = 0;
                for (int d = 3; d <= 6; d++) {
                    char h = js[i + d];
                    lo = lo * 16 + (unsigned)(h >= '0' && h <= '9' ? h - '0' : (h | 32) - 'a' + 10);
                }
                if (lo >= 0xdc00 && lo <= 0xdfff) { cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00); i += 6; }
            }
            o += u8enc(cp, out + o);
            break;
        }
        default: out[o++] = c;
        }
    }
    out[o] = 0;
    return o;
}
/* a string appended to a growing buffer, escaped for JSON */
static void json_esc(char **b, size_t *n, size_t *cap, const char *s) {
    for (; *s; s++) {
        if (*n + 8 > *cap) { *cap = (*cap + 8) * 2; *b = xrealloc(*b, *cap); }
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { (*b)[(*n)++] = '\\'; (*b)[(*n)++] = (char)c; }
        else if (c == '\n') { (*b)[(*n)++] = '\\'; (*b)[(*n)++] = 'n'; }
        else if (c == '\t') { (*b)[(*n)++] = '\\'; (*b)[(*n)++] = 't'; }
        else if (c < 0x20) *n += (size_t)snprintf(*b + *n, 8, "\\u%04x", c);
        else (*b)[(*n)++] = (char)c;
    }
}
static void buf_add(char **b, size_t *n, size_t *cap, const char *s) {
    size_t l = strlen(s);
    if (*n + l + 1 > *cap) { *cap = (*n + l + 1) * 2; *b = xrealloc(*b, *cap); }
    memcpy(*b + *n, s, l + 1);
    *n += l;
}

/* ---------------------------------------------------------------- HTTP to coli serve */

#ifdef _WIN32
typedef SOCKET sock_t;
#define BAD_SOCK INVALID_SOCKET
#define sock_close closesocket
#else
typedef int sock_t;
#define BAD_SOCK (-1)
#define sock_close close
#endif

static char g_host[128] = "127.0.0.1", g_port[16] = "8000";
static char g_key[256]; /* COLI_API_KEY or --api-key: sent as a bearer token */
static int g_connect_ms = 300; /* a port nobody listens on may hang instead of refusing (WSL does): never wait longer */
static int g_live; /* talking to a server rather than playing the demo */

/* one request: the headers, then the body as it arrives (SSE lines are read from it as they come) */
typedef struct {
    sock_t fd;
    int state; /* 0 idle, 1 headers, 2 body, 3 done, -1 failed */
    int status;
    char head[8192];
    size_t hn;
    char *buf;
    size_t n, cap, pos;
} Http;

static void http_close(Http *h) {
    if (h->fd != BAD_SOCK && h->fd != 0) sock_close(h->fd);
    h->fd = BAD_SOCK;
}
static int send_all(sock_t fd, const char *p, size_t n) {
    while (n) {
        int w = (int)send(fd, p, (int)n, 0);
        if (w <= 0) return -1;
        p += w;
        n -= (size_t)w;
    }
    return 0;
}
static int http_start(Http *h, const char *method, const char *path, const char *body) {
    http_close(h);
    h->state = -1;
    h->status = 0;
    h->hn = h->n = h->pos = 0;
    struct addrinfo hint, *ai = NULL;
    memset(&hint, 0, sizeof hint);
    hint.ai_family = AF_UNSPEC;
    hint.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(g_host, g_port, &hint, &ai) || !ai) return -1;
    sock_t fd = BAD_SOCK;
    for (struct addrinfo *a = ai; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd == BAD_SOCK) continue;
        /* a connect that waits at most g_connect_ms */
#ifdef _WIN32
        u_long nb = 1;
        ioctlsocket(fd, FIONBIO, &nb);
#else
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
#endif
        int ok = connect(fd, a->ai_addr, (int)a->ai_addrlen) == 0;
        if (!ok) {
            fd_set w;
            FD_ZERO(&w);
            FD_SET(fd, &w);
            struct timeval tv = {g_connect_ms / 1000, (g_connect_ms % 1000) * 1000};
            int err = 0;
            socklen_t el = sizeof err;
            ok = select((int)fd + 1, NULL, &w, NULL, &tv) == 1 && !getsockopt(fd, SOL_SOCKET, SO_ERROR, (char *)&err, &el) && !err;
        }
#ifdef _WIN32
        nb = 0;
        ioctlsocket(fd, FIONBIO, &nb);
#else
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) & ~O_NONBLOCK);
#endif
        if (ok) break;
        sock_close(fd);
        fd = BAD_SOCK;
    }
    freeaddrinfo(ai);
    if (fd == BAD_SOCK) return -1;
    size_t bl = body ? strlen(body) : 0;
    char hdr[900];
    int hl = snprintf(hdr, sizeof hdr, "%s %s HTTP/1.1\r\nHost: %s:%s\r\nConnection: close\r\nAccept: */*\r\n", method, path, g_host, g_port);
    if (body) hl += snprintf(hdr + hl, sizeof hdr - (size_t)hl, "Content-Type: application/json\r\nContent-Length: %lu\r\n", (unsigned long)bl);
    if (*g_key) hl += snprintf(hdr + hl, sizeof hdr - (size_t)hl, "Authorization: Bearer %.255s\r\n", g_key);
    hl += snprintf(hdr + hl, sizeof hdr - (size_t)hl, "\r\n");
    if (send_all(fd, hdr, (size_t)hl) || (bl && send_all(fd, body, bl))) {
        sock_close(fd);
        return -1;
    }
#ifdef _WIN32
    u_long on = 1;
    ioctlsocket(fd, FIONBIO, &on);
#else
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
#endif
    h->fd = fd;
    h->state = 1;
    return 0;
}
static void http_body_add(Http *h, const char *p, size_t k) {
    if (h->n + k + 1 > h->cap) {
        h->cap = (h->n + k + 1) * 2;
        h->buf = xrealloc(h->buf, h->cap);
    }
    memcpy(h->buf + h->n, p, k);
    h->n += k;
    h->buf[h->n] = 0;
}
/* reads what has arrived: 1 when something new came in */
static int http_pump(Http *h) {
    int got = 0;
    if (h->state != 1 && h->state != 2) return 0;
    char tmp[16384];
    for (;;) {
        int r = (int)recv(h->fd, tmp, (int)sizeof tmp, 0);
        if (r > 0) {
            got = 1;
            if (h->state == 2) { http_body_add(h, tmp, (size_t)r); continue; }
            size_t take = (size_t)r < sizeof h->head - 1 - h->hn ? (size_t)r : sizeof h->head - 1 - h->hn;
            memcpy(h->head + h->hn, tmp, take);
            h->hn += take;
            h->head[h->hn] = 0;
            char *e = strstr(h->head, "\r\n\r\n");
            if (!e) {
                if (h->hn >= sizeof h->head - 1) { h->state = -1; http_close(h); return got; }
                continue;
            }
            size_t hl = (size_t)(e + 4 - h->head), rest = h->hn - hl;
            const char *sp = strchr(h->head, ' ');
            h->status = sp ? atoi(sp + 1) : 0;
            h->state = 2;
            http_body_add(h, h->head + hl, rest);
            if (take < (size_t)r) http_body_add(h, tmp + take, (size_t)r - take);
            continue;
        }
        if (r == 0) {
            h->state = h->state == 2 ? 3 : -1;
            http_close(h);
            break;
        }
#ifdef _WIN32
        if (WSAGetLastError() == WSAEWOULDBLOCK) break;
#else
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) break;
#endif
        h->state = -1;
        http_close(h);
        break;
    }
    return got;
}
static void sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    struct timespec ts = {ms / 1000, (long)(ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
#endif
}
/* a GET that waits for its answer: 0 with a 200 and the body in h->buf */
static int http_get_sync(Http *h, const char *path, double timeout) {
    if (http_start(h, "GET", path, NULL)) return -1;
    double t0 = now_s();
    while (h->state == 1 || h->state == 2) {
        http_pump(h);
        if (now_s() - t0 > timeout) {
            http_close(h);
            h->state = -1;
            return -1;
        }
        if (h->state == 1 || h->state == 2) sleep_ms(5);
    }
    return h->state == 3 && h->status == 200 && h->buf ? 0 : -1;
}

static Http hc_chat, hc_health, hc_profile, hc_experts;

/* ---------------------------------------------------------------- what the terminal can draw */

static int g_sixel_cap, g_cell_w, g_cell_h;

/* asks for the cell size in pixels and the device attributes, reads the answers for half a second */
static void probe_terminal(void) {
    static const char q[] = "\x1b[16t\x1b[c";
    char buf[512];
    int n = 0;
    raw_write(q, sizeof q - 1);
    double t0 = now_s();
    while (now_s() - t0 < 0.5 && n < (int)sizeof buf - 1) {
#ifdef _WIN32
        if (WaitForSingleObject(hin, 50) != WAIT_OBJECT_0) continue;
        INPUT_RECORD r[64];
        DWORD k = 0;
        if (!ReadConsoleInputW(hin, r, 64, &k)) break;
        for (DWORD i = 0; i < k && n < (int)sizeof buf - 1; i++)
            if (r[i].EventType == KEY_EVENT && r[i].Event.KeyEvent.bKeyDown && r[i].Event.KeyEvent.uChar.UnicodeChar &&
                r[i].Event.KeyEvent.uChar.UnicodeChar < 0x80)
                buf[n++] = (char)r[i].Event.KeyEvent.uChar.UnicodeChar;
#else
        struct pollfd p = {0, POLLIN, 0};
        if (poll(&p, 1, 50) <= 0) continue;
        ssize_t r = read(0, buf + n, sizeof buf - 1 - (size_t)n);
        if (r > 0) n += (int)r;
#endif
        buf[n] = 0;
        const char *d = strstr(buf, "\x1b[?");
        if (d && strchr(d, 'c')) break;
    }
    buf[n] = 0;
    const char *d = strstr(buf, "\x1b[?");
    if (d) {
        for (const char *q2 = d + 3; *q2 && *q2 != 'c';) {
            int v = (int)strtol(q2, (char **)&q2, 10);
            if (v == 4) g_sixel_cap = 1;
            if (*q2 == ';') q2++;
            else if (*q2 != 'c') break;
        }
    }
    const char *c = strstr(buf, "\x1b[6;");
    int h = 0, w = 0;
    if (c && sscanf(c, "\x1b[6;%d;%dt", &h, &w) == 2 && h >= 6 && h <= 128 && w >= 3 && w <= 64) {
        g_cell_h = h;
        g_cell_w = w;
    }
}

/* ---------------------------------------------------------------- keys */

enum { K_NONE, K_CHAR, K_ENTER, K_NEWLINE, K_BACK, K_DEL, K_LEFT, K_RIGHT, K_UP, K_DOWN, K_HOME, K_END,
       K_PGUP, K_PGDN, K_TAB, K_BTAB, K_ESC, K_CTRL, K_FN, K_WHEELUP, K_WHEELDOWN, K_CLICK, K_DRAG, K_RELEASE };
enum { KM_SHIFT = 1, KM_ALT = 2, KM_CTRL = 4 };
typedef struct {
    int k, mod, x, y;
    uint32_t cp; /* the character, the letter of a ctrl key, the number of a function key */
} Key;

#ifdef _WIN32
static WCHAR hi_surrogate;
static int win_button;

static int term_keys(Key *out, int max, int ms) {
    if (WaitForSingleObject(hin, (DWORD)ms) != WAIT_OBJECT_0) return 0;
    INPUT_RECORD rec[128];
    DWORD nr = 0;
    if (!ReadConsoleInputW(hin, rec, 128, &nr)) return 0;
    int n = 0;
    for (DWORD i = 0; i < nr && n < max; i++) {
        Key k;
        memset(&k, 0, sizeof k);
        if (rec[i].EventType == MOUSE_EVENT) {
            MOUSE_EVENT_RECORD *m = &rec[i].Event.MouseEvent;
            k.x = m->dwMousePosition.X;
            k.y = m->dwMousePosition.Y - win_top;
            int left = (m->dwButtonState & FROM_LEFT_1ST_BUTTON_PRESSED) != 0;
            if (m->dwEventFlags == MOUSE_WHEELED) k.k = (SHORT)HIWORD(m->dwButtonState) > 0 ? K_WHEELUP : K_WHEELDOWN;
            else if (m->dwEventFlags == MOUSE_MOVED) { if (left) k.k = K_DRAG; }
            else if (left && !win_button) k.k = K_CLICK;
            else if (!left && win_button) k.k = K_RELEASE;
            if (m->dwEventFlags != MOUSE_WHEELED) win_button = left;
            if (k.k) out[n++] = k;
            continue;
        }
        if (rec[i].EventType != KEY_EVENT) continue;
        KEY_EVENT_RECORD *e = &rec[i].Event.KeyEvent;
        if (!e->bKeyDown) continue;
        if (e->uChar.UnicodeChar == 0x1b) { /* ESC [ in the same batch: a terminal's answer, not keys */
            DWORD j = i + 1;
            while (j < nr && (rec[j].EventType != KEY_EVENT || !rec[j].Event.KeyEvent.bKeyDown)) j++;
            if (j < nr && rec[j].Event.KeyEvent.uChar.UnicodeChar == '[') {
                for (j++; j < nr; j++) {
                    WCHAR c = rec[j].Event.KeyEvent.uChar.UnicodeChar;
                    if (rec[j].EventType == KEY_EVENT && rec[j].Event.KeyEvent.bKeyDown && c >= 0x40 && c <= 0x7e) break;
                }
                i = j;
                continue;
            }
        }
        DWORD cs = e->dwControlKeyState;
        int ctrl = (cs & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED)) != 0;
        int alt = (cs & (LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED)) != 0;
        /* AltGr is ctrl+alt: neither, so an Italian keyboard's @ # [ ] arrive as characters */
        k.mod = ((cs & SHIFT_PRESSED) ? KM_SHIFT : 0) | (alt && !ctrl ? KM_ALT : 0) | (ctrl && !alt ? KM_CTRL : 0);
        WCHAR ch = e->uChar.UnicodeChar;
        switch (e->wVirtualKeyCode) {
        case VK_RETURN: {
            /* more keys already queued behind it: a paste, so a line break and not "send" */
            int more = 0;
            for (DWORD j = i + 1; j < nr && !more; j++)
                more = rec[j].EventType == KEY_EVENT && rec[j].Event.KeyEvent.bKeyDown;
            if (more) { k.k = K_CHAR; k.cp = '\n'; }
            else k.k = (k.mod & KM_CTRL) ? K_NEWLINE : K_ENTER;
            break;
        }
        case VK_BACK: k.k = K_BACK; break;
        case VK_DELETE: k.k = K_DEL; break;
        case VK_LEFT: k.k = K_LEFT; break;
        case VK_RIGHT: k.k = K_RIGHT; break;
        case VK_UP: k.k = K_UP; break;
        case VK_DOWN: k.k = K_DOWN; break;
        case VK_HOME: k.k = K_HOME; break;
        case VK_END: k.k = K_END; break;
        case VK_PRIOR: k.k = K_PGUP; break;
        case VK_NEXT: k.k = K_PGDN; break;
        case VK_TAB: k.k = (k.mod & KM_SHIFT) ? K_BTAB : K_TAB; break;
        case VK_ESCAPE: k.k = K_ESC; break;
        default:
            if (e->wVirtualKeyCode >= VK_F1 && e->wVirtualKeyCode <= VK_F12) {
                k.k = K_FN;
                k.cp = (uint32_t)(e->wVirtualKeyCode - VK_F1 + 1);
                break;
            }
            if (!ch) continue;
            if (ch >= 0xd800 && ch <= 0xdbff) { hi_surrogate = ch; continue; }
            uint32_t cp = ch;
            if (ch >= 0xdc00 && ch <= 0xdfff) {
                if (!hi_surrogate) continue;
                cp = 0x10000 + ((uint32_t)(hi_surrogate - 0xd800) << 10) + (uint32_t)(ch - 0xdc00);
                hi_surrogate = 0;
            }
            if (cp < 0x20) {
                if (cp < 1 || cp > 26) continue;
                k.k = K_CTRL;
                k.cp = 'a' + cp - 1;
            } else if ((k.mod & KM_CTRL) && cp < 0x80 && isalpha((int)cp)) {
                k.k = K_CTRL;
                k.cp = (uint32_t)tolower((int)cp);
            } else {
                k.k = K_CHAR;
                k.cp = cp;
            }
        }
        int rep = e->wRepeatCount ? e->wRepeatCount : 1;
        while (rep-- && n < max) out[n++] = k;
    }
    return n;
}
#else
static unsigned char ib[8192];
static int ibn, in_paste;

static int csi_mod(int p) {
    int m = p > 1 ? p - 1 : 0, r = 0;
    if (m & 1) r |= KM_SHIFT;
    if (m & 2) r |= KM_ALT;
    if (m & 4) r |= KM_CTRL;
    return r;
}

/* one key from the front of ib: the bytes it used, or 0 while it is not complete
   (with force, an unfinished sequence counts as the key it starts with) */
static int parse_key(Key *k, int force) {
    const unsigned char *b = ib;
    int n = ibn;
    memset(k, 0, sizeof *k);
    if (b[0] == 0x1b) {
        if (n == 1) {
            if (!force) return 0;
            k->k = K_ESC;
            return 1;
        }
        if (b[1] == '[') {
            int i = 2;
            while (i < n && !(b[i] >= 0x40 && b[i] <= 0x7e)) i++;
            if (i >= n) {
                if (!force && n < 64) return 0;
                k->k = K_ESC;
                return 1;
            }
            int fin = b[i], lt = b[2] == '<', p[4] = {0, 0, 0, 0}, np = 0;
            for (int j = 2 + lt; j < i; j++) {
                if (b[j] >= '0' && b[j] <= '9') p[np] = p[np] * 10 + (b[j] - '0');
                else if (b[j] == ';' && np < 3) np++;
            }
            if (in_paste) {
                if (fin == '~' && p[0] == 201) in_paste = 0;
                return i + 1;
            }
            if (lt) { /* SGR mouse: wheel, left press and release, motion with the left button held */
                if (p[0] & 64) k->k = (p[0] & 1) ? K_WHEELDOWN : K_WHEELUP;
                else if ((p[0] & 3) == 0) k->k = (p[0] & 32) ? K_DRAG : fin == 'M' ? K_CLICK : K_RELEASE;
                k->x = p[1] - 1;
                k->y = p[2] - 1;
                return i + 1;
            }
            k->mod = csi_mod(p[1]);
            switch (fin) {
            case 'A': k->k = K_UP; break;
            case 'B': k->k = K_DOWN; break;
            case 'C': k->k = K_RIGHT; break;
            case 'D': k->k = K_LEFT; break;
            case 'H': k->k = K_HOME; break;
            case 'F': k->k = K_END; break;
            case 'Z': k->k = K_BTAB; break;
            case '~':
                switch (p[0]) {
                case 1: case 7: k->k = K_HOME; break;
                case 4: case 8: k->k = K_END; break;
                case 3: k->k = K_DEL; break;
                case 5: k->k = K_PGUP; break;
                case 6: k->k = K_PGDN; break;
                case 200: in_paste = 1; break;
                default: break;
                }
                break;
            default: break;
            }
            return i + 1;
        }
        if (b[1] == 'O') {
            if (n < 3) {
                if (!force) return 0;
                k->k = K_ESC;
                return 1;
            }
            switch (b[2]) {
            case 'A': k->k = K_UP; break;
            case 'B': k->k = K_DOWN; break;
            case 'C': k->k = K_RIGHT; break;
            case 'D': k->k = K_LEFT; break;
            case 'H': k->k = K_HOME; break;
            case 'F': k->k = K_END; break;
            case 'P': case 'Q': case 'R': case 'S': k->k = K_FN; k->cp = (uint32_t)(b[2] - 'P' + 1); break;
            default: break;
            }
            return 3;
        }
        return 1; /* alt + a key: the key follows on its own */
    }
    if (b[0] == '\r' || b[0] == '\n') {
        if (in_paste) {
            k->k = K_CHAR;
            k->cp = '\n';
            return b[0] == '\r' && n > 1 && b[1] == '\n' ? 2 : 1;
        }
        k->k = b[0] == '\r' ? K_ENTER : K_NEWLINE;
        return 1;
    }
    if (b[0] == '\t') {
        if (in_paste) { k->k = K_CHAR; k->cp = ' '; }
        else k->k = K_TAB;
        return 1;
    }
    if (b[0] == 0x7f || b[0] == 0x08) { k->k = K_BACK; return 1; }
    if (b[0] < 0x20) {
        if (b[0] >= 1 && b[0] <= 26 && !in_paste) { k->k = K_CTRL; k->cp = 'a' + b[0] - 1u; }
        return 1;
    }
    uint32_t c;
    int l = u8dec(b, n, &c);
    if (!l) {
        if (!force) return 0;
        l = 1;
        c = 0xFFFD;
    }
    k->k = K_CHAR;
    k->cp = c;
    return l;
}

static int term_keys(Key *out, int max, int ms) {
    struct pollfd p = {0, POLLIN, 0};
    if (ibn == 0 && poll(&p, 1, ms) <= 0) return 0;
    ssize_t r = read(0, ib + ibn, sizeof ib - (size_t)ibn);
    if (r > 0) ibn += (int)r;
    else if (r == 0 && ibn == 0) { g_eof = 1; return 0; }
    int n = 0;
    while (n < max && ibn > 0) {
        int used = parse_key(&out[n], 0);
        if (!used) {
            if (ibn < (int)sizeof ib && poll(&p, 1, 30) > 0 &&
                (r = read(0, ib + ibn, sizeof ib - (size_t)ibn)) > 0) {
                ibn += (int)r;
                continue;
            }
            used = parse_key(&out[n], 1);
        }
        memmove(ib, ib + used, (size_t)(ibn - used));
        ibn -= used;
        if (out[n].k != K_NONE) n++;
    }
    return n;
}
#endif

/* ---------------------------------------------------------------- the screen */

enum { A_BOLD = 1, A_DIM = 2, A_ITAL = 4 };
#define CONT 0xFFFFFFFFu /* the right half of a wide character */
typedef struct {
    uint32_t cp, fg, bg;
    uint8_t at;
} Cell;

static Cell *scr, *old;
static int W, H, full_redraw = 1;
static int cx0, cy0, cx1, cy1; /* drawing is clipped to this box */
static int cur_show, cur_x, cur_y;

static char *ob;
static size_t obn, obcap;
static void ob_put(const char *s, size_t n) {
    if (obn + n > obcap) {
        obcap = (obn + n) * 2 + 4096;
        ob = xrealloc(ob, obcap);
    }
    memcpy(ob + obn, s, n);
    obn += n;
}
static void ob_s(const char *s) { ob_put(s, strlen(s)); }

static void clip(int x0, int y0, int x1, int y1) {
    cx0 = x0 < 0 ? 0 : x0;
    cy0 = y0 < 0 ? 0 : y0;
    cx1 = x1 > W ? W : x1;
    cy1 = y1 > H ? H : y1;
}
static void clip_all(void) { clip(0, 0, W, H); }

static void put(int x, int y, uint32_t cp, uint32_t fg, uint32_t bg, int at) {
    int w = wcw(cp);
    if (!w || y < cy0 || y >= cy1 || x < cx0 || x >= cx1) return;
    if (w == 2 && x + 1 >= cx1) { cp = ' '; w = 1; }
    Cell *row = &scr[y * W];
    if (row[x].cp == CONT && x > 0) row[x - 1].cp = ' '; /* we cut a wide character in half */
    if (x + w < W && row[x + w].cp == CONT) row[x + w].cp = ' ';
    if (w == 2 && row[x + 1].cp != CONT && x + 2 < W && row[x + 2].cp == CONT) row[x + 2].cp = ' ';
    row[x] = (Cell){cp, fg, bg, (uint8_t)at};
    if (w == 2) row[x + 1] = (Cell){CONT, fg, bg, (uint8_t)at};
}
static void fill(int x, int y, int w, int h, uint32_t bg) {
    for (int j = y; j < y + h; j++)
        for (int i = x; i < x + w; i++) put(i, j, ' ', bg, bg, 0);
}
/* a string in at most maxw columns; when it does not fit it ends with an ellipsis */
static int text(int x, int y, int maxw, const char *s, uint32_t fg, uint32_t bg, int at) {
    if (maxw <= 0) return 0;
    int total = strw(s), lim = total > maxw ? maxw - 1 : maxw, col = 0;
    while (*s) {
        uint32_t c = u8next(&s);
        int cw = wcw(c);
        if (!cw) continue;
        if (col + cw > lim) break;
        put(x + col, y, c, fg, bg, at);
        col += cw;
    }
    if (total > maxw) put(x + col++, y, 0x2026, fg, bg, at);
    return col;
}
/* the same, right-aligned: it ends at column xr (exclusive) */
static int textr(int xr, int y, int maxw, const char *s, uint32_t fg, uint32_t bg, int at) {
    int w = strw(s);
    if (w > maxw) w = maxw;
    return text(xr - w, y, w, s, fg, bg, at);
}
static void hline(int x, int y, int w, uint32_t cp, uint32_t fg, uint32_t bg) {
    for (int i = 0; i < w; i++) put(x + i, y, cp, fg, bg, 0);
}

/* ---------------------------------------------------------------- the bird as a picture */

/* The mascot's palette (the chat page's COL) and the CSS pixels a cell: the sixel pixels of one
   cell, or 3 x 6 when the bird is drawn with half blocks. */
static int g_sixel, g_cw = 10, g_ch = 20; /* CSS pixels a cell */

static const char MPAL_KEYS[] = "DMOWTLRYG";
static const uint32_t MPAL[] = {0x00808f, 0xd43fce, 0xfd8c21, 0xf6fcfc, 0x00a8be, 0x72daeb, 0x6e1e3c, 0xf0a528, 0x969696};
static int mpal(char c) {
    const char *p = c ? strchr(MPAL_KEYS, c) : NULL;
    return p ? (int)(p - MPAL_KEYS) + 1 : 0;
}

/* the picture of this frame and the last one sent: the cell it starts in, its size in pixels,
   palette indices (0 is transparent) */
typedef struct {
    int on, c0, r0, w, h;
    uint8_t px[160 * 160];
} Img;
static Img img_new, img_old;

static int img_same(void) {
    if (img_new.on != img_old.on) return 0;
    if (!img_new.on) return 1;
    return img_new.c0 == img_old.c0 && img_new.r0 == img_old.r0 && img_new.w == img_old.w && img_new.h == img_old.h &&
           !memcmp(img_new.px, img_old.px, (size_t)img_new.w * (size_t)img_new.h);
}

static void ob_f(const char *f, ...) {
    char t[128];
    va_list ap;
    va_start(ap, f);
    int n = vsnprintf(t, sizeof t, f, ap);
    va_end(ap);
    if (n > 0) ob_put(t, (size_t)(n < (int)sizeof t ? n : (int)sizeof t - 1));
}

/* a sixel image with a transparent background: six rows a band, one pass a colour, runs compressed */
static void sixel_out(const Img *im) {
    ob_f("\x1b[%d;%dH\x1bP0;1;0q\"1;1;%d;%d", im->r0 + 1, im->c0 + 1, im->w, im->h);
    for (int i = 0; i < 9; i++)
        ob_f("#%d;2;%u;%u;%u", i + 1, (unsigned)((MPAL[i] >> 16) * 100 / 255), (unsigned)((MPAL[i] >> 8 & 255) * 100 / 255),
             (unsigned)((MPAL[i] & 255) * 100 / 255));
    for (int band = 0; band < im->h; band += 6) {
        for (int col = 1; col <= 9; col++) {
            int last = -1;
            for (int x = 0; x < im->w; x++)
                for (int i = 0; i < 6 && band + i < im->h; i++)
                    if (im->px[(band + i) * im->w + x] == col) last = x;
            if (last < 0) continue;
            ob_f("#%d", col);
            int run = 0, prev = -1;
            for (int x = 0; x <= last + 1; x++) {
                int bits = -1;
                if (x <= last) {
                    bits = 0;
                    for (int i = 0; i < 6 && band + i < im->h; i++)
                        if (im->px[(band + i) * im->w + x] == col) bits |= 1 << i;
                }
                if (bits == prev) { run++; continue; }
                if (run > 3) ob_f("!%d%c", run, 63 + prev);
                else for (int r = 0; r < run; r++) ob_put((const char[]){(char)(63 + prev)}, 1);
                prev = bits;
                run = 1;
            }
            ob_s("$");
        }
        ob_s("-");
    }
    ob_s("\x1b\\");
}

static void sgr(const Cell *c) {
    char t[96];
    int n = snprintf(t, sizeof t, "\x1b[0%s%s%s;38;2;%u;%u;%u;48;2;%u;%u;%um", c->at & A_BOLD ? ";1" : "",
                     c->at & A_DIM ? ";2" : "", c->at & A_ITAL ? ";3" : "", (unsigned)(c->fg >> 16),
                     (unsigned)(c->fg >> 8 & 255), (unsigned)(c->fg & 255), (unsigned)(c->bg >> 16),
                     (unsigned)(c->bg >> 8 & 255), (unsigned)(c->bg & 255));
    ob_put(t, (size_t)n);
}

/* send the cells that changed since the last frame, as one synchronized update */
static void present(const char *title) {
    char t[64];
    ob_s("\x1b[?2026h\x1b[?25l");
    if (title) {
        ob_s("\x1b]0;");
        ob_s(title);
        ob_s("\x07");
    }
    int was_full = full_redraw, changed = !img_same(), touched = 0;
    int ic1 = img_new.c0 + (img_new.w + g_cw - 1) / g_cw, ir1 = img_new.r0 + (img_new.h + g_ch - 1) / g_ch;
    if (changed && img_old.on) /* writing a cell erases the picture under it: rewrite the old one's cells */
        for (int y = img_old.r0; y <= img_old.r0 + (img_old.h + g_ch - 1) / g_ch; y++)
            for (int x = img_old.c0; x <= img_old.c0 + (img_old.w + g_cw - 1) / g_cw; x++)
                if (x >= 0 && x < W && y >= 0 && y < H) old[y * W + x].cp = 0xFFFFFFFEu;
    if (full_redraw) ob_s("\x1b[0m\x1b[2J");
    int px = -1, py = -1;
    Cell pen = {0, 0xffffffffu, 0xffffffffu, 0xff};
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const Cell *c = &scr[y * W + x], *o = &old[y * W + x];
            if (c->cp == CONT) continue;
            int wide = x + 1 < W && scr[y * W + x + 1].cp == CONT;
            if (!full_redraw && c->cp == o->cp && c->fg == o->fg && c->bg == o->bg && c->at == o->at &&
                (!wide || old[y * W + x + 1].cp == CONT))
                continue;
            if (px != x || py != y) ob_put(t, (size_t)snprintf(t, sizeof t, "\x1b[%d;%dH", y + 1, x + 1));
            if (c->fg != pen.fg || c->bg != pen.bg || c->at != pen.at) {
                sgr(c);
                pen = *c;
            }
            char u[4];
            ob_put(u, (size_t)u8enc(c->cp ? c->cp : ' ', u));
            px = x + (wide ? 2 : 1);
            py = y;
            if (img_new.on && x >= img_new.c0 && x < ic1 && y >= img_new.r0 && y < ir1) touched = 1;
        }
    if (img_new.on && (changed || touched || was_full)) sixel_out(&img_new);
    img_old.on = img_new.on;
    if (img_new.on) {
        img_old.c0 = img_new.c0;
        img_old.r0 = img_new.r0;
        img_old.w = img_new.w;
        img_old.h = img_new.h;
        memcpy(img_old.px, img_new.px, (size_t)img_new.w * (size_t)img_new.h);
    }
    if (cur_show) ob_put(t, (size_t)snprintf(t, sizeof t, "\x1b[%d;%dH\x1b[?25h", cur_y + 1, cur_x + 1));
    ob_s("\x1b[?2026l");
    raw_write(ob, obn);
    obn = 0;
    memcpy(old, scr, sizeof(Cell) * (size_t)W * (size_t)H);
    full_redraw = 0;
}

/* ---------------------------------------------------------------- colours */

enum {
    BG = 0x141414, SIDE = 0x1a1a1a, S2 = 0x222222, S3 = 0x2a2a2a, LINE = 0x333333, TRACK = 0x262626,
    INK = 0xececec, SOFT = 0xc8c8c8, MUTED = 0x8b8b8b, FAINT = 0x5c5c5c,
    ACC = 0x38c6c6, ACCT = 0x5fd7d7, PLUM = 0xe07ae0, WARN = 0xf8c96a, CODEBG = 0x0d0d0d, SELBG = 0x1d3535
};

static uint32_t mix(uint32_t a, uint32_t b, float t) {
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    uint32_t r = 0;
    for (int s = 0; s <= 16; s += 8) {
        float ca = (float)(a >> s & 255), cb = (float)(b >> s & 255);
        r |= (uint32_t)(ca + (cb - ca) * t + 0.5f) << s;
    }
    return r;
}

/* ---------------------------------------------------------------- the mascot */

/* The chat page's mascot (ui/index.html): the same poses, palette, blink, moves and timings.
   Everything is in the page's CSS pixels and milliseconds: the bird is a 60 x 60 box (20 x 20
   pixels of 3), it sits on the input box 30 pixels from its right end. Where the terminal draws
   sixel images (Windows Terminal, foot, WezTerm, xterm, mlterm...) a CSS pixel is a sixel pixel,
   so the bird is the page's bird; elsewhere it falls back to half blocks, a cell being one bird
   pixel wide and two tall (3 x 6 CSS pixels). */

#define BOX 60


typedef struct {
    int seq[4], n;
    double fps, t0;
} Anim;

/* an action is a short script of the page's steps */
enum { OP_ANIM, OP_WAIT, OP_HOP, OP_MOVE, OP_HOME, OP_LIFT, OP_Y, OP_FACE, OP_CLIP, OP_ZZ, OP_GOTO, OP_BACK };
typedef struct {
    int op, seq[4], n;
    double fps, a, b; /* a, b: the step's numbers (ms, pixels, pixels a second) */
} Step;
#define AN1(p) {OP_ANIM, {p, 0, 0, 0}, 1, 6, 0, 0}
#define AN2(p, q, f) {OP_ANIM, {p, q, 0, 0}, 2, f, 0, 0}
#define WAIT(ms) {OP_WAIT, {0, 0, 0, 0}, 0, 0, ms, 0}
#define HOP(h, ms) {OP_HOP, {0, 0, 0, 0}, 0, 0, h, ms}
#define MOVE(dx, speed, p, q, f) {OP_MOVE, {p, q, 0, 0}, 2, f, dx, speed}
#define HOME(speed) {OP_HOME, {P_WALK1, P_WALK2, 0, 0}, 2, 7, 0, speed}
#define LIFT(to, ms) {OP_LIFT, {0, 0, 0, 0}, 0, 0, to, ms}
#define YTO(to, ms) {OP_Y, {0, 0, 0, 0}, 0, 0, to, ms}
#define FACE(f) {OP_FACE, {0, 0, 0, 0}, 0, 0, f, 0}
#define CLIP(on) {OP_CLIP, {0, 0, 0, 0}, 0, 0, on, 0}
#define ZZ {OP_ZZ, {0, 0, 0, 0}, 0, 0, 0, 0}
#define GOTO(i) {OP_GOTO, {0, 0, 0, 0}, 0, 0, i, 0}
#define BACK(ms) {OP_BACK, {0, 0, 0, 0}, 0, 0, ms, 0}

static const Step ACT_FLAP[] = {AN2(P_FLAP1, P_STAND, 5), WAIT(1200)};
static const Step ACT_JUMP[] = {AN1(P_CROUCH), WAIT(110), AN1(P_FLAP1), HOP(18, 340), AN1(P_CROUCH), WAIT(90)};
static const Step ACT_CHEER[] = {AN2(P_FLAP1, P_FLAP2, 8), HOP(10, 280), HOP(7, 240)};
static const Step ACT_DANCE[] = {
    AN1(P_DANCE1), FACE(-1), HOP(4, 280), AN1(P_FLAP2), FACE(1), HOP(4, 280),
    AN1(P_DANCE2), FACE(-1), HOP(4, 280), AN1(P_FLAP1), FACE(1), HOP(8, 280),
    AN1(P_DANCE1), FACE(-1), HOP(4, 280), AN1(P_FLAP2), FACE(1), HOP(4, 280),
    AN1(P_DANCE2), FACE(-1), HOP(4, 280), AN1(P_FLAP1), FACE(1), HOP(8, 280), FACE(-1)};
static const Step ACT_SKATE[] = {MOVE(-160, 220, P_SKATE, P_SKATE, 1), WAIT(200), MOVE(0, 220, P_SKATE, P_SKATE, 1), FACE(-1)};
static const Step ACT_FLY[] = {AN2(P_FLY1, P_FLY2, 14), LIFT(-14, 300), WAIT(1200), LIFT(0, 260)};
static const Step ACT_LOOK[] = {AN1(P_SIT), FACE(1), WAIT(900), FACE(-1), WAIT(300)};
static const Step ACT_SWING[] = {AN2(P_SIT, P_SIT2, 3), WAIT(2400)};
static const Step ACT_LEAN[] = {AN1(P_LEAN), WAIT(3000)};
static const Step ACT_READ[] = {AN1(P_READ), WAIT(3000)};
static const Step ACT_TAP[] = {AN2(P_STAND, P_TAP, 4), WAIT(1600)};
static const Step ACT_WALK[] = {MOVE(-80, 60, P_WALK1, P_WALK2, 7), WAIT(400), HOME(90)};
static const Step ACT_HIDE[] = {CLIP(1), AN1(P_STAND), YTO(BOX, 300), WAIT(900), YTO(BOX * 0.55, 260), WAIT(800), YTO(0, 220), CLIP(0)};
static const Step ACT_SLEEP[] = {AN1(P_LIE), ZZ, WAIT(1800), GOTO(1)};
/* while the model works: it reads the prompt, leans while it reasons, flies while it writes */
static const Step WORK_READ[] = {AN1(P_READ), WAIT(-1)};
static const Step WORK_LEAN[] = {AN1(P_LEAN), WAIT(-1)};
static const Step WORK_FLY[] = {AN2(P_FLY1, P_FLY2, 14), LIFT(-8, 300), WAIT(-1)};
static const Step WORK_STOPPED[] = {AN1(P_LIE), WAIT(1800)};
static const Step DRAG_BACK[] = {AN2(P_FLY1, P_FLY2, 14), BACK(650), FACE(-1), AN1(P_CROUCH), WAIT(120)};

typedef struct {
    const char *it, *en;
    const Step *s;
    int n;
} Act;
#define ACT(it, en, arr) {it, en, arr, (int)(sizeof arr / sizeof arr[0])}
static const Act ACTS[] = {
    ACT("Sbatte l'ala", "Flaps a wing", ACT_FLAP), ACT("Salta", "Jumps", ACT_JUMP),
    ACT("Esulta", "Cheers", ACT_CHEER), ACT("Balla", "Dances", ACT_DANCE), ACT("Skate", "Skates", ACT_SKATE),
    ACT("Vola", "Flies", ACT_FLY), ACT("Si guarda intorno", "Looks around", ACT_LOOK),
    ACT("Dondola le zampe", "Swings its legs", ACT_SWING), ACT("Si appoggia", "Leans", ACT_LEAN),
    ACT("Legge", "Reads", ACT_READ), ACT("Batte il piede", "Taps its foot", ACT_TAP),
    ACT("Cammina", "Walks", ACT_WALK), ACT("Si nasconde", "Hides", ACT_HIDE), ACT("Dorme", "Sleeps", ACT_SLEEP),
};
#define NACTS ((int)(sizeof ACTS / sizeof ACTS[0]))
#define N_OF(arr) ((int)(sizeof arr / sizeof arr[0]))

enum { MS_IDLE, MS_BUSY, MS_WORK, MS_SLEEP, MS_DRAG };
static struct {
    double x, y, lift, ax; /* x from the input box's left edge; y down; lift up (negative); ax: x when the action began */
    int face, mode, clip, shown;
    Anim anim;
    double blink_at, next_idle, last_input, next_tick;
    const Step *steps;
    int nsteps, pc, started, prefix;
    double s0, fx, fy, flift, tx, dur; /* the running step */
    /* the mouse: a press on the bird, a drag, a click waiting for a second one */
    int down, dragging, click_pending;
    double ox, oy, dpx, dpy, drag_x, drag_y, click_at;
    /* where it was drawn: the cells it covers, for the mouse */
    int hc0, hr0, hc1, hr1;
    /* the input box in CSS pixels, from the last frame */
    double comp_l, comp_t, comp_w;
} M;

static struct {
    double t0, x, y;
} zz[8];

static double rnd01(void) { return (double)(rnd() % 1000000u) / 1e6; }

static void set_anim(const int *seq, int n, double fps, double now) {
    for (int i = 0; i < n && i < 4; i++) M.anim.seq[i] = seq[i];
    M.anim.n = n;
    M.anim.fps = fps;
    M.anim.t0 = now;
}
static void set_anim1(int p, double now) { set_anim(&p, 1, 6, now); }
static int cur_pose(double now) {
    int i = (int)((now - M.anim.t0) * M.anim.fps);
    if (i < 0) i = 0;
    return M.anim.seq[i % (M.anim.n ? M.anim.n : 1)];
}

static double m_home(void) { return M.comp_w - BOX - 30; }
static double m_clampx(double x) {
    double hi = M.comp_w - BOX;
    return x < 0 ? 0 : x > hi ? hi : x;
}

static void m_idle(double now) {
    M.mode = MS_IDLE;
    M.lift = M.y = 0;
    M.clip = 0;
    M.face = -1;
    M.steps = NULL;
    M.prefix = 0;
    set_anim1(P_SIT, now);
    M.next_idle = now + 20 + rnd01() * 20;
}
static void m_begin(int mode) {
    M.mode = mode;
    M.lift = M.y = 0;
    M.clip = 0;
    M.steps = NULL;
    M.prefix = 0;
}
/* the page's run(): from sitting it crouches first, then the script, then it sits again */
static void m_run(const Step *s, int n, int mode, double now) {
    if (!M.shown) return;
    int sat = M.mode == MS_IDLE && s != ACT_LOOK && s != ACT_SWING;
    m_begin(mode);
    M.steps = s;
    M.nsteps = n;
    M.pc = M.started = 0;
    M.ax = M.x;
    if (sat && mode != MS_SLEEP && mode != MS_DRAG) {
        set_anim1(P_CROUCH, now);
        M.prefix = 1;
        M.s0 = now;
    }
}
static void m_act(int i, double now) { m_run(ACTS[i].s, ACTS[i].n, ACTS[i].s == ACT_SLEEP ? MS_SLEEP : MS_BUSY, now); }
/* a script for the work in progress, without the crouch and without resetting the lift */
static void m_work(const Step *s, int n, double now) {
    (void)now;
    M.steps = s;
    M.nsteps = n;
    M.pc = M.started = 0;
}
static void m_wake(double now) {
    M.last_input = now;
    if (M.mode == MS_SLEEP) m_run(ACT_JUMP, N_OF(ACT_JUMP), MS_BUSY, now);
}

static void zz_spawn(double now) {
    for (int i = 0; i < 8; i++)
        if (now - zz[i].t0 > 2.4) {
            zz[i].t0 = now;
            zz[i].x = M.comp_l + M.x + BOX - 14;
            zz[i].y = M.comp_t - BOX * 0.55;
            return;
        }
}

static void m_step_start(const Step *s, double now) {
    M.s0 = now;
    M.fx = M.x;
    M.fy = M.y;
    M.flift = M.lift;
    M.dur = 0;
    if (s->op == OP_MOVE || s->op == OP_HOME) {
        M.tx = s->op == OP_MOVE ? m_clampx(M.ax + s->a) : m_home();
        double d = M.tx - M.x;
        if (d > 3 || d < -3) {
            M.face = d > 0 ? 1 : -1;
            set_anim(s->seq, s->n, s->fps, now);
            M.dur = (d < 0 ? -d : d) / s->b;
        }
    } else if (s->op == OP_BACK) {
        M.tx = m_home();
        M.face = M.tx > M.x ? 1 : -1;
    }
}
/* one step at this moment: 1 when it is over */
static int m_step_run(const Step *s, double now) {
    double el = now - M.s0, k;
    switch (s->op) {
    case OP_ANIM: set_anim(s->seq, s->n, s->fps, now); return 1;
    case OP_WAIT: return s->a >= 0 && el * 1000 >= s->a;
    case OP_HOP:
        k = el * 1000 / s->b;
        if (k >= 1) { M.y = 0; return 1; }
        M.y = -sin(k * 3.14159265358979) * s->a;
        return 0;
    case OP_MOVE:
    case OP_HOME:
        k = M.dur > 0 ? el / M.dur : 1;
        if (k > 1) k = 1;
        if (M.dur > 0) M.x = M.fx + (M.tx - M.fx) * k;
        if (k >= 1 && s->op == OP_HOME) M.face = -1;
        return k >= 1;
    case OP_LIFT:
        k = el * 1000 / s->b;
        if (k > 1) k = 1;
        M.lift = M.flift + (s->a - M.flift) * k;
        return k >= 1;
    case OP_Y:
        k = el * 1000 / s->b;
        if (k > 1) k = 1;
        M.y = M.fy + (s->a - M.fy) * k;
        return k >= 1;
    case OP_FACE: M.face = (int)s->a; return 1;
    case OP_CLIP: M.clip = (int)s->a; return 1;
    case OP_ZZ: zz_spawn(now); return 1;
    case OP_GOTO: M.pc = (int)s->a - 1; return 1;
    case OP_BACK: {
        k = el * 1000 / s->a;
        if (k > 1) k = 1;
        double e2 = 1 - (1 - k) * (1 - k) * (1 - k);
        M.x = M.fx + (M.tx - M.fx) * e2;
        M.y = M.fy * (1 - e2);
        if (k >= 1) M.y = 0;
        return k >= 1;
    }
    default: return 1;
    }
}

static void m_tick(double now) {
    if (!M.shown) return;
    if (M.prefix) {
        if (now - M.s0 < 0.09) return;
        M.prefix = 0;
    }
    while (M.steps && M.pc < M.nsteps) {
        const Step *s = &M.steps[M.pc];
        if (!M.started) {
            m_step_start(s, now);
            M.started = 1;
        }
        if (!m_step_run(s, now)) break;
        M.pc++;
        M.started = 0;
    }
    if (M.steps && M.pc >= M.nsteps) m_idle(now);
    if (now >= M.next_tick) { /* the page's one-second timer: rare and small, sleep after 90 s */
        M.next_tick = now + 1;
        if (M.mode == MS_IDLE && now - M.last_input > 90) m_act(NACTS - 1, now);
        else if (M.mode == MS_IDLE && now > M.next_idle && !M.dragging) {
            static const Step *const small[] = {ACT_SWING, ACT_SWING, ACT_LOOK, ACT_FLAP};
            const Step *s = small[rnd() % 4];
            m_run(s, s == ACT_FLAP ? N_OF(ACT_FLAP) : s == ACT_LOOK ? N_OF(ACT_LOOK) : N_OF(ACT_SWING), MS_BUSY, now);
        }
    }
    if (M.click_pending && now - M.click_at >= 0.23) { /* one click: a move; two: it dances */
        static const int moves[] = {0, 1, 2, 4, 5, 12};
        M.click_pending = 0;
        m_act(moves[rnd() % 6], now);
    }
}

/* the bird's picture this frame, in CSS pixels, from where its box starts in its first cell */
static uint8_t cv[160 * 160];
static int cv_w, cv_h;
static const int OFFS[P_COUNT] = {[P_SIT] = 13, [P_SIT2] = 13, [P_FLY1] = -2, [P_FLY2] = -2};

static int floordiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

/* compose the frame: the cell its box starts in (c0, r0); 0 when hidden */
static int m_compose(double now, int *c0, int *r0) {
    if (!M.shown) return 0;
    int pose = cur_pose(now);
    if (now > M.blink_at + 4.2) M.blink_at = now;
    int blink = now - M.blink_at < 0.13 && M.mode != MS_SLEEP;
    double hov = pose == P_FLY1 || pose == P_FLY2 ? sin(now * 1000 / 170) * 1.5 : 0, bx, by;
    if (M.dragging) {
        bx = M.drag_x;
        by = M.drag_y;
    } else {
        bx = M.comp_l + M.x;
        by = M.comp_t - BOX + 1 + OFFS[pose] + M.y + M.lift + hov;
    }
    int ix = (int)(bx + (bx < 0 ? -0.5 : 0.5)), iy = (int)(by + (by < 0 ? -0.5 : 0.5));
    *c0 = floordiv(ix, g_cw);
    *r0 = floordiv(iy, g_ch);
    int offx = ix - *c0 * g_cw, offy = iy - *r0 * g_ch;
    cv_w = offx + BOX;
    cv_h = offy + BOX;
    memset(cv, 0, (size_t)cv_w * (size_t)cv_h);
    /* hiding, it sinks into the input box: nothing below its top edge */
    int cliprow = M.clip && !M.dragging ? (int)M.comp_t - *r0 * g_ch : 1 << 20;
    int skip_g = pose == P_SIT || pose == P_SIT2;
    for (int v = 0; v < BOX; v++) {
        if (offy + v >= cliprow) break;
        for (int u = 0; u < BOX; u++) {
            int sx = M.face > 0 ? 19 - u / 3 : u / 3, sy = v / 3;
            char ch = POSES[pose][sy][sx];
            if (ch == '.' || (skip_g && ch == 'G')) continue;
            if (blink && ch == 'W' && sy > 0 && POSES[pose][sy - 1][sx] == 'M') ch = 'T';
            cv[(offy + v) * cv_w + offx + u] = (uint8_t)mpal(ch);
        }
    }
    return 1;
}

/* the frame as the screen's picture (sixel): the part on screen, never on the last row, which
   would make the terminal scroll when the cursor moves below the picture */
static void m_to_img(int c0, int r0) {
    int x0 = 0, y0 = 0, w = cv_w, h = cv_h;
    if (c0 < 0) { x0 = -c0 * g_cw; c0 = 0; }
    if (r0 < 0) { y0 = -r0 * g_ch; r0 = 0; }
    if (x0 >= w || y0 >= h) return;
    w -= x0;
    h -= y0;
    if (c0 * g_cw + w > W * g_cw) w = W * g_cw - c0 * g_cw;
    if (r0 * g_ch + h > (H - 1) * g_ch) h = (H - 1) * g_ch - r0 * g_ch;
    if (w <= 0 || h <= 0) return;
    img_new.on = 1;
    img_new.c0 = c0;
    img_new.r0 = r0;
    img_new.w = w;
    img_new.h = h;
    for (int y = 0; y < h; y++) memcpy(img_new.px + y * w, cv + (y0 + y) * cv_w + x0, (size_t)w);
}
/* the frame in half blocks: a cell is one bird pixel wide and two tall */
static void m_to_cells(int c0, int r0) {
    int cols = (cv_w + g_cw - 1) / g_cw, rows = (cv_h + g_ch - 1) / g_ch;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            int x = c0 + c, y = r0 + r, sx = c * g_cw + g_cw / 2, ty = r * g_ch + g_ch / 4, by = r * g_ch + 3 * g_ch / 4;
            if (x < 0 || x >= W || y < 0 || y >= H) continue;
            int tp = sx < cv_w && ty < cv_h ? cv[ty * cv_w + sx] : 0, bp = sx < cv_w && by < cv_h ? cv[by * cv_w + sx] : 0;
            if (!tp && !bp) continue;
            uint32_t bg = scr[y * W + x].bg;
            if (tp) put(x, y, 0x2580, MPAL[tp - 1], bp ? MPAL[bp - 1] : bg, 0);
            else put(x, y, 0x2584, MPAL[bp - 1], bg, 0);
        }
}
static void m_draw(double now) {
    for (int i = 0; i < 8; i++) { /* the page's zz: up 34 and right 14 pixels in 2.4 s, fading in and out */
        double k = (now - zz[i].t0) / 2.4;
        if (k < 0 || k >= 1) continue;
        double a = k < 0.15 ? k / 0.15 * 0.8 : 0.8 * (1 - (k - 0.15) / 0.85);
        int x = (int)((zz[i].x + 14 * k) / g_cw), y = (int)((zz[i].y - 34 * k) / g_ch);
        if (x >= 0 && x < W && y >= 0 && y < H) put(x, y, 'z', mix(scr[y * W + x].bg, ACCT, (float)a), scr[y * W + x].bg, A_BOLD);
    }
    int c0, r0;
    if (!m_compose(now, &c0, &r0)) return;
    M.hc0 = c0;
    M.hr0 = r0;
    M.hc1 = c0 + (cv_w + g_cw - 1) / g_cw;
    M.hr1 = r0 + (cv_h + g_ch - 1) / g_ch;
    if (g_sixel) m_to_img(c0, r0);
    else m_to_cells(c0, r0);
}

/* ---------------------------------------------------------------- the session */

enum { M_USER, M_ASSIST };
typedef struct {
    int kind, model, streaming, toks, err;
    char *text, *reason;
    double reason_s, tps, secs;
    time_t when;
} Msg;
typedef struct {
    int id, model;
    char title[160];
    long used;
    time_t upd;
    Msg *m;
    int n, cap;
} Conv;

typedef struct {
    const char *name;
    long ctx; /* 0: a decision model, it answers through System One */
} Model;
static const Model MODELS[] = {
    {"Qwen3.8-Flash-Next", 262144}, {"Qwen3.6-35B-A3B", 262144}, {"Qwen3.8-27B", 262144},
    {"Qwen3-Coder-30B-A3B", 262144}, {"Mistral-Large-4", 262144}, {"Muse Spark Open", 262144},
    {"Flint", 262144}, {"LFM3-VL-2B", 32768}, {"OLMoE", 4096}, {"GLM-5.2/5.3", 202752},
    {"DeepSeek V4.1 Flash", 262144}, {"Kimi K3", 262144}, {"MiMo-V2.6 Flash", 262144},
    {"Inkling", 131072}, {"Laya", 0},
};
#define NMODELS ((int)(sizeof MODELS / sizeof MODELS[0]))
#define LIVE_MODEL NMODELS /* the model the server runs, not one of the demo's */
static char live_model[96] = "colibri";
static long live_ctx = 262144;
static const char *mname(int i) { return i >= NMODELS ? live_model : MODELS[i].name; }
static long mctx(int i) { return i >= NMODELS ? live_ctx : MODELS[i].ctx; }

static Conv *conv;
static int nconv, cur, next_id = 1, cur_model;

enum { F_INPUT, F_SIDE };
static int running = 1, focus = F_INPUT, show_side = 1, show_map = 0, show_reason = 0, scroll, chat_maxscroll;
static double t_start, toast_until;
static char toast_s[200];
static int del_pending; /* deleting asks first: the conversation's id waiting for enter, or 0 */

static void toast(const char *s) {
    snprintf(toast_s, sizeof toast_s, "%s", s);
    toast_until = now_s() + 2.5;
}

static int find_model(const char *name) {
    for (int i = 0; i < NMODELS; i++)
        if (!strcmp(mname(i), name)) return i;
    return 0;
}

static Conv *conv_by_id(int id) {
    for (int i = 0; i < nconv; i++)
        if (conv[i].id == id) return &conv[i];
    return NULL;
}

/* a new conversation at the top of the list */
static Conv *conv_push_top(const char *title, int model) {
    conv = xrealloc(conv, sizeof(Conv) * (size_t)(nconv + 1));
    memmove(conv + 1, conv, sizeof(Conv) * (size_t)nconv);
    nconv++;
    Conv *c = &conv[0];
    memset(c, 0, sizeof *c);
    c->id = next_id++;
    c->model = model;
    c->upd = time(NULL);
    snprintf(c->title, sizeof c->title, "%s", title);
    return c;
}
static Conv *conv_push_end(const char *title, const char *model, long used, long age_s) {
    conv = xrealloc(conv, sizeof(Conv) * (size_t)(nconv + 1));
    Conv *c = &conv[nconv++];
    memset(c, 0, sizeof *c);
    c->id = next_id++;
    c->model = find_model(model);
    c->used = used;
    c->upd = time(NULL) - age_s;
    snprintf(c->title, sizeof c->title, "%s", title);
    return c;
}
static void conv_to_top(int i) {
    if (i <= 0) return;
    Conv c = conv[i];
    memmove(conv + 1, conv, sizeof(Conv) * (size_t)i);
    conv[0] = c;
    cur = 0;
}

static Msg *msg_add(Conv *c, int kind) {
    if (c->n == c->cap) {
        c->cap = c->cap ? c->cap * 2 : 8;
        c->m = xrealloc(c->m, sizeof(Msg) * (size_t)c->cap);
    }
    Msg *m = &c->m[c->n++];
    memset(m, 0, sizeof *m);
    m->kind = kind;
    m->model = c->model;
    m->when = time(NULL);
    return m;
}
static void say(Conv *c, const char *s, long ago) {
    Msg *m = msg_add(c, M_USER);
    m->text = xstrdup(s);
    m->when = time(NULL) - ago;
}
static void answer(Conv *c, const char *s, const char *reason, double reason_s, int toks, double tps) {
    Msg *m = msg_add(c, M_ASSIST);
    m->text = xstrdup(s);
    m->reason = reason ? xstrdup(reason) : NULL;
    m->reason_s = reason_s;
    m->toks = toks;
    m->tps = tps;
    m->secs = reason_s + toks / tps + 0.3;
}
/* the demo's conversations: what the real ones will look like */
static void demo_init(void) {
    Conv *c = conv_push_end(T("Perché col tier va più veloce?", "Why is it faster with the tier?"),
                            "Qwen3.8-Flash-Next", 6200, 120);
    say(c, T("Perché col tier il modello va più veloce?", "Why does the model run faster with the tier?"), 130);
    answer(c,
           T("Il tier tiene in **VRAM** gli esperti che il modello usa di più e lascia gli altri in RAM o sul "
             "disco.\n\n"
             "- **VRAM**: gli esperti caldi, letti dalla GPU a ogni token\n"
             "- **RAM**: quelli tiepidi, copiati in VRAM quando servono\n"
             "- **disco**: il resto, letto solo quando un token lo chiede\n\n"
             "Le promozioni si adattano mentre la conversazione va avanti: dopo qualche centinaio di token gli "
             "esperti giusti sono già in VRAM. Con **ctrl+e** vedi la mappa.",
             "The tier keeps the experts the model uses most in **VRAM** and leaves the others in RAM or on "
             "disk.\n\n"
             "- **VRAM**: the hot experts, read by the GPU at every token\n"
             "- **RAM**: the warm ones, copied to VRAM when they are needed\n"
             "- **disk**: the rest, read only when a token asks for it\n\n"
             "Promotions adapt as the conversation goes on: after a few hundred tokens the right experts are "
             "already in VRAM. **ctrl+e** shows the map."),
           T("La domanda è sul tier: dove stanno gli esperti e perché questo cambia la velocità. Spiego i tre "
             "livelli e le promozioni.",
             "The question is about the tier: where the experts live and why that changes the speed. I explain "
             "the three levels and the promotions."),
           4.1, 118, 45.7);

    c = conv_push_end(T("Le Chonk con 24 GB", "Le Chonk on 24 GB"), "Mistral-Large-4", 18400, 26 * 3600);
    say(c, T("Ce la fa Mistral Large 4 con una scheda da 24 GB?", "Can Mistral Large 4 run on a 24 GB card?"),
        26 * 3600 + 60);
    answer(c,
           T("Sì, dai GGUF di unsloth. Con **UD-IQ1_S** gli esperti che non stanno in VRAM si leggono "
             "dall'NVMe: su una scheda da 24 GB fa **1,34-1,36 tok/s**, con testo, immagini, ragionamento e "
             "strumenti.",
             "Yes, from unsloth's GGUFs. With **UD-IQ1_S** the experts that do not fit in VRAM are read from "
             "NVMe: on a 24 GB card it does **1.34-1.36 tok/s**, with text, pictures, reasoning and tools."),
           NULL, 0, 52, 1.35);

    c = conv_push_end(T("README in giapponese", "README in Japanese"), "LFM3-VL-2B", 2900, 50 * 3600);
    say(c,
        T("Traduci in giapponese: colibri fa girare modelli enormi sul tuo computer.",
          "Translate to Japanese: colibri runs huge models on your computer."),
        50 * 3600 + 30);
    answer(c, "colibriは、巨大なモデルをあなたのコンピューターで動かします。", NULL, 0, 21, 235.1);

    cur = 0;
    cur_model = conv[0].model;
}

/* ---------------------------------------------------------------- the fake answer */

static const char *DEMO_IT =
    "Questa è la **demo**: nessun `coli serve` ha risposto, quindi la risposta è finta e i numeri sono "
    "simulati. Le schermate sono quelle vere.\n\n"
    "Per parlare con un modello vero:\n\n"
    "- `coli tui --model <cartella>` avvia il server e apre la TUI\n"
    "- oppure, con `coli serve` già acceso, `coli tui` si collega\n\n"
    "Intanto prova **ctrl+p** per i comandi, **F2** per il monitor, **ctrl+e** per la mappa degli esperti e "
    "**tab** per passare da una conversazione all'altra.";
static const char *DEMO_EN =
    "This is the **demo**: no `coli serve` answered, so the answer is made up and the numbers are "
    "simulated. The screens are the real ones.\n\n"
    "To talk to a real model:\n\n"
    "- `coli tui --model <dir>` starts the server and opens the TUI\n"
    "- or, with `coli serve` already running, `coli tui` attaches to it\n\n"
    "Meanwhile try **ctrl+p** for commands, **F2** for the monitor, **ctrl+e** for the expert map and "
    "**tab** to move between conversations.";

static struct {
    int on, cid, mi, phase, toks;
    double t0, t2, prompt_s, reason_s, tps;
    const char *full;
    size_t len, pos;
} gen;

/* the expert map: activity per layer and expert, cooling every frame */
static float *hm;        /* activity per layer and expert, cooling every frame */
static int hm_rows, hm_cols, hm_tiers[3]; /* experts on disk, in RAM, in VRAM */
static uint8_t *hb;      /* the experts the last turn used (the engine's HITS), when it sends them */
static int hb_count;
static uint8_t hotset[40][6];
static void hm_alloc(int rows, int cols) {
    if (rows == hm_rows && cols == hm_cols && hm) return;
    hm_rows = rows;
    hm_cols = cols;
    hm = xrealloc(hm, sizeof(float) * (size_t)rows * (size_t)cols);
    memset(hm, 0, sizeof(float) * (size_t)rows * (size_t)cols);
    hb = xrealloc(hb, (size_t)rows * (size_t)cols);
    memset(hb, 0, (size_t)rows * (size_t)cols);
    hb_count = 0;
}
static void hm_hit(int i) {
    if (i >= 0 && i < hm_rows * hm_cols && (hm[i] += 0.25f) > 2.0f) hm[i] = 2.0f;
}

static void map_init(void) {
    hm_alloc(40, 128);
    hm_tiers[2] = 3072;
    hm_tiers[1] = 2048;
    hm_tiers[0] = 0;
    for (int l = 0; l < 40; l++)
        for (int i = 0; i < 6; i++) hotset[l][i] = (uint8_t)(rnd() % 128);
}
static void map_token(void) {
    for (int l = 0; l < 40; l++)
        for (int k = 0; k < 8; k++) hm_hit(l * 128 + (rnd() % 100 < 60 ? hotset[l][rnd() % 6] : (int)(rnd() % 128)));
}
static void map_cool(double dt) {
    float f = 1.0f - (float)dt * 5.0f;
    if (f < 0) f = 0;
    for (int i = 0; i < hm_rows * hm_cols; i++) hm[i] *= f;
}

static void gen_start(Conv *c, double t) {
    Msg *m = msg_add(c, M_ASSIST);
    m->streaming = 1;
    gen.full = T(DEMO_IT, DEMO_EN);
    gen.len = strlen(gen.full);
    m->text = xrealloc(NULL, gen.len + 1);
    m->text[0] = 0;
    m->reason = xstrdup(T("L'utente sta provando la TUI. Spiego cosa è finto in questa demo e quali tasti provare.",
                          "The user is trying the TUI. I explain what is fake in this demo and which keys to try."));
    gen.on = 1;
    gen.cid = c->id;
    gen.mi = c->n - 1;
    gen.t0 = t;
    gen.phase = 0;
    gen.prompt_s = 0.35;
    gen.reason_s = 1.6;
    gen.tps = 45.7;
    gen.pos = 0;
    gen.toks = 0;
    m_begin(MS_WORK);
    m_work(WORK_READ, N_OF(WORK_READ), t);
}

static void mon_turn_done(int prompt, int completion, double wall, int stopped);

static void gen_finish(Conv *c, Msg *m, double t, int stopped) {
    mon_turn_done((int)c->used, gen.toks, t - gen.t0, stopped);
    m->streaming = 0;
    m->toks = gen.toks;
    m->tps = t > gen.t2 && gen.phase == 2 ? gen.toks / (t - gen.t2) : 0;
    m->secs = t - gen.t0;
    if (gen.phase < 2) m->reason_s = 0;
    c->used += gen.toks;
    c->upd = time(NULL);
    gen.on = 0;
    if (stopped) m_run(WORK_STOPPED, N_OF(WORK_STOPPED), MS_BUSY, t);
    else m_run(ACT_CHEER, N_OF(ACT_CHEER), MS_BUSY, t);
}

static void gen_tick(double t) {
    if (!gen.on || g_live) return;
    Conv *c = conv_by_id(gen.cid);
    if (!c) { gen.on = 0; return; }
    Msg *m = &c->m[gen.mi];
    double e = t - gen.t0;
    if (gen.phase == 0 && e >= gen.prompt_s) {
        gen.phase = 1;
        if (M.mode == MS_WORK) m_work(WORK_LEAN, N_OF(WORK_LEAN), t);
    }
    if (gen.phase == 1 && e >= gen.prompt_s + gen.reason_s) {
        gen.phase = 2;
        if (M.mode == MS_WORK) m_work(WORK_FLY, N_OF(WORK_FLY), t);
        gen.t2 = t;
        m->reason_s = gen.reason_s;
    }
    if (gen.phase != 2) return;
    int want = (int)((t - gen.t2) * gen.tps);
    while (gen.toks < want && gen.pos < gen.len) {
        /* one token: a leading space and up to four characters of a word, as a BPE vocabulary roughly cuts */
        size_t p = gen.pos;
        int chars = 0;
        do {
            p++;
            while (p < gen.len && ((unsigned char)gen.full[p] & 0xc0) == 0x80) p++;
            chars++;
        } while (p < gen.len && chars < 4 && gen.full[p] != ' ' && gen.full[p] != '\n');
        gen.pos = p;
        gen.toks++;
        map_token();
    }
    memcpy(m->text, gen.full, gen.pos);
    m->text[gen.pos] = 0;
    if (gen.pos >= gen.len) gen_finish(c, m, t, 0);
}

static void gen_stop(double t) {
    if (!gen.on) return;
    if (g_live) { /* closing the stream is how the engine hears "stop" */
        http_close(&hc_chat);
        hc_chat.state = 0;
    }
    Conv *c = conv_by_id(gen.cid);
    if (c) gen_finish(c, &c->m[gen.mi], t, 1);
    gen.on = 0;
}

/* ---------------------------------------------------------------- the monitor */

/* The chat page's monitor (ui/index.html, view-monitor): Overview and Profiling, the same cards,
   numbers and charts, plus the GPU the page does not show. In the demo the engine is simulated:
   the numbers move with the fake answers, every finished answer is a turn. */

typedef struct {
    int prompt, completion, forwards;
    double wall, wait, matmul, attn, lmhead, other, disk, toks, tpf;
} Turn;
static Turn turns[64];
static int nturns;
static float lvh[120], qhist[40]; /* tok/s a second for two minutes; requests a second */
static int nlvh, nqh;
/* what the monitor shows: /health in live mode, simulated in the demo (the gpu server's shape) */
static struct {
    int ok, sys_ok, gpu_ok, tiers_ok, cores, kv_slots, active, capacity, queued, max_queue, completed, failed, cancelled;
    long ctx, t_vram, t_ram, t_disk;
    double cpu_pct, rss, ram_total, ram_avail, load1, gpu_util, vram_used, vram_total, gpu_temp, gpu_power, gpu_power_max,
        t_vram_gb, t_ram_gb, next;
    char arch[32], gpu_short[48];
    /* the turn the engine is on (live): reading or writing, its rates and counts */
    int lv, lv_writing;
    double lv_tok_now, lv_tok_avg, lv_elapsed, lv_read;
    long lv_prompt, lv_tokens, lv_forwards;
} ST = {.ok = 1, .sys_ok = 1, .gpu_ok = 1, .tiers_ok = 1, .cores = 16, .kv_slots = 4, .capacity = 4, .max_queue = 8,
        .ctx = 262144, .t_vram = 3072, .t_ram = 2048, .ram_total = 62.6, .vram_total = 24.0, .gpu_power_max = 140,
        .t_vram_gb = 7.0, .t_ram_gb = 4.7, .arch = "qwen38", .gpu_short = "RTX PRO 4000"};

static int show_mon, mon_tab, mon_scroll, mon_maxscroll;

/* the page's phases: their names and colours */
static const char *const PH_IT[] = {"Attesa I/O", "Matmul esperti", "Attenzione", "LM head", "Altro"};
static const char *const PH_EN[] = {"I/O wait", "Expert matmul", "Attention", "LM head", "Other"};
static const uint32_t PH_COL[] = {0x3987e5, 0x199e70, 0xc98500, 0x008300, 0x9085e9};
static double ph_of(const Turn *t, int i) {
    return i == 0 ? t->wait : i == 1 ? t->matmul : i == 2 ? t->attn : i == 3 ? t->lmhead : t->other;
}

static void turn_add(int prompt, int completion, double wall, double tpf) {
    if (nturns == 64) {
        memmove(turns, turns + 1, sizeof(Turn) * 63);
        nturns--;
    }
    Turn *t = &turns[nturns++];
    t->prompt = prompt;
    t->completion = completion;
    t->wall = wall > 0.05 ? wall : 0.05;
    t->tpf = tpf;
    t->forwards = (int)(completion / tpf + 0.5);
    t->wait = t->wall * (0.05 + 0.07 * rnd01());
    t->matmul = t->wall * (0.34 + 0.10 * rnd01());
    t->attn = t->wall * (0.12 + 0.08 * rnd01());
    t->lmhead = t->wall * (0.04 + 0.02 * rnd01());
    t->other = t->wall - t->wait - t->matmul - t->attn - t->lmhead;
    if (t->other < 0) t->other = 0;
    t->disk = t->wall * (0.30 + 0.20 * rnd01());
    t->toks = completion / t->wall;
}

static void mon_init(void) {
    for (int i = 0; i < 14; i++) {
        int prompt = 300 + (int)(rnd01() * rnd01() * 9000), completion = 40 + (int)(rnd01() * 700);
        turn_add(prompt, completion, prompt / (150 + 40 * rnd01()) + completion / (44.0 + 3 * rnd01()), 1.0);
    }
    ST.completed = 14;
    ST.rss = 11.4;
    ST.ram_avail = 40.3;
}

/* once a second, as the page polls /health and /profile */
static void mon_tick(double now) {
    if (now < ST.next) return;
    ST.next = now + 1;
    if (g_live) {
        if (nlvh == 120) memmove(lvh, lvh + 1, sizeof(float) * 119), nlvh--;
        lvh[nlvh++] = (float)(ST.lv && ST.lv_writing ? ST.lv_tok_now : 0);
        if (nqh == 40) memmove(qhist, qhist + 1, sizeof(float) * 39), nqh--;
        qhist[nqh++] = (float)(ST.active + ST.queued);
        return;
    }
    ST.active = gen.on;
    int writing = gen.on && gen.phase >= 1, busy = gen.on;
    double tps = writing && gen.phase == 2 && now - gen.t2 > 0.3 ? gen.toks / (now - gen.t2) : 0;
    if (nlvh == 120) memmove(lvh, lvh + 1, sizeof(float) * 119), nlvh--;
    lvh[nlvh++] = (float)tps;
    if (nqh == 40) memmove(qhist, qhist + 1, sizeof(float) * 39), nqh--;
    qhist[nqh++] = (float)busy;
    ST.cpu_pct = busy ? 420 + 140 * rnd01() : 1 + 2 * rnd01();
    ST.load1 = busy ? 8.5 + 2 * rnd01() : 1.6 + 0.6 * rnd01();
    ST.rss = 11.4 + 0.15 * rnd01();
    ST.ram_avail = 40.1 + 0.4 * rnd01();
    ST.gpu_util = busy ? 88 + 9 * rnd01() : rnd01() * 2;
    ST.vram_used = 21.8 + (busy ? 0.15 * rnd01() : 0);
    ST.gpu_temp = busy ? 66 + 5 * rnd01() : 38 + 2 * rnd01();
    ST.gpu_power = busy ? 118 + 20 * rnd01() : 11 + 2 * rnd01();
}

static double next_profile;
static void mon_turn_done(int prompt, int completion, double wall, int stopped) {
    if (g_live) { /* the engine's own record of the turn comes with the next /profile */
        next_profile = 0;
        return;
    }
    turn_add(prompt, completion, wall, 1.0);
    if (stopped) ST.cancelled++;
    else ST.completed++;
}

/* numbers as the page writes them: 1.234 (it) or 1,234 (en) */
static const char *nf(long v) {
    char raw[32], *out = fmt("%s", "");
    snprintf(raw, sizeof raw, "%ld", v < 0 ? -v : v);
    int n = (int)strlen(raw), o = 0;
    if (v < 0) out[o++] = '-';
    for (int i = 0; i < n; i++) {
        if (i && (n - i) % 3 == 0) out[o++] = IT ? '.' : ',';
        out[o++] = raw[i];
    }
    out[o] = 0;
    return out;
}

/* an area chart in eighths of a cell: the cells under the line dim, the top one bright */
static void spark(int x, int y, int w, int h, const float *a, int n, int slots, uint32_t color, uint32_t bg) {
    float hi = 0;
    for (int i = 0; i < n; i++)
        if (a[i] > hi) hi = a[i];
    if (n < 1) return;
    uint32_t dim = mix(bg, color, 0.32f);
    for (int c = 0; c < w; c++) {
        /* the newest value at the right edge, as the page's chart fills from the right */
        int i = slots > 0 ? n - 1 - (int)((double)(w - 1 - c) * slots / w) : (int)((double)c * (n - 1) / (w > 1 ? w - 1 : 1));
        if (i < 0 || i >= n) continue;
        int e = hi > 0 ? (int)(a[i] / hi * (float)(h * 8) + 0.5f) : 0;
        if (a[i] > 0 && e < 1) e = 1;
        int top = (e + 7) / 8 - 1;
        for (int r = 0; r < h; r++) {
            int f = e - r * 8;
            if (f <= 0) break;
            uint32_t col = r == top ? color : dim;
            put(x + c, y + h - 1 - r, f >= 8 ? 0x2588 : (uint32_t)(0x2580 + f), col, bg, 0);
        }
    }
}

static void progress(int x, int y, int w, double frac, uint32_t color) {
    if (frac < 0) frac = 0;
    if (frac > 1) frac = 1;
    int e = (int)(frac * w * 8 + 0.5);
    for (int i = 0; i < w; i++) {
        int f = e - i * 8;
        if (f >= 8) put(x + i, y, 0x2588, color, S3, 0);
        else if (f <= 0) put(x + i, y, ' ', S3, S3, 0);
        else put(x + i, y, (uint32_t)(0x2590 - f), color, S3, 0);
    }
}

static void card(int x, int y, int w, int h) { fill(x, y, w, h, S2); }

/* the state of the model: what the engine does now */
static int live_state(void) {
    if (g_live) return ST.lv ? (ST.lv_writing ? 2 : 1) : gen.on ? (gen.phase == 0 ? 1 : 2) : ST.queued ? 3 : 0;
    return !gen.on ? 0 : gen.phase == 0 ? 1 : 2;
}

static int mon_overview(int x, int y, int w, double t) {
    int y0 = y, cols = w >= 96 ? 4 : 2;
    /* ---- the state of the model, live */
    int cellw = (w - 4) / cols, cellrows = 8 / cols, h = 5 + cellrows * 4 + 6;
    card(x, y, w, h);
    int col = text(x + 2, y + 1, w - 4, T("Stato del modello", "Model state"), INK, S2, A_BOLD) + 3;
    static const char *const bit[] = {"Fermo", "Legge", "Scrive", "In coda", "Errore"};
    static const char *const ben[] = {"Idle", "Reading", "Writing", "Queued", "Error"};
    static const uint32_t bcol[] = {SOFT, 0x7cc4ff, ACCT, SOFT, 0xff7a88};
    int st = live_state();
    for (int i = 0; i < 5; i++) {
        const char *b = fmt(" %s ", IT ? bit[i] : ben[i]);
        uint32_t fg = i == st ? bcol[i] : FAINT, bg = i == st ? mix(S2, bcol[i], 0.22f) : S3;
        col += text(x + 2 + col, y + 1, w - 4 - col, b, fg, bg, i == st ? A_BOLD : 0) + 1;
    }
    textr(x + w - 2, y + 1, w - 6 - col, fmt("%s \xc2\xb7 %d slot KV", mname(conv[cur].model), ST.kv_slots), MUTED, S2, 0);
    static const char *const lit[] = {"Fermo, aspetta una richiesta", "Sta leggendo il prompt", "Sta scrivendo"};
    static const char *const len_[] = {"Idle, waiting for a request", "Reading the prompt", "Writing"};
    col = text(x + 2, y + 2, w - 4, IT ? lit[st] : len_[st], INK, S2, A_BOLD) + 2;
    text(x + 2 + col, y + 2, w - 4 - col,
         fmt(T("%d attive su %d \xc2\xb7 %d in coda \xc2\xb7 %s completate \xc2\xb7 %d fallite \xc2\xb7 %d annullate",
               "%d active of %d \xc2\xb7 %d queued \xc2\xb7 %s completed \xc2\xb7 %d failed \xc2\xb7 %d cancelled"),
             ST.active, ST.capacity, ST.queued, nf(ST.completed), ST.failed, ST.cancelled),
         MUTED, S2, 0);
    double el = gen.on ? t - gen.t0 : 0, wt = gen.on && gen.phase == 2 ? t - gen.t2 : 0;
    double tnow = wt > 0.3 ? gen.toks / wt : 0;
    int prompt_live = gen.on ? (int)conv_by_id(gen.cid)->used : 0;
    long ltoks = gen.toks, lfw = 0;
    if (g_live && ST.lv) {
        tnow = ST.lv_writing ? ST.lv_tok_now : 0;
        prompt_live = (int)ST.lv_prompt;
        ltoks = ST.lv_tokens;
        lfw = ST.lv_forwards;
        el = ST.lv_elapsed;
        wt = ST.lv_elapsed - ST.lv_read;
    }
    const char *cells[8][3] = {
        {T("Velocità adesso", "Speed now"), st == 2 && tnow > 0 ? dec(tnow, 1) : "-",
         st == 2 ? fmt("tok/s \xc2\xb7 %s %s", T("media", "mean"), dec(tnow, 1)) : st == 1 ? T("legge il prompt", "reading the prompt") : T("fermo", "idle")},
        {T("Token scritti", "Tokens written"), (gen.on || ST.lv) && ltoks ? nf(ltoks) : "-", st == 2 ? fmt(T("in %s s", "in %s s"), dec(wt, 1)) : ""},
        {T("Prompt", "Prompt"), gen.on ? nf(prompt_live) : "-",
         !gen.on ? "token" : st == 1 ? fmt(T("lo legge da %s s", "reading for %s s"), dec(el, 1)) : fmt(T("letto in %s s", "read in %s s"), dec(gen.prompt_s, 1))},
        {T("Token per passo", "Tokens per step"), g_live ? (lfw ? dec((double)ltoks / (double)lfw, 2) : "-") : gen.on && gen.toks ? dec(1.0, 2) : "-",
         g_live ? (lfw ? fmt(T("%s token in %s passi", "%s tokens in %s steps"), nf(ltoks), nf(lfw)) : "MTP")
                : gen.on && gen.toks ? fmt(T("%s token in %s passi", "%s tokens in %s steps"), nf(gen.toks), nf(gen.toks)) : "MTP"},
        {T("CPU del motore", "Engine CPU"), ST.sys_ok && ST.cores ? fmt("%d%%", (int)(ST.cpu_pct / ST.cores + 0.5)) : "-",
         cellw >= 30 ? fmt(T("%s core su %d \xc2\xb7 carico %s", "%s cores of %d \xc2\xb7 load %s"), dec(ST.cpu_pct / 100, 1), ST.cores, dec(ST.load1, 1))
                     : fmt(T("%s core su %d", "%s cores of %d"), dec(ST.cpu_pct / 100, 1), ST.cores)},
        {T("RAM del motore", "Engine RAM"), ST.sys_ok ? fmt("%s GB", dec(ST.rss, 1)) : "-", fmt(T("libera %s di %s GB", "%s of %s GB free"), dec(ST.ram_avail, 1), dec(ST.ram_total, 1))},
        {"GPU", ST.gpu_ok ? fmt("%d%%", (int)(ST.gpu_util + 0.5)) : "-", ST.gpu_ok ? fmt("%d \xc2\xb0""C \xc2\xb7 %d W", (int)ST.gpu_temp, (int)ST.gpu_power) : ""},
        {"VRAM", ST.gpu_ok ? fmt("%s GB", dec(ST.vram_used, 1)) : "-", ST.gpu_ok ? fmt(T("di %s GB \xc2\xb7 %s", "of %s GB \xc2\xb7 %s"), dec(ST.vram_total, 1), ST.gpu_short) : ""},
    };
    for (int i = 0; i < 8; i++) {
        int cx = x + 2 + (i % cols) * cellw, cy = y + 4 + (i / cols) * 4;
        text(cx, cy, cellw - 2, cells[i][0], MUTED, S2, 0);
        text(cx, cy + 1, cellw - 2, cells[i][1], INK, S2, A_BOLD);
        text(cx, cy + 2, cellw - 2, cells[i][2], FAINT, S2, 0);
    }
    int chy = y + 4 + cellrows * 4;
    spark(x + 2, chy, w - 4, 4, lvh, nlvh, 120, ACC, S2);
    float pk = 0;
    for (int i = 0; i < nlvh; i++)
        if (lvh[i] > pk) pk = lvh[i];
    text(x + 2, chy + 4, w / 2, T("tok/s, ultimi 2 minuti", "tok/s, last 2 minutes"), FAINT, S2, 0);
    if (pk > 0) textr(x + w - 2, chy + 4, w / 2, fmt(T("picco %s tok/s", "peak %s tok/s"), dec(pk, 1)), FAINT, S2, 0);
    y += h + 1;

    /* ---- the last finished turn: eight cards, each with the last 40 turns */
    text(x, y, w, T("Ultimo turno finito", "Last finished turn"), INK, BG, A_BOLD);
    y += 2;
    const Turn *L = nturns ? &turns[nturns - 1] : NULL;
    int n40 = nturns < 40 ? nturns : 40, first = nturns - n40;
    float ser[8][40];
    for (int i = 0; i < n40; i++) {
        const Turn *u = &turns[first + i];
        ser[0][i] = (float)u->toks;
        ser[1][i] = (float)u->tpf;
        ser[2][i] = (float)u->prompt;
        ser[3][i] = (float)u->wall;
        ser[4][i] = (float)u->disk;
        ser[5][i] = (float)u->wait;
        ser[7][i] = (float)(u->prompt + u->completion);
    }
    for (int i = 0; i < nqh && i < 40; i++) ser[6][i] = qhist[i];
    if (L) {
        const char *mc[8][4] = {
            {T("Velocità del turno", "Turn speed"), dec(L->toks, 1), "tok/s", T("lettura del prompt compresa", "prompt reading included")},
            {T("Token per passo", "Tokens per step"), dec(L->tpf, 2), "MTP", fmt(T("%s token in %s passi", "%s tokens in %s steps"), nf(L->completion), nf(L->forwards))},
            {"Prompt", nf(L->prompt), "token", T("letti nell'ultimo turno", "read in the last turn")},
            {T("Durata", "Duration"), dec(L->wall, 1), "s", fmt(T("%s token scritti", "%s tokens written"), nf(L->completion))},
            {T("Lettura esperti", "Expert reads"), dec(L->disk, 2), "s", T("dal disco, sovrapposta al calcolo", "from disk, overlapped with compute")},
            {T("Attesa I/O", "I/O wait"), dec(L->wait, 2), "s", T("quella che si sente davvero", "the part you actually feel")},
            {T("Coda", "Queue"), nf(ST.active + ST.queued), T("richieste", "requests"),
             fmt(T("capienza %d, coda massima %d", "capacity %d, max queue %d"), ST.capacity, ST.max_queue)},
            {T("Contesto", "Context"), nf(ST.ctx), "token", T("riassume da solo quando si riempie", "summarises by itself when full")},
        };
        static const uint32_t mcol[] = {ACC, PLUM, 0x3aa9ff, 0xf5a524, PLUM, 0x3987e5, ACC, 0x3aa9ff};
        int cw = (w - (cols - 1) * 2) / cols;
        for (int i = 0; i < 8; i++) {
            int cx = x + (i % cols) * (cw + 2), cy = y + (i / cols) * 8;
            card(cx, cy, cw, 7);
            text(cx + 2, cy + 1, cw - 4, mc[i][0], MUTED, S2, 0);
            int vc = text(cx + 2, cy + 2, cw - 4, mc[i][1], INK, S2, A_BOLD);
            text(cx + 3 + vc, cy + 2, cw - 5 - vc, mc[i][2], MUTED, S2, 0);
            text(cx + 2, cy + 3, cw - 4, mc[i][3], FAINT, S2, 0);
            spark(cx + 2, cy + 4, cw - 4, 2, ser[i], i == 6 ? (nqh < 40 ? nqh : 40) : n40, 0, mcol[i], S2);
        }
        y += 8 * (8 / cols);
    } else {
        card(x, y, w, 3);
        text(x + 2, y + 1, w - 4, T("Nessun turno ancora: scrivi qualcosa in chat e i numeri appaiono qui.",
                                    "No turns yet: write something in the chat and the numbers show up here."), MUTED, S2, 0);
        y += 4;
    }

    /* ---- the context and the memory; the recent turns */
    int two = w >= 90, lw = two ? w * 2 / 5 : w, rw = two ? w - lw - 2 : w, rx = two ? x + lw + 2 : x;
    int used = g_live && ST.lv ? (int)(ST.lv_prompt + ST.lv_tokens) : gen.on ? (int)conv_by_id(gen.cid)->used + gen.toks : L ? L->prompt + L->completion : 0;
    double pct = 100.0 * used / (double)(ST.ctx > 0 ? ST.ctx : 1);
    int lh = 6 + 2 * 7 + 1;
    card(x, y, lw, lh);
    text(x + 2, y + 1, lw - 4, gen.on ? T("Contesto del turno in corso", "Context of the turn in progress") : T("Contesto dell'ultimo turno", "Context of the last turn"),
         INK, S2, A_BOLD);
    int pc = text(x + 2, y + 3, 8, fmt("%s%%", pct < 10 ? dec(pct, 1) : fmt("%d", (int)(pct + 0.5))), INK, S2, A_BOLD);
    text(x + 4 + pc, y + 3, lw - 6 - pc, fmt("%sk / %dk", dec(used / 1000.0, 1), (int)(ST.ctx / 1000)), MUTED, S2, 0);
    progress(x + 2, y + 4, lw - 4, pct / 100, ACC);
    struct { const char *l, *r; double f; uint32_t c; int on; } bars[] = {
        {T("Esperti in VRAM", "Experts in VRAM"), fmt("%s \xc2\xb7 %s GB", nf(ST.t_vram), dec(ST.t_vram_gb, 1)), (double)ST.t_vram / (double)(ST.t_vram + ST.t_ram + ST.t_disk + 1), ACC, ST.tiers_ok},
        {T("Esperti in RAM", "Experts in RAM"), fmt("%s \xc2\xb7 %s GB", nf(ST.t_ram), dec(ST.t_ram_gb, 1)), (double)ST.t_ram / (double)(ST.t_vram + ST.t_ram + ST.t_disk + 1), 0x3aa9ff, ST.tiers_ok},
        {T("RAM del motore", "Engine RAM"), fmt("%s GB", dec(ST.rss, 1)), ST.rss / (ST.ram_total + 1e-9), PLUM, ST.sys_ok},
        {T("RAM della macchina", "Machine RAM"), fmt("%s / %s GB", dec(ST.ram_total - ST.ram_avail, 1), dec(ST.ram_total, 1)), (ST.ram_total - ST.ram_avail) / (ST.ram_total + 1e-9), 0xf5a524, ST.sys_ok},
        {T("CPU del motore", "Engine CPU"), fmt(T("%s core su %d", "%s cores of %d"), dec(ST.cpu_pct / 100, 1), ST.cores), ST.cpu_pct / 100 / (ST.cores + 1e-9), 0x3aa9ff, ST.sys_ok},
        {"GPU", fmt("%d%% \xc2\xb7 %d W / %d W", (int)(ST.gpu_util + 0.5), (int)ST.gpu_power, (int)ST.gpu_power_max), ST.gpu_util / 100, ACC, ST.gpu_ok},
        {"VRAM", fmt("%s / %s GB", dec(ST.vram_used, 1), dec(ST.vram_total, 1)), ST.vram_used / (ST.vram_total + 1e-9), PLUM, ST.gpu_ok},
    };
    for (int i = 0, row = 0; i < 7; i++) {
        if (!bars[i].on) continue;
        int by = y + 6 + row++ * 2;
        text(x + 2, by, lw / 2, bars[i].l, SOFT, S2, 0);
        textr(x + lw - 2, by, lw / 2 - 2, bars[i].r, MUTED, S2, 0);
        progress(x + 2, by + 1, lw - 4, bars[i].f, bars[i].c);
    }
    if (!two) y += lh + 1;
    int ry = two ? y : y, rh = 4 + 10 + 3;
    card(rx, ry, rw, rh);
    text(rx + 2, ry + 1, rw - 4, T("Turni recenti", "Recent turns"), INK, S2, A_BOLD);
    static const char *const hit[] = {"Turno", "Prompt", "Output", "Tok/s", "Tok/passo", "Durata", "Disco"};
    static const char *const hen[] = {"Turn", "Prompt", "Output", "Tok/s", "Tok/step", "Time", "Disk"};
    int cwid[7] = {5, 8, 7, 7, 10, 8, 8}, total = 0;
    for (int i = 0; i < 7; i++) total += cwid[i];
    int xs = rx + 2, avail = rw - 4;
    for (int i = 0, cx = xs; i < 7 && cx - xs + cwid[i] <= avail; cx += cwid[i], i++) {
        if (i == 0) text(cx, ry + 3, cwid[i], IT ? hit[i] : hen[i], MUTED, S2, 0);
        else textr(cx + cwid[i] - 1, ry + 3, cwid[i] - 1, IT ? hit[i] : hen[i], MUTED, S2, 0);
    }
    hline(xs, ry + 4, avail, 0x2500, LINE, S2);
    for (int k = 0; k < 10 && k < nturns; k++) {
        const Turn *u = &turns[nturns - 1 - k];
        const char *v[7] = {fmt("%d", nturns - k), nf(u->prompt), nf(u->completion), dec(u->toks, 1),
                            dec(u->tpf, 2), fmt("%s s", dec(u->wall, 1)), fmt("%s s", dec(u->disk, 2))};
        for (int i = 0, cx = xs; i < 7 && cx - xs + cwid[i] <= avail; cx += cwid[i], i++) {
            if (i == 0) text(cx, ry + 5 + k, cwid[i], v[i], SOFT, S2, 0);
            else textr(cx + cwid[i] - 1, ry + 5 + k, cwid[i] - 1, v[i], SOFT, S2, 0);
        }
    }
    text(rx + 2, ry + rh - 2, rw - 4, T("Tok/s qui è sul turno intero, lettura del prompt compresa.",
                                        "Tok/s here is over the whole turn, prompt reading included."), FAINT, S2, 0);
    (void)total;
    y += (two ? (lh > rh ? lh : rh) : rh) + 1;
    return y - y0;
}

static void share_bar(int x, int y, int w, const char *label, const Turn *ts, int n) {
    double tot = 0, v[5] = {0};
    for (int k = 0; k < n; k++) {
        tot += ts[k].wall;
        for (int i = 0; i < 5; i++) v[i] += ph_of(&ts[k], i);
    }
    if (tot <= 0) tot = 1;
    text(x, y, w / 2, label, SOFT, S2, 0);
    textr(x + w, y, w / 2, fmt("%ss", dec(tot, 1)), ACCT, S2, 0);
    int cx = x;
    for (int i = 0; i < 5; i++) {
        int sw = (int)(v[i] / tot * w + 0.5);
        if (i == 4) sw = x + w - cx;
        if (sw <= 0) continue;
        fill(cx, y + 1, sw, 1, PH_COL[i]);
        double sh = v[i] / tot;
        if (sh >= 0.09) {
            const char *p = fmt("%d%%", (int)(sh * 100 + 0.5));
            if (strw(p) < sw) text(cx + (sw - strw(p)) / 2, y + 1, sw, p, 0xffffff, PH_COL[i], A_BOLD);
        }
        cx += sw;
    }
}

static int mon_profiling(int x, int y, int w, double t) {
    (void)t;
    int y0 = y;
    const Turn *L = nturns ? &turns[nturns - 1] : NULL;
    card(x, y, w, 3);
    int col = text(x + 2, y + 1, w - 4, T("Dove il motore spende ogni turno", "Where the engine spends each turn"), INK, S2, A_BOLD) + 3;
    for (int i = 0; i < 5; i++) {
        put(x + 2 + col, y + 1, 0x2588, PH_COL[i], S2, 0);
        col += 2 + text(x + 4 + col, y + 1, w - 6 - col, IT ? PH_IT[i] : PH_EN[i], MUTED, S2, 0) + 1;
    }
    y += 4;
    if (!L) {
        card(x, y, w, 3);
        text(x + 2, y + 1, w - 4, T("Nessun turno profilato: invia un messaggio e i dettagli appaiono qui.",
                                    "No profiled turn: send a message and the details show up here."), MUTED, S2, 0);
        return y + 4 - y0;
    }
    /* four tiles */
    int cols = w >= 80 ? 4 : 2, tw = (w - (cols - 1) * 2) / cols;
    const char *tiles[4][3] = {
        {T("Ultimo turno", "Last turn"), dec(L->toks, 1), "tok/s"},
        {T("Tempo totale", "Total time"), fmt("%ss", dec(L->wall, 1)), fmt("%s \xe2\x86\x92 %s token", nf(L->prompt), nf(L->completion))},
        {"Batching", dec(L->tpf, 2), "token / forward"},
        {T("Servizio disco", "Disk service"), fmt("%ss", dec(L->disk, 2)), T("sovrapposto al calcolo", "overlapped with compute")},
    };
    for (int i = 0; i < 4; i++) {
        int cx = x + (i % cols) * (tw + 2), cy = y + (i / cols) * 6;
        card(cx, cy, tw, 5);
        text(cx + 2, cy + 1, tw - 4, tiles[i][0], MUTED, S2, 0);
        text(cx + 2, cy + 2, tw - 4, tiles[i][1], INK, S2, A_BOLD);
        text(cx + 2, cy + 3, tw - 4, tiles[i][2], FAINT, S2, 0);
    }
    y += 6 * (4 / cols);
    /* the shares */
    int sh = nturns > 1 ? 7 : 4;
    card(x, y, w, sh);
    share_bar(x + 2, y + 1, w - 4, T("Ultimo turno", "Last turn"), L, 1);
    if (nturns > 1) share_bar(x + 2, y + 4, w - 4, fmt(T("Finestra \xc2\xb7 ultimi %d turni", "Window \xc2\xb7 last %d turns"), nturns), turns, nturns);
    y += sh + 1;
    /* two charts: speed per turn, time per turn by phase */
    int two = w >= 90, cw = two ? (w - 2) / 2 : w, chh = 6, n = nturns < 40 ? nturns : 40, first = nturns - n;
    double pk = 1e-9, pw = 1e-9;
    for (int i = first; i < nturns; i++) {
        if (turns[i].toks > pk) pk = turns[i].toks;
        if (turns[i].wall > pw) pw = turns[i].wall;
    }
    for (int ch = 0; ch < 2; ch++) {
        int cx = two ? x + ch * (cw + 2) : x, cy = two ? y : y + ch * (chh + 5);
        card(cx, cy, cw, chh + 4);
        text(cx + 2, cy + 1, cw - 4, ch ? T("Tempo per turno per fase (s)", "Time per turn by phase (s)") : T("Velocità per turno (tok/s)", "Speed per turn (tok/s)"),
             SOFT, S2, A_BOLD);
        int aw = cw - 4, bw = aw / (n ? n : 1), gap = bw >= 3 ? 1 : 0;
        if (bw < 1) bw = 1;
        for (int k = 0; k < n && (k + 1) * bw <= aw; k++) {
            const Turn *u = &turns[first + k];
            int bx = cx + 2 + k * bw;
            if (!ch) {
                int e = (int)(u->toks / pk * chh * 8 + 0.5);
                for (int r = 0; r < chh; r++) {
                    int f = e - r * 8;
                    if (f <= 0) break;
                    for (int q = 0; q < bw - gap; q++) put(bx + q, cy + 2 + chh - 1 - r, f >= 8 ? 0x2588 : (uint32_t)(0x2580 + f), ACC, S2, 0);
                }
            } else { /* stacked: each cell the colour of the phase at its middle */
                double lim[5], acc = 0;
                for (int i = 0; i < 5; i++) lim[i] = (acc += ph_of(u, i)) / pw * chh;
                for (int r = 0; r < chh; r++) {
                    double mid = r + 0.5;
                    if (mid > lim[4]) break;
                    int i = 0;
                    while (i < 4 && mid > lim[i]) i++;
                    for (int q = 0; q < bw - gap; q++) put(bx + q, cy + 2 + chh - 1 - r, 0x2588, PH_COL[i], S2, 0);
                }
            }
        }
        const char *peak = ch ? fmt(T("picco %ss", "peak %ss"), dec(pw, 1)) : fmt(T("picco %s tok/s", "peak %s tok/s"), dec(pk, 1));
        int pkw = strw(peak);
        textr(cx + cw - 2, cy + chh + 2, pkw, peak, FAINT, S2, 0);
        text(cx + 2, cy + chh + 2, cw - 6 - pkw, n > 1 ? fmt(T("%d turni \xc2\xb7 dal meno al più recente", "%d turns \xc2\xb7 oldest to newest"), n) : T("1 turno", "1 turn"), FAINT, S2, 0);
    }
    y += two ? chh + 5 : 2 * (chh + 5);
    /* the table */
    int rows = nturns < 12 ? nturns : 12, th = rows + 6;
    card(x, y, w, th);
    int cwid[10] = {6, 15, 7, 8, 9, 9, 9, 9, 9, 10}, xs = x + 2, avail = w - 4;
    const char *hd[10] = {T("Turno", "Turn"), "Token", "tok/s", T("Totale", "Total"), "I/O", "Matmul", T("Attenz.", "Attn"),
                          "LM head", T("Altro", "Other"), T("Disco", "Disk")};
    for (int i = 0, cx = xs; i < 10 && cx - xs + cwid[i] <= avail; cx += cwid[i], i++) {
        if (i < 2) text(cx, y + 1, cwid[i] - 1, hd[i], MUTED, S2, 0);
        else textr(cx + cwid[i] - 1, y + 1, cwid[i] - 1, hd[i], i >= 4 && i <= 8 ? PH_COL[i - 4] : MUTED, S2, 0);
    }
    hline(xs, y + 2, avail, 0x2500, LINE, S2);
    for (int k = 0; k < rows; k++) {
        const Turn *u = &turns[nturns - 1 - k];
        const char *v[10] = {fmt("%d", nturns - k), fmt("%s \xe2\x86\x92 %s", nf(u->prompt), nf(u->completion)), dec(u->toks, 1),
                             fmt("%ss", dec(u->wall, 1)), fmt("%ss", dec(u->wait, 2)), fmt("%ss", dec(u->matmul, 2)), fmt("%ss", dec(u->attn, 2)),
                             fmt("%ss", dec(u->lmhead, 2)), fmt("%ss", dec(u->other, 2))};
        v[9] = fmt("%ss", dec(u->disk, 2));
        for (int i = 0, cx = xs; i < 10 && cx - xs + cwid[i] <= avail; cx += cwid[i], i++) {
            if (i < 2) text(cx, y + 3 + k, cwid[i] - 1, v[i], SOFT, S2, 0);
            else textr(cx + cwid[i] - 1, y + 3 + k, cwid[i] - 1, v[i], SOFT, S2, 0);
        }
    }
    text(x + 2, y + th - 2, w - 4, T("Il servizio disco è il tempo speso a leggere gli esperti sui thread di I/O: si sovrappone al calcolo.",
                                     "Disk service is the time spent reading experts on the I/O threads: it overlaps with compute."), FAINT, S2, 0);
    y += th + 1;
    return y - y0;
}

static void draw_monitor(int x0, double t) {
    int px = x0 + 3, pr = W - 3, w = pr - px;
    clip(x0, 0, W, H);
    int col = text(px, 1, 10, "Monitor", INK, BG, A_BOLD) + 3;
    static const char *const tabs_it[] = {"Panoramica", "Profiling"}, *const tabs_en[] = {"Overview", "Profiling"};
    for (int i = 0; i < 2; i++) {
        const char *s = fmt(" %s ", IT ? tabs_it[i] : tabs_en[i]);
        col += text(px + col, 1, w - col, s, i == mon_tab ? ACCT : MUTED, i == mon_tab ? SELBG : BG, i == mon_tab ? A_BOLD : 0) + 1;
    }
    const char *stt = !g_live ? fmt(T("demo: numeri simulati \xc2\xb7 %s", "demo: simulated numbers \xc2\xb7 %s"), mname(conv[cur].model))
                      : ST.ok ? fmt(T("motore raggiungibile \xc2\xb7 %s", "engine reachable \xc2\xb7 %s"), ST.arch)
                              : T("il motore non risponde", "the engine does not answer");
    int sw = strw(stt);
    textr(pr, 1, sw, stt, MUTED, BG, 0);
    put(pr - sw - 2, 1, 0x25cf, g_live && !ST.ok ? 0xff7a88 : gen.on ? ACC : 0x4ad08a, BG, 0);
    /* the page scrolls: so does this */
    int top = 3, bottom = H - 2;
    clip(x0, top, W, bottom);
    int h = mon_tab ? mon_profiling(px, top - mon_scroll, w, t) : mon_overview(px, top - mon_scroll, w, t);
    mon_maxscroll = h > bottom - top ? h - (bottom - top) : 0;
    if (mon_scroll > mon_maxscroll) mon_scroll = mon_maxscroll;
    clip(x0, 0, W, H);
    if (mon_scroll < mon_maxscroll) textr(pr, bottom - 1, 30, fmt(" \xe2\x86\x93 %s ", T("altro sotto", "more below")), INK, S3, 0);
    /* the status line */
    text(px, H - 1, w / 2, fmt("Monitor \xc2\xb7 %s", mname(conv[cur].model)), MUTED, BG, 0);
    const char *keys = T("tab schede \xc2\xb7 \xe2\x86\x91\xe2\x86\x93 scorri \xc2\xb7 esc chat", "tab tabs \xc2\xb7 \xe2\x86\x91\xe2\x86\x93 scroll \xc2\xb7 esc chat");
    textr(pr, H - 1, w / 2, keys, MUTED, BG, 0);
}

/* ---------------------------------------------------------------- live: the answers and the numbers from coli serve */

static double next_health, next_experts, exp_full_at;
static long exp_seq = -1;
static double gen_tr0;            /* the first reasoning token */
static long gen_ptoks;            /* the prompt's tokens, as the engine counted them */

static void str_add(char **s, const char *add) {
    size_t a = *s ? strlen(*s) : 0, b = strlen(add);
    *s = xrealloc(*s, a + b + 1);
    memcpy(*s + a, add, b + 1);
}

/* the conversation so far as an OpenAI chat request, streamed */
static void live_send(Conv *c, double t) {
    Msg *m = msg_add(c, M_ASSIST);
    m->streaming = 1;
    m->text = xstrdup("");
    char *b = NULL;
    size_t n = 0, cap = 0;
    buf_add(&b, &n, &cap, "{\"model\":\"");
    json_esc(&b, &n, &cap, live_model);
    buf_add(&b, &n, &cap, "\",\"stream\":true,\"messages\":[");
    for (int i = 0, first = 1; i < c->n - 1; i++) {
        const Msg *x = &c->m[i];
        if ((x->kind != M_USER && x->kind != M_ASSIST) || !x->text || !*x->text || x->err) continue;
        buf_add(&b, &n, &cap, first ? "{\"role\":\"" : ",{\"role\":\"");
        first = 0;
        buf_add(&b, &n, &cap, x->kind == M_USER ? "user\",\"content\":\"" : "assistant\",\"content\":\"");
        json_esc(&b, &n, &cap, x->text);
        buf_add(&b, &n, &cap, "\"}");
    }
    buf_add(&b, &n, &cap, "]}");
    gen.on = 1;
    gen.cid = c->id;
    gen.mi = c->n - 1;
    gen.t0 = t;
    gen.t2 = 0;
    gen.phase = 0;
    gen.toks = 0;
    gen.prompt_s = 0;
    gen_tr0 = 0;
    gen_ptoks = 0;
    m_begin(MS_WORK);
    m_work(WORK_READ, N_OF(WORK_READ), t);
    int bad = http_start(&hc_chat, "POST", "/v1/chat/completions", b);
    free(b);
    if (bad) {
        free(m->text);
        m->text = xstrdup(T("Il motore non risponde: è acceso `coli serve`?", "The engine does not answer: is `coli serve` running?"));
        m->err = 1;
        gen_finish(c, m, t, 1);
    }
    next_experts = 0;
}

/* a token arrived: the phases move as the page moves them */
static void live_token(Msg *m, double t, int content) {
    if (!gen.toks) {
        gen.prompt_s = t - gen.t0;
        gen.t2 = t;
    }
    gen.toks++;
    if (!content && gen.phase < 1) {
        gen.phase = 1;
        gen_tr0 = t;
        if (M.mode == MS_WORK) m_work(WORK_LEAN, N_OF(WORK_LEAN), t);
    }
    if (content && gen.phase < 2) {
        if (gen.phase == 1) m->reason_s = t - gen_tr0;
        gen.phase = 2;
        if (M.mode == MS_WORK) m_work(WORK_FLY, N_OF(WORK_FLY), t);
    }
}

static void live_fail(Conv *c, Msg *m, const char *why, double t) {
    char *e = NULL;
    if (m->text && *m->text) {
        str_add(&e, m->text);
        str_add(&e, "\n\n");
    }
    str_add(&e, why);
    free(m->text);
    m->text = e;
    m->err = 1;
    gen_finish(c, m, t, 1);
}

static void live_stream(double t) {
    if (!gen.on) return;
    Conv *c = conv_by_id(gen.cid);
    if (!c) { gen.on = 0; http_close(&hc_chat); return; }
    Msg *m = &c->m[gen.mi];
    http_pump(&hc_chat);
    if (hc_chat.state == -1) { live_fail(c, m, T("La connessione col motore si è interrotta.", "The connection to the engine dropped."), t); return; }
    if (hc_chat.state < 2) return;
    if (hc_chat.status != 200) {
        if (hc_chat.state != 3) return;
        char why[600] = "";
        if (json_parse(hc_chat.buf, (int)hc_chat.n) > 0) jstr(jget(jget(0, "error"), "message"), why, sizeof why);
        live_fail(c, m, fmt("%s %d: %s", T("Errore", "Error"), hc_chat.status, *why ? why : T("il motore ha rifiutato la richiesta", "the engine refused the request")), t);
        return;
    }
    static char piece[65536];
    int done = 0;
    while (!done) {
        char *s = hc_chat.buf + hc_chat.pos, *e = memchr(s, '\n', hc_chat.n - hc_chat.pos);
        if (!e) break;
        size_t len = (size_t)(e - s);
        hc_chat.pos += len + 1;
        if (len && s[len - 1] == '\r') len--;
        if (len < 6 || memcmp(s, "data: ", 6)) continue;
        s += 6;
        len -= 6;
        if (len == 6 && !memcmp(s, "[DONE]", 6)) { done = 1; break; }
        if (json_parse(s, (int)len) <= 0) continue;
        int er = jget(0, "error");
        if (er >= 0) {
            jstr(jget(er, "message"), piece, sizeof piece);
            live_fail(c, m, fmt("%s: %s", T("Errore", "Error"), piece), t);
            return;
        }
        int ch = jat(jget(0, "choices"), 0), d = jget(ch, "delta");
        int rc = jget(d, "reasoning_content"), ct = jget(d, "content");
        if (rc >= 0 && jstr(rc, piece, sizeof piece) > 0) {
            str_add(&m->reason, piece);
            live_token(m, t, 0);
        }
        if (ct >= 0 && jstr(ct, piece, sizeof piece) > 0) {
            const char *q = piece;
            if (!m->text || !*m->text)
                while (*q == '\n' || *q == ' ') q++;
            if (*q) str_add(&m->text, q);
            live_token(m, t, 1);
        }
        if (ST.lv && ST.lv_prompt) gen_ptoks = ST.lv_prompt;
    }
    if (done || hc_chat.state == 3) {
        http_close(&hc_chat);
        hc_chat.state = 0;
        if (gen.phase == 1 && !m->reason_s) m->reason_s = t - gen_tr0;
        long used = gen_ptoks ? gen_ptoks : c->used;
        gen_finish(c, m, t, 0);
        c->used = used + m->toks;
        if (!m->text || !*m->text) {
            free(m->text);
            m->text = xstrdup(T("(nessun testo: il modello ha solo ragionato, ctrl+r lo mostra)", "(no text: the model only reasoned, ctrl+r shows it)"));
        }
    }
}

/* /health: the engine's state, its machine, the turn in flight */
static void parse_health(void) {
    if (json_parse(hc_health.buf, (int)hc_health.n) <= 0) return;
    char tmp[96];
    ST.ok = 1;
    if (jstr(jget(0, "arch"), tmp, sizeof tmp)) snprintf(ST.arch, sizeof ST.arch, "%.31s", tmp);
    int sc = jget(0, "scheduler");
    ST.active = (int)jnum(jget(sc, "active"), 0);
    ST.queued = (int)jnum(jget(sc, "queued"), 0);
    ST.capacity = (int)jnum(jget(sc, "capacity"), 1);
    ST.max_queue = (int)jnum(jget(sc, "max_queue"), 0);
    ST.completed = (int)jnum(jget(sc, "completed"), 0);
    ST.failed = (int)jnum(jget(sc, "failed"), 0);
    ST.cancelled = (int)jnum(jget(sc, "cancelled"), 0);
    ST.kv_slots = (int)jnum(jget(0, "kv_slots"), 1);
    ST.ctx = (long)jnum(jget(0, "context_tokens"), (double)ST.ctx);
    live_ctx = ST.ctx;
    int ti = jget(0, "tiers");
    ST.tiers_ok = ti >= 0;
    if (ti >= 0) {
        ST.t_vram = (long)jnum(jget(ti, "vram"), 0);
        ST.t_ram = (long)jnum(jget(ti, "ram"), 0);
        ST.t_disk = (long)jnum(jget(ti, "disk"), 0);
        ST.t_vram_gb = jnum(jget(ti, "vram_gb"), 0);
        ST.t_ram_gb = jnum(jget(ti, "ram_gb"), 0);
    }
    int lv = jat(jget(0, "live"), 0);
    ST.lv = lv >= 0;
    if (lv >= 0) {
        jstr(jget(lv, "state"), tmp, sizeof tmp);
        ST.lv_writing = !strcmp(tmp, "writing");
        ST.lv_elapsed = jnum(jget(lv, "elapsed_s"), 0);
        ST.lv_read = jnum(jget(lv, "read_s"), 0);
        ST.lv_prompt = (long)jnum(jget(lv, "prompt_tokens"), 0);
        ST.lv_tokens = (long)jnum(jget(lv, "tokens"), 0);
        ST.lv_forwards = (long)jnum(jget(lv, "forwards"), 0);
        ST.lv_tok_avg = jnum(jget(lv, "tok_s"), 0);
        ST.lv_tok_now = jnum(jget(lv, "tok_s_now"), 0);
    }
    int sy = jget(0, "system");
    ST.sys_ok = sy >= 0 && jget(sy, "ram_total_gb") >= 0;
    if (ST.sys_ok) {
        ST.ram_total = jnum(jget(sy, "ram_total_gb"), 0);
        ST.ram_avail = jnum(jget(sy, "ram_avail_gb"), 0);
        ST.load1 = jnum(jget(sy, "load1"), 0);
        ST.cores = (int)jnum(jget(sy, "cores"), 1);
        ST.rss = jnum(jget(sy, "engine_rss_gb"), 0);
        ST.cpu_pct = jnum(jget(sy, "engine_cpu_pct"), 0);
    }
    int g = jat(jget(sy, "gpus"), 0);
    ST.gpu_ok = g >= 0;
    if (g >= 0) {
        ST.gpu_util = jnum(jget(g, "util"), 0);
        ST.vram_used = jnum(jget(g, "mem_used_gb"), 0);
        ST.vram_total = jnum(jget(g, "mem_total_gb"), 0);
        ST.gpu_temp = jnum(jget(g, "temp"), 0);
        ST.gpu_power = jnum(jget(g, "power_w"), 0);
        ST.gpu_power_max = jnum(jget(g, "power_max_w"), 0);
        if (jstr(jget(g, "name"), tmp, sizeof tmp)) {
            const char *p = strstr(tmp, "RTX") ? strstr(tmp, "RTX") : tmp; /* "NVIDIA RTX PRO 4000 ..." -> "RTX PRO 4000 ..." */
            snprintf(ST.gpu_short, sizeof ST.gpu_short, "%.20s", p);
        }
    }
}

/* /profile: the engine's record of every turn, the monitor's table and charts */
static void parse_profile(void) {
    if (json_parse(hc_profile.buf, (int)hc_profile.n) <= 0) return;
    int a = jget(0, "turns"), n = jlen(a), first = n > 64 ? n - 64 : 0;
    nturns = 0;
    for (int i = first; i < n; i++) {
        int t = jat(a, i);
        Turn *u = &turns[nturns++];
        u->prompt = (int)jnum(jget(t, "prompt_tokens"), 0);
        u->completion = (int)jnum(jget(t, "completion_tokens"), 0);
        u->forwards = (int)jnum(jget(t, "forwards"), 0);
        u->wall = jnum(jget(t, "wall_s"), 0);
        u->wait = jnum(jget(t, "expert_wait_s"), 0);
        u->matmul = jnum(jget(t, "expert_matmul_s"), 0);
        u->attn = jnum(jget(t, "attention_s"), 0);
        u->lmhead = jnum(jget(t, "lm_head_s"), 0);
        u->disk = jnum(jget(t, "expert_disk_s"), 0);
        u->other = u->wall - u->wait - u->matmul - u->attn - u->lmhead;
        if (u->other < 0) u->other = 0;
        u->toks = u->wall > 0 ? u->completion / u->wall : 0;
        u->tpf = u->forwards > 0 ? (double)u->completion / u->forwards : 0;
    }
}

/* /experts: the whole map (where each expert lives), then the steps since the last poll */
static void parse_experts(void) {
    if (json_parse(hc_experts.buf, (int)hc_experts.n) <= 0) return;
    int rows = (int)jnum(jget(0, "rows"), 0), cols = (int)jnum(jget(0, "cols"), 0);
    if (rows <= 0 || cols <= 0) return;
    hm_alloc(rows, cols);
    long seq = (long)jnum(jget(0, "live_seq"), 0);
    int mp = jget(0, "map");
    if (mp >= 0) { /* two hex digits an expert: the tier in the top two bits */
        hm_tiers[0] = hm_tiers[1] = hm_tiers[2] = 0;
        for (int k = jt[mp].start; k + 1 < jt[mp].end; k += 2) {
            char h = js[k];
            int v = h >= '0' && h <= '9' ? h - '0' : (h | 32) - 'a' + 10;
            int tier = (v << 4) >> 6;
            if (tier >= 0 && tier < 3) hm_tiers[tier]++;
        }
        int ht = jget(0, "hits"), nexp = rows * cols;
        if (ht >= 0 && jt[ht].end - jt[ht].start >= (nexp + 7) / 8 * 2) { /* a byte a pair of hex digits, the low bit first */
            hb_count = 0;
            for (int i = 0; i < nexp; i++) {
                const char *q = js + jt[ht].start + (i >> 3) * 2;
                int hi = q[0] <= '9' ? q[0] - '0' : (q[0] | 32) - 'a' + 10, lo = q[1] <= '9' ? q[1] - '0' : (q[1] | 32) - 'a' + 10;
                hb[i] = (uint8_t)((((hi << 4) | lo) >> (i & 7)) & 1);
                hb_count += hb[i];
            }
        }
        if (exp_seq < 0) exp_seq = seq;
        return;
    }
    int st = jget(0, "steps"), n = jlen(st);
    for (int i = 0; i < n; i++) {
        int k = jat(st, i);
        for (int p = jt[k].start; p + 4 <= jt[k].end; p += 4) {
            int idx = 0;
            for (int d = 0; d < 4; d++) {
                char h = js[p + d];
                idx = idx * 16 + (h >= '0' && h <= '9' ? h - '0' : (h | 32) - 'a' + 10);
            }
            hm_hit(idx);
        }
    }
    exp_seq = seq;
}

/* every frame: the answer streaming in, the polls the page makes */
static void live_tick(double t) {
    if (!g_live) return;
    live_stream(t);
    if (hc_health.state == 1 || hc_health.state == 2) http_pump(&hc_health);
    if (hc_health.state == 3) { if (hc_health.status == 200) parse_health(); else ST.ok = 0; hc_health.state = 0; }
    if (hc_health.state == -1) { ST.ok = 0; hc_health.state = 0; }
    if (hc_health.state == 0 && t >= next_health) {
        next_health = t + (gen.on || show_mon ? 1.0 : 2.0);
        if (http_start(&hc_health, "GET", "/health", NULL)) ST.ok = 0;
    }
    if (hc_profile.state == 1 || hc_profile.state == 2) http_pump(&hc_profile);
    if (hc_profile.state == 3) { if (hc_profile.status == 200) parse_profile(); hc_profile.state = 0; }
    if (hc_profile.state == -1) hc_profile.state = 0;
    if (hc_profile.state == 0 && t >= next_profile && (show_mon || !nturns)) {
        next_profile = t + 2.0;
        http_start(&hc_profile, "GET", "/profile", NULL);
    }
    if (hc_experts.state == 1 || hc_experts.state == 2) http_pump(&hc_experts);
    if (hc_experts.state == 3) { if (hc_experts.status == 200) parse_experts(); hc_experts.state = 0; }
    if (hc_experts.state == -1) hc_experts.state = 0;
    if (hc_experts.state == 0 && t >= next_experts && (show_map || gen.on)) {
        next_experts = t + 0.4;
        if (!hm || exp_seq < 0 || t >= exp_full_at) { /* the whole map now and then: the tiers and the turn's experts */
            exp_full_at = t + 2;
            http_start(&hc_experts, "GET", "/experts", NULL);
        } else {
            http_start(&hc_experts, "GET", fmt("/experts?since=%ld", exp_seq), NULL);
        }
    }
}

/* is there a coli serve at g_host:g_port? Its model's id, if so */
static int live_probe(void) {
    Http h;
    memset(&h, 0, sizeof h);
    h.fd = BAD_SOCK;
    if (http_get_sync(&h, "/health", 3.0)) { free(h.buf); return 0; }
    int ok = json_parse(h.buf, (int)h.n) > 0 && jget(0, "status") >= 0;
    if (ok && !http_get_sync(&h, "/v1/models", 3.0) && json_parse(h.buf, (int)h.n) > 0) {
        char id[96];
        if (jstr(jget(jat(jget(0, "data"), 0), "id"), id, sizeof id)) snprintf(live_model, sizeof live_model, "%s", id);
    }
    free(h.buf);
    return ok;
}

/* ---------------------------------------------------------------- rich text */

/* a message's text as styled characters, then cut into lines that fit */
typedef struct {
    uint32_t cp;
    uint8_t st;
} SC;
typedef struct {
    int a, b, ind, code;
} VL;
enum { ST_N, ST_B, ST_CODE, ST_DIM };
static SC *sc;
static int scn, sccap;
static VL *vl;
static int vln, vlcap;

static void sc_put(uint32_t cp, int st) {
    if (scn == sccap) {
        sccap = sccap ? sccap * 2 : 2048;
        sc = xrealloc(sc, sizeof(SC) * (size_t)sccap);
    }
    sc[scn].cp = cp;
    sc[scn++].st = (uint8_t)st;
}
static void vl_put(int a, int b, int ind, int code) {
    if (vln == vlcap) {
        vlcap = vlcap ? vlcap * 2 : 256;
        vl = xrealloc(vl, sizeof(VL) * (size_t)vlcap);
    }
    vl[vln++] = (VL){a, b, ind, code};
}

static void wrap(int a, int b, int w, int ind, int code) {
    if (a == b) { vl_put(a, a, 0, code); return; }
    for (int first = 1; a < b; first = 0) {
        int avail = w - (first ? 0 : ind), col = 0, i = a, sp = -1;
        if (avail < 1) avail = 1;
        while (i < b) {
            int cw = wcw(sc[i].cp);
            if (col + cw > avail) break;
            if (sc[i].cp == ' ') sp = i;
            col += cw;
            i++;
        }
        int e = i >= b ? b : (sp > a && !code) ? sp : (i > a ? i : a + 1);
        vl_put(a, e, first ? 0 : ind, code);
        a = e;
        if (!code)
            while (a < b && sc[a].cp == ' ') a++;
    }
}

/* the markdown the models write most: **bold**, `code`, ``` blocks, lists, headings */
static void md_layout(const char *s, int w, int base) {
    scn = vln = 0;
    int code = 0;
    if (w < 4) w = 4;
    while (*s) {
        const char *e = strchr(s, '\n');
        size_t len = e ? (size_t)(e - s) : strlen(s);
        const char *p = s, *end = s + len;
        if (len >= 3 && !strncmp(s, "```", 3)) {
            code = !code;
        } else if (code) {
            int a = scn;
            while (p < end) sc_put(u8next(&p), ST_CODE);
            wrap(a, scn, w - 2, 0, 1);
        } else {
            int a = scn, ind = 0, bold = 0, inl = 0;
            if (len >= 2 && (p[0] == '-' || p[0] == '*') && p[1] == ' ') {
                sc_put(0x2022, base);
                sc_put(' ', base);
                p += 2;
                ind = 2;
            } else {
                const char *q = p;
                while (q < end && *q >= '0' && *q <= '9') q++;
                if (q > p && q + 1 < end && q[0] == '.' && q[1] == ' ') ind = (int)(q - p) + 2;
            }
            if (p < end && *p == '#') {
                while (p < end && *p == '#') p++;
                while (p < end && *p == ' ') p++;
                bold = 1;
            }
            while (p < end) {
                if (!inl && p + 1 < end && p[0] == '*' && p[1] == '*') { bold = !bold; p += 2; continue; }
                if (*p == '`') { inl = !inl; p++; continue; }
                sc_put(u8next(&p), inl ? ST_CODE : bold ? ST_B : base);
            }
            wrap(a, scn, w, ind, 0);
        }
        s += len;
        if (e) s++;
    }
}

static int draw_vl(const VL *v, int x, int y, uint32_t fg, uint32_t bg, int w) {
    int col = v->ind;
    if (v->code) {
        fill(x, y, w, 1, CODEBG);
        col += 2;
    }
    for (int i = v->a; i < v->b; i++) {
        int cw = wcw(sc[i].cp), at = 0;
        if (!cw) continue;
        uint32_t f = fg, b = v->code ? (uint32_t)CODEBG : bg;
        switch (sc[i].st) {
        case ST_B: f = INK; at = A_BOLD; break;
        case ST_CODE: f = ACCT; if (!v->code) b = S2; break;
        case ST_DIM: f = MUTED; at = A_ITAL; break;
        default: break;
        }
        put(x + col, y, sc[i].cp, f, b, at);
        col += cw;
    }
    return col;
}

/* ---------------------------------------------------------------- drawing */

static uint32_t spin_cp(double t) {
    static const uint32_t f[] = {0x2581, 0x2583, 0x2585, 0x2587, 0x2585, 0x2583};
    return f[(int)(t * 10) % 6];
}

static int live_msg(const Conv *c, int mi) { return gen.on && gen.cid == c->id && gen.mi == mi; }

/* one message at (x, y), w columns: its height; with draw 0 it only measures */
static int msg_draw(const Conv *c, int mi, int x, int y, int w, int draw, double t) {
    const Msg *m = &c->m[mi];
    switch (m->kind) {
    case M_USER: {
        md_layout(m->text, w - 4, ST_N);
        int h = vln + 2;
        if (draw) {
            fill(x, y, w, h, S2);
            for (int r = 0; r < h; r++) put(x, y + r, 0x258e, ACC, S2, 0);
            for (int i = 0; i < vln; i++) draw_vl(&vl[i], x + 2, y + 1 + i, INK, S2, w - 4);
            struct tm *tm = localtime(&m->when);
            if (tm) textr(x + w - 2, y + h - 1, 8, fmt("%02d:%02d", tm->tm_hour, tm->tm_min), MUTED, S2, 0);
        }
        return h;
    }
    case M_ASSIST: {
        int live = live_msg(c, mi), h = 1;
        if (draw) {
            int col = text(x, y, w, mname(m->model), MUTED, BG, 0);
            const char *meta = NULL;
            if (live && gen.phase == 0) meta = T("legge il prompt", "reading the prompt");
            else if (live && gen.phase == 1)
                meta = fmt("%s %s s", T("ragiona", "reasoning"), dec(t - gen.t0 - gen.prompt_s, 1));
            else if (m->reason_s > 0)
                meta = fmt("%s %s s", T("ragionamento", "reasoning"), dec(m->reason_s, 1));
            if (meta) {
                col += text(x + col, y, w - col, " \xc2\xb7 ", MUTED, BG, 0);
                if (live && gen.phase < 2) {
                    put(x + col, y, spin_cp(t), ACC, BG, 0);
                    col += 2;
                }
                text(x + col, y, w - col, meta, MUTED, BG, 0);
            }
        }
        if (show_reason && m->reason && m->reason_s > 0) {
            md_layout(m->reason, w - 2, ST_DIM);
            if (draw)
                for (int i = 0; i < vln; i++) {
                    put(x, y + h + i, 0x2502, FAINT, BG, 0);
                    draw_vl(&vl[i], x + 2, y + h + i, MUTED, BG, w - 2);
                }
            h += vln;
        }
        md_layout(m->text ? m->text : "", w, ST_N);
        int endc = 0;
        if (draw)
            for (int i = 0; i < vln; i++) endc = draw_vl(&vl[i], x, y + h + i, m->err ? 0xff7a88 : SOFT, BG, w);
        if (live && gen.phase == 2) { /* the cursor where the next token lands */
            if (vln) {
                if (draw) put(x + endc, y + h + vln - 1, 0x258d, ACC, BG, 0);
            } else {
                if (draw) put(x, y + h, 0x258d, ACC, BG, 0);
                h++;
            }
        }
        return h + vln;
    }
    default: return 0;
    }
}

/* a text cut into lines that fit w, each line centred: the lines it took */
static int centered(int x, int y, int w, const char *s, uint32_t fg) {
    md_layout(s, w, ST_N);
    for (int i = 0; i < vln; i++) {
        int lw = 0;
        for (int j = vl[i].a; j < vl[i].b; j++) lw += wcw(sc[j].cp);
        draw_vl(&vl[i], x + (w - lw) / 2, y + i, fg, BG, w);
    }
    return vln;
}

static void draw_welcome(int x, int y0, int y1, int w) {
    int y = y0 + (y1 - y0) / 2 - 3;
    const char *l1 = T("Di cosa parliamo?", "What shall we talk about?");
    text(x + (w - strw(l1)) / 2 > x ? x + (w - strw(l1)) / 2 : x, y, w, l1, INK, BG, A_BOLD);
    y += 2;
    y += centered(x, y, w, T("Risponde il modello sul tuo computer, niente esce da qui.",
                             "The model on your computer answers; nothing leaves this machine."), MUTED);
    centered(x, y + 1, w, T("ctrl+p comandi \xc2\xb7 ctrl+e esperti \xc2\xb7 tab conversazioni",
                            "ctrl+p commands \xc2\xb7 ctrl+e experts \xc2\xb7 tab conversations"), FAINT);
}

static void draw_chat(const Conv *c, int x, int y0, int y1, int w, double t) {
    clip(x, y0, x + w, y1);
    if (c->n == 0) {
        draw_welcome(x, y0, y1, w);
        chat_maxscroll = 0;
        clip_all();
        return;
    }
    int total = 0;
    for (int i = 0; i < c->n; i++) total += msg_draw(c, i, x, 0, w, 0, t) + (i ? 1 : 0);
    int vh = y1 - y0, maxs = total > vh ? total - vh : 0;
    chat_maxscroll = maxs;
    if (scroll > maxs) scroll = maxs;
    int y = total > vh ? y1 - total + scroll : y0;
    for (int i = 0; i < c->n; i++) {
        if (i) y++;
        int h = msg_draw(c, i, x, y, w, 0, t);
        if (y + h > y0 && y < y1) msg_draw(c, i, x, y, w, 1, t);
        y += h;
    }
    clip_all();
    if (scroll > 0) {
        const char *s = fmt(" \xe2\x86\x93 %s ", T("più recenti: pagina giù", "newer: page down"));
        textr(x + w, y1 - 1, w, s, INK, S3, 0);
    }
}

static int side_w, side_new_y = -1, side_item_y[64], side_item_c[64], side_items, box_top;

static const char *age(time_t when) {
    long s = (long)(time(NULL) - when);
    if (s < 60) return T("ora", "now");
    if (s < 3600) return fmt("%ldm", s / 60);
    if (s < 86400) return fmt("%ldh", s / 3600);
    return fmt(IT ? "%ldg" : "%ldd", s / 86400);
}

static void draw_sidebar(int sw, double t) {
    clip(0, 0, sw, H);
    fill(0, 0, sw, H, SIDE);
    int col = text(2, 1, sw - 4, "colibri", ACC, SIDE, A_BOLD);
    text(3 + col, 1, 12, COLI_VERSION, FAINT, SIDE, 0);
    side_new_y = 3;
    put(2, 3, '+', ACC, SIDE, A_BOLD);
    text(4, 3, sw - 13, T("Nuova conversazione", "New conversation"), SOFT, SIDE, 0);
    textr(sw - 2, 3, 7, "ctrl+n", FAINT, SIDE, 0);

    int top = 5, vis = (H - 1 - top) / 3;
    if (vis < 1) vis = 1;
    int first = cur - vis + 1;
    if (first < 0) first = 0;
    side_items = 0;
    for (int i = first; i < nconv && i < first + vis && side_items < 64; i++) {
        const Conv *c = &conv[i];
        int y = top + (i - first) * 3, sel = i == cur, working = gen.on && gen.cid == c->id;
        uint32_t bg = sel ? (del_pending == c->id ? 0x4a2229 : focus == F_SIDE ? 0x323232 : S3) : SIDE;
        if (sel) {
            fill(0, y, sw, 2, bg);
            put(0, y, 0x258c, ACC, bg, 0);
            put(0, y + 1, 0x258c, ACC, bg, 0);
        }
        int rw;
        if (working) {
            const char *wk = T("al lavoro", "working");
            rw = strw(wk) + 2;
            put(sw - 2 - rw, y, spin_cp(t), ACC, bg, 0);
            textr(sw - 2, y, rw, wk, ACCT, bg, 0);
        } else {
            const char *a = age(c->upd);
            rw = strw(a);
            textr(sw - 2, y, rw, a, MUTED, bg, 0);
        }
        text(2, y, sw - 5 - rw, c->title, sel ? INK : SOFT, bg, A_BOLD);
        const char *ctx = mctx(c->model) ? fmt("%s/%s", ksize(c->used), ksize(mctx(c->model)))
                                               : "System One";
        int cw = strw(ctx);
        textr(sw - 2, y + 1, cw, ctx, MUTED, bg, 0);
        text(2, y + 1, sw - 5 - cw, mname(c->model), MUTED, bg, 0);
        side_item_y[side_items] = y;
        side_item_c[side_items++] = i;
    }
    clip_all();
}

static uint32_t heat_rgb(float v) {
    if (v < 0.04f) return 0x1f2626;
    if (v < 0.5f) return mix(0x1f2a2a, 0x177272, v / 0.5f);
    if (v < 1.2f) return mix(0x177272, ACC, (v - 0.5f) / 0.7f);
    return mix(ACC, PLUM, (v - 1.2f) / 0.6f);
}

static void draw_map(int x, int y, int w, int h) {
    clip(x, y, x + w, y + h);
    fill(x, y, w, h, SIDE);
    text(x + 2, y + 1, w - 4, T("Mappa degli esperti", "Expert map"), INK, SIDE, A_BOLD);
    text(x + 2, y + 2, w - 4, fmt(T("%d strati \xc2\xb7 %d esperti", "%d layers \xc2\xb7 %d experts"), hm_rows, hm_cols),
         MUTED, SIDE, 0);
    int gx = x + 5, gy = y + 4, gw = w - 7, gh = h - 10;
    if (gh > (hm_rows + 1) / 2) gh = (hm_rows + 1) / 2;
    if (!hm) text(gx, gy, gw, T("In attesa del motore\xe2\x80\xa6", "Waiting for the engine\xe2\x80\xa6"), MUTED, SIDE, 0);
    else if (gw > 2 && gh > 1) {
        int per = (hm_cols + gw - 1) / gw;
        for (int r = 0; r < gh; r++) {
            if ((2 * r) % 10 == 0) textr(gx - 1, gy + r, 3, fmt("%d", 2 * r), FAINT, SIDE, 0);
            for (int cc = 0; cc < gw; cc++) {
                /* live pulses: the hottest expert of the group; the turn's experts: how many of the group it used */
                float a = 0, b = 0;
                int n = 0, ua = 0, ub = 0;
                for (int e = cc * per; e < (cc + 1) * per && e < hm_cols; e++, n++) {
                    int i0 = 2 * r * hm_cols + e, i1 = i0 + hm_cols;
                    if (hm[i0] > a) a = hm[i0];
                    ua += hb[i0];
                    if (2 * r + 1 < hm_rows) {
                        if (hm[i1] > b) b = hm[i1];
                        ub += hb[i1];
                    }
                }
                if (hb_count && n) {
                    float fa = 1.8f * (float)ua / (float)n, fb = 1.8f * (float)ub / (float)n;
                    if (fa > a) a = fa;
                    if (fb > b) b = fb;
                }
                put(gx + cc, gy + r, 0x2580, heat_rgb(a), heat_rgb(b), 0);
            }
        }
    }
    int ly = y + h - 4;
    int col = text(x + 2, ly, 8, T("freddo ", "cold "), MUTED, SIDE, 0);
    for (int i = 0; i < 8; i++) put(x + 2 + col + i, ly, 0x2588, heat_rgb(0.25f * (float)i + 0.05f), SIDE, 0);
    text(x + 2 + col + 9, ly, 8, T("caldo", "hot"), MUTED, SIDE, 0);
    col = 0;
    static const uint32_t tc[] = {ACC, WARN, MUTED};
    int tot = hm_tiers[1] + hm_tiers[2] ? hm_tiers[0] + hm_tiers[1] + hm_tiers[2] : 0;
    if (!tot && hb_count) {
        int c2 = text(x + 2, ly + 2, w - 4, fmt(T("%s di %s", "%s of %s"), nf(hb_count), nf((long)hm_rows * hm_cols)), INK, SIDE, A_BOLD);
        text(x + 3 + c2, ly + 2, w - 5 - c2, fmt("(%d%%)", (int)(100.0 * hb_count / ((double)hm_rows * hm_cols) + 0.5)), MUTED, SIDE, 0);
        text(x + 2, ly + 3, w - 4, T("esperti usati nell'ultimo turno", "experts used in the last turn"), MUTED, SIDE, 0);
    }
    for (int i = 0; i < 3 && tot; i++) {
        int k = 2 - i, pct = (int)(100.0 * hm_tiers[k] / tot + 0.5);
        const char *name = k == 2 ? "VRAM" : k == 1 ? "RAM" : T("disco", "disk");
        put(x + 2 + col, ly + 2, 0x2588, tc[i], SIDE, 0);
        col += 2 + text(x + 4 + col, ly + 2, w - 4 - col, fmt("%s %d%%", name, pct), SOFT, SIDE, 0) + 1;
    }
    clip_all();
}

/* ---------------------------------------------------------------- the input */

static char inb[4096];
static int inlen, incur;

static void in_insert(uint32_t cp) {
    char t[4];
    int l = u8enc(cp, t);
    if (inlen + l >= (int)sizeof inb) return;
    memmove(inb + incur + l, inb + incur, (size_t)(inlen - incur));
    memcpy(inb + incur, t, (size_t)l);
    inlen += l;
    incur += l;
    inb[inlen] = 0;
}
static int prev_char(int i) {
    if (i <= 0) return 0;
    i--;
    while (i > 0 && (inb[i] & 0xc0) == 0x80) i--;
    return i;
}
static int next_char(int i) {
    if (i >= inlen) return inlen;
    i++;
    while (i < inlen && (inb[i] & 0xc0) == 0x80) i++;
    return i;
}
static void in_cut(int a, int b) {
    memmove(inb + a, inb + b, (size_t)(inlen - b));
    inlen -= b - a;
    inb[inlen] = 0;
    incur = a;
}

/* the input's visual lines (their first bytes) and the cursor's line and column */
static int in_layout(int w, int *starts, int maxl, int *crow, int *ccol) {
    int n = 1, col = 0;
    starts[0] = 0;
    *crow = *ccol = 0;
    const char *s = inb;
    for (;;) {
        int i = (int)(s - inb);
        if (i == incur) { *crow = n - 1; *ccol = col; }
        if (!*s) break;
        if (*s == '\n') {
            s++;
            if (n < maxl) starts[n++] = (int)(s - inb);
            col = 0;
            continue;
        }
        const char *p = s;
        int cw = wcw(u8next(&p));
        if (col + cw > w) {
            if (n < maxl) starts[n++] = i;
            col = 0;
            if (i == incur) { *crow = n - 1; *ccol = 0; }
        }
        col += cw;
        s = p;
    }
    return n;
}

static void draw_input(int x, int y, int w, int rows, int top, const int *starts, int n, int crow, int ccol) {
    int h = rows + 2;
    uint32_t barc = focus == F_INPUT ? ACC : FAINT;
    fill(x, y, w, h, S2);
    for (int r = 0; r < h; r++) put(x, y + r, 0x258e, barc, S2, 0);
    if (!inlen) {
        text(x + 2, y + 1, w - 4, T("Chiedi qualcosa", "Ask anything"), MUTED, S2, 0);
    } else {
        for (int r = 0; r < rows && top + r < n; r++) {
            int a = starts[top + r], b = top + r + 1 < n ? starts[top + r + 1] : inlen, col = 0;
            const char *s = inb + a;
            while (s < inb + b) {
                uint32_t c = u8next(&s);
                if (c == '\n') continue;
                put(x + 2 + col, y + 1 + r, c, INK, S2, 0);
                col += wcw(c);
            }
        }
    }
    if (focus == F_INPUT) {
        cur_show = 1;
        cur_x = x + 2 + ccol;
        cur_y = y + 1 + (crow - top);
    }
}

/* ---------------------------------------------------------------- the palette */

typedef struct {
    const char *it, *en, *key;
    int id;
} Cmd;
static const Cmd CMDS[] = {
    {"Nuova conversazione", "New conversation", "ctrl+n", 1},
    {"Elimina la conversazione", "Delete the conversation", "tab, canc", 11},
    {"Cambia modello", "Switch model", "", 2},
    {"Monitor: CPU, GPU, memoria, turni", "Monitor: CPU, GPU, memory, turns", "F2", 10},
    {"Mappa degli esperti", "Expert map", "ctrl+e", 3},
    {"Mostra il ragionamento", "Show the reasoning", "ctrl+r", 4},
    {"Conversazioni a lato", "Conversations sidebar", "ctrl+b", 5},
    {"Colibrì: le sue mosse", "Hummingbird: its moves", "", 8},
    {"Mostra o nascondi il colibrì", "Show or hide the hummingbird", "", 9},
    {"Language: English", "Lingua: italiano", "", 6},
    {"Esci", "Quit", "ctrl+q", 7},
};
#define NCMDS ((int)(sizeof CMDS / sizeof CMDS[0]))

static struct {
    int open, mode, sel, fn;
    char f[64];
} pal;

static int contains(const char *hay, const char *needle) {
    if (!*needle) return 1;
    for (; *hay; hay++) {
        const char *a = hay, *b = needle;
        while (*a && *b && tolower((unsigned char)*a) == tolower((unsigned char)*b)) a++, b++;
        if (!*b) return 1;
    }
    return 0;
}
static int pal_items(int *ids) {
    int n = 0;
    if (pal.mode == 0) {
        for (int i = 0; i < NCMDS; i++)
            if (contains(IT ? CMDS[i].it : CMDS[i].en, pal.f)) ids[n++] = i;
    } else if (pal.mode == 1) {
        for (int i = g_live ? LIVE_MODEL : 0; i < (g_live ? LIVE_MODEL + 1 : NMODELS); i++)
            if (contains(mname(i), pal.f)) ids[n++] = i;
    } else {
        for (int i = 0; i < NACTS; i++)
            if (contains(IT ? ACTS[i].it : ACTS[i].en, pal.f)) ids[n++] = i;
    }
    return n;
}
static void pal_open(int mode) {
    memset(&pal, 0, sizeof pal);
    pal.open = 1;
    pal.mode = mode;
    if (mode == 1) pal.sel = conv[cur].model;
}

static void new_conv(void) {
    if (nconv && conv[0].n == 0) { cur = 0; }
    else {
        conv_push_top(T("Nuova conversazione", "New conversation"), cur_model);
        cur = 0;
    }
    scroll = 0;
    focus = F_INPUT;
    m_act(0, now_s());
}

static void pal_run(int id) {
    if (pal.mode == 2) {
        pal.open = 0;
        m_act(id, now_s());
        return;
    }
    if (pal.mode == 1) {
        conv[cur].model = cur_model = id;
        pal.open = 0;
        toast(fmt("%s: %s", T("Modello", "Model"), mname(id)));
        return;
    }
    pal.open = 0;
    switch (CMDS[id].id) {
    case 1: new_conv(); break;
    case 2: pal_open(1); break;
    case 3: show_map = !show_map; break;
    case 4: show_reason = !show_reason; break;
    case 5: show_side = !show_side; break;
    case 6: IT = !IT; toast(T("Lingua: italiano", "Language: English")); break;
    case 7: running = 0; break;
    case 8: pal_open(2); break;
    case 10: show_mon = !show_mon; mon_scroll = 0; break;
    case 11: del_pending = conv[cur].id; break;
    case 9:
        M.shown = !M.shown;
        if (M.shown) m_idle(now_s());
        break;
    default: break;
    }
}

static void pal_key(const Key *k) {
    int ids[64], n = pal_items(ids);
    switch (k->k) {
    case K_ESC: pal.open = 0; break;
    case K_UP: if (pal.sel > 0) pal.sel--; break;
    case K_DOWN: if (pal.sel < n - 1) pal.sel++; break;
    case K_PGUP: pal.sel = 0; break;
    case K_PGDN: pal.sel = n ? n - 1 : 0; break;
    case K_ENTER: if (n) pal_run(ids[pal.sel < n ? pal.sel : n - 1]); break;
    case K_BACK:
        if (pal.fn) {
            do pal.fn--;
            while (pal.fn > 0 && (pal.f[pal.fn] & 0xc0) == 0x80);
            pal.f[pal.fn] = 0;
            pal.sel = 0;
        }
        break;
    case K_CHAR:
        if (k->cp >= 0x20 && k->cp != '\n' && pal.fn < (int)sizeof pal.f - 5) {
            pal.fn += u8enc(k->cp, pal.f + pal.fn);
            pal.f[pal.fn] = 0;
            pal.sel = 0;
        }
        break;
    case K_CTRL:
        if (k->cp == 'q') running = 0;
        if (k->cp == 'p' || k->cp == 'c' || k->cp == 'q') pal.open = 0;
        break;
    default: break;
    }
}

static void draw_palette(void) {
    int ids[64], n = pal_items(ids), rows = n < 10 ? (n ? n : 1) : 10;
    int w = W - 8 < 62 ? W - 8 : 62, h = rows + 5, x = (W - w) / 2, y = H / 5;
    if (y + h > H - 1) y = H - 1 - h;
    if (y < 1) y = 1;
    clip_all();
    fill(x, y, w, h, S2);
    put(x, y, 0x250c, LINE, S2, 0);
    put(x + w - 1, y, 0x2510, LINE, S2, 0);
    put(x, y + h - 1, 0x2514, LINE, S2, 0);
    put(x + w - 1, y + h - 1, 0x2518, LINE, S2, 0);
    hline(x + 1, y, w - 2, 0x2500, LINE, S2);
    hline(x + 1, y + h - 1, w - 2, 0x2500, LINE, S2);
    for (int r = 1; r < h - 1; r++) {
        put(x, y + r, 0x2502, LINE, S2, 0);
        put(x + w - 1, y + r, 0x2502, LINE, S2, 0);
    }
    text(x + 2, y + 1, w - 10, pal.mode == 2 ? T("Colibrì", "Hummingbird") : pal.mode ? T("Modelli", "Models") : T("Comandi", "Commands"),
         INK, S2, A_BOLD);
    textr(x + w - 2, y + 1, 4, "esc", FAINT, S2, 0);
    put(x + 2, y + 2, 0x203a, ACC, S2, A_BOLD);
    if (pal.fn) text(x + 4, y + 2, w - 6, pal.f, INK, S2, 0);
    else text(x + 4, y + 2, w - 6, T("cerca", "search"), FAINT, S2, 0);
    cur_show = 1;
    cur_x = x + 4 + strw(pal.f);
    cur_y = y + 2;
    if (!n) {
        text(x + 4, y + 3, w - 6, T("nessun risultato", "no results"), MUTED, S2, 0);
        return;
    }
    if (pal.sel >= n) pal.sel = n - 1;
    int first = pal.sel - rows + 1;
    if (first < 0) first = 0;
    for (int r = 0; r < rows && first + r < n; r++) {
        int i = ids[first + r], sel = first + r == pal.sel, yy = y + 3 + r;
        uint32_t bg = sel ? SELBG : S2;
        if (sel) fill(x + 1, yy, w - 2, 1, bg);
        const char *label, *right;
        if (pal.mode == 0) {
            label = IT ? CMDS[i].it : CMDS[i].en;
            right = CMDS[i].key;
        } else if (pal.mode == 2) {
            label = IT ? ACTS[i].it : ACTS[i].en;
            right = "";
        } else {
            label = mname(i);
            right = i == conv[cur].model ? T("attuale", "current")
                    : mctx(i)      ? ksize(mctx(i))
                                         : "System One";
        }
        int rw = strw(right);
        text(x + 4, yy, w - 8 - rw, label, sel ? ACCT : SOFT, bg, sel ? A_BOLD : 0);
        textr(x + w - 3, yy, rw, right, FAINT, bg, 0);
    }
}

/* ---------------------------------------------------------------- the main view */

static void draw_main(int x0, double t) {
    if (show_mon) {
        draw_monitor(x0, t);
        return;
    }
    const Conv *c = &conv[cur];
    int px = x0 + 3, pr = W - 3;
    int mapw = show_map && W - x0 >= 84 ? 34 : 0, cr = mapw ? pr - mapw - 3 : pr;
    clip(x0, 0, W, H);

    /* the title and the model, or a toast, or the question before a deletion */
    if (del_pending) {
        const char *q = fmt(T("Eliminare \xc2\xab%s\xc2\xbb? invio s\xc3\xac \xc2\xb7 esc no", "Delete \xc2\xab%s\xc2\xbb? enter yes \xc2\xb7 esc no"), c->title);
        text(px, 1, pr - px, q, 0xff7a88, BG, A_BOLD);
    } else if (t < toast_until) {
        int tw = strw(toast_s) + 2;
        text(px, 1, cr - px - tw - 2, c->title, INK, BG, A_BOLD);
        textr(pr, 1, tw, fmt(" %s ", toast_s), INK, S3, 0);
    } else {
        const char *mn = mname(c->model);
        int mw = strw(mn);
        text(px, 1, cr - px - mw - 2, c->title, INK, BG, A_BOLD);
        textr(cr, 1, mw, mn, MUTED, BG, 0);
    }

    /* from the bottom: the status line, the input box, the activity line */
    int starts[256], crow, ccol, iw = pr - px - 3;
    int n = in_layout(iw, starts, 256, &crow, &ccol);
    if (ccol >= iw) { crow++; ccol = 0; }
    int lines = n > crow + 1 ? n : crow + 1, rows = lines < 6 ? lines : 6;
    int top = crow - rows + 1 > 0 ? crow - rows + 1 : 0;
    int status_y = H - 1, box_y = status_y - (rows + 2), act_y = box_y - 1;
    int ax = px, zone = M.shown ? (BOX + 18 + g_ch - 1) / g_ch : 0;
    int chat_y1 = zone ? box_y - zone : act_y - 1, alim = M.shown ? pr - ax - (BOX + 40) / g_cw - 2 : pr - ax;
    if (chat_y1 < 5) chat_y1 = 5;
    box_top = box_y;
    M.comp_l = px * g_cw;
    M.comp_t = box_y * g_ch;
    M.comp_w = (pr - px) * g_cw;
    if (M.mode == MS_IDLE && !M.steps) M.x = m_home();

    draw_chat(c, px, 3, chat_y1, cr - px, t);
    if (mapw) draw_map(pr - mapw, 3, mapw, chat_y1 - 3);

    /* the activity line: live numbers while it writes, the last answer's after */
    clip(x0, 0, W, H);
    if (gen.on && gen.cid == c->id) {
        put(ax, act_y, spin_cp(t), ACC, BG, 0);
        const char *s;
        if (gen.phase == 0) s = T("legge il prompt", "reading the prompt");
        else if (gen.phase == 1) s = fmt("%s %s s", T("ragiona", "reasoning"), dec(t - gen.t0 - gen.prompt_s, 1));
        else {
            double el = t - gen.t2;
            s = fmt("%s tok/s \xc2\xb7 %d token", dec(el > 0.2 ? gen.toks / el : gen.tps, 1), gen.toks);
        }
        text(ax + 2, act_y, alim - 2, s, SOFT, BG, 0);
    } else {
        for (int i = c->n - 1; i >= 0; i--) {
            const Msg *m = &c->m[i];
            if (m->kind != M_ASSIST) continue;
            if (m->toks)
                text(ax, act_y, alim,
                     fmt("%d token \xc2\xb7 %s tok/s \xc2\xb7 %s s", m->toks, dec(m->tps, 1), dec(m->secs, 1)),
                     FAINT, BG, 0);
            break;
        }
    }

    draw_input(px, box_y, pr - px, rows, top, starts, n, crow, ccol);
    clip_all();
    if (M.shown) m_draw(t);
    clip(x0, 0, W, H);

    /* the status line */
    const char *mn = mname(c->model);
    int col = text(px, status_y, pr - px, mn, MUTED, BG, 0);
    text(px + col + 2, status_y, 6, g_live ? "live" : "demo", g_live ? (ST.ok ? 0x4ad08a : 0xff7a88) : PLUM, BG, 0);
    const char *ctx = mctx(c->model) ? fmt("%s/%s", ksize(c->used), ksize(mctx(c->model))) : "System One";
    int working = gen.on && gen.cid == c->id;
    const char *l1 = fmt("%s \xc2\xb7 ", ctx), *l2 = working ? "esc" : "ctrl+p",
               *l3 = fmt(" %s", working ? T("ferma", "stop") : T("comandi", "commands"));
    int x = pr - strw(l1) - strw(l2) - strw(l3);
    x += text(x, status_y, pr - x, l1, MUTED, BG, 0);
    x += text(x, status_y, pr - x, l2, INK, BG, A_BOLD);
    text(x, status_y, pr - x, l3, MUTED, BG, 0);
}

static char last_title[200];

static void render(double t) {
    int w, h;
    term_size(&w, &h);
    if (w != W || h != H || !scr) {
        W = w;
        H = h;
        scr = xrealloc(scr, sizeof(Cell) * (size_t)W * (size_t)H);
        old = xrealloc(old, sizeof(Cell) * (size_t)W * (size_t)H);
        full_redraw = 1;
    }
    clip_all();
    for (int i = 0; i < W * H; i++) scr[i] = (Cell){' ', INK, BG, 0};
    cur_show = 0;
    img_new.on = 0;
    if (W < 60 || H < 16) {
        const char *s = T("Allarga il terminale (almeno 60 x 16)", "Make the terminal larger (60 x 16 at least)");
        text((W - strw(s)) / 2 > 0 ? (W - strw(s)) / 2 : 0, H / 2, W, s, INK, BG, 0);
        present(NULL);
        return;
    }
    side_w = show_side && W >= 90 ? 34 : 0;
    if (side_w) draw_sidebar(side_w, t);
    draw_main(side_w, t);
    if (pal.open) draw_palette();

    /* the window's title follows the conversation */
    char title[200];
    snprintf(title, sizeof title, "colibri: %s", conv[cur].title);
    for (char *p = title; *p; p++)
        if ((unsigned char)*p < 0x20) *p = ' ';
    int changed = strcmp(title, last_title) != 0;
    if (changed) snprintf(last_title, sizeof last_title, "%s", title);
    present(changed ? title : NULL);
}

/* ---------------------------------------------------------------- keys to actions */

static void send_input(double t) {
    if (!inlen) return;
    if (gen.on) {
        toast(T("Aspetta la fine della risposta (esc la ferma)", "Wait for the answer to end (esc stops it)"));
        return;
    }
    Conv *c = &conv[cur];
    if (c->n == 0) {
        int i = 0, o = 0;
        while (inb[i] && o < 60) {
            int nx = next_char(i);
            for (int j = i; j < nx; j++) c->title[o++] = inb[j] == '\n' ? ' ' : inb[j];
            i = nx;
        }
        c->title[o] = 0;
        if (inb[i]) snprintf(c->title + o, sizeof c->title - (size_t)o, "\xe2\x80\xa6");
    }
    Msg *u = msg_add(c, M_USER);
    u->text = xstrdup(inb);
    c->used += inlen / 4 + 1;
    c->upd = time(NULL);
    inlen = incur = 0;
    inb[0] = 0;
    scroll = 0;
    if (g_live) live_send(c, t);
    else gen_start(c, t);
    conv_to_top(cur);
}

static void conv_delete(int id, double t) {
    int i = 0;
    while (i < nconv && conv[i].id != id) i++;
    if (i == nconv) return;
    if (gen.on && gen.cid == id) gen_stop(t);
    Conv *c = &conv[i];
    for (int k = 0; k < c->n; k++) {
        Msg *m = &c->m[k];
        free(m->text);
        free(m->reason);
    }
    free(c->m);
    memmove(conv + i, conv + i + 1, sizeof(Conv) * (size_t)(nconv - i - 1));
    nconv--;
    if (!nconv) conv_push_top(T("Nuova conversazione", "New conversation"), cur_model);
    if (cur >= nconv) cur = nconv - 1;
    scroll = 0;
}

static void switch_conv(int to) {
    if (to < 0 || to >= nconv || to == cur) return;
    cur = to;
    scroll = 0;
}

static void on_key(const Key *k, double t) {
    m_wake(t);
    if (del_pending) {
        int id = del_pending;
        del_pending = 0;
        if (k->k == K_ENTER || (k->k == K_CHAR && (k->cp == 's' || k->cp == 'y'))) {
            conv_delete(id, t);
            toast(T("Conversazione eliminata", "Conversation deleted"));
        }
        return;
    }
    if (k->k == K_CLICK && M.shown && !show_mon && k->x >= M.hc0 && k->x < M.hc1 && k->y >= M.hr0 && k->y < M.hr1 && !pal.open) {
        M.down = 1;
        M.dpx = k->x * g_cw + g_cw / 2.0;
        M.dpy = k->y * g_ch + g_ch / 2.0;
        M.drag_x = M.comp_l + M.x;
        M.drag_y = M.comp_t - BOX + 1 + M.y + M.lift;
        M.ox = M.dpx - M.drag_x;
        M.oy = M.dpy - M.drag_y;
        return;
    }
    if (k->k == K_DRAG) {
        double px = k->x * g_cw + g_cw / 2.0, py = k->y * g_ch + g_ch / 2.0;
        if (!M.down) return;
        if (!M.dragging && hypot(px - M.dpx, py - M.dpy) > 6) {
            static const int fly[] = {P_FLY1, P_FLY2};
            M.dragging = 1;
            m_begin(MS_DRAG);
            set_anim(fly, 2, 14, t);
        }
        if (M.dragging) {
            double nx = px - M.ox;
            if (nx != M.drag_x) M.face = nx > M.drag_x ? 1 : -1;
            M.drag_x = nx;
            M.drag_y = py - M.oy;
        }
        return;
    }
    if (k->k == K_RELEASE) {
        if (M.dragging) { /* it flies back to its place */
            double sx = M.drag_x - M.comp_l, sy = M.drag_y - (M.comp_t - BOX + 1);
            M.dragging = 0;
            m_run(DRAG_BACK, N_OF(DRAG_BACK), MS_BUSY, t);
            M.x = sx;
            M.y = sy;
        } else if (M.down) {
            if (M.click_pending) { M.click_pending = 0; m_act(3, t); }
            else { M.click_pending = 1; M.click_at = t; }
        }
        M.down = 0;
        return;
    }
    if (pal.open) { pal_key(k); return; }
    if ((k->k == K_FN && k->cp == 2) || (k->k == K_CTRL && k->cp == 'o')) {
        show_mon = !show_mon;
        mon_scroll = 0;
        return;
    }
    if (show_mon) {
        int page = H / 2;
        switch (k->k) {
        case K_ESC: show_mon = 0; return;
        case K_TAB: case K_BTAB: case K_LEFT: case K_RIGHT: mon_tab = !mon_tab; mon_scroll = 0; return;
        case K_UP: mon_scroll = mon_scroll > 0 ? mon_scroll - 1 : 0; return;
        case K_DOWN: if (mon_scroll < mon_maxscroll) mon_scroll++; return;
        case K_WHEELUP: mon_scroll = mon_scroll > 3 ? mon_scroll - 3 : 0; return;
        case K_WHEELDOWN: mon_scroll = mon_scroll + 3 < mon_maxscroll ? mon_scroll + 3 : mon_maxscroll; return;
        case K_PGUP: mon_scroll = mon_scroll > page ? mon_scroll - page : 0; return;
        case K_PGDN: mon_scroll = mon_scroll + page < mon_maxscroll ? mon_scroll + page : mon_maxscroll; return;
        case K_HOME: mon_scroll = 0; return;
        case K_END: mon_scroll = mon_maxscroll; return;
        case K_CHAR: case K_ENTER: show_mon = 0; break; /* typing goes back to the chat */
        default: if (k->k != K_CTRL) return;
        }
    }
    if (k->k == K_CTRL) {
        switch (k->cp) {
        case 'q': running = 0; break;
        case 'c':
            if (gen.on) gen_stop(t);
            else if (inlen) { inlen = incur = 0; inb[0] = 0; }
            else running = 0;
            break;
        case 'd': if (!inlen) running = 0; break;
        case 'p': pal_open(0); break;
        case 'n': new_conv(); break;
        case 'e': show_map = !show_map; break;
        case 'b': show_side = !show_side; break;
        case 'r':
            show_reason = !show_reason;
            toast(show_reason ? T("Ragionamento visibile", "Reasoning shown") : T("Ragionamento nascosto", "Reasoning hidden"));
            break;
        case 'l': full_redraw = 1; break;
        case 'j': if (focus == F_INPUT) in_insert('\n'); break;
        case 'u': inlen = incur = 0; inb[0] = 0; break;
        case 'w': {
            int i = incur;
            while (i > 0 && inb[i - 1] == ' ') i--;
            while (i > 0 && inb[i - 1] != ' ' && inb[i - 1] != '\n') i--;
            in_cut(i, incur);
            break;
        }
        case 'a': incur = 0; break;
        default: break;
        }
        return;
    }
    switch (k->k) {
    case K_FN: if (k->cp == 1) pal_open(0); return;
    case K_ESC:
        if (gen.on) gen_stop(t);
        else if (show_map) show_map = 0;
        else focus = F_INPUT;
        return;
    case K_TAB:
    case K_BTAB:
        if (side_w) focus = focus == F_INPUT ? F_SIDE : F_INPUT;
        return;
    case K_WHEELUP: scroll += 3; return;
    case K_WHEELDOWN: scroll = scroll > 3 ? scroll - 3 : 0; return;
    case K_PGUP: scroll += H / 2; return;
    case K_PGDN: scroll = scroll > H / 2 ? scroll - H / 2 : 0; return;
    case K_CLICK:
        if (side_w && k->x < side_w) {
            if (k->y == side_new_y) new_conv();
            for (int i = 0; i < side_items; i++)
                if (k->y == side_item_y[i] || k->y == side_item_y[i] + 1) switch_conv(side_item_c[i]);
            focus = F_INPUT;
        } else if (k->y >= box_top) {
            focus = F_INPUT;
        }
        return;
    default: break;
    }
    if ((k->k == K_UP || k->k == K_DOWN) && (k->mod & (KM_CTRL | KM_ALT))) {
        switch_conv(cur + (k->k == K_UP ? -1 : 1));
        return;
    }
    if (focus == F_SIDE) {
        if (k->k == K_DEL || k->k == K_BACK) { del_pending = conv[cur].id; return; }
        if (k->k == K_UP) { switch_conv(cur - 1); return; }
        if (k->k == K_DOWN) { switch_conv(cur + 1); return; }
        focus = F_INPUT;
        if (k->k == K_ENTER) return;
    }
    switch (k->k) {
    case K_CHAR: in_insert(k->cp); break;
    case K_NEWLINE: in_insert('\n'); break;
    case K_ENTER: send_input(t); break;
    case K_BACK: if (incur) in_cut(prev_char(incur), incur); break;
    case K_DEL: if (incur < inlen) in_cut(incur, next_char(incur)); break;
    case K_LEFT: incur = prev_char(incur); break;
    case K_RIGHT: incur = next_char(incur); break;
    case K_HOME: incur = 0; break;
    case K_END: incur = inlen; break;
    case K_UP: if (scroll < chat_maxscroll) scroll += 2; break;
    case K_DOWN: scroll = scroll > 2 ? scroll - 2 : 0; break;
    default: break;
    }
}

/* ---------------------------------------------------------------- main */

static int italian_by_default(void) {
    const char *v = getenv("COLI_LANG");
    if (!v || !*v) v = getenv("LC_ALL");
    if (!v || !*v) v = getenv("LANG");
    if (v && *v) return !strncmp(v, "it", 2);
#ifdef _WIN32
    return PRIMARYLANGID(GetUserDefaultUILanguage()) == LANG_ITALIAN;
#else
    return 0;
#endif
}

int main(int argc, char **argv) {
    int force_sixel = -1, demo_only = 0;
    const char *server = getenv("COLI_SERVER"), *key = getenv("COLI_API_KEY");
    const char *env = getenv("COLI_TUI_SIXEL");
    if (env && *env) force_sixel = *env != '0';
    IT = italian_by_default();
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--lang") && i + 1 < argc) IT = !strncmp(argv[++i], "it", 2);
        else if (!strcmp(argv[i], "--version")) { printf("colibri-tui %s\n", COLI_VERSION); return 0; }
        else if (!strcmp(argv[i], "--demo")) demo_only = 1;
        else if (!strcmp(argv[i], "--server") && i + 1 < argc) server = argv[++i];
        else if (!strcmp(argv[i], "--api-key") && i + 1 < argc) key = argv[++i];
        else if (!strcmp(argv[i], "--sixel")) force_sixel = 1;
        else if (!strcmp(argv[i], "--no-sixel")) force_sixel = 0;
        else {
            printf("colibri-tui %s: the terminal interface of colibri (docs/tui.md)\n\n"
                   "  --server URL   a coli serve to talk to (default http://127.0.0.1:8000 when it\n"
                   "                 answers, the demo otherwise)\n"
                   "  --api-key KEY  the server's key (default: COLI_API_KEY)\n"
                   "  --demo         the demo, without looking for a server\n"
                   "  --lang it|en   the interface's language (default: the system's)\n"
                   "  --sixel, --no-sixel   draw the hummingbird as a picture or in half blocks\n"
                   "                 (default: a picture when the terminal draws sixel)\n"
                   "  --version      the version\n\n"
                   "  enter send, ctrl+j new line, ctrl+p commands, ctrl+n new conversation,\n"
                   "  ctrl+e expert map, ctrl+r reasoning, ctrl+b sidebar, tab conversations,\n"
                   "  F2 or ctrl+o monitor, esc stop, pgup/pgdn scroll, ctrl+q quit\n", COLI_VERSION);
            return strcmp(argv[i], "--help") && strcmp(argv[i], "-h") ? 2 : 0;
        }
    }
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    if (key) snprintf(g_key, sizeof g_key, "%s", key);
    if (server && *server) { /* http://host:port */
        const char *h = strstr(server, "://") ? strstr(server, "://") + 3 : server, *c = strrchr(h, ':');
        size_t hl = c ? (size_t)(c - h) : strcspn(h, "/");
        snprintf(g_host, sizeof g_host, "%.*s", (int)(hl < sizeof g_host ? hl : sizeof g_host - 1), h);
        if (c) snprintf(g_port, sizeof g_port, "%.*s", (int)strcspn(c + 1, "/"), c + 1);
    }
    g_connect_ms = server && *server ? 1500 : 500;
    g_live = demo_only ? 0 : live_probe();
    g_connect_ms = 300;
    if (!g_live && server && *server) {
        fprintf(stderr, "colibri-tui: no coli serve answers at %s:%s\n", g_host, g_port);
        return 1;
    }
    int e = term_init();
    if (e) {
        fprintf(stderr, e == -2 ? "colibri-tui: this console has no VT mode (Windows 10 or later is needed)\n"
                                : "colibri-tui: run it in a terminal\n");
        return 1;
    }
    atexit(term_restore);
    probe_terminal();
    g_sixel = force_sixel >= 0 ? force_sixel : g_sixel_cap;
    if (g_sixel && g_cell_w) {
        g_cw = g_cell_w;
        g_ch = g_cell_h;
    } else if (!g_sixel) {
        g_cw = 3;
        g_ch = 6;
    }
    if (g_live) {
        memset(&ST, 0, sizeof ST);
        ST.capacity = 1;
        ST.ctx = live_ctx;
        conv_push_top(T("Nuova conversazione", "New conversation"), LIVE_MODEL);
        cur = 0;
        cur_model = LIVE_MODEL;
    } else {
        demo_init();
        map_init();
        mon_init();
    }
    t_start = now_s();
    M.shown = 1;
    M.last_input = t_start;
    M.next_tick = t_start + 1;
    m_idle(t_start);
    if (g_live) toast(fmt(T("Collegata a %s su %s:%s", "Connected to %s at %s:%s"), live_model, g_host, g_port));
    else toast(T("Modalità demo: le risposte sono finte. ctrl+p per i comandi", "Demo mode: the answers are fake. ctrl+p for commands"));
    toast_until = t_start + 5;
    double last = t_start, last_frame = 0;
    while (running && !g_eof) {
        Key keys[64];
        int n = term_keys(keys, 64, 15);
        double t = now_s();
        for (int i = 0; i < n && running; i++) on_key(&keys[i], t);
        gen_tick(t);
        live_tick(t);
        m_tick(t);
        mon_tick(t);
        map_cool(t - last);
        last = t;
        if (n || t - last_frame >= 1.0 / 30) {
            render(t);
            last_frame = t;
        }
    }
    term_restore();
    return 0;
}
