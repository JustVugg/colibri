/* qwen36_serve.c — OpenAI-compatible HTTP server for the colibri qwen36 engine.
 *
 * Model is loaded ONCE and kept resident in memory; each HTTP request reuses it
 * (KV cache is reused across requests, DeltaNet recurrent state is reset per
 * request). Implements a minimal pure-C HTTP/1.1 server speaking the OpenAI
 * Chat Completions protocol:
 *
 *   POST /v1/chat/completions   {messages:[{role,content}], stream?, max_tokens?, model?}
 *   GET  /health                -> {"status":"ok"}
 *   GET  /v1/models             -> {"object":"list","data":[{"id":...,"object":"model"}]}
 *
 * Output is byte-identical to the CLI's OPENAI=1 mode (same emit_openai_result).
 * Requests are handled serially (correct for CPU-only 16GB hardware).
 *
 * Cross-platform sockets: POSIX (Linux/macOS) or Winsock (Windows).
 *
 * Build (Linux):
 *   gcc -O2 -I. qwen36_serve.c -o qwen36_serve -lm -fopenmp
 * Build (Windows / MinGW-w64):
 *   gcc -D_FILE_OFFSET_BITS=64 -O2 -I. qwen36_serve.c -o qwen36_serve.exe -lws2_32 -lpsapi -lm -fopenmp -static -Wl,--stack,16777216
 *
 * Run:  SNAP=../modles/int8 PORT=8000 ./qwen36_serve 16 8
 *        (argv[1]=expert LRU cap/layer [default 16], argv[2]=quant bits [default 8])
 */
#if defined(_WIN32)
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <windows.h>
  typedef SOCKET  sock_t;
  #define SOCK_CLOSE(s)  closesocket(s)
  #define SOCK_INVALID   INVALID_SOCKET
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <unistd.h>
  #include <signal.h>
  typedef int     sock_t;
  #define SOCK_CLOSE(s)  close(s)
  #define SOCK_INVALID   (-1)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QWEN36_NO_MAIN
#include "qwen36.c"

/* ---- socket emit: routed via g_sock_send (defined in qwen36.c) ---- */
static void sock_send(long long fd, const char *buf, int n){
    sock_t s = (sock_t)fd;
    int sent = 0;
    while (sent < n){
        int r = (int)send(s, buf + sent, n - sent, 0);
        if (r <= 0) break;
        sent += r;
    }
}

static Model g_m;

/* ---------- tiny HTTP helpers ---------- */

static int recv_exact(sock_t s, char *buf, int n){
    int got = 0;
    while (got < n){
        int r = (int)recv(s, buf + got, n - got, 0);
        if (r <= 0) return got;
        got += r;
    }
    return got;
}

/* Read one full HTTP request (headers + body) into a malloc'd buffer.
 * Returns total length (>0) or <=0 on failure. Caller frees with free().
 *
 * Robust header-terminator detection: accepts BOTH CRLF ("\r\n\r\n") and
 * LF-only ("\n\n") line endings (some clients/proxies normalize to LF). */
static int http_recv_request(sock_t s, char **out){
    char hdr[16384];
    int hn = 0;
    int boundary = 0;   /* 4 = CRLF, 2 = LF, 0 = none yet */
    while (hn < (int)sizeof(hdr) - 1){
        char b;
        int r = (int)recv(s, &b, 1, 0);
        if (r <= 0) return -1;
        hdr[hn++] = b;
        if (hn >= 4 && hdr[hn-4]=='\r' && hdr[hn-3]=='\n' && hdr[hn-2]=='\r' && hdr[hn-1]=='\n'){ boundary = 4; break; }
        if (hn >= 2 && boundary == 0 && hdr[hn-2]=='\n' && hdr[hn-1]=='\n'){ boundary = 2; break; }
    }
    hdr[hn] = 0;

    /* case-insensitive content-length */
    char lh[16384]; int li = 0;
    for (int i=0;i<hn;i++){ char c=hdr[i]; if(c>='A'&&c<='Z') c=(char)(c+32); lh[li++]=c; } lh[li]=0;
    int cl = 0;
    char *clp = strstr(lh, "content-length:");
    if (clp){ cl = atoi(clp + 15); }
    if (cl < 0) cl = 0;
    if (cl > 64*1024*1024) cl = 64*1024*1024;   /* sanity cap on body */

    /* Reply to HTTP/1.1 Expect: 100-continue so clients (e.g. .NET / PowerShell
     * Invoke-RestMethod, and some OpenAI SDK builds) actually send the request
     * body. Without this the client waits for a 100 response that never arrives
     * and the body never comes -> the parser sees only the request line. */
    if (strstr(lh, "expect: 100-continue")) {
        static const char cont[] = "HTTP/1.1 100 Continue\r\n\r\n";
        send(s, cont, (int)sizeof(cont) - 1, 0);
    }

    int total = hn + cl;
    char *buf = (char*)malloc((size_t)total + 1);
    if (!buf) return -1;
    memcpy(buf, hdr, (size_t)hn);
    if (cl > 0){
        int got = recv_exact(s, buf + hn, cl);
        if (got < cl){ free(buf); return -1; }
    }
    buf[total] = 0;
    *out = buf;
    return total;
}

static void send_str(sock_t s, const char *str){
    send(s, str, (int)strlen(str), 0);
}

/* Build a Qwen chat-template prompt from OpenAI-style messages.
 * template: <|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n */
static void build_prompt(jval *messages, char *out, int outcap){
    int o = 0;
    if (messages && messages->t == J_ARR){
        for (int i=0;i<messages->len;i++){
            jval *m = messages->kids[i];
            if (!m || m->t != J_OBJ) continue;
            const char *role = jstr(m, "role");
            const char *content = jstr(m, "content");
            if (!role) role = "user";
            if (!content) content = "";
            int need = (int)strlen(role) + (int)strlen(content) + 32;
            if (o + need < outcap){
                o += snprintf(out+o, outcap-o, "<|im_start|>%s\n%s<|im_end|>\n", role, content);
            }
        }
    }
    const char *tail = "<|im_start|>assistant\n";
    if (o + (int)strlen(tail) < outcap) o += snprintf(out+o, outcap-o, "%s", tail);
    out[o] = 0;
}

static void send_error(sock_t s, int code, const char *msg){
    char body[512];
    int bl = snprintf(body, sizeof body,
        "{\"error\":{\"message\":\"%s\",\"type\":\"invalid_request_error\"}}", msg);
    char head[256];
    snprintf(head, sizeof head,
        "HTTP/1.1 %d Bad Request\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n",
        code, bl);
    send_str(s, head);
    send(s, body, bl, 0);
}

/* ---------- request dispatch ---------- */

static void handle_conn(sock_t s){
    char *req = NULL;
    int len = http_recv_request(s, &req);
    if (len <= 0){ if(req) free(req); SOCK_CLOSE(s); return; }

    /* method + path from first line (parsed into local arrays so `req`
     * stays intact for body parsing below) */
    char method[16], path[256]; method[0] = path[0] = 0;
    char *sp1 = strchr(req, ' ');
    if (!sp1){ free(req); SOCK_CLOSE(s); return; }
    int ml = (int)(sp1 - req); if (ml >= (int)sizeof(method)) ml = (int)sizeof(method)-1;
    memcpy(method, req, ml); method[ml] = 0;
    char *p = sp1 + 1;
    char *sp2 = strchr(p, ' ');
    int pl = sp2 ? (int)(sp2 - p) : (int)strlen(p);
    if (pl >= (int)sizeof(path)) pl = (int)sizeof(path)-1;
    memcpy(path, p, pl); path[pl] = 0;

    if (strcmp(method, "GET") == 0){
        if (strcmp(path, "/health") == 0){
            send_str(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}");
        } else if (strcmp(path, "/v1/models") == 0){
            char body[256];
            int bl = snprintf(body, sizeof body,
                "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"colibri\"}]}",
                g_model);
            char head[256];
            snprintf(head, sizeof head,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n", bl);
            send_str(s, head); send(s, body, bl, 0);
        } else {
            send_error(s, 404, "not found");
        }
        free(req); SOCK_CLOSE(s); return;
    }

    if (strcmp(method, "POST") != 0 || strcmp(path, "/v1/chat/completions") != 0){
        send_error(s, 404, "not found");
        free(req); SOCK_CLOSE(s); return;
    }

    /* parse JSON body. Body starts right after the header terminator, which
     * may be either "\r\n\r\n" (CRLF) or "\n\n" (LF-only). */
    char *arena = NULL;
    char *body = req;
    {
        char *term = strstr(req, "\r\n\r\n");
        int toff = 4;
        if (!term){ char *t2 = strstr(req, "\n\n"); if (t2){ term = t2; toff = 2; } }
        body = term ? term + toff : req;
    }
    jval *root = json_parse(body, &arena);
    if (!root){
        fprintf(stderr, "[serve] invalid JSON body | head=%.80s\n", body);
        send_error(s, 400, "invalid JSON body");
        free(req); free(arena); SOCK_CLOSE(s); return;
    }

    jval *messages = json_get(root, "messages");
    if (!messages || messages->t != J_ARR){
        fprintf(stderr, "[serve] field 'messages' missing/invalid | body_head=%.120s\n", body);
        send_error(s, 400, "field 'messages' (array) is required");
        free(req); free(arena); SOCK_CLOSE(s); return;
    }
    int stream = 0;
    jval *st = json_get(root, "stream");
    if (st && st->t == J_BOOL) stream = st->boolean;
    int max_tokens = (int)jnum(root, "max_tokens");
    int n_new = (max_tokens > 0) ? max_tokens : 256;
    if (n_new > 8192) n_new = 8192;          /* hard cap to protect memory */

    fprintf(stderr, "[serve] chat: stream=%d max_tokens=%d\n", stream, n_new);

    /* build prompt from chat template and encode */
    char prompt[1<<16];   /* 64KB; chat prompts are small. build_prompt truncates if exceeded. */
    build_prompt(messages, prompt, sizeof prompt);
    int *ids = NULL, np = 0;
    encode_text(prompt, &ids, &np);
    if (np <= 0){
        send_error(s, 400, "tokenizer produced no prompt tokens");
        free(req); free(arena); free(ids); SOCK_CLOSE(s); return;
    }

    /* set up OpenAI emit to this socket */
    g_openai = 1;
    g_stream = stream ? 1 : 0;
    g_sock_out = (long long)s;
    g_sock_send = sock_send;
    g_ttft = -1;
    g_gen_t0 = now_s();
    g_oa_created = (long)time(NULL);
    snprintf(g_oa_id, sizeof g_oa_id, "chatcmpl-%ld%04d", g_oa_created, (int)(now_s()*1000) % 10000);

    /* HTTP header (no content-length; body ends at connection close) */
    if (stream)
        send_str(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n");
    else
        send_str(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nConnection: close\r\n\r\n");

    int *out = (int*)malloc((size_t)(np + n_new) * sizeof(int));
    /* COLIBRI_RESIDENT: re-collect this request's experts (model is shared across requests) */
    if (g_m.resident_mode) { g_m.first_step = 1; g_m.resident_collecting = 0; }
    generate(&g_m, ids, np, n_new, out);
    emit_openai_result(out, np, n_new, g_stream);

    free(out); free(ids); free(req); free(arena);
    g_sock_out = -1; g_sock_send = NULL;
    SOCK_CLOSE(s);
}

int main(int argc, char **argv){
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }

#if defined(_WIN32)
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2,2), &wsa) != 0){ fprintf(stderr, "WSAStartup failed\n"); return 1; }
#endif

    g_pilot = getenv("PILOT") ? atoi(getenv("PILOT")) : 0;
    g_wide  = getenv("WIDE")  ? atoi(getenv("WIDE"))  : 1;
    if (g_wide < 1) g_wide = 1; if (g_wide > 4) g_wide = 4;
    const char *mv = getenv("MODEL"); if (mv && *mv) g_model = mv;
    int hot_n = getenv("HOT") ? atoi(getenv("HOT")) : 0;
    int cap   = argc > 1 ? atoi(argv[1]) : 16;
    int bits  = argc > 2 ? atoi(argv[2]) : 8;
    if (bits < 2 || bits > 8) { fprintf(stderr, "quant_bits must be 2..8 (got %d)\n", bits); return 1; }
    int port  = getenv("PORT") ? atoi(getenv("PORT")) : 8000;

    /* load tokenizer (reuse engine loader) */
    {
        const char *tokpath = getenv("TOK");
        if (tokpath && *tokpath) load_tokenizer(tokpath);
        else { char tpb[2048]; snprintf(tpb,sizeof tpb,"%s/tokenizer.json",snap); load_tokenizer(tpb); }
        if (!g_tok) { fprintf(stderr, "[serve] FATAL: tokenizer failed to load from %s\n", snap); return 1; }
    }

    /* load model ONCE (resident) */
    double t0 = now_s();
    model_init(&g_m, snap, cap, bits);

    /* optional transparent Vulkan compute backend for the routed-expert GEMVs.
     * Same path as qwen36.c main(): probe, and only engage if a compute-capable
     * Vulkan device exists; silently fall back to CPU on any failure. */
    gpu_probe();
    if (g_gpu_backend == GPU_BACKEND_VULKAN) {
        vg_cfg vc;
        vc.n_layers = g_m.c.n_layers;
        vc.hidden   = g_m.c.hidden;
        vc.inter    = g_m.c.inter;
        vc.cap      = cap;
        vc.topk     = g_m.c.topk;
        vc.weight_bits = container_is_int4(&g_m) ? 4 : 8;  /* by on-disk size, not meta.ebits */
        if (vg_init(&vc) == 0) {
            fprintf(stderr, "[serve][gpu] MoE expert GEMVs -> Vulkan compute device\n");
            atexit(vg_shutdown);
        } else {
            fprintf(stderr, "[serve][gpu] vg_init failed -> keeping CPU MoE path\n");
            g_gpu_backend = GPU_BACKEND_CPU;
        }
    }
    double tload = now_s() - t0;
    fprintf(stderr, "[serve] model resident in %.1fs | RSS %.2f GB | listening on http://0.0.0.0:%d  (PORT=%d)\n",
            tload, rss_gb(), port, port);
    (void)hot_n;

    sock_t srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv == SOCK_INVALID){ fprintf(stderr, "socket() failed\n"); return 1; }
    int opt = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const void*)&opt, sizeof opt);
    struct sockaddr_in addr; memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((unsigned short)port);
    if (bind(srv, (struct sockaddr*)&addr, sizeof addr) == SOCK_INVALID){
        fprintf(stderr, "bind() to port %d failed (in use?)\n", port); SOCK_CLOSE(srv); return 1;
    }
    if (listen(srv, 16) == SOCK_INVALID){ fprintf(stderr, "listen() failed\n"); SOCK_CLOSE(srv); return 1; }

#if !defined(_WIN32)
    signal(SIGPIPE, SIG_IGN);   /* don't die if a client disconnects mid-stream */
#endif

    while (1){
        sock_t c = accept(srv, NULL, NULL);
        if (c == SOCK_INVALID) continue;
        handle_conn(c);   /* serial; closes c internally */
    }
    /* unreachable */
    SOCK_CLOSE(srv);
#if defined(_WIN32)
    WSACleanup();
#endif
    return 0;
}
