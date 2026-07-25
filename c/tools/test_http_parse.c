/* Faithful replica of qwen36_serve.c's http_recv_request body-extraction +
 * json_parse path, to isolate the "messages required" bug without a model.
 * Compile: gcc -I. test_http_parse.c -o test_http_parse
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "json.h"

/* Replicate handle_conn's body extraction exactly */
static void test_one(const char *raw, const char *label){
    int len = (int)strlen(raw);
    char *req = (char*)raw;  /* in real code this is malloc'd; same bytes */
    char *arena = NULL;
    char *body = req;
    {
        char *term = strstr(req, "\r\n\r\n");
        int toff = 4;
        if (!term){ char *t2 = strstr(req, "\n\n"); if (t2){ term = t2; toff = 2; } }
        body = term ? term + toff : req;
    }
    jval *root = json_parse(body, &arena);
    if (!root){ printf("[%s] json_parse returned NULL (invalid JSON)\n", label); return; }
    printf("[%s] root type=%d (J_OBJ=5)  body_head=[%.40s]\n", label, root->t, body);
    jval *messages = json_get(root, "messages");
    if (!messages || messages->t != J_ARR){
        printf("[%s] >>> messages MISSING or not array (got %p type=%d)\n",
               label, (void*)messages, messages?messages->t:-1);
    } else {
        printf("[%s] >>> messages OK, len=%d\n", label, messages->len);
    }
    free(arena);
}

int main(void){
    /* Case A: realistic curl request with Chinese content, CRLF line endings */
    const char *A =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost:8000\r\n"
        "User-Agent: curl/8.0\r\n"
        "Accept: */*\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 102\r\n"
        "\r\n"
        "{\"model\":\"qwen36-35b-a3b\",\"messages\":[{\"role\":\"user\",\"content\":\"请用一句话介绍杭州。\"}],\"stream\":false,\"max_tokens\":64}";
    test_one(A, "A-curl-crlf-chinese");

    /* Case B: LF-only line endings (some clients) */
    const char *B =
        "POST /v1/chat/completions HTTP/1.1\n"
        "Content-Length: 102\n"
        "\n"
        "{\"model\":\"qwen36-35b-a3b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8}";
    test_one(B, "B-curl-lf");

    /* Case C: content-length WRONG (too small -> truncated body) */
    const char *C =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Content-Length: 20\r\n"
        "\r\n"
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hello world this is truncated\"}]}";
    test_one(C, "C-truncated-body");

    return 0;
}
