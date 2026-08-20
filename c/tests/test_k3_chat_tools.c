/* chat_build_wire's tool-call records (#1143) against the tiny Kimi tokenizer.
 *
 * The expected XTML below is derived from Moonshot's reference renderer
 * (encoding_k3.py in the Kimi-K3 checkpoint repo): tool declarations are a
 * `message role="system" type="tool-declare"` turn, assistant calls render as
 * tools > call > argument tags with per-argument XTML types, tool results are
 * `message role="tool" tool="..." index="..."` turns, and attribute values
 * escape & -> &amp; and " -> &quot;. Only <|open|>/<|close|>/<|sep|>/
 * <|end_of_msg|> are special tokens; everything else is ordinary text, so the
 * round trip below (encode via chat_build_wire, decode id-by-id) must
 * reproduce the reference byte stream exactly. */
#define main coli_k3_main_unused
#include "../kimi_k3.c"
#undef main

#include <stdio.h>
#include <string.h>

static int fails;

static void render(Tok *T, const char *wire, char *out, size_t cap){
    int ids[4096], sp[4], thinking=0;
    int n=chat_build_wire(T,wire,(int)strlen(wire),&thinking,ids,4096,sp);
    if(n<0){ snprintf(out,cap,"<chat_build_wire failed: %d>",n); return; }
    size_t w=0; out[0]=0;
    for(int i=0;i<n && w+64<cap;i++){
        const char *lit = ids[i]==sp[0]?"<|open|>" : ids[i]==sp[1]?"<|close|>" :
                          ids[i]==sp[2]?"<|sep|>"  : ids[i]==sp[3]?"<|end_of_msg|>" : NULL;
        if(lit){ size_t l=strlen(lit); memcpy(out+w,lit,l); w+=l; out[w]=0; continue; }
        char buf[64];
        int nb=tok_decode(T,&ids[i],1,buf,sizeof(buf)-1);
        if(nb>0){ memcpy(out+w,buf,(size_t)nb); w+=(size_t)nb; out[w]=0; }
    }
}

static void expect(Tok *T, const char *what, const char *wire, const char *want){
    char got[8192];
    render(T,wire,got,sizeof(got));
    if(strcmp(got,want)){
        printf("  FAIL %s\n    want: %s\n    got:  %s\n",what,want,got);
        fails++;
    } else printf("  ok   %s\n",what);
}

int main(void){
    Tok T;
    tok_load(&T,"tests/tok_kimi_tiny.json");

    /* Every wire ends with the generation prompt the caller appends. */
    const char *tail_think="<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>";
    const char *tail_plain="<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>";

    /* Y: typed system message (tool-declare shape; body abbreviated). */
    {
        char want[1024];
        snprintf(want,sizeof(want),
            "<|open|>message role=\"system\" type=\"tool-declare\"<|sep|>"
            "# Tools body"
            "<|close|>message<|sep|><|end_of_msg|>%s",tail_plain);
        expect(&T,"Y renders a typed system message",
               "K3CHAT1\nY 12 12\ntool-declare# Tools body"
               "G 0\n",want);
    }

    /* O: tool result with name and index attributes. */
    {
        char want[1024];
        snprintf(want,sizeof(want),
            "<|open|>message role=\"tool\" tool=\"get_weather\" index=\"1\"<|sep|>"
            "sunny"
            "<|close|>message<|sep|><|end_of_msg|>%s",tail_plain);
        expect(&T,"O renders a tool-result message",
               "K3CHAT1\nO 1 11 5\nget_weathersunny"
               "G 0\n",want);
    }

    /* B+F+V: assistant turn with one call, two typed arguments, thinking on. */
    {
        char want[2048];
        snprintf(want,sizeof(want),
            "<|open|>message role=\"assistant\"<|sep|>"
            "<|open|>think<|sep|>why<|close|>think<|sep|>"
            "<|open|>response<|sep|>ok<|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
            "<|open|>argument key=\"city\" type=\"string\"<|sep|>Rome<|close|>argument<|sep|>"
            "<|open|>argument key=\"days\" type=\"number\"<|sep|>1e2<|close|>argument<|sep|>"
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
            "<|close|>message<|sep|><|end_of_msg|>%s",tail_think);
        expect(&T,"B/F/V render calls with typed arguments",
               "K3CHAT1\n"
               "B 1 3 2 1\nwhyok"
               "F 11 2\nget_weather"
               "V 4 6 4\ncitystringRome"
               "V 4 6 3\ndaysnumber1e2"
               "G 1\n",want);
    }

    /* J: json fallback block, plus attribute escaping in the tool name. */
    {
        char want[1024];
        snprintf(want,sizeof(want),
            "<|open|>message role=\"assistant\"<|sep|>"
            "<|open|>response<|sep|><|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            "<|open|>call tool=\"a&amp;b&quot;c\" index=\"1\"<|sep|>"
            "<|open|>json type=\"object\"<|sep|>{\"x\":1}<|close|>json<|sep|>"
            "<|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
            "<|close|>message<|sep|><|end_of_msg|>%s",tail_plain);
        expect(&T,"J renders a json block and escapes attributes",
               "K3CHAT1\n"
               "B 0 0 0 1\n"
               "J 5 7\na&b\"c{\"x\":1}"
               "G 0\n",want);
    }

    /* Malformed records are refused, not misread. */
    {
        int ids[256], sp[4], thinking=0;
        const char *bad[]={
            "K3CHAT1\nY 200 4\nshortG 0\n",              /* lengths past the payload */
            "K3CHAT1\nO 0 3 2\nfooxxG 0\n",              /* index < 1 */
            "K3CHAT1\nB 0 0 0 1\nX 3 0\nfooG 0\n",       /* unknown call record */
        };
        for(size_t i=0;i<sizeof(bad)/sizeof(*bad);i++){
            int n=chat_build_wire(&T,bad[i],(int)strlen(bad[i]),&thinking,ids,256,sp);
            if(n!=-1){ printf("  FAIL malformed record %zu accepted (n=%d)\n",i,n); fails++; }
            else printf("  ok   malformed record %zu refused\n",i);
        }
    }

    printf(fails?"test_k3_chat_tools: %d failure(s)\n":"test_k3_chat_tools: ok\n",fails);
    return fails!=0;
}
