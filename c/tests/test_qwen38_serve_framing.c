/* Qwen3.8 must share the gateway's exact byte-framed SUBMIT/STOP/CANCEL
 * contract, including consuming a busy SUBMIT payload before the next frame. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#define QWEN38_TEST_SERVE
#include "../qwen38.c"

#define CHECK(x) do { if(!(x)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#x);return 1; \
} } while(0)

static FILE *input_with(const char *bytes){
    FILE *file=tmpfile();
    if(!file)return NULL;
    size_t n=strlen(bytes);
    if(fwrite(bytes,1,n,file)!=n||fseek(file,0,SEEK_SET)){fclose(file);return NULL;}
    return file;
}

int main(void){
    CHECK(q38_reference_mode("ref.json",0)==1);
    CHECK(q38_reference_mode("prompt.txt",0)==0);
    CHECK(q38_reference_mode("ref.json",1)==0);
    CHECK(fabs(q38_decode_rate(3,0.5)-4.0)<1e-12);
    CHECK(q38_decode_rate(1,0.5)==0.0);
    CHECK(q38_decode_rate(3,0.0)==0.0);
    FILE *out=tmpfile();CHECK(out!=NULL);
    FILE *in=input_with("SUBMIT req 0 5 2 0 1\nHello\n");CHECK(in!=NULL);
    ServeReq q={0};
    CHECK(serve_read_req(in,out,&q,NULL)==2);
    CHECK(!strcmp(q.id,"req")&&q.slot==0&&q.max_tok==2&&q.plen==5&&!strcmp(q.payload,"Hello"));
    free(q.payload);fclose(in);

    /* The wire codec must consume a valid frame before the one-slot policy
     * rejects a nonzero cache slot, so the following command remains aligned. */
    in=input_with("SUBMIT wrong 7 5 1 0 1\nHello\nCANCEL wrong\n");CHECK(in!=NULL);
    CHECK(serve_read_req(in,out,NULL,NULL)==0);
    CHECK(serve_read_req(in,out,NULL,"wrong")==3);
    fclose(in);

    in=input_with("STOP req\nCANCEL req\n");CHECK(in!=NULL);
    CHECK(serve_read_req(in,out,NULL,"req")==1);
    CHECK(serve_read_req(in,out,NULL,"req")==3);
    fclose(in);

    /* The rejected payload must be drained or the following CANCEL would be
     * parsed from "Hello" and silently lost. */
    in=input_with("SUBMIT queued 0 5 1 0 1\nHello\nCANCEL req\n");CHECK(in!=NULL);
    CHECK(serve_read_req(in,out,NULL,"req")==0);
    CHECK(serve_read_req(in,out,NULL,"req")==3);
    fclose(in);

    in=input_with("SUBMIT bad 0 0 1 nan 1\n\n");CHECK(in!=NULL);
    CHECK(serve_read_req(in,out,&q,NULL)==0);
    fclose(in);fclose(out);
    puts("qwen38 serve framing: mode, decode rate, submit, busy drain, stop, cancel: ok");
    return 0;
}
