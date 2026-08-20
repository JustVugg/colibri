/* Freeze Kimi K3's existing request framing before adopting serve_codec.h. */
#define main kimi_k3_main_unused
#include "../kimi_k3.c"
#undef main

#include <assert.h>

static void binary_stream(FILE *stream)
{
#ifdef _WIN32
    assert(_setmode(_fileno(stream), _O_BINARY) != -1);
#else
    (void)stream;
#endif
}

static FILE *bytes_input(const void *data, size_t size)
{
    FILE *stream=tmpfile(); assert(stream); binary_stream(stream);
    assert(fwrite(data,1,size,stream)==size); rewind(stream); return stream;
}

static size_t read_output(FILE *stream, char *output, size_t capacity)
{
    assert(fflush(stream)==0); rewind(stream);
    return fread(output,1,capacity,stream);
}

static void test_submit_preserves_k3_wire_bytes(void)
{
    static const char payload[]=
        "K3CHAT1\nM system 11\nBe precise."
        "M user 11\n你好\nKimi"
        "A 7 9\nbecause你好。"
        "M user 8\nContinueG 1\n";
    char frame[512];
    int header=snprintf(frame,sizeof(frame),"SUBMIT req-7 0 %zu 4 0.7 0.95\n",
                        sizeof(payload)-1);
    assert(header>0);
    memcpy(frame+header,payload,sizeof(payload)-1);
    frame[header+sizeof(payload)-1]='\n';
    FILE *in=bytes_input(frame,(size_t)header+sizeof(payload));
    FILE *out=tmpfile(); assert(out); binary_stream(out);
    ServeReq request={0};

    assert(serve_read_req(in,out,&request,NULL)==2);
    assert(strcmp(request.id,"req-7")==0);
    assert(request.plen==(int)sizeof(payload)-1);
    assert(memcmp(request.payload,payload,sizeof(payload)-1)==0);
    assert(request.max_tok==4);
    assert(request.temp>.69f&&request.top_p>.94f);
    assert(fgetc(in)==EOF);
    char output[16]; assert(read_output(out,output,sizeof(output))==0);
    free(request.payload); fclose(in); fclose(out);
}

static void test_controls_match_only_active_request(void)
{
    static const char *commands[]={"STOP req-7\n","CANCEL req-7\n"};
    FILE *out=tmpfile(); assert(out); binary_stream(out);
    for(size_t i=0;i<2;i++){
        ServeReq request={0};
        FILE *match=bytes_input(commands[i],strlen(commands[i]));
        assert(serve_read_req(match,out,&request,"req-7")==1); fclose(match);
        FILE *other=bytes_input(commands[i],strlen(commands[i]));
        assert(serve_read_req(other,out,&request,"req-8")==0); fclose(other);
    }
    fclose(out);
}

static void test_errors_and_busy_payload_are_byte_exact(void)
{
    static const char bad[]="SUBMIT bad 0 -1 4 0.7 0.95\n";
    FILE *in=bytes_input(bad,sizeof(bad)-1);
    FILE *out=tmpfile(); assert(out); binary_stream(out);
    ServeReq request={0};
    assert(serve_read_req(in,out,&request,NULL)==-2);
    char output[128]; size_t count=read_output(out,output,sizeof(output));
    static const char want[]="ERROR bad bad submit header\n";
    assert(count==sizeof(want)-1&&memcmp(output,want,count)==0);
    fclose(in); fclose(out);

    static const char busy[]="SUBMIT busy 0 3 1 0 1\nx\ny\n";
    in=bytes_input(busy,sizeof(busy)-1); out=tmpfile(); assert(out); binary_stream(out);
    assert(serve_read_req(in,out,&request,"live")==2);
    assert(request.plen==3&&memcmp(request.payload,"x\ny",3)==0);
    fprintf(out,"ERROR %s engine busy\n",request.id); fflush(out);
    free(request.payload);
    count=read_output(out,output,sizeof(output));
    static const char busy_want[]="ERROR busy engine busy\n";
    assert(count==sizeof(busy_want)-1&&memcmp(output,busy_want,count)==0);
    assert(fgetc(in)==EOF);
    fclose(in); fclose(out);
}

static void test_malformed_submit_cannot_become_a_control(void)
{
    static const char *frames[]={
        "SUBMIT bad 0 12 0 0.7 0.95\nCANCEL live\n",
        "SUBMIT bad 0 12 4 0.7 0.95 extra extra\nCANCEL live\n",
    };
    for(size_t i=0;i<sizeof(frames)/sizeof(frames[0]);i++){
        FILE *in=bytes_input(frames[i],strlen(frames[i]));
        FILE *out=tmpfile(); assert(out); binary_stream(out);
        ServeReq request={0};
        assert(serve_read_req(in,out,&request,"live")==-2);
        assert(fgetc(in)=='C');
        char output[64]; size_t count=read_output(out,output,sizeof(output));
        static const char want[]="ERROR bad bad submit header\n";
        assert(count==sizeof(want)-1&&memcmp(output,want,count)==0);
        fclose(in); fclose(out);
    }

    static const char bad_term[]=
        "SUBMIT broken 0 1 4 0.7 0.95\nxXSTOP live\n";
    FILE *in=bytes_input(bad_term,sizeof(bad_term)-1);
    FILE *out=tmpfile(); assert(out); binary_stream(out);
    ServeReq request={0};
    assert(serve_read_req(in,out,&request,"live")==-2);
    assert(fgetc(in)=='S');
    fclose(in); fclose(out);
}

int main(void)
{
    test_submit_preserves_k3_wire_bytes();
    test_controls_match_only_active_request();
    test_errors_and_busy_payload_are_byte_exact();
    test_malformed_submit_cannot_become_a_control();
    puts("kimi serve framing baseline: ok");
    return 0;
}
