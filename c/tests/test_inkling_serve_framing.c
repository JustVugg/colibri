/* Freeze Inkling's text + binary DMel framing before adopting serve_codec.h. */
#define main inkling_main_unused
#include "../inkling.c"
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

static size_t read_output(FILE *stream, unsigned char *output, size_t capacity)
{
    assert(fflush(stream)==0); rewind(stream);
    return fread(output,1,capacity,stream);
}

static void reset_queue(void)
{
    for(int i=0;i<g_qn;i++) serve_request_dispose(&g_q[i]);
    memset(g_q,0,sizeof(g_q)); g_qn=0;
}

static void test_text_and_audio_submits_are_exact(void)
{
    static const unsigned char audio[]={0,1,2,3,'\n',5,6,7,'\r',9,10,11,12,13,14,15};
    static const char prompt[]="Hé\nx";
    unsigned char frame[256];
    int header=snprintf((char*)frame,sizeof(frame),"SUBMIT audio 0 %zu 4 0.25 0.9 %zu\n",
                        sizeof(prompt)-1,sizeof(audio));
    assert(header>0);
    memcpy(frame+header,prompt,sizeof(prompt)-1);
    memcpy(frame+header+sizeof(prompt)-1,audio,sizeof(audio));
    frame[header+sizeof(prompt)-1+sizeof(audio)]='\n';
    FILE *input=bytes_input(frame,(size_t)header+sizeof(prompt)+sizeof(audio));
    FILE *output=tmpfile(); assert(output); binary_stream(output);
    assert(serve_read_cmd(input,output,NULL)==0);
    assert(g_qn==1&&strcmp(g_q[0].id,"audio")==0);
    assert(g_q[0].plen==(int)sizeof(prompt)-1&&memcmp(g_q[0].payload,prompt,sizeof(prompt)-1)==0);
    assert(g_q[0].alen==(int)sizeof(audio)&&memcmp(g_q[0].audio,audio,sizeof(audio))==0);
    assert(fgetc(input)==EOF);
    unsigned char bytes[16]; assert(read_output(output,bytes,sizeof(bytes))==0);
    fclose(input); fclose(output); reset_queue();

    static const char text[]="SUBMIT text 7 3 2 0 1\na\nb\n";
    input=bytes_input(text,sizeof(text)-1); output=tmpfile(); assert(output); binary_stream(output);
    assert(serve_read_cmd(input,output,NULL)==0);
    assert(g_qn==1&&g_q[0].alen==0&&memcmp(g_q[0].payload,"a\nb",3)==0);
    fclose(input); fclose(output); reset_queue();
}

static void test_controls_and_queue_full_preserve_behavior(void)
{
    FILE *output=tmpfile(); assert(output); binary_stream(output);
    static const char cancel[]="CANCEL live\n";
    FILE *input=bytes_input(cancel,sizeof(cancel)-1);
    assert(serve_read_cmd(input,output,"live")==1); fclose(input);
    static const char stop[]="STOP live\n";
    input=bytes_input(stop,sizeof(stop)-1);
    assert(serve_read_cmd(input,output,"live")==0); fclose(input);

    g_qn=SRV_QMAX;
    static const char full[]="SUBMIT full 0 1 1 0 1 4\nxWXYZ\n";
    input=bytes_input(full,sizeof(full)-1);
    assert(serve_read_cmd(input,output,NULL)==0);
    assert(fgetc(input)==EOF);
    unsigned char bytes[64]; size_t count=read_output(output,bytes,sizeof(bytes));
    static const char want[]="ERROR full queue full\n";
    assert(count==sizeof(want)-1&&memcmp(bytes,want,count)==0);
    fclose(input); fclose(output); g_qn=0;
}

static void test_malformed_submit_cannot_become_control(void)
{
    static const char frame[]=
        "SUBMIT bad 0 12 0 0.7 0.95 4\nCANCEL liveWXYZ\n";
    FILE *input=bytes_input(frame,sizeof(frame)-1);
    FILE *output=tmpfile(); assert(output); binary_stream(output);
    assert(serve_read_cmd(input,output,"live")==-2);
    assert(fgetc(input)=='C');
    fclose(input); fclose(output);

    static const char bad_term[]=
        "SUBMIT broken 0 1 4 0.7 0.95 4\nxWXYZXSTOP live\n";
    input=bytes_input(bad_term,sizeof(bad_term)-1);
    output=tmpfile(); assert(output); binary_stream(output);
    assert(serve_read_cmd(input,output,"live")==-2);
    assert(fgetc(input)=='S');
    fclose(input); fclose(output);
}

int main(void)
{
    test_text_and_audio_submits_are_exact();
    test_controls_and_queue_full_preserve_behavior();
    test_malformed_submit_cannot_become_control();
    puts("inkling serve framing baseline: ok");
    return 0;
}
