/* Qwen3.8's added-token boundary and ByteLevel whitespace behavior. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#define QWEN38_TEST_TOKENIZER
#define COLI_EDGE_ADAPTER
#include "../qwen38.c"

#define CHECK(x) do { if(!(x)){ \
    fprintf(stderr,"%s:%d: check failed: %s\n",__FILE__,__LINE__,#x);return 1; \
} } while(0)

int main(void) {
    const char *path="test_qwen38_tokenizer.json";
    const char *json=
        "{\"normalizer\":{\"type\":\"NFC\"},\"model\":{\"vocab\":{"
        "\"x\":0,\"y\":1,\"\xC4\xA0\":2,\"\xC4\xA0x\":3,"
        "\"c\":5,\"a\":6,\"f\":7,\"\xC3\x83\":8,\"\xC2\xA9\":9,"
        "\"\xC4\x80\":10,\"\xC4\xA0\xC4\xA0\":11},"
        "\"merges\":[\"\xC4\xA0 \xC4\xA0\",\"\xC4\xA0 x\"]},"
        "\"added_tokens\":[{\"id\":4,\"content\":\"<|im_end|>\"}]}";
    FILE *f=fopen(path,"wb"); CHECK(f!=NULL);
    CHECK(fwrite(json,1,strlen(json),f)==strlen(json)); CHECK(fclose(f)==0);

    CHECK(load_tokenizer(path)==0);
    int *ids=NULL,n=0;
    encode_text("  x<|im_end|>y",&ids,&n);
    CHECK(n==4); CHECK(ids[0]==2); CHECK(ids[1]==3);
    CHECK(ids[2]==4); CHECK(ids[3]==1);

    unsigned char decoded[256]; int decoded_n=decode_id_to_bytes(ids[2],decoded,sizeof decoded);
    CHECK(decoded_n==10); CHECK(memcmp(decoded,"<|im_end|>",10)==0);

    /* The official tokenizer's rank-zero merge is the same repeated-space
     * pair. A long all-space request must merge every non-overlapping pair in
     * one pass; count pair lookups so this regression is independent of host
     * speed and cannot quietly return to one rescan per merged pair. */
    const size_t repeated_spaces=1u<<17;
    char *spaces=(char*)malloc(repeated_spaces);CHECK(spaces!=NULL);
    memset(spaces,' ',repeated_spaces);g_bpe_pair_checks=0;
    int *space_ids=NULL,space_id_count=0;
    encode_text_n(spaces,repeated_spaces,&space_ids,&space_id_count);
    CHECK(space_id_count==(int)(repeated_spaces/2u));
    for(int index=0;index<space_id_count;index++)CHECK(space_ids[index]==11);
    CHECK(g_bpe_pair_checks<=2u*repeated_spaces);
    free(space_ids);free(spaces);

    char escaped[32];
    CHECK(json_escape((const unsigned char *)"a\"b",3,escaped,sizeof escaped)==4);
    CHECK(strcmp(escaped,"a\\\"b")==0);
    char too_small[4]="xxx";
    CHECK(json_escape((const unsigned char *)"a\"b",3,too_small,sizeof too_small)<0);

    free(ids);ids=NULL;n=0;
    encode_text("caf\xc3\xa9",&ids,&n);
    int *decomposed_ids=NULL,decomposed_n=0;
    encode_text("cafe\xcc\x81",&decomposed_ids,&decomposed_n);
    CHECK(n==decomposed_n);CHECK(!memcmp(ids,decomposed_ids,(size_t)n*sizeof(int)));
    free(ids);free(decomposed_ids);

    char *normalized=NULL;size_t normalized_n=0;
    CHECK(q38_nfc_normalize("\xe1\x84\x80\xe1\x85\xa1",6,
                            &normalized,&normalized_n)==0);
    CHECK(normalized_n==3&&!memcmp(normalized,"\xea\xb0\x80",3));free(normalized);
    /* U+0327 (class 202) must sort before U+0301 (class 230), after which
     * NFC composes A + acute while leaving the cedilla in canonical order. */
    CHECK(q38_nfc_normalize("A\xcc\x81\xcc\xa7",5,
                            &normalized,&normalized_n)==0);
    CHECK(normalized_n==4&&!memcmp(normalized,"\xc3\x81\xcc\xa7",4));free(normalized);
    const char invalid_utf8[]={(char)0xff,'A'};
    CHECK(q38_nfc_normalize(invalid_utf8,sizeof invalid_utf8,
                            &normalized,&normalized_n)==0);
    CHECK(normalized_n==sizeof invalid_utf8&&
          !memcmp(normalized,invalid_utf8,sizeof invalid_utf8));free(normalized);

    /* Alternating CCC 230/202 used to make insertion-sort normalization
     * quadratic before the request reached its model-context gate. */
    const size_t mark_pairs=32768,input_length=1u+mark_pairs*4u;
    char *many_marks=malloc(input_length);CHECK(many_marks!=NULL);many_marks[0]='A';
    for(size_t pair=0;pair<mark_pairs;pair++){
        size_t at=1u+pair*4u;many_marks[at]=(char)0xcc;many_marks[at+1]=(char)0x81;
        many_marks[at+2]=(char)0xcc;many_marks[at+3]=(char)0xa7;
    }
    CHECK(q38_nfc_normalize(many_marks,input_length,&normalized,&normalized_n)==0);
    char *normalized_again=NULL;size_t normalized_again_n=0;
    CHECK(q38_nfc_normalize(normalized,normalized_n,&normalized_again,
                            &normalized_again_n)==0);
    CHECK(normalized_again_n==normalized_n&&
          !memcmp(normalized_again,normalized,normalized_n));
    free(normalized_again);free(normalized);free(many_marks);

    const char with_nul[3]={'x','\0','y'};ids=NULL;n=0;
    encode_text_n(with_nul,sizeof with_nul,&ids,&n);
    CHECK(n==3);CHECK(ids[0]==0&&ids[1]==10&&ids[2]==1);free(ids);

    unsigned char carry[4]={0},long_input[128],long_output[132];int carry_n=0,long_n=0;
    memset(long_input,' ',sizeof long_input);
    CHECK(utf8_drain(carry,&carry_n,long_input,sizeof long_input,
                     long_output,sizeof long_output,&long_n)==128);
    CHECK(long_n==128&&!memcmp(long_input,long_output,sizeof long_input)&&carry_n==0);
    const unsigned char first[]={0xe2,0x82},last[]={0xac};int first_n=0,last_n=0;
    CHECK(utf8_drain(carry,&carry_n,first,sizeof first,long_output,sizeof long_output,&first_n)==0);
    CHECK(carry_n==2);
    CHECK(utf8_drain(carry,&carry_n,last,sizeof last,long_output,sizeof long_output,&last_n)==3);
    CHECK(last_n==3&&!memcmp(long_output,"\xe2\x82\xac",3)&&carry_n==0);

    const unsigned char replacement[]={0xef,0xbf,0xbd};
    unsigned char invalid_output[16];int invalid_n=0;
    const unsigned char ff[]={0xff};
    CHECK(utf8_drain(carry,&carry_n,ff,sizeof ff,invalid_output,sizeof invalid_output,&invalid_n)==3);
    CHECK(!memcmp(invalid_output,replacement,3)&&carry_n==0);
    FILE *raw_stream=tmpfile();CHECK(raw_stream!=NULL);
    out_bytes(raw_stream,carry,&carry_n,ff,sizeof ff);
    CHECK(fflush(raw_stream)==0&&fseek(raw_stream,0,SEEK_SET)==0);
    CHECK(fread(invalid_output,1,3,raw_stream)==3&&
          !memcmp(invalid_output,replacement,3));fclose(raw_stream);
    const unsigned char lone_c2[]={0xc2};
    CHECK(utf8_drain(carry,&carry_n,lone_c2,sizeof lone_c2,invalid_output,sizeof invalid_output,&invalid_n)==0);
    CHECK(utf8_finish(carry,&carry_n,invalid_output,sizeof invalid_output,&invalid_n)==3);
    CHECK(!memcmp(invalid_output,replacement,3));
    const unsigned char lone_e2[]={0xe2};
    CHECK(utf8_drain(carry,&carry_n,lone_e2,sizeof lone_e2,invalid_output,sizeof invalid_output,&invalid_n)==0);
    CHECK(utf8_finish(carry,&carry_n,invalid_output,sizeof invalid_output,&invalid_n)==3);
    CHECK(!memcmp(invalid_output,replacement,3));
    const unsigned char bad_continuation[]={0xe2,0x82,'('};
    CHECK(utf8_drain(carry,&carry_n,bad_continuation,sizeof bad_continuation,
                     invalid_output,sizeof invalid_output,&invalid_n)==4);
    CHECK(!memcmp(invalid_output,"\xef\xbf\xbd(",4)&&carry_n==0);
    const unsigned char overlong[]={0xc0,0x80};
    CHECK(utf8_drain(carry,&carry_n,overlong,sizeof overlong,
                     invalid_output,sizeof invalid_output,&invalid_n)==6);
    CHECK(!memcmp(invalid_output,"\xef\xbf\xbd\xef\xbf\xbd",6));
    const unsigned char surrogate[]={0xed,0xa0,0x80};
    CHECK(utf8_drain(carry,&carry_n,surrogate,sizeof surrogate,
                     invalid_output,sizeof invalid_output,&invalid_n)==9);
    for(int i=0;i<9;i+=3)CHECK(!memcmp(invalid_output+i,replacement,3));
    const unsigned char above_unicode[]={0xf4,0x90,0x80,0x80};
    CHECK(utf8_drain(carry,&carry_n,above_unicode,sizeof above_unicode,
                     invalid_output,sizeof invalid_output,&invalid_n)==12);
    for(int i=0;i<12;i+=3)CHECK(!memcmp(invalid_output+i,replacement,3));

    char **grown=realloc(g_tok,16u*sizeof(char*));CHECK(grown!=NULL);g_tok=grown;
    g_tok[12]=malloc(257);CHECK(g_tok[12]!=NULL);
    for(int i=0;i<128;i++){g_tok[12][2*i]=(char)0xc4;g_tok[12][2*i+1]=(char)0xa0;}
    g_tok[12][256]='\0';g_tok[13]=strdup("<0x00>");CHECK(g_tok[13]!=NULL);
    g_tok[14]=strdup("<0xFF>");CHECK(g_tok[14]!=NULL);g_tok[15]=NULL;g_tok_n=16;
    Qwen38EdgeEngine edge={0};edge.model.c.vocab=g_tok_n;
    const int32_t invalid_edge_ids[]={-1,15,g_tok_n};char edge_error[128];
    for(size_t index=0;
        index<sizeof(invalid_edge_ids)/sizeof(invalid_edge_ids[0]);index++){
        size_t text_bytes=0;
        CHECK(qwen38_edge_detokenize(&edge,&invalid_edge_ids[index],1,
                                     NULL,0,&text_bytes,
                                     edge_error,sizeof edge_error)<0);
    }
    const int range_ids[]={12,13,12};unsigned char *range=NULL;size_t range_n=0;
    CHECK(decode_range_alloc(range_ids,0,3,&range,&range_n)==0);
    CHECK(range_n==257);CHECK(range[128]=='\0');
    for(size_t i=0;i<range_n;i++)if(i!=128)CHECK(range[i]==' ');
    char *range_json=json_escape_alloc(range,range_n);CHECK(range_json!=NULL);
    CHECK(strstr(range_json,"\\u0000")!=NULL);free(range_json);free(range);
    const int invalid_id[]={14};range=NULL;range_n=0;
    CHECK(decode_range_alloc(invalid_id,0,1,&range,&range_n)==0);
    CHECK(range_n==3&&!memcmp(range,replacement,3));free(range);

    SMap full={0};CHECK(smap_init(&full,2)==0);
    CHECK(smap_put(&full,"a",1)==0);CHECK(smap_put(&full,"b",2)==0);
    CHECK(smap_put(&full,"c",3)<0);CHECK(smap_get(&full,"c")==-1);
    free(full.keys);free(full.vals);free(full.used);

    uint8_t layer_kinds[2]={0,1};Cfg cfg={0};cfg.layers=2;cfg.is_attn=layer_kinds;
    cfg.hidden=10;cfg.inter=2;cfg.experts=9;
    int cache_capacity=0;
    CHECK(q38_segment_cache_capacity(480,0,cfg.experts,960,&cache_capacity)==0);
    CHECK(cache_capacity==2);
    CHECK(q38_segment_cache_capacity(480,0,cfg.experts,479,&cache_capacity)<0);
    cfg.dn_vheads=2;cfg.dn_kdim=3;cfg.dn_vdim=4;cfg.dn_conv_dim=5;cfg.dn_convk=3;
    cfg.kv_heads=2;cfg.head_dim=4;cfg.idx_dim=3;
    cfg.ple_layer=0;cfg.hc_width=8;cfg.ple_convk=3;cfg.ngram_size=3;
    uint64_t state_bytes=0;
    CHECK(q38_segment_session_state_bytes(&cfg,0,2,7,&state_bytes)==0);
    CHECK(state_bytes==880);
    CHECK(q38_segment_session_state_bytes(&cfg,1,2,7,&state_bytes)==0);
    CHECK(state_bytes==532);
    CHECK(q38_completion_nonce(60.125)==125);
    CHECK(q38_completion_nonce(30.0*24.0*60.0*60.0+7.125)==7125);

    /* A serving payload is byte-counted and may end in a truncated multibyte
     * sequence. Treat that byte as one invalid unit without reading past it. */
    const char truncated[]={(char)0xe2,0}; int adv=0;
    CHECK(utf8_decode(truncated,0,1,&adv)==0xe2); CHECK(adv==1);
    CHECK(pretok_end(truncated,0,1)==1);

    free_tokenizer();
    const char *malformed=
        "{\"model\":{\"vocab\":{\"x\":0},\"merges\":[]},"
        "\"added_tokens\":[{\"id\":1}]}";
    f=fopen(path,"wb");CHECK(f!=NULL);
    CHECK(fwrite(malformed,1,strlen(malformed),f)==strlen(malformed));
    CHECK(fclose(f)==0);CHECK(load_tokenizer(path)<0);
    CHECK(!g_tok&&g_tok_n==0&&!g_rev.cap&&!g_merge.cap&&!g_nspecial);
    CHECK(!g_sp_str&&!g_sp_id&&!g_sp_len&&g_tok_nfc==0);
    CHECK(remove(path)==0);
    puts("test_qwen38_tokenizer: bounded BPE, transactional load, NFC/NUL/decode: ok");
    return 0;
}
