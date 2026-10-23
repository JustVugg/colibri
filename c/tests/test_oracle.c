#include <stdio.h>
#include <limits.h>
#include "../oracle.h"

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#c); return 1; \
} } while (0)

int main(void) {
    OracleRef r;
    const char *good="{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[2,3]}";
    CHECK(oracle_ref_parse(good,4,1,&r));
    CHECK(r.np==1 && r.nfull==2 && r.prompt[0]==1 && r.full[1]==2 && r.tf[1]==3);
    oracle_ref_free(&r);
    CHECK(oracle_ref_parse("{\"prompt_ids\":[1],\"full_ids\":[1,2]}",4,0,&r));
    CHECK(r.tf==NULL);
    oracle_ref_free(&r);

    /* Missing/truncated/wrongly typed predictions used to reach tf[i] unchecked. */
    const char *bad[]={
        "{}",
        "[]",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[2]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[2,3,1]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":{\"x\":2,\"y\":3}}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[2,3]",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[2,3}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[2,3,]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[2,3]} garbage",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[2 3]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\" [2,3]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[2,3],\"tf_pred\":[1,1]}",
        "{\"prompt_ids\":[1,2],\"full_ids\":[1],\"tf_pred\":[2]}",
        "{\"prompt_ids\":[2],\"full_ids\":[1,2],\"tf_pred\":[2,3]}",
        "{\"prompt_ids\":[],\"full_ids\":[1,2],\"tf_pred\":[2,3]}",
        "{\"prompt_ids\":[-1],\"full_ids\":[-1,2],\"tf_pred\":[2,3]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,4],\"tf_pred\":[2,3]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[true,3]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[\"2\",3]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[2.5,3]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[1e999,3]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[NaN,3]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[+2,3]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[02,3]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[2.,3]}",
        "{\"prompt_ids\":[1],\"full_ids\":[1,2],\"tf_pred\":[0x2,3]}",
    };
    for (size_t i=0; i<sizeof(bad)/sizeof(bad[0]); i++) {
        CHECK(!oracle_ref_parse(bad[i],4,1,&r));
        CHECK(!r.prompt && !r.full && !r.tf);
    }
    CHECK(!oracle_ref_parse(good,0,1,&r));
    CHECK(!oracle_ref_parse(NULL,4,1,&r));
    CHECK(!oracle_ref_parse("{\"prompt_ids\":[1],\"full_ids\":[1]}",4,0,&r));
    CHECK(oracle_ref_parse("{\"prompt_ids\":[1],\"full_ids\":[1],\"tf_pred\":[2]}",4,1,&r));
    oracle_ref_free(&r);

    /* Even a non-winning NaN/Inf must invalidate the model output. */
    float logits[]={2.0f,1.0f,-3.0f};
    CHECK(oracle_logits_finite(logits,3));
    logits[2]=NAN; CHECK(!oracle_logits_finite(logits,3));
    logits[2]=INFINITY; CHECK(!oracle_logits_finite(logits,3));
    logits[2]=-INFINITY; CHECK(!oracle_logits_finite(logits,3));

    int allowed=-1;
    CHECK(oracle_tf_allowance(NULL,32,&allowed) && allowed==0);
    CHECK(oracle_tf_allowance("0",32,&allowed) && allowed==0);
    CHECK(oracle_tf_allowance("2",32,&allowed) && allowed==2);
    CHECK(oracle_tf_allowance("31",32,&allowed) && allowed==31);
    CHECK(oracle_tf_allowance("0",1,&allowed) && allowed==0);
    CHECK(!oracle_tf_allowance("2",2,&allowed));
    CHECK(!oracle_tf_allowance(NULL,0,&allowed));
    const char *bad_allowance[]={"", "-1", "+2", " 2", "2 ", "2x", "2.0", "32", "33",
                                 "9999999999999999999999999999999999999999"};
    for (size_t i=0; i<sizeof(bad_allowance)/sizeof(bad_allowance[0]); i++)
        CHECK(!oracle_tf_allowance(bad_allowance[i],32,&allowed) && allowed==0);
    char large[32];
    snprintf(large,sizeof(large),"%d",INT_MAX-1);
    CHECK(oracle_tf_allowance(large,INT_MAX,&allowed) && allowed==INT_MAX-1);
    puts("oracle input and finite-logit tests: ok");
    return 0;
}
