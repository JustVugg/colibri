/* Managed server placement needs exact Linux range-list masks.  Keep this test
 * independent of sysfs and machine topology: it exercises the production
 * parser/mask representation directly, including sparse IDs beyond one
 * unsigned-long word. */
#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#ifdef __linux__
#include <sys/wait.h>
#endif

#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int failures;

#define CHECK(cond, fmt, ...) do {                                             \
    if(!(cond)){                                                               \
        fprintf(stderr, "FAIL: " fmt "\n", ##__VA_ARGS__);                    \
        failures++;                                                            \
    }                                                                          \
} while(0)

#ifdef __linux__
static void expect_invalid(const char *spec, unsigned long max_id){
    ColiIdMask mask={0};
    errno=0;
    CHECK(coli_idmask_parse(spec,max_id,&mask)<0,
          "accepted invalid range list %s",spec?spec:"(null)");
    CHECK(mask.words==NULL && mask.nwords==0 && mask.maxnode==0 && mask.count==0,
          "invalid parse left a partial mask for %s",spec?spec:"(null)");
    coli_idmask_free(&mask);
}
#endif

int main(void){
#ifndef __linux__
    puts("test_resource_masks: skipped (Linux only)");
    return 0;
#else
    const unsigned long word_bits=(unsigned long)(sizeof(unsigned long)*CHAR_BIT);
    ColiIdMask sparse={0};
    CHECK(coli_idmask_parse("0,2,65,130-132",4096,&sparse)==0,
          "sparse range list did not parse");
    CHECK(sparse.count==6,"selected count=%zu, expected 6",sparse.count);
    CHECK(sparse.maxnode==133,"maxnode=%lu, expected 133",sparse.maxnode);
    CHECK(sparse.nwords==(133+word_bits-1)/word_bits,
          "nwords=%zu does not cover maxnode 133",sparse.nwords);
    CHECK(coli_numa_policy_mode(1,1)==
              (COLI_MPOL_BIND|COLI_MPOL_F_STATIC_NODES),
          "explicit one-node policy is not a static bind");
    CHECK(coli_numa_policy_mode(2,1)==
              (COLI_MPOL_INTERLEAVE|COLI_MPOL_F_STATIC_NODES),
          "explicit multi-node policy is not static interleave");
    CHECK(coli_numa_policy_mode(1,0)==COLI_MPOL_INTERLEAVE,
          "legacy implicit policy unexpectedly changed");
    {
        const unsigned long present[]={0,2,65,130,131,132};
        const unsigned long absent[]={1,64,66,129,133};
        for(size_t i=0;i<sizeof(present)/sizeof(present[0]);i++)
            CHECK(coli_idmask_has(&sparse,present[i]),
                  "sparse mask lost ID %lu",present[i]);
        for(size_t i=0;i<sizeof(absent)/sizeof(absent[0]);i++)
            CHECK(!coli_idmask_has(&sparse,absent[i]),
                  "sparse mask gained ID %lu",absent[i]);
    }

    ColiIdMask allowed={0},outside={0};
    CHECK(coli_idmask_parse("0-2,65,130-132",4096,&allowed)==0,
          "allowed mask did not parse");
    CHECK(coli_idmask_is_subset(&sparse,&allowed),
          "valid selected mask is not a subset");
    CHECK(coli_idmask_parse("0,3",4096,&outside)==0,
          "outside mask did not parse");
    CHECK(!coli_idmask_is_subset(&outside,&allowed),
          "out-of-domain ID passed subset validation");

    ColiIdMask zero={0};
    CHECK(coli_idmask_parse("0",0,&zero)==0 && zero.count==1 && zero.maxnode==1,
          "zero-only domain did not parse");
    coli_idmask_free(&zero);

    ColiIdMask allowed_mems={0};
    CHECK(coli_idmask_read_status_field("Mems_allowed_list",
                                        COLI_IDMASK_MAX_ID,&allowed_mems)==0,
          "cannot parse /proc/self/status Mems_allowed_list: %s",strerror(errno));
    CHECK(allowed_mems.count>0,"Mems_allowed_list parsed as empty");

    expect_invalid(NULL,4096);
    expect_invalid("",4096);
    expect_invalid(" ",4096);
    expect_invalid(",0",4096);
    expect_invalid("0,",4096);
    expect_invalid("0,,1",4096);
    expect_invalid("-1",4096);
    expect_invalid("+1",4096);
    expect_invalid("1-",4096);
    expect_invalid("2-1",4096);
    expect_invalid("1--2",4096);
    expect_invalid("1 2",4096);
    expect_invalid("1, 2",4096);
    expect_invalid("1,1",4096);
    expect_invalid("1-3,3-5",4096);
    expect_invalid("1",0);
    expect_invalid("4097",4096);
    expect_invalid("999999999999999999999999999999",ULONG_MAX-1);

    /* Exercise the production apply+readback path in a child so this test does
     * not alter the test runner's own affinity.  sched_getcpu() necessarily
     * returns a CPU allowed to this process, making the case topology-neutral. */
    pid_t child=fork();
    CHECK(child>=0,"fork failed: %s",strerror(errno));
    if(child==0){
        int cpu=sched_getcpu();
        char spec[32];
        if(cpu<0) _exit(10);
        snprintf(spec,sizeof(spec),"%d",cpu);
        _exit(coli_cpu_affinity_apply(spec)?11:0);
    } else if(child>0){
        int status=0;
        CHECK(waitpid(child,&status,0)==child,"waitpid failed: %s",strerror(errno));
        CHECK(WIFEXITED(status) && WEXITSTATUS(status)==0,
              "managed CPU apply/readback child status=%d",status);
    }

    /* The engine entry point must honor the managed mask even when every OMP
     * tuning/re-exec gate is disabled. This was previously nested inside that
     * optional branch and silently bypassed by server-style overrides. */
    cpu_set_t inherited;
    CPU_ZERO(&inherited);
    if(!sched_getaffinity(0,sizeof(inherited),&inherited) &&
       CPU_COUNT(&inherited)>=2){
        int target=-1;
        for(int cpu=0;cpu<CPU_SETSIZE;cpu++)
            if(CPU_ISSET(cpu,&inherited)){ target=cpu; break; }
        pid_t gated=fork();
        CHECK(gated>=0,"affinity-gate fork failed: %s",strerror(errno));
        if(gated==0){
            char spec[32];
            char *argv[]={(char*)"colibri",NULL};
            snprintf(spec,sizeof(spec),"%d",target);
            setenv("COLI_CPU_AFFINITY",spec,1);
            setenv("COLI_NO_OMP_TUNE","1",1);
            setenv("COLI_CUDA","0",1);
            unsetenv("SNAP");
            int rc=coli_glm_main_unused(1,argv);
            cpu_set_t applied;
            CPU_ZERO(&applied);
            if(rc!=1 || sched_getaffinity(0,sizeof(applied),&applied) ||
               CPU_COUNT(&applied)!=1 || !CPU_ISSET(target,&applied))
                _exit(12);
            _exit(0);
        } else if(gated>0){
            int status=0;
            CHECK(waitpid(gated,&status,0)==gated,
                  "affinity-gate waitpid failed: %s",strerror(errno));
            CHECK(WIFEXITED(status) && WEXITSTATUS(status)==0,
                  "OMP-gated managed affinity child status=%d",status);
        }
    }

    coli_idmask_free(&outside);
    coli_idmask_free(&allowed);
    coli_idmask_free(&allowed_mems);
    coli_idmask_free(&sparse);
    if(failures) return 1;
    puts("OK exact CPU/NUMA range-list masks");
    return 0;
#endif
}
