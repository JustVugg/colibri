/* fmt=8 (native FP8-e4m3 passthrough) loader-seam tests.
 *
 * fmt=8, PUBLIC ordinal: this format was minted fmt=6 during original
 * development of this branch, before dev's own #465 (E8/IQ3) claimed that
 * ordinal upstream and merged it into dev as a REAL fmt=6 (see quant.h's E8
 * constants and e8_ helper functions, and qt_resolve_fmt's ns==4-tag early
 * check); re-tagged fmt=100 (PRIVATE ORDINAL BLOCK, see colibri.c's QT struct
 * comment) from that point forward -- there was never a build in this branch's
 * history where this format was reachable as fmt=6 -- graduated to fmt=7 when
 * the maintainer assigned that ordinal on #524, and renumbered to fmt=8 after
 * #705 merged claiming 7 for MXFP4 (see colibri.c's QT comment).
 *
 * Part A: qt_resolve_fmt disambiguation suite -- THE DESIGN LANDMINE. fmt=8
 * weight bytes are byte-identical to fmt=1 (int8): both are O*I raw bytes.
 * The two are told apart ONLY by the scale array's byte count (per-row O*4 for
 * fmt=1, per-128x128-block ceil(O/128)*ceil(I/128)*4 for fmt=8, THIS build's
 * implemented f32 scale encoding). For some shapes those two counts coincide
 * exactly -- INVERSION (maintainer review, #528): qt_resolve_fmt used to
 * REFUSE (exit(1)) this ambiguous case; it now resolves to fmt=1 (the
 * incumbent, already-on-disk, decodable format) instead, because the
 * collision is not hypothetical (GLM-5.2's own self_attn.o_proj.weight hits
 * it, see qt_resolve_fmt's own "REVIEW FINDING"/"INVERSION" comment) and the
 * writer side (repack_fp8_passthrough.py's _check_geometry) now refuses to
 * ever EMIT an fmt=8 container at this same shape, so an unstamped ambiguous
 * tensor reaching this function is never a genuine fmt=8 candidate. The
 * former refusal-testing convention (fork()+waitpid(), mirroring
 * tests/test_st_pread.c's exit(1)-path idiom) is kept for the OTHER
 * refusing cases below (Part A2's fmt=6 collision, Part A3's UE8M0
 * recognized-not-implemented refusal, and the generic garbage-byte-count
 * refusal) -- only the is_row&&is_blk collision in this Part flipped from
 * expect_refuse to expect_fmt(...,1,...).
 *
 * Part A2: fmt=6 (E8/IQ3, upstream #465) vs fmt=8 collision at [O<=128 or
 * O in a 128-block-count range, I=98] -- SECOND DESIGN LANDMINE. Unchanged by
 * the #528 inversion above (a different collision predicate, still refused).
 *
 * Part A3: fmt=8's scale ENCODING is a declared property, not a hardcoded
 * constant -- f32 (Part A/A2 above) is what this build implements. A UE8M0
 * (1 byte/block) encoding is a REAL, distinct byte signature (the DeepSeek-V4
 * checkpoint format for this identical weight geometry) this build recognizes
 * and refuses BY NAME rather than misreading. Unchanged by the #528
 * inversion (a stamp confirms the WEIGHT format, never a decoder this build
 * doesn't have -- see qt_resolve_fmt's own comment).
 *
 * Part B: qt_from_disk loader-seam -- writes a real single-shard .safetensors
 * file containing an fmt=8 tensor (U8 weight + per-block F32 .qs) next to an
 * int8 control tensor of a DIFFERENT, non-colliding shape, loads both through
 * qt_from_disk, and checks the byte-count/.qs-size inference picks fmt=8 vs
 * fmt=1 correctly and the loaded weights dequantize identically to a reference.
 * Mirrors tests/test_int3_load.c's structure for fmt=5.
 *
 * Part C: qt_bytes()/qt_scale_bytes() byte-accounting for fmt=8, plus
 * qt_wire_split() -- the shared weight/scale byte-range split qt_wire_mmap
 * and qt_unwire_mmap both now call (maintainer review, #528: qt_scale_bytes()
 * existed correctly but neither call site actually used it, a defect only
 * -Wno-unused-function's suppression let compile clean). check_wire_split()
 * exercises qt_wire_split() itself directly; test_wire_site_regression()
 * (FIX ROUND, validator finding) additionally exercises the real
 * qt_wire_mmap()/qt_unwire_mmap() call sites through a mem_wire()/munlock()
 * observer seam (defined right below the #include below) -- a mutation that
 * reverts ONLY those two call sites back to the old scale_b=(int64_t)t->O*4
 * hardcode, leaving qt_wire_split() itself untouched, is invisible to
 * check_wire_split() but fails test_wire_site_regression() (proven by
 * actually running that exact mutation -- see the report).
 *
 * Part D: metadata-stamp TRUST-VERIFY-REFUSE (qt_verify_fmt_stamp, colibri.c)
 * and stamp-RESOLVES-ambiguity (qt_resolve_fmt's `stamped_name` parameter) --
 * this PR (registry + metadata stamp) makes the stamp load-bearing: Parts A
 * and A2's collision cases gain stamped variants below (a stamp naming
 * exactly one live candidate resolves what an absent stamp still refuses --
 * or, for Part A's is_row&&is_blk collision specifically, what an absent
 * stamp now resolves to fmt=1 anyway, see the #528 INVERSION note above; the
 * stamped sub-case there is UNCHANGED by that inversion, still
 * TRUST-VERIFY-REFUSE), and Part D covers the four stamp outcomes on an
 * otherwise-unambiguous tensor (agreeing/mismatching/unrecognized-name/
 * absent). A stamp can never grant this build a decoder it doesn't have:
 * Part A3's UE8M0 refusals stay refusals even when correctly stamped
 * "fp8-e4m3-b128" (see test_ue8m0_scale_refusal's stamped cases). */
/* WIRE-SITE REGRESSION SEAM (FIX ROUND, validator finding). First attempt
 * (superseded, kept as a note): renaming mem_wire() itself via macro does
 * NOT work as an observer seam -- mem_wire is a real, internally-defined
 * static function, so the SAME rename that frees up the name "mem_wire"
 * for a shadow ALSO renames every CALL SITE (qt_wire_mmap's) to the new
 * name, meaning qt_wire_mmap ends up calling the renamed-but-still-real
 * function directly, bypassing any shadow defined under the old name
 * entirely (confirmed by inspecting the preprocessed output -- caught
 * before it could hide a broken test). The seam that actually works is one
 * level lower: mlock()/munlock() themselves are EXTERNAL POSIX library
 * functions with no body anywhere in this translation unit (only a
 * declaration, via <sys/mman.h>, plus mem_wire's/qt_unwire_mmap's own call
 * sites) -- renaming them redirects those call sites to a name THIS FILE
 * provides its own (self-contained) definition for, with no real
 * implementation being shadowed out of existence. This observes the EXACT
 * (addr,len) qt_wire_mmap (via mem_wire) and qt_unwire_mmap actually pass
 * down to the platform lock/unlock call -- not a reimplementation of what
 * they SHOULD pass, and immune to a future mem_wire refactor since the
 * seam sits at the syscall boundary, not the wrapper. #ifndef _WIN32:
 * mlock/munlock are only called on this `#if defined(__APPLE__) ||
 * defined(__linux__) || defined(__FreeBSD__)` arm; Windows uses
 * compat_mlock/compat_munlock instead (untouched here, matching this
 * file's existing POSIX-only test-seam convention -- the fork/pipe/waitpid
 * refusal tests below skip analogously on Windows). */
#ifndef _WIN32
#define mlock test_mlock_seam
#define munlock test_munlock_seam
#endif
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main
#ifndef _WIN32
#undef mlock
#undef munlock
#endif

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>      /* open(/dev/null) in Part G's sweep probe */
#endif

/* Shadow definitions for the seam above -- must come after the #include so
 * mem_wire's (unmodified, real) call to mlock() and qt_unwire_mmap's
 * (unmodified, real) call to munlock() have already been renamed to these
 * names by the #define above. Neither mlock() nor munlock() has a body
 * anywhere in this translation unit (both are declared only, via
 * <sys/mman.h>) -- these are the ONLY definitions the renamed call sites
 * can resolve to, both self-contained: returning 0 (success) without
 * actually locking/unlocking anything is fine for this test, since the
 * buffer under test is never really mlocked in the first place (mirrors
 * check_fp8_bytes'/check_wire_split's own stated reasoning that the real
 * RLIMIT_MEMLOCK-gated syscall's success is environment-dependent and not
 * what any of these tests need to prove). Non-static: both must match the
 * extern linkage of the (renamed) declarations <sys/mman.h> already left in
 * this translation unit -- a static definition here would conflict with
 * that non-static declaration ("static declaration follows non-static
 * declaration"). g_seam_wire_* observes mem_wire's (hence qt_wire_mmap's)
 * calls; g_seam_unwire_* observes qt_unwire_mmap's direct calls. */
#ifndef _WIN32
static void *g_seam_wire_addr[4]; static size_t g_seam_wire_len[4]; static int g_seam_wire_n;
int test_mlock_seam(const void *addr, size_t len){
    if(g_seam_wire_n < 4){ g_seam_wire_addr[g_seam_wire_n]=(void*)addr; g_seam_wire_len[g_seam_wire_n]=len; }
    g_seam_wire_n++;
    return 0;
}
static void *g_seam_unwire_addr[4]; static size_t g_seam_unwire_len[4]; static int g_seam_unwire_n;
int test_munlock_seam(const void *addr, size_t len){
    if(g_seam_unwire_n < 4){ g_seam_unwire_addr[g_seam_unwire_n]=(void*)addr; g_seam_unwire_len[g_seam_unwire_n]=len; }
    g_seam_unwire_n++;
    return 0;
}
#endif

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

static uint64_t rng = 0xFEEDFACE0DDBA11ull;
static float rndf(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return ((int64_t)(rng & 0xFFFFF) - 0x80000) / (float)0x80000; }
static uint8_t rndbyte_nonan(void){
    for(;;){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        uint8_t b = (uint8_t)(rng & 0xFF);
        if(b != 0x7F && b != 0xFF) return b; }
}

/* ---- Part A: qt_resolve_fmt disambiguation (in-process for non-refusing
 * cases, fork+waitpid for refusing ones -- qt_resolve_fmt exit(1)s in place,
 * it does not return an error code). ---- */

static int expect_fmt_stamped(int O, int I, int64_t nb, int64_t ns, const char *stamped_name, int expect_fmt_val, const char *tag){
    int gs=0;
    int fmt = qt_resolve_fmt(tag, O, I, nb, ns, &gs, stamped_name);
    if(fmt != expect_fmt_val){
        printf("FAIL %s: got fmt=%d, expected fmt=%d (O=%d I=%d nb=%lld ns=%lld)\n",
               tag, fmt, expect_fmt_val, O, I, (long long)nb, (long long)ns);
        return 0;
    }
    return 1;
}
static int expect_fmt(int O, int I, int64_t nb, int64_t ns, int expect_fmt_val, const char *tag){
    return expect_fmt_stamped(O, I, nb, ns, NULL, expect_fmt_val, tag);
}

static int expect_refuse_stamped(int O, int I, int64_t nb, int64_t ns, const char *stamped_name, const char *tag){
#ifndef _WIN32
    int pipefd[2]; if(pipe(pipefd)!=0) return 0;
    pid_t pid = fork();
    if(pid < 0) return 0;
    if(pid == 0){
        dup2(pipefd[1],2); close(pipefd[0]); close(pipefd[1]);
        int gs=0;
        qt_resolve_fmt(tag, O, I, nb, ns, &gs, stamped_name);   /* must exit(1) inside; must NOT return */
        _exit(42);                                  /* reaching here is the bug */
    }
    close(pipefd[1]);
    char err[1024]={0}; size_t eoff=0; ssize_t n; /* drain to EOF: a single read() can return SHORT on Linux pipes (glibc unbuffered stderr arrives in chunks) -- truncated the refusal message past the marker, CI-caught */ while(eoff<sizeof(err)-1 && (n=read(pipefd[0],err+eoff,sizeof(err)-1-eoff))>0) eoff+=(size_t)n;
    close(pipefd[0]);
    int status=0; waitpid(pid,&status,0);
    int ok = WIFEXITED(status) && WEXITSTATUS(status)==1;
    if(!ok){
        printf("FAIL %s: expected exit(1) refusal, got status=%d, stderr=%.200s\n", tag, status, err);
        return 0;
    }
    /* Search the ENGINE's message, not this test's own words: qt_resolve_fmt
     * prints "%s: ..." with the caller-supplied name -- which is `tag` here --
     * so a tag containing "refuses" would satisfy a naive strstr all by
     * itself and turn this assertion into a tautology (fix round 1, found
     * while pinning the ue8m0 family: the ue8m0 refusal message itself
     * lacked any "refus" wording and the check never noticed). Skip the
     * echoed tag prefix before searching. */
    const char *msg = err; size_t tl = strlen(tag);
    if(!strncmp(err, tag, tl)) msg = err + tl;
    if(!strstr(msg,"refus")){
        printf("FAIL %s: exited(1) but message lacked a refusal explanation: %.200s\n", tag, err);
        return 0;
    }
    return 1;
#else
    /* fork/pipe/waitpid are POSIX -- mirrors tests/test_st_pread.c's own
     * Windows arm: skip the exit(1) subprocess check, print an explicit
     * (never silent) skip line per case, count it as passing rather than
     * failing. Every non-refusing case in this file (expect_fmt and its
     * callers) still runs and asserts for real on Windows. */
    printf("skipped on Windows (no fork): %s\n", tag);
    (void)O; (void)I; (void)nb; (void)ns; (void)stamped_name;
    return 1;
#endif
}
static int expect_refuse(int O, int I, int64_t nb, int64_t ns, const char *tag){
    return expect_refuse_stamped(O, I, nb, ns, NULL, tag);
}

static void test_disambiguation(void){
    /* --- non-degenerate golden paths (unambiguous either way) --- */
    CHECK(expect_fmt(4096,4096,(int64_t)4096*4096,4096*4,1,"plain int8 4096x4096"));
    /* spec's own worked example: [2048,6144] expert -> block scale [16,48] */
    CHECK(expect_fmt(2048,6144,(int64_t)2048*6144,16LL*48*4,8,"fp8 2048x6144 (spec example)"));
    CHECK(expect_fmt(384,6144,(int64_t)384*6144,3LL*48*4,8,"fp8 384x6144 non-square block grid"));

    /* --- REGRESSION (maintainer review, #528): GLM-5.2's own
     * self_attn.o_proj.weight, [D,H*v_head]=[6144,16384]. nblkO=ceil(6144/128)=48,
     * nblkI=ceil(16384/128)=128, product=6144==O -- this shape hits the
     * is_row&&is_blk collision below on every GLM-5.2 checkpoint, and is a
     * REAL, pre-existing, valid int8-row tensor, not a hypothetical. Confirmed
     * against this repo's own v1 container header (o_proj weight U8
     * 100,663,296 B == 6144*16384; .qs scale blob 24,576 B == 6144*4 ==
     * 48*128*4, both at once) -- see the CENSUS SCAN
     * (tools/fp8_collision_census.py) for the full enumerated family. An
     * earlier revision of qt_resolve_fmt refused this shape unconditionally
     * (exit(1) at load time on an ordinary model); the INVERSION resolves it
     * to fmt=1 instead -- this is that non-refusal, asserted directly. */
    CHECK(expect_fmt(6144,16384,(int64_t)6144*16384,6144LL*4,1,
        "GLM-5.2 o_proj shape [6144,16384]: valid int8-row, ambiguous-by-byte-count, resolves fmt=1 (NOT a refusal)"));

    /* --- degenerate shapes: O<=128 makes nblkO==1, so ns_blk==nblkI*4 can
     * equal ns_row==O*4 whenever nblkI==O. Exhaustive-in-spirit sweep of the
     * boundary the design landmine describes. INVERSION (maintainer review,
     * #528): every one of these used to be an expect_refuse (exit(1)) -- the
     * general family this collision predicate describes (any int8 tensor
     * where O==ceil(O/128)*ceil(I/128)) includes real, valid, pre-existing
     * tensors (see the o_proj case just above), so refusing was the wrong
     * call across the board, not just for the three shapes the maintainer's
     * review named explicitly ([128,16384],[256,16384],[384,16384] below) --
     * every case in this sweep hits the exact same is_row&&is_blk branch and
     * is flipped here for the same reason. --- */
    CHECK(expect_fmt(1,1,      1,        4,  1, "degenerate O=1 I=1 (nblkI=1=O) -> fmt=1, was refuse"));
    CHECK(expect_fmt(1,128,    128,      4,  1, "degenerate O=1 I=128 (nblkI=1=O, I at block edge) -> fmt=1, was refuse"));
    CHECK(expect_fmt(2,256,    2LL*256,  8,  1, "degenerate O=2 I=256 (nblkI=2=O) -> fmt=1, was refuse"));
    CHECK(expect_fmt(6,768,    6LL*768,  24, 1, "degenerate O=6 I=768 (nblkI=6=O) -> fmt=1, was refuse"));
    /* the three shapes the maintainer's review named explicitly (his exact expect_refuse
     * calls, O=128/256/384 at I=16384) -- FLIPPED to assert fmt=1. */
    CHECK(expect_fmt(128,16384,128LL*16384, 512, 1, "degenerate O=128 I=16384 (nblkO=1,nblkI=128=O) -> fmt=1, was refuse"));
    /* O>128 degenerate case: nblkO=2, need nblkI=O/2 -- O=256,I=16384 -> nblkI=128, 2*128=256=O */
    CHECK(expect_fmt(256,16384,256LL*16384, 1024, 1, "degenerate O=256 I=16384 (nblkO=2,nblkI=128, product=O) -> fmt=1, was refuse"));
    /* k=3: the ambiguity isn't a one-off O=256 coincidence -- it's structural for ANY
     * O that's a multiple of 128 (nblkO=k), since nblkI==128 (I in (16256,16384]) always
     * makes nblkO*nblkI == k*128 == O. One more multiple (O=384=3*128) confirms the
     * condition generalizes past the k=2 worked example, not just a re-derivation. */
    CHECK(expect_fmt(384,16384,384LL*16384, 1536, 1, "degenerate O=384 I=16384 (nblkO=3,nblkI=128, k=3, product=O) -> fmt=1, was refuse"));

    /* --- boundary-ADJACENT non-degenerate cases: one step past each
     * degenerate case above, both interpretations now legitimately resolve. --- */
    CHECK(expect_fmt(1,129, 129,   4, 1, "adjacent O=1 I=129 as fmt=1 (ns=row)"));
    CHECK(expect_fmt(1,129, 129,   8, 8, "adjacent O=1 I=129 as fmt=8 (ns=block, nblkI=2)"));
    CHECK(expect_fmt(2,257, 2LL*257, 8,  1, "adjacent O=2 I=257 as fmt=1 (ns=row)"));
    CHECK(expect_fmt(2,257, 2LL*257, 12, 8, "adjacent O=2 I=257 as fmt=8 (ns=block, nblkI=3)"));

    /* --- neither interpretation matches: garbage .qs size, must still refuse
     * (the pre-existing generic-mismatch path, exercised through the fmt=8-aware
     * function to confirm the new code didn't disturb it). --- */
    CHECK(expect_refuse(10,10, 100, 999, "garbage ns matches neither row nor block layout"));
}

/* ---- Part A2: fmt=6 (E8/IQ3, upstream #465, merged into dev) vs fmt=8
 * (this branch's fp8-e4m3-b128) collision -- SECOND DESIGN LANDMINE, see the
 * derivation in qt_resolve_fmt's own comment. e8_rowbytes(I) is the constant
 * 98 for every I in (0,256], so dev's fmt=6 tag check (ns==4 &&
 * nb==O*e8_rowbytes(I)) collapses to nb==O*98 -- which coincides with
 * fp8-e4m3-b128's raw weight bytes (O*I) at the ONE value I==98, where a
 * SINGLE-BLOCK (O<=128) fp8 tensor with f32 block scales, OR a FOUR-BLOCK
 * fp8 tensor with ue8m0 (1 byte/block) scales, ALSO carries exactly ns==4,
 * same as the E8 tag. An unstamped [I=98] fp8-e4m3-b128 tensor at either of
 * those shapes is therefore byte-for-byte indistinguishable from a genuine
 * fmt=6 tensor: nb AND ns both coincide, not just ns (contrast the fmt=1/
 * fmt=8 collision in Part A, where only ns ever coincides). O==1 stacks a
 * THIRD candidate: fmt=1's per-row ns (O*4) is also 4 there. Unstamped, every
 * one of these refuses; a stamp naming exactly one live candidate resolves
 * it -- EXCEPT the ue8m0-scaled fp8 candidate, which stays refused even when
 * correctly stamped "fp8-e4m3-b128" (a stamp cannot grant a decoder this
 * build doesn't have). */
static void test_fmt6_fp8_collision(void){
    /* O=64, I=98: nblkO=nblkI=1 (fmt=8, single block, f32 scales) -> ns=4;
     * e8_blocks(98)=1 -> nb=O*98=6272 for BOTH interpretations, and fmt=6's
     * .qs tag is always exactly one f32 -> ns=4 too. Unstamped: must refuse. */
    int64_t nb64=(int64_t)64*98;
    CHECK(expect_refuse(64,98, nb64, 4, "fmt=6/fmt=8(f32) collision O=64 I=98 (unstamped)"));
    /* boundary O=128 variant: still nblkO=1 for fmt=8 (128<=128), same collision. */
    int64_t nb128=(int64_t)128*98;
    CHECK(expect_refuse(128,98, nb128, 4, "fmt=6/fmt=8(f32) collision O=128 I=98 (unstamped)"));

    /* stamped: the ONLY way to break this collision. Both directions must resolve
     * to the STAMPED format, not whichever the byte-arithmetic-only check would
     * have unconditionally picked (fmt=6, since it runs first in the function). */
    CHECK(expect_fmt_stamped(64,98, nb64, 4, "fp8-e4m3-b128", 8,
        "fmt=6/fmt=8(f32) collision O=64 I=98, stamped fp8-e4m3-b128 -> resolves to fmt=8"));
    CHECK(expect_fmt_stamped(64,98, nb64, 4, "e8-iq3-lattice", 6,
        "fmt=6/fmt=8(f32) collision O=64 I=98, stamped e8-iq3-lattice -> resolves to fmt=6"));
    CHECK(expect_fmt_stamped(128,98, nb128, 4, "fp8-e4m3-b128", 8,
        "fmt=6/fmt=8(f32) collision O=128 I=98, stamped fp8-e4m3-b128 -> resolves to fmt=8"));
    CHECK(expect_fmt_stamped(128,98, nb128, 4, "e8-iq3-lattice", 6,
        "fmt=6/fmt=8(f32) collision O=128 I=98, stamped e8-iq3-lattice -> resolves to fmt=6"));

    /* a stamp naming something else entirely does NOT resolve the ambiguity --
     * same "refuse rather than guess" posture as an absent stamp. */
    CHECK(expect_refuse_stamped(64,98, nb64, 4, "int4-row",
        "fmt=6/fmt=8(f32) collision O=64 I=98, stamped with an UNRELATED format -> still refuses"));

    /* O=1, I=98: a THIRD candidate stacks on (fmt=1 plain int8 per-row, ns==O*4==4
     * too) -- a genuine three-way ambiguity. Unstamped refuses; stamped resolves
     * to whichever of the three the stamp names. */
    int64_t nb1=(int64_t)1*98;
    CHECK(expect_refuse(1,98, nb1, 4, "fmt=1/fmt=6/fmt=8(f32) THREE-way collision O=1 I=98 (unstamped)"));
    CHECK(expect_fmt_stamped(1,98, nb1, 4, "int8-row", 1,
        "three-way collision O=1 I=98, stamped int8-row -> resolves to fmt=1"));
    CHECK(expect_fmt_stamped(1,98, nb1, 4, "fp8-e4m3-b128", 8,
        "three-way collision O=1 I=98, stamped fp8-e4m3-b128 -> resolves to fmt=8"));
    CHECK(expect_fmt_stamped(1,98, nb1, 4, "e8-iq3-lattice", 6,
        "three-way collision O=1 I=98, stamped e8-iq3-lattice -> resolves to fmt=6"));

    /* O in (384,512], I=98: nblkO=4, nblkI=1, product=4 -- a fmt=8 tensor with
     * UE8M0 (1 byte/block) scales also lands at ns==4*1==4 here, the SAME tag
     * fmt=6 uses. Unstamped: must refuse. Stamped "e8-iq3-lattice": resolves to
     * fmt=6 (a real, decodable format). Stamped "fp8-e4m3-b128": still refuses
     * -- the stamp confirms the WEIGHT format, but this build has no UE8M0
     * decoder, so it cannot grant a resolution the byte layout itself can't
     * support (contrast the [64,98]/[128,98] cases above, where the SAME stamp
     * name resolves cleanly because those are the f32-scaled candidate). */
    int64_t nb400=(int64_t)400*98;
    CHECK(expect_refuse(400,98, nb400, 4, "fmt=6/fmt=8(ue8m0, 4-block) collision O=400 I=98 (unstamped)"));
    CHECK(expect_fmt_stamped(400,98, nb400, 4, "e8-iq3-lattice", 6,
        "fmt=6/fmt=8(ue8m0) collision O=400 I=98, stamped e8-iq3-lattice -> resolves to fmt=6"));
    CHECK(expect_refuse_stamped(400,98, nb400, 4, "fp8-e4m3-b128",
        "fmt=6/fmt=8(ue8m0) collision O=400 I=98, stamped fp8-e4m3-b128 -> STILL refuses (no ue8m0 decoder)"));

    /* regression guard: a GENUINE (non-colliding) fmt=6 fixture -- I!=98, so
     * e8_rowbytes(I)==98 does NOT equal O*I -- must keep resolving to fmt=6 with
     * NO stamp at all. Mirrors test_e8_kernel.c's own O=24,I=512 shape and a
     * small single-super-block shape (I=256, inside the (0,256] range where
     * e8_rowbytes(I)==98 but I!=98, so still non-colliding). */
    CHECK(expect_fmt(24,512, (int64_t)24*e8_rowbytes(512), 4, 6,
        "genuine fmt=6 (non-colliding) O=24 I=512, unstamped -> still resolves to fmt=6"));
    CHECK(expect_fmt(64,256, (int64_t)64*e8_rowbytes(256), 4, 6,
        "genuine fmt=6 (non-colliding) O=64 I=256 (I!=98, no collision), unstamped -> fmt=6"));
}

/* ---- Part A3: fmt=8's scale ENCODING is a declared property -- UE8M0
 * recognized, refused by name (not implemented in this build). A stamp
 * cannot resolve any of these: "fp8-e4m3-b128" confirms the WEIGHT format,
 * not a scale encoding this build can decode, so the stamped cases below
 * refuse exactly like their unstamped counterparts -- the one exception is
 * the small-O collision, where a DIFFERENT stamp ("int8-row", naming the
 * OTHER live candidate) legitimately resolves it, because that candidate
 * really is decodable. ---- */
static void test_ue8m0_scale_refusal(void){
    /* [2048,6144] (spec example shape, same as Part A's fmt=8/f32 golden
     * path): nblkO=16, nblkI=48, product=768 blocks. A UE8M0 sidecar is
     * exactly 1 byte/block -> ns=768, distinct from BOTH fmt=1's per-row
     * count (O*4=8192) and this build's f32 block-scale count (768*4=3072).
     * Clean, unambiguous UE8M0 signature -- must name-refuse, not silently
     * treat it as a truncated/corrupt f32 array or match it to fmt=1. */
    int64_t nb=(int64_t)2048*6144;
    CHECK(expect_refuse(2048,6144, nb, 768,
        "fp8-e4m3-b128 with ue8m0 scales (spec-shaped, non-degenerate), unstamped -> recognized, refused by name"));
    CHECK(expect_refuse_stamped(2048,6144, nb, 768, "fp8-e4m3-b128",
        "fp8-e4m3-b128 with ue8m0 scales, stamped fp8-e4m3-b128 -> STILL refuses (stamp names the weight format, not a decoder)"));

    /* O=1, I=400: nblkO=1, nblkI=ceil(400/128)=4, product=4 -- a UE8M0 sidecar
     * here is ns=4*1=4, which ALSO equals fmt=1's per-row count (O*4=4): the
     * same small-O regime that produces the f32-vs-fmt=1 collision in Part A
     * produces a ue8m0-vs-fmt=1 collision too. Unstamped: must still refuse
     * (combined message), not silently pick fmt=1. Stamped "int8-row" DOES
     * resolve it -- that candidate is real, decodable plain int8, and the
     * stamp confirms it's the one on disk. Stamped "fp8-e4m3-b128" does NOT
     * resolve it -- same "no decoder" refusal as the clean case above. */
    int64_t nb1=(int64_t)1*400;
    CHECK(expect_refuse(1,400, nb1, 4,
        "fp8-e4m3-b128 with ue8m0 scales, ALSO colliding with fmt=1 per-row (O=1), unstamped -> refused"));
    CHECK(expect_fmt_stamped(1,400, nb1, 4, "int8-row", 1,
        "ue8m0/fmt=1 collision O=1 I=400, stamped int8-row -> resolves to fmt=1 (real, decodable candidate)"));
    CHECK(expect_refuse_stamped(1,400, nb1, 4, "fp8-e4m3-b128",
        "ue8m0/fmt=1 collision O=1 I=400, stamped fp8-e4m3-b128 -> STILL refuses (no ue8m0 decoder)"));
}

/* ---- Part A3b (fix round 1): the ue8m0-vs-fmt=1 unstamped-refusal FAMILY,
 * pinned at two more member shapes. Membership is exactly:
 *     nb == O*I  &&  ns == O*4  &&  ceil(O/128)*ceil(I/128) == 4*O
 * (qt_resolve_fmt's is_row && is_blk_ue8m0; is_blk can never co-hold, since
 * nblk==O and nblk==4*O are disjoint for O>=1). The [1,400] case above is
 * the O=1 member; these pin O=2 (nblkO=1) and O=129 (nblkO=2, past the
 * block edge) so the family's O-dependence is exercised, not one corner.
 * CURRENT POLARITY, pinned deliberately: an UNSTAMPED tensor whose bytes
 * are genuine plain int8 at a member shape REFUSES here (the named ue8m0
 * refusal, carrying its "ALSO matches per-row int8" clause), while a build
 * without this PR's fp8 branches loads the same bytes as fmt=1 -- this is
 * a knowing, disclosed strictness trade for untrusted containers, and the
 * stamp ("int8-row") is the designed escape hatch, verified here too.
 * No GLM-5.2 resident tensor is a member: at I<=16384 (nblkI<=128),
 * membership forces O<=32, and every repack-eligible GLM-5.2 role has
 * O>=576 (tools/fp8_collision_census.py enumerates the roles). */
static void test_ue8m0_family_sweep(void){
    /* [2,1024]: nblkO=1, nblkI=8 -> nblk=8 == 4*O. ns = O*4 = nblk = 8. */
    CHECK(expect_refuse(2,1024, (int64_t)2*1024, 8,
        "ue8m0 family [2,1024] (nblk=8==4*O), unstamped int8-shaped -> refuses (current polarity)"));
    CHECK(expect_fmt_stamped(2,1024, (int64_t)2*1024, 8, "int8-row", 1,
        "ue8m0 family [2,1024], stamped int8-row -> resolves to fmt=1 (the escape hatch)"));
    /* [129,33000]: nblkO=2, nblkI=258 -> nblk=516 == 4*129. ns = 516. */
    CHECK(expect_refuse(129,33000, (int64_t)129*33000, 516,
        "ue8m0 family [129,33000] (nblk=516==4*O, nblkO=2), unstamped int8-shaped -> refuses (current polarity)"));
    CHECK(expect_fmt_stamped(129,33000, (int64_t)129*33000, 516, "int8-row", 1,
        "ue8m0 family [129,33000], stamped int8-row -> resolves to fmt=1"));
}

/* ---- Part B: qt_from_disk loader-seam (real safetensors file) ---- */

static void deq_fmt8(const QT *t, float *dq){
    int64_t nblkI = fp8_nblk(t->I);
    for(int o=0;o<t->O;o++){
        int64_t blkO = o/FP8_BLOCK; const float *scl = t->s + blkO*nblkI;
        for(int i=0;i<t->I;i++){
            int64_t bi = i/FP8_BLOCK;
            dq[(int64_t)o*t->I+i] = e4m3_decode(t->q8[(int64_t)o*t->I+i]) * scl[bi];
        }
    }
}
static void deq_fmt1(const QT *t, float *dq){
    for(int o=0;o<t->O;o++){ float s=t->s[o];
        for(int i=0;i<t->I;i++) dq[(int64_t)o*t->I+i]=(float)t->q8[(int64_t)o*t->I+i]*s; }
}

#define CDIV(n,d) (((n)+(d)-1)/(d))

static void test_loader_seam(void){
    enum { O7=8, I7=256 };                        /* nblkO=1, nblkI=2 -> 2 block scales total */
    enum { O1=5, I1=64 };                        /* DIFFERENT shape from the fp8 tensor: O1*I1=320
                                                   * bytes, ns=O1*4=20 -- neither collides with the
                                                   * fp8 tensor's own byte counts (kept deliberately
                                                   * distinct so this is a plain, non-degenerate
                                                   * negative control, not another landmine case). */
    enum { NBLK7 = CDIV(O7,128) * CDIV(I7,128) };  /* must be a compile-time constant expression for
                                                     * the static array below -- fp8_nblk() is a real
                                                     * function (runtime, not constexpr), so it can't
                                                     * size a `static` array even with literal inputs. */
    static uint8_t q7[O7*I7]; static float s7[NBLK7];
    for(int i=0;i<O7*I7;i++) q7[i]=rndbyte_nonan();
    for(int i=0;i<(int)(sizeof s7/sizeof *s7);i++) s7[i]=0.01f+0.001f*(float)i;

    static int8_t q1[O1*I1]; static float s1[O1];
    for(int i=0;i<O1*I1;i++) q1[i]=(int8_t)(rndbyte_nonan()-128);
    for(int i=0;i<O1;i++) s1[i]=0.02f+0.001f*(float)i;

    const char *dir="tests/tmp_fp8_snap";
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir,0755);
#endif
    char path[300]; snprintf(path,sizeof path,"%s/model.safetensors",dir);
    int64_t nb7=(int64_t)O7*I7, ns7=(int64_t)(sizeof s7);
    int64_t nb1=(int64_t)O1*I1, ns1=(int64_t)O1*4;
    char hdr[1024];
    int hl=snprintf(hdr,sizeof hdr,
        "{\"w7\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[0,%lld]},"
        "\"w7.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"w1\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]},"
        "\"w1.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}}",
        (long long)nb7,(long long)nb7,
        (long long)(ns7/4),(long long)nb7,(long long)(nb7+ns7),
        (long long)nb1,(long long)(nb7+ns7),(long long)(nb7+ns7+nb1),
        (long long)O1,(long long)(nb7+ns7+nb1),(long long)(nb7+ns7+nb1+ns1));
    FILE *f=fopen(path,"wb");
    if(!f){ printf("FAIL: cannot create %s (run from c/, like tools/run_tests.py does)\n", path); fails++; return; }
    uint64_t hlen=(uint64_t)hl;
    fwrite(&hlen,8,1,f); fwrite(hdr,1,hl,f);
    fwrite(q7,1,(size_t)nb7,f); fwrite(s7,1,(size_t)ns7,f);
    fwrite(q1,1,(size_t)nb1,f); fwrite(s1,1,(size_t)ns1,f);
    fclose(f);

    static Model gm;                             /* only gm.S is used by qt_from_disk */
    st_init(&gm.S, dir);

    QT t7; memset(&t7,0,sizeof t7);
    qt_from_disk(&gm,"w7",O7,I7,8,0,&t7);
    CHECK(t7.fmt==8);
    CHECK(t7.q8!=NULL && t7.s!=NULL);            /* both weight and scale allocated (qalloc, not falloc) */
    static float dq_load[O7*I7], dq_ref[O7*I7];
    deq_fmt8(&t7,dq_load);
    QT tr7={.fmt=8,.q8=(int8_t*)q7,.s=s7,.O=O7,.I=I7};
    deq_fmt8(&tr7,dq_ref);
    CHECK(memcmp(dq_load,dq_ref,sizeof dq_ref)==0);

    QT t1; memset(&t1,0,sizeof t1);
    qt_from_disk(&gm,"w1",O1,I1,8,0,&t1);
    CHECK(t1.fmt==1);                            /* negative control: plain int8 still resolves as fmt=1 */
    static float dq_load1[O1*I1], dq_ref1[O1*I1];
    deq_fmt1(&t1,dq_load1);
    QT tr1={.fmt=1,.q8=q1,.s=s1,.O=O1,.I=I1};
    deq_fmt1(&tr1,dq_ref1);
    CHECK(memcmp(dq_load1,dq_ref1,sizeof dq_ref1)==0);

    unlink(path); rmdir(dir);
}

/* ---- Part C: qt_bytes()/qt_scale_bytes() byte-accounting for fmt=8 ----
 *
 * qt_bytes() must not fall through to the fmt=2 (packed int4, O*ceil(I/2)+O*4)
 * default -- for a real fp8 tensor that would undercount the resident byte
 * count by roughly half (an AUTOPIN/RAM-budget-feeding hazard). qt_wire_mmap/
 * qt_unwire_mmap must not hardcode scale_b=O*4 (per-row) either -- wrong for
 * fmt=8's per-128x128-block scale array -- hence the dedicated
 * qt_scale_bytes() helper shared by qt_bytes() and both wire functions so
 * there is exactly one place that knows each format's scale geometry. This
 * test exercises the arithmetic directly (no qt_from_disk/disk I/O, no mlock
 * syscall -- qt_wire_mmap's actual mem_wire() call is environment-dependent
 * (RLIMIT_MEMLOCK) and not what changed; the byte-count formula is) across
 * the shapes already used elsewhere in this file plus a block-edge
 * (non-128-multiple) case. */
static void check_fp8_bytes(int O, int I, const char *tag){
    QT t; memset(&t,0,sizeof t); t.fmt=8; t.O=O; t.I=I; t.gs=0;
    int64_t nblkO=fp8_nblk(O), nblkI=fp8_nblk(I), nblk=nblkO*nblkI;
    int64_t want_total = (int64_t)O*I + nblk*4;
    int64_t want_scale = nblk*4;
    int64_t got_total = qt_bytes(&t);
    int64_t got_scale = qt_scale_bytes(&t);
    if(got_total != want_total)
        printf("FAIL %s: qt_bytes=%lld want=%lld\n", tag, (long long)got_total, (long long)want_total);
    CHECK(got_total == want_total);
    if(got_scale != want_scale)
        printf("FAIL %s: qt_scale_bytes=%lld want=%lld\n", tag, (long long)got_scale, (long long)want_scale);
    CHECK(got_scale == want_scale);
    /* weight_b, as qt_wire_mmap/qt_unwire_mmap now compute it, must land on the exact
     * O*I raw-byte weight region -- not short (partial mlock) or long (mlock past the
     * allocation, undefined behavior) by even one byte. */
    CHECK(got_total - got_scale == (int64_t)O*I);
    /* regression guard: the fmt=2 (packed-nibble) formula must NOT be what fmt=8
     * returns -- confirm the value has actually MOVED off it (catches a silent
     * revert of the fmt==8 branch order/placement, not just a formula typo). For
     * O=1,I=1 the two formulas coincide by coincidence (both give 1+4=5), so that
     * shape is skipped for this particular guard -- the other three shapes below are
     * chosen to avoid the coincidence. */
    int64_t old_wrong = (int64_t)O*((I+1)/2) + (int64_t)O*4;
    if(!(O==1 && I==1)) CHECK(got_total != old_wrong);
}

/* qt_wire_mmap()/qt_unwire_mmap() (colibri.c) both compute weight_b/scale_b via
 * qt_wire_split(t,&weight_b,&scale_b) -- this calls that EXACT shared function
 * directly (no mlock syscall: same reasoning as check_fp8_bytes above, mem_wire's
 * actual RLIMIT_MEMLOCK behavior is environment-dependent and not what changed)
 * so a regression at qt_wire_split() itself -- e.g. reverting to the
 * scale_b=(int64_t)t->O*4 hardcode the maintainer's #528 review found dead-coded
 * behind an unused qt_scale_bytes() (compiling clean only because
 * -Wno-unused-function suppressed the warning that should have caught it) --
 * fails here. Covers every format qt_wire_split's callers can see in practice
 * (fmt=1 per-row unaffected; fmt=4/5 grouped-scale formats qt_scale_bytes'
 * own comment names as previously-broken too; fmt=6 (E8/IQ3, FIX ROUND,
 * audit finding: a FIXED 4-byte tag, not O*4 -- see qt_scale_bytes' own
 * comment for why this is reachable, not dead code); fmt=8 nblk>O, the
 * shape this review round is about). */
static void check_wire_split(int fmt, int O, int I, int gs, const char *tag){
    /* self-guard (fix round 1, reviewer finding): the fmt ARGUMENT is the
     * load-bearing input here, and a label that says fmt=8 with the argument
     * left at a stale ordinal turns every assertion below into a tautology
     * (want and got both fall to the same per-row default together). Pin the
     * argument to the supported set so label/argument drift dies loudly at
     * the call site instead. */
    if(fmt!=1 && fmt!=4 && fmt!=5 && fmt!=6 && fmt!=8){
        printf("FAIL %s: check_wire_split fmt=%d not in supported set {1,4,5,6,8} -- stale call site?\n", tag, fmt);
        CHECK(0); return;
    }
    QT t; memset(&t,0,sizeof t); t.fmt=fmt; t.O=O; t.I=I; t.gs=gs;
    int64_t want_scale = qt_scale_bytes(&t);
    int64_t want_weight = qt_bytes(&t) - want_scale;
    int64_t got_weight=-1, got_scale=-1;
    qt_wire_split(&t,&got_weight,&got_scale);
    if(got_scale != want_scale)
        printf("FAIL %s: qt_wire_split scale_b=%lld want=%lld\n", tag, (long long)got_scale, (long long)want_scale);
    CHECK(got_scale == want_scale);
    if(got_weight != want_weight)
        printf("FAIL %s: qt_wire_split weight_b=%lld want=%lld\n", tag, (long long)got_weight, (long long)want_weight);
    CHECK(got_weight == want_weight);
    /* the regression this test exists to catch: a per-row-only scale_b==O*4
     * must NOT be what qt_wire_split returns for a format whose real scale
     * cardinality differs from O (fmt=4/5/6/8 here) -- confirm the value has
     * actually moved off that old constant, not just matched qt_scale_bytes()
     * by coincidence at a degenerate shape. */
    int64_t old_wrong_scale = (int64_t)O*4;
    if((fmt==4 || fmt==5 || fmt==6 || fmt==8) && want_scale != old_wrong_scale)
        CHECK(got_scale != old_wrong_scale);
    /* fmt=6's scale is a FIXED 4 bytes regardless of [O,I] -- assert the
     * literal value directly too, not just "moved off O*4", since a future
     * regression that made it O-dependent in some OTHER wrong way would
     * still pass the generic check above. */
    if(fmt==6) CHECK(want_scale == 4);
    /* fmt=8: same independent-pin discipline as the fmt==6 literal above.
     * want and got BOTH flow through the engine's qt_scale_bytes() (the test
     * computes want from it, and qt_wire_split() calls it), so a reverted or
     * deleted fmt==8 branch would move both sides to O*4 TOGETHER: the
     * got==want checks stay green and the moved-off-O*4 check self-disables
     * (its want_scale != O*4 guard goes false). Recomputing the expected
     * block-grid value here, independently of the engine, is what makes a
     * regression of that branch actually FAIL this test (mutation-proven,
     * fix round 1). */
    if(fmt==8){
        int64_t nblk = (((int64_t)O+127)/128) * (((int64_t)I+127)/128);
        CHECK(want_scale == nblk*4);
    }
}

/* ---- Site-level wire regression (FIX ROUND, validator finding: mutation-
 * proven gap). check_wire_split() above calls qt_wire_split() directly --
 * it would NOT notice a mutation that reverts ONLY qt_wire_mmap's and
 * qt_unwire_mmap's call sites back to the old inline
 * `scale_b=(int64_t)t->O*4` hardcode, leaving qt_wire_split() itself
 * intact (the two sites would simply stop CALLING the now-orphaned-again
 * helper). This test calls the real qt_wire_mmap()/qt_unwire_mmap()
 * functions and asserts, through the mlock()/munlock() observer seam
 * defined above (right after the #include), that the (addr,len) each site
 * ACTUALLY passed down to the platform lock call matches
 * qt_scale_bytes()/qt_bytes() -- i.e. it observes the call sites' own
 * behavior, not a reimplementation of it. Shape chosen so a reverted site
 * is unmistakably wrong, not coincidentally right: O=2, I=16384 ->
 * nblkO=1, nblkI=128, nblk=128, scale_b=512B; the old hardcode would
 * compute O*4=8B instead, a 64x difference. Proven to bite: see the
 * report's mutation output (the exact single-line revert, applied and
 * reverted, with the resulting FAIL line pasted in). POSIX-only (like this
 * file's fork-based refusal tests): the observer seam only intercepts the
 * `#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)`
 * arm both qt_wire_mmap (via mem_wire) and qt_unwire_mmap take -- Windows
 * takes a different call (compat_mlock/compat_munlock) not intercepted
 * here. */
static void test_wire_site_regression(void){
#ifndef _WIN32
    g_seam_wire_n = 0; g_seam_unwire_n = 0;
    enum { O=2, I=16384 };
    enum { NBLKO = CDIV(O,128), NBLKI = CDIV(I,128), NBLK = NBLKO*NBLKI };
    static uint8_t q7[O*I]; static float s7[NBLK];
    for(int i=0;i<O*I;i++) q7[i]=rndbyte_nonan();
    for(int i=0;i<NBLK;i++) s7[i]=0.01f+0.001f*(float)i;

    QT t; memset(&t,0,sizeof t);
    t.fmt=8; t.O=O; t.I=I; t.gs=0; t.q8=(int8_t*)q7; t.s=s7;

    int64_t want_scale = qt_scale_bytes(&t);
    int64_t want_weight = qt_bytes(&t) - want_scale;
    /* sanity: this shape must actually distinguish the fix from the old bug,
     * or the test below would pass either way and prove nothing. */
    CHECK(want_scale != (int64_t)O*4);

    /* qt_wire_mmap doesn't itself gate on g_mmap/mem_should_wire (its
     * caller, pin_wire, does) -- calling it directly always attempts to
     * wire, which is exactly what this test wants. */
    int64_t wired=0; long failed=0;
    qt_wire_mmap(&t, &wired, &failed);
    if(g_seam_wire_n != 2)
        printf("FAIL wire-site regression: qt_wire_mmap called mlock() %d times, expected 2\n", g_seam_wire_n);
    CHECK(g_seam_wire_n == 2);
    if(g_seam_wire_n >= 1 && (int64_t)g_seam_wire_len[0] != want_weight)
        printf("FAIL wire-site regression: qt_wire_mmap's WEIGHT mlock() call got len=%lld want=%lld\n",
               (long long)g_seam_wire_len[0], (long long)want_weight);
    CHECK(g_seam_wire_n>=1 && (int64_t)g_seam_wire_len[0]==want_weight);
    if(g_seam_wire_n >= 2 && (int64_t)g_seam_wire_len[1] != want_scale)
        printf("FAIL wire-site regression: qt_wire_mmap's SCALE mlock() call got len=%lld want=%lld "
               "(this is exactly what a reverted scale_b=O*4 hardcode breaks)\n",
               (long long)g_seam_wire_len[1], (long long)want_scale);
    CHECK(g_seam_wire_n>=2 && (int64_t)g_seam_wire_len[1]==want_scale);

    /* qt_unwire_mmap early-returns unless g_mmap && mem_should_wire() are
     * both true -- force both on for this call, then restore, so this test
     * doesn't change global state for any test that runs after it. */
    int saved_g_mmap = g_mmap, saved_g_mlock = g_mlock;
    g_mmap = 1; g_mlock = 1;
    qt_unwire_mmap(&t);
    g_mmap = saved_g_mmap; g_mlock = saved_g_mlock;

    if(g_seam_unwire_n != 2)
        printf("FAIL wire-site regression: qt_unwire_mmap called munlock() %d times, expected 2\n", g_seam_unwire_n);
    CHECK(g_seam_unwire_n == 2);
    if(g_seam_unwire_n >= 1 && (int64_t)g_seam_unwire_len[0] != want_weight)
        printf("FAIL wire-site regression: qt_unwire_mmap's WEIGHT munlock() call got len=%lld want=%lld\n",
               (long long)g_seam_unwire_len[0], (long long)want_weight);
    CHECK(g_seam_unwire_n>=1 && (int64_t)g_seam_unwire_len[0]==want_weight);
    if(g_seam_unwire_n >= 2 && (int64_t)g_seam_unwire_len[1] != want_scale)
        printf("FAIL wire-site regression: qt_unwire_mmap's SCALE munlock() call got len=%lld want=%lld "
               "(this is exactly what a reverted scale_b=O*4 hardcode breaks)\n",
               (long long)g_seam_unwire_len[1], (long long)want_scale);
    CHECK(g_seam_unwire_n>=2 && (int64_t)g_seam_unwire_len[1]==want_scale);
#else
    printf("skipped on Windows (no mlock/munlock observer seam): wire-site regression\n");
#endif
}

/* ---- Part F (micro-round 2): Metal fused-path POSITIVE allowlist ----
 * The two fused-decode gates (attention_rows / layer_forward_rows) are only
 * reachable at model scale (GLM-5.2 dims + a live Metal context), so the
 * honest toy-scale decomposition is: (1) unit-test the predicate's truth
 * table -- metal_fused_fmt_ok is defined unconditionally in colibri.c for
 * exactly this reason -- and (2) pin, by reading colibri.c's own source,
 * that BOTH gate sites actually consult the predicate on every fused-path
 * tensor and that the old fail-open `l-><t>.fmt!=8` idiom is gone. The pin
 * is a source-text assertion, disclosed as such: it proves wiring, not
 * runtime routing; the truth table proves the routing decision itself. */
static void test_metal_fused_allowlist(void){
    /* (1) truth table: admits EXACTLY the shader's explicit integer-format
     * set {1,2,3,4} (fmt=3 was legal-and-correct on the fused path before
     * the allowlist existed; fmt=0 never actually fused -- WP_ hands the
     * shader a NULL q4 for it). A negative-guard regression (e.g.
     * `return fmt!=8;`) admits 5/6 -- the silent-f32-misread hazard -- and
     * fails here. */
    for(int fmt=-1; fmt<=9; fmt++){
        int want = (fmt==1 || fmt==2 || fmt==3 || fmt==4);
        if(metal_fused_fmt_ok(fmt) != want)
            printf("FAIL metal_fused_fmt_ok(%d)=%d, want %d\n", fmt, metal_fused_fmt_ok(fmt), want);
        CHECK(metal_fused_fmt_ok(fmt) == want);
    }
    CHECK(!metal_fused_fmt_ok(100));   /* private-block ordinals stay out too */
    /* (2) grep-pin: run_tests.py runs us with cwd=c/ (Part B's fixtures rely
     * on the same fact). The per-tensor calls live in ONE place now --
     * metal_fused_layer_fmt_miss(), the shared per-layer predicate both gates
     * and metal_fmt_gate_notice consume -- so the pin is: exactly 7
     * metal_fused_fmt_ok(l->...) call sites (q_a/q_b/kv_a/o +
     * sh_gate/sh_up/sh_down, each once, inside the helper; kv_b uses its own
     * two-format+mode term there, not the allowlist), and BOTH gate sites
     * consult the helper against the mask of tensors their kernel binds.
     * (test_kvb_notice.c pins the helper's own truth table.) */
    FILE *f = fopen("colibri.c", "r");
    if(!f){ printf("FAIL Part F grep-pin: cannot open colibri.c (cwd not c/?)\n"); CHECK(0); return; }
    static char src[16*1024*1024];
    size_t n = fread(src, 1, sizeof(src)-1, f); src[n]=0; fclose(f);
    CHECK(n > 100000);                      /* sanity: we read the real file */
    int calls=0; const char *p=src;
    while((p=strstr(p,"metal_fused_fmt_ok(l->"))!=NULL){ calls++; p++; }
    if(calls!=7) printf("FAIL Part F grep-pin: %d metal_fused_fmt_ok(l->...) call sites, want 7\n", calls);
    CHECK(calls==7);
    int attn=0; p=src;
    while((p=strstr(p,"metal_fused_layer_fmt_miss(l) & METAL_FUSED_ATTN_TENSORS"))!=NULL){ attn++; p++; }
    CHECK(attn==1);                    /* attention_rows' gate consults the shared predicate */
    int lay=0; p=src;
    while((p=strstr(p,"metal_fused_layer_fmt_miss(l) & METAL_FUSED_LAYER_TENSORS"))!=NULL){ lay++; p++; }
    CHECK(lay==1);                     /* layer_forward_rows' gate consults the shared predicate */
    /* the old fail-open idiom must be gone from the gate sites (the CUDA
     * eligibility guard's `w->fmt!=8` is a different site and untouched). */
    CHECK(strstr(src,"l->q_a.fmt!=8")==NULL);
    CHECK(strstr(src,"l->sh_gate.fmt!=8")==NULL);
}

/* ---- Part G (micro-round 2): fmt=7 is unreachable as a qt_resolve_fmt
 * return value -- property sweep. The fmt=7 ordinal belongs to dev's MXFP4
 * (Vulkan tier, never a QT format); matmul_qt_ex's int4 tail is therefore
 * unreachable for it, and this sweep is the RAN half of that argument (the
 * other half is the producer enumeration in the PR verification matrix:
 * every QT.fmt producer is qt_resolve_fmt, qt_alloc's 0/1/2/3/5 ladder, or
 * a literal 1..6/8 assignment). Sweep a representative + adversarial
 * (O, I, nb, ns) grid across every byte-count class the resolver knows
 * (row formats, grouped, int3-g64, E8/IQ3, fp8 f32/ue8m0, collision-family
 * members, degenerate dims) and assert every non-refusing resolution is in
 * {1,2,3,4,5,6,8} -- never 7, never anything else. */
static int resolved_fmt_probe(int O, int I, int64_t nb, int64_t ns, const char *stamped){
#ifndef _WIN32
    fflush(stdout);   /* a refusing child exits via exit(1), which flushes
                       * inherited stdio -- don't let it replay our buffer */
    int pipefd[2]; if(pipe(pipefd)!=0) return -2;
    pid_t pid = fork();
    if(pid < 0) return -2;
    if(pid == 0){
        int devnull = open("/dev/null", O_WRONLY);
        if(devnull>=0) dup2(devnull,2);                 /* silence refusal chatter */
        close(pipefd[0]);
        int gs=0;
        int fmt = qt_resolve_fmt("sweep", O, I, nb, ns, &gs, stamped);
        unsigned char b = (unsigned char)fmt;
        ssize_t wr = write(pipefd[1], &b, 1); (void)wr;
        _exit(0);
    }
    close(pipefd[1]);
    unsigned char b=0; ssize_t got = read(pipefd[0], &b, 1);
    close(pipefd[0]);
    int status=0; waitpid(pid,&status,0);
    if(WIFEXITED(status) && WEXITSTATUS(status)==0 && got==1) return (int)b;
    if(WIFEXITED(status) && WEXITSTATUS(status)==1) return -1;   /* refused */
    return -2;                                                    /* crash/protocol: always a failure */
#else
    (void)O;(void)I;(void)nb;(void)ns;
    return -3;                                                    /* sweep skipped on Windows */
#endif
}
static void test_fmt7_unreachable_sweep(void){
#ifndef _WIN32
    static const int Os[] = {1,2,3,24,32,33,64,98,127,128,129,130,384,400,576,2048};
    static const int Is[] = {1,64,98,128,129,200,256,257,384,400,512,1024,6144,16384,33000};
    int probes=0, resolved=0, refused=0;
    for(size_t oi=0; oi<sizeof(Os)/sizeof(Os[0]); oi++){
        for(size_t ii=0; ii<sizeof(Is)/sizeof(Is[0]); ii++){
            int O=Os[oi], I=Is[ii];
            int64_t nblk = fp8_nblk(O)*fp8_nblk(I);
            /* every (nb, ns) class the resolver's ladder can see, incl. the
             * adversarial cross-wired ones (row-shaped ns on block-shaped nb
             * and vice versa): */
            const int64_t cand[][2] = {
                { (int64_t)O*I,              (int64_t)O*4 },        /* int8-row shaped   */
                { (int64_t)O*I,              nblk*4       },        /* fp8 f32 block     */
                { (int64_t)O*I,              nblk         },        /* fp8 ue8m0 block   */
                { (int64_t)O*((I+1)/2),      (int64_t)O*4 },        /* int4-row          */
                { (int64_t)O*((I+3)/4),      (int64_t)O*4 },        /* int2-row          */
                { (int64_t)O*((I+1)/2),      (int64_t)O*((I+63)/64)*4 }, /* int4-grouped g64 */
                { (int64_t)O*i3_rowbytes(I), (int64_t)O*i3_groups(I)*4 },/* int3-g64      */
                { (int64_t)O*e8_rowbytes(I), 4 },                   /* E8/IQ3 tag        */
                { (int64_t)O*I,              4 },                   /* cross-wired tag   */
                { (int64_t)O*I,              0 },                   /* degenerate ns     */
                { 0,                          (int64_t)O*4 },       /* degenerate nb     */
            };
            /* STAMPED arm (this commit threads stamped_name into
             * qt_resolve_fmt): the same grid under every stamp class --
             * agreeing/disagreeing recognized names, and an unrecognized
             * one. A stamp can steer WHICH member of {1,2,3,4,5,6,8} wins
             * (or force a refusal); it must never mint 7 or anything else. */
            static const char *stamps[] = { NULL, "int8-row", "fp8-e4m3-b128",
                                            "e8-iq3-lattice", "quantum-format-9000" };
            for(size_t st=0; st<sizeof(stamps)/sizeof(stamps[0]); st++){
                for(size_t c=0; c<sizeof(cand)/sizeof(cand[0]); c++){
                    int f = resolved_fmt_probe(O, I, cand[c][0], cand[c][1], stamps[st]);
                    probes++;
                    if(f==-2){ printf("FAIL sweep O=%d I=%d nb=%lld ns=%lld stamp=%s: crash/protocol error\n",
                                      O,I,(long long)cand[c][0],(long long)cand[c][1],
                                      stamps[st]?stamps[st]:"(none)"); CHECK(0); continue; }
                    if(f==-1){ refused++; continue; }
                    resolved++;
                    int ok = (f==1||f==2||f==3||f==4||f==5||f==6||f==8);
                    if(!ok) printf("FAIL sweep O=%d I=%d nb=%lld ns=%lld stamp=%s: resolved to fmt=%d (outside {1,2,3,4,5,6,8})\n",
                                   O,I,(long long)cand[c][0],(long long)cand[c][1],
                                   stamps[st]?stamps[st]:"(none)",f);
                    CHECK(ok);
                    CHECK(f != 7);
                }
            }
        }
    }
    /* the sweep must have actually exercised both outcomes across both the
     * stamped and unstamped arms, or it proved nothing. */
    CHECK(probes >= 10000 && resolved >= 500 && refused >= 500);
#else
    printf("skipped on Windows (no fork): fmt=7 unreachability sweep\n");
#endif
}

/* ---- Part D: metadata-stamp TRUST-VERIFY-REFUSE (qt_verify_fmt_stamp, colibri.c) ----
 * The stamp lives in the safetensors __metadata__["colibri.fmt"] value -- itself
 * JSON text (a {tensor_name: format_name} map, matching what
 * tools/repack_fp8_passthrough.py writes and st_fmt_stamp_ingest (st.h) parses).
 * Three outcomes: an AGREEING stamp is a silent no-op (loads exactly as it would
 * unstamped); a MISMATCHING stamp (a recognized name naming a DIFFERENT format,
 * or a name this build doesn't recognize at all) refuses (exit(1)); an UNSTAMPED
 * container (no __metadata__ block at all) infers exactly as before this
 * feature existed -- already exercised incidentally by every Part B/C fixture
 * above (none of them write __metadata__), but given an explicit case here too
 * so all three outcomes live together, self-documenting. */

/* Writes a single-tensor fp8 shard (O=8,I=256 -> nblkO=1,nblkI=2, the same
 * non-degenerate shape Part B's w7 fixture uses) with an OPTIONAL __metadata__
 * block. `meta_json_string_literal`, if non-NULL, must already be a quoted/
 * escaped JSON STRING TOKEN placed verbatim after "colibri.fmt": in the header
 * -- callers pass a C string literal like "\"{\\\"w\\\":\\\"fp8-e4m3-b128\\\"}\""
 * so the raw header bytes end up with colibri.fmt's VALUE being the JSON text
 * {"w":"fp8-e4m3-b128"} (double-JSON-encoded: a JSON string whose CONTENT is
 * itself JSON -- exactly what safetensors.save_file(metadata=...) produces,
 * since __metadata__ values must be plain strings). */
static void write_stamp_fixture(const char *dir, const char *meta_json_string_literal){
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir,0755);
#endif
    enum { O=8, I=256 };
    enum { NBLK = CDIV(O,128) * CDIV(I,128) };
    static uint8_t q[O*I]; static float s[NBLK];
    for(int i=0;i<O*I;i++) q[i]=rndbyte_nonan();
    for(int i=0;i<NBLK;i++) s[i]=0.01f+0.001f*(float)i;
    char path[300]; snprintf(path,sizeof path,"%s/model.safetensors",dir);
    int64_t nb=(int64_t)O*I, ns=(int64_t)sizeof(s);
    char hdr[2048]; int hl;
    if(meta_json_string_literal)
        hl=snprintf(hdr,sizeof hdr,
            "{\"__metadata__\":{\"colibri.fmt\":%s},"
            "\"w\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[0,%lld]},"
            "\"w.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}}",
            meta_json_string_literal,
            (long long)nb,(long long)nb,
            (long long)(ns/4),(long long)nb,(long long)(nb+ns));
    else
        hl=snprintf(hdr,sizeof hdr,
            "{\"w\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[0,%lld]},"
            "\"w.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}}",
            (long long)nb,(long long)nb,
            (long long)(ns/4),(long long)nb,(long long)(nb+ns));
    FILE *f=fopen(path,"wb");
    if(!f){ printf("FAIL: cannot create %s (run from c/, like tools/run_tests.py does)\n", path); fails++; return; }
    uint64_t hlen=(uint64_t)hl;
    fwrite(&hlen,8,1,f); fwrite(hdr,1,hl,f);
    fwrite(q,1,(size_t)nb,f); fwrite(s,1,(size_t)ns,f);
    fclose(f);
}

/* Fork+waitpid, mirroring expect_refuse above: st_init+qt_from_disk run in the
 * child (isolating a parsed-and-possibly-poisoned `shards` struct from the
 * rest of the suite) and must exit(1) with a "refus"-containing stderr message. */
static int expect_stamp_refuse(const char *dir, const char *tag){
#ifndef _WIN32
    int pipefd[2]; if(pipe(pipefd)!=0) return 0;
    pid_t pid = fork();
    if(pid < 0) return 0;
    if(pid == 0){
        dup2(pipefd[1],2); close(pipefd[0]); close(pipefd[1]);
        static Model gm; memset(&gm,0,sizeof gm);
        st_init(&gm.S, dir);
        QT t; memset(&t,0,sizeof t);
        qt_from_disk(&gm,"w",8,256,8,0,&t);   /* must exit(1) inside qt_verify_fmt_stamp */
        _exit(42);                             /* reaching here is the bug */
    }
    close(pipefd[1]);
    char err[1024]={0}; size_t eoff=0; ssize_t n; /* drain to EOF (Linux pipe short-reads; see expect_refuse) */ while(eoff<sizeof(err)-1 && (n=read(pipefd[0],err+eoff,sizeof(err)-1-eoff))>0) eoff+=(size_t)n;
    close(pipefd[0]);
    int status=0; waitpid(pid,&status,0);
    int ok = WIFEXITED(status) && WEXITSTATUS(status)==1;
    if(!ok){
        printf("FAIL %s: expected exit(1) refusal, got status=%d, stderr=%.200s\n", tag, status, err);
        return 0;
    }
    if(!strstr(err,"refus")){
        printf("FAIL %s: exited(1) but message lacked a refusal explanation: %.200s\n", tag, err);
        return 0;
    }
    return 1;
#else
    /* same Windows arm as expect_refuse_stamped above: no fork, visible skip. */
    printf("skipped on Windows (no fork): %s\n", tag);
    (void)dir;
    return 1;
#endif
}

static void test_stamp_agreeing(void){
    const char *dir="tests/tmp_fp8_stamp_agree";
    write_stamp_fixture(dir, "\"{\\\"w\\\":\\\"fp8-e4m3-b128\\\"}\"");
    static Model gm; memset(&gm,0,sizeof gm);
    st_init(&gm.S, dir);
    QT t; memset(&t,0,sizeof t);
    qt_from_disk(&gm,"w",8,256,8,0,&t);
    CHECK(t.fmt==8);                   /* stamp agreed -- loads exactly as unstamped would */
    CHECK(t.q8!=NULL && t.s!=NULL);
    char p[300]; snprintf(p,sizeof p,"%s/model.safetensors",dir); unlink(p); rmdir(dir);
}

static void test_stamp_mismatching(void){
    const char *dir="tests/tmp_fp8_stamp_mismatch";
    /* byte-arithmetic says fmt=8 (fp8); the stamp names a REAL, recognized,
     * but DIFFERENT format -- exercises "found but disagrees", not just "name
     * not found at all" (see test_stamp_unrecognized_name below for that case). */
    write_stamp_fixture(dir, "\"{\\\"w\\\":\\\"int8-row\\\"}\"");
    CHECK(expect_stamp_refuse(dir, "stamped-mismatching: stamp says int8-row, bytes say fp8-e4m3-b128"));
    char p[300]; snprintf(p,sizeof p,"%s/model.safetensors",dir); unlink(p); rmdir(dir);
}

static void test_stamp_unrecognized_name(void){
    const char *dir="tests/tmp_fp8_stamp_unknown";
    write_stamp_fixture(dir, "\"{\\\"w\\\":\\\"quantum-format-9000\\\"}\"");
    CHECK(expect_stamp_refuse(dir, "stamped with a format name this build doesn't recognize"));
    char p[300]; snprintf(p,sizeof p,"%s/model.safetensors",dir); unlink(p); rmdir(dir);
}

static void test_stamp_absent(void){
    const char *dir="tests/tmp_fp8_stamp_absent";
    write_stamp_fixture(dir, NULL);       /* no __metadata__ block at all */
    static Model gm; memset(&gm,0,sizeof gm);
    st_init(&gm.S, dir);
    QT t; memset(&t,0,sizeof t);
    qt_from_disk(&gm,"w",8,256,8,0,&t);
    CHECK(t.fmt==8);                   /* unstamped: byte-arithmetic inference alone decides */
    CHECK(t.q8!=NULL && t.s!=NULL);
    char p[300]; snprintf(p,sizeof p,"%s/model.safetensors",dir); unlink(p); rmdir(dir);
}

/* ---- Duplicate stamp claims (micro-round 2, user-ratified design D8) ----
 * At most one DISTINCT format claim per tensor name, container-wide:
 * conflicting claims refuse at ingest (naming the tensor and both format
 * names); agreeing duplicates are tolerated (idempotent). There is no
 * locality constraint -- shard 2 below stamps "w", a tensor it does NOT
 * contain, which is legal by design (centralized-manifest writers) and is
 * exactly what makes cross-shard conflicts possible in the first place. */

/* Shard 1 = write_stamp_fixture's single-tensor fp8 shard ("w" + "w.qs"),
 * renamed into a two-shard layout; shard 2 = one tiny aux f32 tensor of its
 * own plus a colibri.fmt map stamping "w". */
static void write_second_stamp_shard(const char *dir, const char *meta_json_string_literal){
    char path[300]; snprintf(path,sizeof path,"%s/model-extra.safetensors",dir);
    static float aux[4] = {1.f,2.f,3.f,4.f};
    char hdr[1024]; int hl;
    hl=snprintf(hdr,sizeof hdr,
        "{\"__metadata__\":{\"colibri.fmt\":%s},"
        "\"aux\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}",
        meta_json_string_literal);
    FILE *f=fopen(path,"wb");
    if(!f){ printf("FAIL: cannot create %s\n", path); fails++; return; }
    uint64_t hlen=(uint64_t)hl;
    fwrite(&hlen,8,1,f); fwrite(hdr,1,hl,f); fwrite(aux,1,sizeof aux,f);
    fclose(f);
}
static void rm_two_shard_fixture(const char *dir){
    char p[300];
    snprintf(p,sizeof p,"%s/model.safetensors",dir); unlink(p);
    snprintf(p,sizeof p,"%s/model-extra.safetensors",dir); unlink(p);
    rmdir(dir);
}
static void test_stamp_conflicting_duplicate(void){
#ifndef _WIN32
    const char *dir="tests/tmp_fp8_stamp_conflict";
    write_stamp_fixture(dir, "\"{\\\"w\\\":\\\"fp8-e4m3-b128\\\"}\"");
    write_second_stamp_shard(dir, "\"{\\\"w\\\":\\\"int8-row\\\"}\"");
    /* refusal fires at DISCOVERY time (st_init's header-parse loop), before
     * any tensor is resolved -- fork st_init alone and capture stderr. */
    fflush(stdout);
    int pipefd[2]; if(pipe(pipefd)!=0){ CHECK(0); return; }
    pid_t pid = fork();
    if(pid < 0){ CHECK(0); return; }
    if(pid == 0){
        dup2(pipefd[1],2); close(pipefd[0]); close(pipefd[1]);
        static Model gm; memset(&gm,0,sizeof gm);
        st_init(&gm.S, dir);                   /* must exit(1) inside the ingest */
        _exit(42);                              /* surviving discovery is the bug */
    }
    close(pipefd[1]);
    char err[2048]={0}; size_t eoff=0; ssize_t n;
    while(eoff<sizeof(err)-1 && (n=read(pipefd[0],err+eoff,sizeof(err)-1-eoff))>0) eoff+=(size_t)n;
    close(pipefd[0]);
    int status=0; waitpid(pid,&status,0);
    int ok = WIFEXITED(status) && WEXITSTATUS(status)==1;
    if(!ok) printf("FAIL stamp-conflict: expected exit(1) at discovery, got status=%d, stderr=%.300s\n", status, err);
    CHECK(ok);
    /* the refusal must NAME the tensor and BOTH claims (either shard may be
     * enumerated first -- readdir order -- so assert both names, not roles). */
    CHECK(strstr(err,"conflicting format claims")!=NULL);
    CHECK(strstr(err,"'w'")!=NULL);
    CHECK(strstr(err,"fp8-e4m3-b128")!=NULL && strstr(err,"int8-row")!=NULL);
    CHECK(strstr(err,"refus")!=NULL);
    rm_two_shard_fixture(dir);
#else
    printf("skipped on Windows (no fork): stamp-conflict refusal\n");
#endif
}
static void test_stamp_agreeing_duplicate(void){
    const char *dir="tests/tmp_fp8_stamp_dupok";
    write_stamp_fixture(dir, "\"{\\\"w\\\":\\\"fp8-e4m3-b128\\\"}\"");
    write_second_stamp_shard(dir, "\"{\\\"w\\\":\\\"fp8-e4m3-b128\\\"}\"");
    static Model gm; memset(&gm,0,sizeof gm);
    st_init(&gm.S, dir);                        /* agreeing duplicate: no refusal */
    CHECK(gm.S.fmt_n == 1);                     /* collapsed to ONE entry, not two */
    QT t; memset(&t,0,sizeof t);
    qt_from_disk(&gm,"w",8,256,8,0,&t);
    CHECK(t.fmt==8);                            /* loads exactly as single-stamped */
    CHECK(t.q8!=NULL && t.s!=NULL);
    rm_two_shard_fixture(dir);
}

/* ---- Stamp-map scan bound (maintainer review, #529): st_fmt_stamp_ingest
 * (st.h) caps the number of stamped-tensor entries it will ingest across a
 * container's shards at ST_FMT_STAMP_MAX and refuses (exit(1)) past it --
 * see st.h's own comment and docs/FORMATS.md's "Stamp-map scan bound" for
 * why (stamps are a resident-tensor convention, never a bulk migration path
 * for the tens of thousands of routed-expert tensors a large MoE model
 * carries). Writes a minimal ONE-tensor shard (the shard's actual tensor
 * content is irrelevant to this check -- the cap fires during
 * st_init_multi's header-parse loop, at CONTAINER DISCOVERY time, before any
 * tensor is resolved against the model, see st_fmt_stamp_ingest's own
 * "DISCOVERY-TIME ABORT SURFACE" comment) carrying an oversized
 * __metadata__["colibri.fmt"] blob of N distinct, entirely fictitious tensor
 * names -- st_fmt_stamp_ingest parses that blob independently of the
 * shard's real tensor list, so the names never need to correspond to
 * anything real for the cap to trigger. Tests BOTH sides of the boundary:
 * exactly ST_FMT_STAMP_MAX entries must still load fine (no off-by-one
 * false refusal), one more must refuse. */
static void write_stamp_cap_fixture(const char *dir, int n_entries){
#ifdef _WIN32
    mkdir(dir);
#else
    mkdir(dir,0755);
#endif
    size_t inner_cap = (size_t)n_entries*40 + 64;
    char *inner = malloc(inner_cap);
    size_t p = 0; inner[p++]='{';
    for(int i=0;i<n_entries;i++)
        p += (size_t)snprintf(inner+p, inner_cap-p, "%s\"stamp_%d\":\"int8-row\"", i?",":"", i);
    inner[p++]='}'; inner[p]=0;

    /* JSON-escape `inner` into a quoted string literal for the OUTER header's
     * colibri.fmt VALUE (double-JSON-encoding, same shape write_stamp_fixture's
     * callers hand-escape for small fixtures -- built programmatically here
     * since n_entries is too many to hand-escape). */
    char *esc = malloc(p*2 + 4);
    size_t q = 0; esc[q++]='"';
    for(size_t i=0;i<p;i++){
        char c = inner[i];
        if(c=='"' || c=='\\') esc[q++]='\\';
        esc[q++]=c;
    }
    esc[q++]='"'; esc[q]=0;
    free(inner);

    enum { O=1, I=1 };
    static uint8_t q1[O*I]; static float s1[O];
    q1[0]=1; s1[0]=0.01f;
    char path[300]; snprintf(path,sizeof path,"%s/model.safetensors",dir);
    int64_t nb=(int64_t)O*I, ns=(int64_t)O*4;
    char *hdr = malloc(q + 512);
    int hl = snprintf(hdr, q+512,
        "{\"__metadata__\":{\"colibri.fmt\":%s},"
        "\"w\":{\"dtype\":\"U8\",\"shape\":[%lld],\"data_offsets\":[0,%lld]},"
        "\"w.qs\":{\"dtype\":\"F32\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}}",
        esc,
        (long long)nb,(long long)nb,
        (long long)(ns/4),(long long)nb,(long long)(nb+ns));
    free(esc);
    FILE *f=fopen(path,"wb");
    if(!f){ printf("FAIL: cannot create %s (run from c/, like tools/run_tests.py does)\n", path); fails++; free(hdr); return; }
    uint64_t hlen=(uint64_t)hl;
    fwrite(&hlen,8,1,f); fwrite(hdr,1,(size_t)hl,f);
    fwrite(q1,1,(size_t)nb,f); fwrite(s1,1,(size_t)ns,f);
    fclose(f);
    free(hdr);
}

static void test_stamp_map_cap_boundary_ok(void){
    const char *dir="tests/tmp_fp8_stamp_cap_ok";
    write_stamp_cap_fixture(dir, ST_FMT_STAMP_MAX);   /* exactly at the cap: must NOT refuse */
    static Model gm; memset(&gm,0,sizeof gm);
    st_init(&gm.S, dir);
    CHECK(gm.S.fmt_n == ST_FMT_STAMP_MAX);
    char p[300]; snprintf(p,sizeof p,"%s/model.safetensors",dir); unlink(p); rmdir(dir);
}

static void test_stamp_map_cap_exceeded(void){
    const char *dir="tests/tmp_fp8_stamp_cap_over";
    write_stamp_cap_fixture(dir, ST_FMT_STAMP_MAX+1);  /* one past the cap: must refuse */
#ifndef _WIN32
    int pipefd[2];
    if(pipe(pipefd)==0){
        pid_t pid = fork();
        if(pid == 0){
            dup2(pipefd[1],2); close(pipefd[0]); close(pipefd[1]);
            static Model gm; memset(&gm,0,sizeof gm);
            st_init(&gm.S, dir);   /* must exit(1) inside st_fmt_stamp_ingest's cap check */
            _exit(42);              /* reaching here is the bug */
        } else if(pid > 0){
            close(pipefd[1]);
            char err[1024]={0}; size_t eoff=0; ssize_t n; /* drain to EOF (Linux pipe short-reads; see expect_refuse) */ while(eoff<sizeof(err)-1 && (n=read(pipefd[0],err+eoff,sizeof(err)-1-eoff))>0) eoff+=(size_t)n;
            close(pipefd[0]);
            int status=0; waitpid(pid,&status,0);
            int ok = WIFEXITED(status) && WEXITSTATUS(status)==1;
            if(!ok) printf("FAIL stamp-map cap exceeded: expected exit(1), got status=%d, stderr=%.200s\n", status, err);
            CHECK(ok);
            if(ok && !strstr(err,"refus")){
                printf("FAIL stamp-map cap exceeded: exited(1) but message lacked a refusal explanation: %.200s\n", err);
                fails++;
            }
        } else fails++;
    } else fails++;
#else
    printf("skipped on Windows (no fork): stamp-map cap exceeded\n");
#endif
    char p[300]; snprintf(p,sizeof p,"%s/model.safetensors",dir); unlink(p); rmdir(dir);
}

int main(void){
    test_disambiguation();
    test_fmt6_fp8_collision();
    test_ue8m0_scale_refusal();
    test_ue8m0_family_sweep();
    test_loader_seam();
    check_fp8_bytes(2048,6144, "qt_bytes fmt=8 gate/up-shaped O=2048 I=6144 (spec example)");
    check_fp8_bytes(6144,2048, "qt_bytes fmt=8 down-shaped O=6144 I=2048");
    check_fp8_bytes(130,200,   "qt_bytes fmt=8 block edges O,I both non-mult-128");
    check_fp8_bytes(1,1,       "qt_bytes fmt=8 degenerate 1x1");
    check_wire_split(1, 4096,4096, 0,  "qt_wire_split fmt=1 plain int8 (per-row scale, unaffected by the fix)");
    check_wire_split(4, 2048,6144, 64, "qt_wire_split fmt=4 grouped int4 (O*ceil(I/gs) scale, not O*4)");
    check_wire_split(5, 2048,6144, 0,  "qt_wire_split fmt=5 int3-g64 (O*ceil(I/64) scale, not O*4)");
    check_wire_split(6, 2048,6144, 0,  "qt_wire_split fmt=6 E8/IQ3 (FIXED 4-byte tag, not O*4=8192B)");
    check_wire_split(6, 1,1,       0,  "qt_wire_split fmt=6 E8/IQ3 degenerate O=1 (O*4 would coincidentally also be 4 -- exercises the literal-4 assert, not just the moved-off-O*4 one)");
    check_wire_split(8, 2,16384, 0,    "qt_wire_split fmt=8 nblk(128) >> O(2): scale=512B, NOT O*4=8B");
    check_wire_split(8, 2048,6144, 0,  "qt_wire_split fmt=8 spec example: scale=3072B, NOT O*4=8192B");
    test_wire_site_regression();
    test_metal_fused_allowlist();
    test_fmt7_unreachable_sweep();
    test_stamp_agreeing();
    test_stamp_mismatching();
    test_stamp_unrecognized_name();
    test_stamp_absent();
    test_stamp_conflicting_duplicate();
    test_stamp_agreeing_duplicate();
    test_stamp_map_cap_boundary_ok();
    test_stamp_map_cap_exceeded();
    if(fails){ printf("fp8 loader-seam tests: %d FAILED\n", fails); return 1; }
    printf("fp8 loader-seam tests: ok\n");
    return 0;
}
