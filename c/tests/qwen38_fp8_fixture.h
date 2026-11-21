/* Diagnostic interchange, little-endian IEEE binary32. No engine ABI changes.
 * Header: magic[8], uint32 version, uint32 case_count.
 * Each case: uint32 id,D,I,S,pattern; gate(q[D*I],scale[nbD*nbI]),
 * up(q[D*I],scale[nbD*nbI]), down(q[I*D],scale[nbI*nbD]); x[S*D];
 * vector gate/up/hidden[S*I], vector down[S*D]; then the same four
 * outputs from the existing table kernel. Down scales are row-major nbD x nbI;
 * gate/up scales are row-major nbI x nbD. No padding or trailing bytes.
 * CPU fixtures are diagnostics; they do not replace native quality references.
 */
#ifndef QWEN38_FP8_FIXTURE_H
#define QWEN38_FP8_FIXTURE_H
#define QFS_MAGIC "Q38F8D1\0"
#define QFS_INPUT_MAGIC "Q38F8I1\0"
#define QFS_NATIVE_PATTERN 3u
#define QFS_VERSION 1u
#define QFS_MAX_CASES 64u
#define QFS_MAX_DIM 4096u
#define QFS_MAX_ROWS 9u
#define QFS_MAX_BYTES (256u * 1024u * 1024u)
#endif
