#!/usr/bin/env python3
"""Extract the actual MXFP4 unpacker and check it against the E2M1 formula.

Run inside the CPU-only builder, after the Light performance run has stopped.
The generated C and logs are evidence artifacts. Nothing touches model files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

C_TEST=r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
UNPACK_FUNCTION
static int expected(unsigned code) {
    unsigned e=code&7;
    int magnitude=e<2 ? (int)e : (2+(int)(e&1)) << ((e>>1)-1);
    return code&8 ? -magnitude : magnitude;
}
int main(void) {
    const int widths[]={0,1,2,3,14,16,18,30,31,32,33,34,62,64,65,66,96,128,512,2048,4096,65536};
    unsigned long cases=0,values=0;
    for (unsigned wi=0;wi<sizeof(widths)/sizeof(*widths);wi++) {
        int n=widths[wi], bytes=n/2, seeds=n>128?16:256, offsets=n>128?4:32;
        size_t insz=(size_t)bytes+96, outsz=(size_t)n+96;
        uint8_t *in=malloc(insz),*saved=malloc(insz),*out=malloc(outsz);
        if (!in || !saved || !out) return 2;
        for (int offset=0;offset<offsets;offset++) for (int seed=0;seed<seeds;seed++) {
            memset(in,0xa5,insz);memset(out,0x55,outsz);
            uint8_t *row=in+32+offset;int8_t *w=(int8_t *)(out+32+31-offset);
            for (int j=0;j<bytes;j++) row[j]=(uint8_t)(j*17+seed);
            memcpy(saved,in,insz);dev_codes_fp4(w,row,n);
            for (int j=0;j<2*bytes;j++) {
                int want=expected((row[j/2] >> (4*(j&1)))&15);
                if (w[j]!=want) { fprintf(stderr,"value n=%d offset=%d seed=%d j=%d got=%d want=%d\n",n,offset,seed,j,w[j],want);return 1; }
                values++;
            }
            size_t first=(uint8_t *)w-out,last=first+(size_t)2*bytes;
            for (size_t j=0;j<outsz;j++) if ((j<first || j>=last) && out[j]!=0x55) {
                fprintf(stderr,"overwrite n=%d offset=%d at=%zu\n",n,offset,j);return 1;
            }
            if (memcmp(in,saved,insz)) { fputs("input modified\n",stderr);return 1; }
            cases++;
        }
        free(in);free(saved);free(out);
    }
    printf("PASS %lu cases, %lu exact E2M1 values; unaligned input/output, signed zero, all codes, vector/tail boundaries, guards and input preservation\n",cases,values);
    return 0;
}
'''


def main():
    a=argparse.ArgumentParser();a.add_argument('source',type=Path);a.add_argument('output',type=Path)
    a.add_argument('--cc',default='gcc');a.add_argument('--arch',default='x86-64-v3');args=a.parse_args()
    source=args.source.read_text();start=source.index('static void dev_codes_fp4(')
    end=source.index('\nstatic float dev_fp8',start)
    function=source[start:end].strip()
    args.output.mkdir(parents=True,exist_ok=False)
    generated=args.output/'probe.c';generated.write_text(C_TEST.replace('UNPACK_FUNCTION',function))
    binary=args.output/'probe'
    cmd=[args.cc,'-std=c11','-O3',f'-march={args.arch}','-Wall','-Wextra','-Werror',str(generated),'-o',str(binary)]
    compile_run=subprocess.run(cmd,text=True,capture_output=True)
    (args.output/'compile.log').write_text(compile_run.stdout+compile_run.stderr)
    compile_run.check_returncode()
    result=subprocess.run([str(binary)],text=True,capture_output=True,timeout=120)
    (args.output/'result.log').write_text(result.stdout+result.stderr)
    (args.output/'meta.json').write_text(json.dumps(dict(source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        extracted_sha256=hashlib.sha256(function.encode()).hexdigest(),command=cmd,exit_code=result.returncode),indent=2)+'\n')
    print(result.stdout+result.stderr,end='');result.check_returncode()


if __name__=='__main__':main()
