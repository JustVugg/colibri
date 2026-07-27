/* Cross-check tok.h's kimi family + rank-BPE against tiktoken.
 * Input: argv[1] = tokenizer.json, argv[2] = cases.bin
 *   cases.bin: u32 n, then n x { u32 len, bytes }  (raw UTF-8, newlines included)
 * Output: one line per case: space-separated token ids.
 * The python driver (tools/k3_tokenizer.py --ctest) generates cases.bin,
 * runs tiktoken on the same texts and diffs. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../tok.h"

int main(int argc, char **argv){
    if(argc<3){ fprintf(stderr,"usage: %s tokenizer.json cases.bin\n",argv[0]); return 2; }
    Tok T; tok_load(&T,argv[1]);
    fprintf(stderr,"family=%s rankbpe=%d\n",T.kimi?"kimi":(T.o200k?"o200k":"cl100k"),T.rankbpe);
    FILE *f=fopen(argv[2],"rb"); if(!f){ perror(argv[2]); return 2; }
    unsigned n;
    if(fread(&n,4,1,f)!=1){ fprintf(stderr,"bad cases file\n"); return 2; }
    for(unsigned i=0;i<n;i++){
        unsigned len;
        if(fread(&len,4,1,f)!=1||len>(1u<<20)){ fprintf(stderr,"bad case %u\n",i); return 2; }
        char *buf=malloc(len+1);
        if(fread(buf,1,len,f)!=len){ fprintf(stderr,"short case %u\n",i); return 2; }
        buf[len]=0;
        int ids[65536];
        int m=tok_encode(&T,buf,(int)len,ids,65536);
        for(int j=0;j<m;j++) printf("%d%c",ids[j],j+1<m?' ':'\n');
        if(m==0) printf("\n");
        free(buf);
    }
    fclose(f);
    return 0;
}
