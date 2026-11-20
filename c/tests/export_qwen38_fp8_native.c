/* Matched native input packets -> the existing stage fixture format.
 * Use the actual loader's BF16/F16/F32 scale conversion and actual CPU math.
 * No model load or GPU initialization. The original native NLL stays the gate.
 */
#define QFS_NO_MAIN
#include "export_qwen38_fp8_stages.c"

static size_t qfn_read_bytes,qfn_input_bytes;
static void qfn_read(FILE *f,void *out,size_t bytes) {
    qfs_require(bytes<=qfn_input_bytes-qfn_read_bytes,"input packet length");
    qfs_require(fread(out,1,bytes,f)==bytes,"complete input read");qfn_read_bytes+=bytes;
}
static void qfn_case(FILE *input,FILE *output,uint32_t expected_id) {
    uint32_t h[5];qfn_read(input,h,sizeof(h));
    qfs_require(h[0]==expected_id&&h[1]==2560&&h[2]==640&&h[3]==1&&
        h[4]==QFS_NATIVE_PATTERN,"reviewed native case geometry/order");
    const int D=(int)h[1],I=(int)h[2],S=(int)h[3];
    const size_t nq=(size_t)D*I,ns=(size_t)((D+127)/128)*((I+127)/128);
    const size_t nx=(size_t)S*D,nh=(size_t)S*I;
    uint8_t *q[3];float *sc[3];
    for(int w=0;w<3;w++){
        q[w]=qfs_alloc(nq);qfn_read(input,q[w],nq);
        qfs_require(!memchr(q[w],0x7f,nq)&&!memchr(q[w],0xff,nq),"finite native FP8 weights");
        uint32_t dtype;qfn_read(input,&dtype,sizeof(dtype));
        qfs_require(dtype<=2,"native scale dtype (BF16/F16/F32)");
        size_t nb=ns*(dtype==2?4:2);uint8_t *raw=qfs_alloc(nb);qfn_read(input,raw,nb);
        sc[w]=qfs_alloc(ns*sizeof(float));
        st_tensor tensor={0};tensor.dtype=(int)dtype;tensor.numel=(int64_t)ns;tensor.nbytes=(int64_t)nb;
        q38_decode_scale_tensor(sc[w],raw,&tensor);
        for(size_t j=0;j<ns;j++)qfs_require(isfinite(sc[w][j])&&sc[w][j]>0,"finite positive converted scales");
        free(raw);
    }
    float *x=qfs_alloc(nx*sizeof(float));qfn_read(input,x,nx*sizeof(float));
    for(size_t j=0;j<nx;j++)qfs_require(isfinite(x[j]),"finite native activation");
    float *v[4],*t[4];
    for(int k=0;k<4;k++){
        size_t n=k==3?nx:nh;v[k]=qfs_alloc(n*sizeof(float));t[k]=qfs_alloc(n*sizeof(float));
    }
    printf("NATIVE_CPU_CASE START id=%u D=%d I=%d S=%d\n",h[0],D,I,S);fflush(stdout);
    q38_matmul_fp8_vec(v[0],x,q[0],sc[0],S,D,I);
    q38_matmul_fp8_vec(v[1],x,q[1],sc[1],S,D,I);
    matmul_fp8(t[0],x,q[0],sc[0],S,D,I);
    matmul_fp8(t[1],x,q[1],sc[1],S,D,I);
    for(size_t j=0;j<nh;j++){
        v[2][j]=q38_silu(v[0][j])*v[1][j];
        t[2][j]=q38_silu(t[0][j])*t[1][j];
    }
    q38_matmul_fp8_vec(v[3],v[2],q[2],sc[2],S,I,D);
    matmul_fp8(t[3],t[2],q[2],sc[2],S,I,D);
    const char *names[4]={"gate","up","hidden","down"};
    for(int k=0;k<4;k++)qfs_compare(names[k],v[k],t[k],k==3?nx:nh);
    qfs_write(output,h,4,5);
    for(int w=0;w<3;w++){qfs_write(output,q[w],1,nq);qfs_write(output,sc[w],4,ns);}
    qfs_write(output,x,4,nx);
    for(int k=0;k<4;k++)qfs_write(output,v[k],4,k==3?nx:nh);
    for(int k=0;k<4;k++)qfs_write(output,t[k],4,k==3?nx:nh);
    for(int w=0;w<3;w++){free(q[w]);free(sc[w]);}
    for(int k=0;k<4;k++){free(v[k]);free(t[k]);}free(x);
    printf("NATIVE_CPU_CASE END id=%u bytes_so_far=%zu\n",h[0],qfs_written);fflush(stdout);
}
int main(int argc,char **argv) {
    qfs_require(argc==3,"usage: export_qwen38_fp8_native INPUT.bin NEW_OUTPUT.bin");
    uint32_t endian=1;
    qfs_require(*(uint8_t*)&endian==1&&sizeof(float)==4&&FLT_RADIX==2&&FLT_MANT_DIG==24,
        "little-endian IEEE binary32 host");
    qfs_environment();
    FILE *input=fopen(argv[1],"rb");qfs_require(input!=NULL,"input file");
    struct stat st;qfs_require(fstat(fileno(input),&st)==0&&st.st_size>=16&&st.st_size<=QFS_MAX_BYTES,
        "bounded input size");qfn_input_bytes=(size_t)st.st_size;
    char magic[8];uint32_t version,count;
    qfn_read(input,magic,8);qfn_read(input,&version,4);qfn_read(input,&count,4);
    qfs_require(!memcmp(magic,QFS_INPUT_MAGIC,8)&&version==QFS_VERSION&&count>0&&count<=QFS_MAX_CASES,
        "native input header");
    char part[PATH_MAX],dir[PATH_MAX];qfs_require(strlen(argv[2])+6<sizeof(part),"output path bound");
    errno=0;qfs_require(lstat(argv[2],&st)<0&&errno==ENOENT,"output must not exist");
    snprintf(part,sizeof(part),"%s.part",argv[2]);
    int fd=open(part,O_WRONLY|O_CREAT|O_EXCL,0600);qfs_require(fd>=0,"exclusive partial file");
    FILE *output=fdopen(fd,"wb");qfs_require(output!=NULL,"fdopen");
    qfs_write(output,QFS_MAGIC,1,8);qfs_write(output,&version,4,1);qfs_write(output,&count,4,1);
    for(uint32_t id=0;id<count;id++)qfn_case(input,output,id);
    qfs_require(qfn_read_bytes==qfn_input_bytes&&fgetc(input)==EOF&&!ferror(input),"no trailing input");
    qfs_require(fclose(input)==0,"close input");
    qfs_require(fflush(output)==0&&fsync(fd)==0,"flush complete output");
    qfs_require(fclose(output)==0,"close output");
    qfs_require(link(part,argv[2])==0,"publish without replacement");qfs_require(unlink(part)==0,"remove partial name");
    strcpy(dir,argv[2]);char *slash=strrchr(dir,'/');
    if(!slash)strcpy(dir,".");else if(slash==dir)slash[1]=0;else *slash=0;
    int dfd=open(dir,O_RDONLY|O_DIRECTORY);qfs_require(dfd>=0,"output directory");
    qfs_require(fsync(dfd)==0&&close(dfd)==0,"sync output directory");
    printf("NATIVE_CPU_FIXTURE COMPLETE cases=%u bytes=%zu path=%s\n",count,qfs_written,argv[2]);
    return 0;
}
