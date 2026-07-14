#include "../backend_cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#ifdef _WIN32
/* MSVC has no POSIX setenv/unsetenv */
static int setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite; return _putenv_s(name, value);
}
static int unsetenv(const char *name) { return _putenv_s(name, ""); }
#endif

static int close_enough(const float *got, const float *want, int n) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(got[i] - want[i]) > 1e-4f) {
            std::fprintf(stderr, "mismatch %d: got %.6f want %.6f\n", i, got[i], want[i]);
            return 0;
        }
    }
    return 1;
}

static int relative_rms(const float *got,const float *want,int n,float limit){
    double err=0,ref=0; for(int i=0;i<n;i++){double d=got[i]-want[i];err+=d*d;ref+=(double)want[i]*want[i];}
    float r=(float)std::sqrt(err/(ref+1e-20));
    if(r>limit){std::fprintf(stderr,"relative RMS %.5f exceeds %.5f\n",r,limit);return 0;} return 1;
}

int main(int argc, char **argv) {
    int devices[COLI_CUDA_MAX_DEVICES], ndev = argc > 1 ? argc - 1 : 1;
    if (ndev > COLI_CUDA_MAX_DEVICES) return 2;
    for (int i = 0; i < ndev; i++) devices[i] = argc > 1 ? std::atoi(argv[i + 1]) : 0;
    if (!coli_cuda_init(devices, ndev)) return 77;
    if (coli_cuda_device_count() != ndev) return 1;
    int d0 = devices[0], d1 = devices[ndev > 1 ? 1 : 0];
    size_t count = 99, bytes = 99;
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;
    const float x[8] = {1, -2, 3, -4, 2, 1, -1, 0.5f};
    float got[4];

    const int8_t q8[8] = {1, 2, 3, 4, -1, 2, -3, 4};
    const float s8[2] = {0.5f, 2.0f};
    const float want8[4] = {-5.0f, -60.0f, 1.5f, 10.0f};
    ColiCudaTensor *t8 = nullptr;
    if (!coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d0)) return 1;
    if (coli_cuda_tensor_upload(&t8, q8, s8, 1, 5, 2, d0)) return 1;
    if (ndev > 1 && coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d1)) return 1;
    if (!coli_cuda_matmul(&t8, got, x, q8, s8, 1, 2, 4, 2, d0) || !close_enough(got, want8, 4)) return 1;
    const int8_t q8b[8]={-1,-2,-3,-4, 1,-2,3,-4};
    const float s8b[2]={1.f,.5f},want8b[4]={10.f,15.f,-3.f,-2.5f};
    if(!coli_cuda_tensor_update(t8,q8b,s8b)||
       !coli_cuda_matmul(&t8,got,x,q8b,s8b,1,2,4,2,d0)||
       !close_enough(got,want8b,4))return 1;

    /* Rows [-8,-1,0,7] and [1,2,3,4], packed low nibble first. */
    const uint8_t q4[4] = {0x70, 0xf8, 0xa9, 0xcb};
    const float s4[2] = {1.0f, 0.25f};
    const float want4[2] = {-34.0f, -2.5f};
    ColiCudaTensor *t4 = nullptr;
    if (!coli_cuda_matmul(&t4, got, x, q4, s4, 2, 1, 4, 2, d1) || !close_enough(got, want4, 2)) return 1;

    const uint8_t q2[2] = {0xe4, 0x1b};
    const float s2[2] = {0.5f, 2.0f};
    const float want2[2] = {-2.0f, 12.0f};
    ColiCudaTensor *t2 = nullptr;
    if (!coli_cuda_matmul(&t2, got, x, q2, s2, 3, 1, 4, 2, d1) || !close_enough(got, want2, 2)) return 1;

    const float wf[8] = {1, 0, -1, 2, 0.5f, 0.5f, 0.5f, 0.5f};
    const float wantf[2] = {-10.0f, -1.0f};
    ColiCudaTensor *tf = nullptr;
    if (!coli_cuda_matmul(&tf, got, x, wf, nullptr, 0, 1, 4, 2, d0) || !close_enough(got, wantf, 2)) return 1;

    const float eg[8] = {1,0,0,0, 0,1,0,0};
    const float eu[8] = {1,0,0,0, 0,1,0,0};
    const float ed[8] = {1,0, 0,1, 1,1, 1,-1};
    ColiCudaTensor *tg=nullptr,*tu=nullptr,*td=nullptr;
    if (!coli_cuda_tensor_upload(&tg,eg,nullptr,0,4,2,d0) ||
        !coli_cuda_tensor_upload(&tu,eu,nullptr,0,4,2,d0) ||
        !coli_cuda_tensor_upload(&td,ed,nullptr,0,2,4,d0)) return 1;
    float expert[8], want_expert[8];
    for(int s=0;s<2;s++){
        float a=x[s*4], b=x[s*4+1];
        a=(a/(1.0f+std::exp(-a)))*a; b=(b/(1.0f+std::exp(-b)))*b;
        want_expert[s*4]=a; want_expert[s*4+1]=b;
        want_expert[s*4+2]=a+b; want_expert[s*4+3]=a-b;
    }
    if (!coli_cuda_expert_mlp(tg,tu,td,expert,x,2) ||
        !close_enough(expert,want_expert,8)) return 1;
    ColiCudaTensor *gates[2]={tg,tg},*ups[2]={tu,tu},*downs[2]={td,td};
    int group_rows[2]={1,1}; float grouped[8];
    if (!coli_cuda_expert_group(gates,ups,downs,group_rows,2,grouped,x) ||
        !close_enough(grouped,want_expert,8)) return 1;

    const float aw[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    const float aq[4]={1,2,.5f,-.5f},al[12]={1,0,0,0, 0,1,0,0, 0,0,1,0};
    const float ar[6]={1,0, 0,1, 1,1};float actx[2],aref[2];
    ColiCudaTensor *at=nullptr;if(!coli_cuda_tensor_upload(&at,aw,nullptr,0,4,4,d0))return 1;
    float score[3];for(int t=0;t<3;t++)score[t]=aq[0]*al[t*4]+aq[1]*al[t*4+1]+aq[2]*ar[t*2]+aq[3]*ar[t*2+1];
    float mx=score[0],z=0;for(int t=1;t<3;t++)mx=score[t]>mx?score[t]:mx;
    for(int t=0;t<3;t++){score[t]=std::exp(score[t]-mx);z+=score[t];}for(int t=0;t<3;t++)score[t]/=z;
    for(int v=0;v<2;v++){aref[v]=0;for(int t=0;t<3;t++)aref[v]+=score[t]*al[t*4+2+v];}
    if(!coli_cuda_attention_absorb(at,actx,aq,al,ar,1,2,2,2,4,3,1.f)||
       !close_enough(actx,aref,2))return 1;
    coli_cuda_tensor_free(at);

    /* Native s4 WMMA path: compare the quantized-activation result against the
       existing FP32-activation/s4-weight grouped implementation. */
    uint8_t w4[32*32/2]; float ws4[32], gx4[64], scalar4[64], tensor4[64];
    for(int i=0;i<(int)sizeof(w4);i++){
        int lo=((i%15)-7)&15,hi=(((i*3)%15)-7)&15;
        w4[i]=(uint8_t)(lo|(hi<<4));
    }
    for(int i=0;i<32;i++)ws4[i]=0.01f+(i%5)*0.002f;
    for(int i=0;i<64;i++)gx4[i]=std::sin((float)(i+1)*0.17f)*2.f;
    ColiCudaTensor *g4=nullptr,*u4=nullptr,*d4=nullptr;
    if(!coli_cuda_tensor_upload(&g4,w4,ws4,2,32,32,d0)||
       !coli_cuda_tensor_upload(&u4,w4,ws4,2,32,32,d0)||
       !coli_cuda_tensor_upload(&d4,w4,ws4,2,32,32,d0))return 1;
    ColiCudaTensor *gg4[2]={g4,g4},*ug4[2]={u4,u4},*dg4[2]={d4,d4};
    if(!coli_cuda_expert_group(gg4,ug4,dg4,group_rows,2,scalar4,gx4))return 1;
    setenv("COLI_CUDA_TC_INT4","1",1);
    setenv("COLI_CUDA_TC_MIN_ROWS","1",1);
    if(!coli_cuda_expert_group(gg4,ug4,dg4,group_rows,2,tensor4,gx4)||
       !relative_rms(tensor4,scalar4,64,0.30f))return 1;
    unsetenv("COLI_CUDA_TC_INT4");
    unsetenv("COLI_CUDA_TC_MIN_ROWS");
    coli_cuda_tensor_free(g4);coli_cuda_tensor_free(u4);coli_cuda_tensor_free(d4);
    uint64_t group_calls=0,group_experts=0,group_total_rows=0;
    coli_cuda_group_stats(&group_calls,&group_experts,&group_total_rows,nullptr,nullptr,nullptr);
    if(group_calls!=3||group_experts!=6||group_total_rows!=6) return 1;

    coli_cuda_stats(-1, &count, &bytes);
    if (count != 7 || bytes != 166) {
        std::fprintf(stderr, "unexpected CUDA stats: %zu tensors, %zu bytes\n", count, bytes);
        return 1;
    }
    if (coli_cuda_tensor_device(t8) != d0 || coli_cuda_tensor_device(tf) != d0 ||
        coli_cuda_tensor_device(t4) != d1 || coli_cuda_tensor_device(t2) != d1) return 1;
    coli_cuda_stats(d0, &count, &bytes);
    if (ndev > 1) {
        if (count != 5 || bytes != 144) return 1;
        coli_cuda_stats(d1, &count, &bytes);
        if (count != 2 || bytes != 22) return 1;
    } else if (count != 7 || bytes != 166) return 1;

    coli_cuda_tensor_free(t8);
    coli_cuda_tensor_free(t4);
    coli_cuda_tensor_free(t2);
    coli_cuda_tensor_free(tf);
    coli_cuda_tensor_free(tg);
    coli_cuda_tensor_free(tu);
    coli_cuda_tensor_free(td);
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;
    coli_cuda_shutdown();
    std::printf("cuda backend: q8/q4/q2/f32 correctness ok on %d device(s)\n", ndev);
    return 0;
}
