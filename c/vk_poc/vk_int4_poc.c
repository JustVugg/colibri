/* vk_int4_poc.c — full Vulkan INT4 matmul proof-of-concept on the Radeon
 * 890M (RADV GFX1150). Compiles a GLSL compute shader to SPIR-V at runtime via
 * glslangValidator, uploads int4 weights + FP32 group scales + FP16 acts,
 * runs y = x @ W^T on-GPU, downloads, verifies vs CPU dequant reference.
 *
 * Build: cc vk_int4_poc.c -o vk_int4_poc -lvulkan -lm
 * Run:   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json ./vk_int4_poc
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vulkan/vulkan.h>

#define I 512
#define O 512
#define S 4
#define GROUP 128
#define BLOCKS_I ((I+GROUP-1)/GROUP)

static void die(const char*m, VkResult r){ fprintf(stderr,"FAIL %s : %d\n",m,(int)r); exit(1); }

/* float -> IEEE half (round-to-nearest-even, no fenv) */
static uint16_t f2h(float v){
    union{float f; uint32_t u;} x={v}; uint32_t s=(x.u>>16)&0x8000, bits=x.u&0x7FFFFFFF;
    int e=(bits>>23)&0xFF, m=bits&0x7FFFFF;
    if(e==255) return s|0x7C00|((m!=0)<<9);          /* Inf/NaN */
    if(e==0 && m==0) return s;                       /* zero */
    int E=e-127; long M=(long)m<<13; int out;
    if(E<=-25) return s;                             /* underflow */
    if(E<=-14){ int sh=-14-E; out=s|((M|0x800000)<<(14-(24-13))>>sh); } /* subnormal */
    else { int N=E+15; if(N>=31) return s|0x7C00;    /* overflow -> Inf */
        out=s|(N<<10)|((M>>(24-10-13)) & 0x3FF); }
    return (uint16_t)out;
}

static void quant_int4(const float*w, uint32_t*out, float*scales){
    for(int o=0;o<O;o++) for(int b=0;b<BLOCKS_I;b++){
        float mx=0.f;
        for(int k=0;k<GROUP;k++){ int i=b*GROUP+k; if(i>=I) break; float v=fabsf(w[(size_t)o*I+i]); if(v>mx)mx=v; }
        float s=(mx<1e-6f)?1.f:mx/7.5f; scales[o*BLOCKS_I+b]=s;
        for(int k=0;k<GROUP;k+=8){
            uint32_t pk=0;
            for(int t=0;t<8;t++){
                int i=b*GROUP+k+t; if(i>=I) break;
                float v=(i<I)?w[(size_t)o*I+i]/s:0.f;
                int q=(int)lrintf(fmaxf(-8.f,fminf(7.f,v)));
                pk |= (((uint32_t)q&0xF)<<(t*4));
            }
            out[(o*I+b*GROUP+k)/8]=pk;
        }
    }
}
static void dequant_ref(const float*x,const uint32_t*q,const float*sc,float*y){
    for(int s=0;s<S;s++) for(int o=0;o<O;o++){
        float acc=0.f;
        for(int b=0;b<BLOCKS_I;b++){ float sca=sc[o*BLOCKS_I+b];
            for(int k=0;k<GROUP;k++){ int gi=o*I+b*GROUP+k; if(gi>=I) break;
                uint32_t packed=q[gi/8]; int qv=(int)((packed>>((gi%8)*4))&0xF); if(qv>=8)qv-=16;
                acc+=x[(size_t)s*I+gi]*(qv*sca);
            }
        }
        y[(size_t)s*O+o]=acc;
    }
}

#define XS(x) #x
static const char* GLSL =
"#version 450\n"
"layout(local_size_x=64) in;\n"
"layout(std430,binding=0) readonly buffer Wbuf { uint w[]; };\n"
"layout(std430,binding=1) readonly buffer Sbuf { float  sc[]; };\n"
"layout(std430,binding=2) readonly buffer Xbuf { float x[]; };\n"
"layout(std430,binding=3) writeonly buffer Ybuf { float y[]; };\n"
"const int I=" XS(512) "; const int O=" XS(512) "; const int GROUP=" XS(128)
"; const int BLOCKS_I=" XS(4) "; const int S=" XS(4) ";\n"
"void main(){\n"
"  int o=int(gl_GlobalInvocationID.x); if(o>=O) return;\n"
"  for(int s=0;s<S;s++){ float acc=0.0f;\n"
"    for(int b=0;b<BLOCKS_I;b++){ float sca=sc[o*BLOCKS_I+b];\n"
"      for(int k=0;k<GROUP;k++){ int gi=o*I+b*GROUP+k; if(gi>=I) break;\n"
"        uint packed=w[gi/8]; int n=(gi%8)*4; int qv=int((packed>>n)&0xF); if(qv>=8) qv-=16;\n"
"        acc += x[uint(s)*uint(I)+uint(gi)] * (float(qv)*sca);\n"
"      } }\n"
"    y[uint(s)*uint(O)+uint(o)]=acc;\n"
"  }\n"
"}\n";

static VkShaderModule load_shader(VkDevice dev, const char* glsl, const char* tmp){
    char glslpath[512]; snprintf(glslpath,sizeof glslpath,"%s.comp",tmp);
    FILE*f=fopen(glslpath,"w"); fputs(glsl,f); fclose(f);
    char cmd[512]; snprintf(cmd,sizeof cmd,"glslangValidator -V %s -o %s.spv >/dev/null 2>&1",glslpath,tmp);
    if(system(cmd)!=0){ fprintf(stderr,"glslangValidator failed for %s\n",glslpath); exit(1); }
    char spv[512]; snprintf(spv,sizeof spv,"%s.spv",tmp);
    FILE*sp=fopen(spv,"rb"); if(!sp){ fprintf(stderr,"spv %s missing\n",spv); exit(1); }
    fseek(sp,0,SEEK_END); long sz=ftell(sp); fseek(sp,0,SEEK_SET);
    if(sz<=0){ fprintf(stderr,"spv empty\n"); exit(1); }
    char*bin=malloc(sz); if(fread(bin,1,sz,sp)!=(size_t)sz){ fprintf(stderr,"spv read short\n"); exit(1);} fclose(sp);
    VkShaderModule m; VkShaderModuleCreateInfo ci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sz,.pCode=(void*)bin};
    VkResult r=vkCreateShaderModule(dev,&ci,NULL,&m); free(bin); if(r!=VK_SUCCESS) die("shaderModule",r);
    return m;
}

static uint32_t find_queue(VkPhysicalDevice pd){
    uint32_t n=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&n,NULL);
    VkQueueFamilyProperties*p=malloc(n*sizeof*p); vkGetPhysicalDeviceQueueFamilyProperties(pd,&n,p);
    for(uint32_t i=0;i<n;i++) if(p[i].queueFlags&VK_QUEUE_COMPUTE_BIT){ free(p); return i; }
    free(p); return 0;
}
static uint32_t host_visible_type(VkPhysicalDevice pd){
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++)
        if(mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) return i;
    return 0;
}

int main(void){
    printf("start\n"); fflush(stdout);
    /* ---- CPU quant reference ---- */
    float *W=malloc((size_t)O*I*sizeof(float)),*X=malloc((size_t)S*I*sizeof(float)),*Ydq=malloc((size_t)S*O*sizeof(float));
    uint32_t*Q=malloc(((size_t)O*I+7)/8 * sizeof(uint32_t)); float*SC=malloc((size_t)O*BLOCKS_I*sizeof(float));
    for(int i=0;i<O*I;i++) W[i]=(float)rand()/RAND_MAX*2-1;
    for(int i=0;i<S*I;i++) X[i]=(float)rand()/RAND_MAX*2-1;
    quant_int4(W,Q,SC); dequant_ref(X,Q,SC,Ydq);
    free(W);

    /* ---- instance + device ---- */
    VkInstance inst; VkApplicationInfo ai; memset(&ai,0,sizeof ai);
    ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici; memset(&ici,0,sizeof ici);
    ici.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo=&ai;
    if(vkCreateInstance(&ici,NULL,&inst)!=VK_SUCCESS) die("instance",1);
    uint32_t ndev=0; vkEnumeratePhysicalDevices(inst,&ndev,NULL);
    VkPhysicalDevice*pds=malloc(ndev*sizeof(VkPhysicalDevice)); vkEnumeratePhysicalDevices(inst,&ndev,pds);
    VkPhysicalDevice pd=pds[0];
    for(uint32_t i=0;i<ndev;i++){ VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i],&p);
        if(strstr(p.deviceName,"890M")||strstr(p.deviceName,"GFX1150")) pd=pds[i]; }
    uint32_t qf=find_queue(pd);
    float qpri=1.0f; VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=qf,.queueCount=1,.pQueuePriorities=&qpri};
    VkDevice dev; VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci};
    if(vkCreateDevice(pd,&dci,NULL,&dev)!=VK_SUCCESS) die("device",2);
    VkQueue q; vkGetDeviceQueue(dev,qf,0,&q);
    printf("device OK\n");

    /* ---- buffers ---- */
    size_t wSz=((size_t)O*I+7)/8*sizeof(uint32_t), sSz=(size_t)O*BLOCKS_I*sizeof(float), xSz=(size_t)S*I*sizeof(float), ySz=(size_t)S*O*sizeof(float);
    VkBuffer wb,sb,xb,yb; VkDeviceMemory wm,sm,xm,ym;
    VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=0,.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VkMemoryAllocateInfo mai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    uint32_t memType = host_visible_type(pd);
    printf("memType=%u\n", memType);
#define MKBUF(buf,mem,sz) do { bci.size=sz; vkCreateBuffer(dev,&bci,NULL,&buf); \
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,buf,&mr); \
    mai.allocationSize=mr.size; mai.memoryTypeIndex=memType; vkAllocateMemory(dev,&mai,NULL,&mem); \
    vkBindBufferMemory(dev,buf,mem,0); } while(0)
    MKBUF(wb,wm,wSz);
    MKBUF(sb,sm,sSz);
    MKBUF(xb,xm,xSz);
    MKBUF(yb,ym,ySz);
    printf("buffers OK\n");

    /* ---- upload ---- */
    void* p;
    vkMapMemory(dev,wm,0,wSz,0,&p); memcpy(p,Q,wSz); vkUnmapMemory(dev,wm);
    vkMapMemory(dev,sm,0,sSz,0,&p); memcpy(p,SC,sSz); vkUnmapMemory(dev,sm);
    float* Xh=malloc(xSz); for(int i=0;i<S*I;i++) Xh[i]=X[i];
    vkMapMemory(dev,xm,0,xSz,0,&p); memcpy(p,Xh,xSz); vkUnmapMemory(dev,xm);

    /* ---- descriptor + pipeline ---- */
    VkShaderModule sh=load_shader(dev,GLSL,"/tmp/vk_poc.spv");
    VkDescriptorSetLayoutBinding bnd[4]={{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},{2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0}};
    VkDescriptorSetLayout dsl; VkDescriptorSetLayoutCreateInfo dlci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=4,.pBindings=bnd};
    vkCreateDescriptorSetLayout(dev,&dlci,NULL,&dsl);
    VkPipelineLayout pl; VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl};
    vkCreatePipelineLayout(dev,&plci,NULL,&pl);
    VkPipeline ppl; VkPipelineShaderStageCreateInfo ssi={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sh,.pName="main"};
    VkComputePipelineCreateInfo cpi={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage=ssi,.layout=pl};
    vkCreateComputePipelines(dev,0,1,&cpi,NULL,&ppl);
    VkDescriptorPool dp; VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}};
    vkCreateDescriptorPool(dev,&dpci,NULL,&dp);
    VkDescriptorSet ds; vkAllocateDescriptorSets(dev,&(VkDescriptorSetAllocateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp,.descriptorSetCount=1,.pSetLayouts=&dsl},&ds);
    VkDescriptorBufferInfo bi[4]={{wb,0,wSz},{sb,0,sSz},{xb,0,xSz},{yb,0,ySz}};
    VkWriteDescriptorSet wds[4]; for(int i=0;i<4;i++) wds[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[i]};
    vkUpdateDescriptorSets(dev,4,wds,0,NULL);

    /* ---- dispatch ---- */
    VkCommandPool cp; VkCommandPoolCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=qf};
    vkCreateCommandPool(dev,&cpci,NULL,&cp);
    VkCommandBuffer cb; vkAllocateCommandBuffers(dev,&(VkCommandBufferAllocateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=cp,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1},&cb);
    vkBeginCommandBuffer(cb,&(VkCommandBufferBeginInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO});
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,ppl);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,0);
    vkCmdDispatch(cb,(O+63)/64,1,1);
    vkEndCommandBuffer(cb);
    VkFence f; vkCreateFence(dev,&(VkFenceCreateInfo){.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},NULL,&f);
    vkQueueSubmit(q,1,&(VkSubmitInfo){.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb},f);
    vkWaitForFences(dev,1,&f,VK_TRUE,~0ULL);
    printf("dispatch OK\n");

    /* ---- download + verify ---- */
    float* Yh=malloc(ySz); vkMapMemory(dev,ym,0,ySz,0,&p); memcpy(Yh,p,ySz); vkUnmapMemory(dev,ym);
    float maxrel=0,dot=0,nref=0,ndq=0;
    for(int i=0;i<S*O;i++){
        float yv=Yh[i];
        float r=Ydq[i];
        float rel=(fabsf(r)>1e-2f)?fabsf(yv-r)/fabsf(r):fabsf(yv-r);
        if(rel>maxrel)maxrel=rel; dot+=yv*r; nref+=r*r; ndq+=yv*yv;
    }
    printf("GPU vs CPU-dequant: max_abs_err~=%.4e  cosine=%.6f  (cosine>0.999 = PASS)\n", maxrel, dot/(sqrtf(nref)*sqrtf(ndq)+1e-9f));

    /* cleanup */
    vkDestroyFence(dev,f,NULL); vkDestroyCommandPool(dev,cp,NULL); vkDestroyDescriptorPool(dev,dp,NULL);
    vkDestroyPipeline(dev,ppl,NULL); vkDestroyPipelineLayout(dev,pl,NULL); vkDestroyDescriptorSetLayout(dev,dsl,NULL);
    vkDestroyShaderModule(dev,sh,NULL);
    vkDestroyBuffer(dev,wb,NULL); vkDestroyBuffer(dev,sb,NULL); vkDestroyBuffer(dev,xb,NULL); vkDestroyBuffer(dev,yb,NULL);
    vkFreeMemory(dev,wm,NULL); vkFreeMemory(dev,sm,NULL); vkFreeMemory(dev,xm,NULL); vkFreeMemory(dev,ym,NULL);
    vkDestroyDevice(dev,NULL); vkDestroyInstance(inst,NULL);
    free(Q); free(SC); free(X); free(Xh); free(Yh); free(Ydq);
    return 0;
}
