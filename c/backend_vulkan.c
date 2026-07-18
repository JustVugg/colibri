/* backend_vulkan.c — colibri Vulkan backend for the Radeon 890M / any Vulkan GPU.
 *
 * Implements the SAME ABI as backend_cuda.h (the coli_cuda_* symbols) but
 * against Vulkan instead of CUDA. glm.c's existing #ifdef COLI_CUDA dispatch
 * calls these names unchanged; we build with -DCOLI_CUDA and link this file
 * instead of backend_cuda.cu, so the engine needs zero changes.
 *
 * Scope (first working version):
 *   - coli_cuda_init / shutdown / mem_info / stats / device_count
 *   - coli_cuda_tensor_upload (alloc + upload int4 weights + per-row scales)
 *   - coli_cuda_matmul        (fmt=2 int4; others fall back to CPU via return 0)
 *   - coli_cuda_expert_mlp    (int4 gate/up/silu/down; covers the MoE experts)
 *   - coli_cuda_expert_group  -> returns 0 (glm.c fans out to per-expert expert_mlp)
 *   - all other coli_cuda_*    -> return 0 / no-op (engine keeps CPU path)
 *
 * The int4 math exactly mirrors glm.c matmul_i4: 2 nibbles/byte (low=first,
 * signed -8..+7), one FP32 scale per output row applied to the whole dot.
 *
 * Shaders are compiled at init via glslangValidator (dev build). For a shippable
 * binary, precompile to SPIR-V and embed the byte arrays (see plan doc).
 *
 * Build (colibri/c):
 *   make glm VULKAN=1
 * Run (force the AMD ICD on machines with multiple GPUs):
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json ./glm ...
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

/* ---- opaque tensor (engine stores the pointer, never dereferences) ---- */
struct ColiCudaTensor {
    int device;
    int fmt;            /* 0=f32 1=int8 2=int4 3=int2 */
    int I, O;
    VkBuffer wbuf;      /* packed weights (q4/q8) */
    VkDeviceMemory wmem;
    VkBuffer sbuf;      /* FP32 scales (O floats) */
    VkDeviceMemory smem;
    size_t wbytes, sbytes;
};
typedef struct ColiCudaTensor ColiCudaTensor;

/* ---- Vulkan singleton ---- */
static VkInstance   g_inst = NULL;
static VkPhysicalDevice g_pd = NULL;
static VkDevice     g_dev = NULL;
static VkQueue      g_queue = NULL;
static uint32_t     g_qf = 0;
static VkCommandPool g_pool = NULL;
static VkFence      g_fence = NULL;
static int          g_enabled = 0;
static int          g_devices[16];
static int          g_ndev = 0;

/* stats */
static uint64_t g_calls=0, g_experts=0, g_rows=0;
static double   g_h2d_ms=0, g_kernel_ms=0, g_d2h_ms=0;

/* ---- shaders (GLSL, compiled to SPIR-V at init via glslangValidator) ---- */
static const char* S_MATMUL =
"#version 450\n"
"layout(local_size_x=64) in;\n"
"layout(std430,binding=0) readonly buffer Wbuf { uint w[]; };\n"
"layout(std430,binding=1) readonly buffer Sbuf { float sc[]; };\n"
"layout(std430,binding=2) readonly buffer Xbuf { float x[]; };\n"
"layout(std430,binding=3) writeonly buffer Ybuf { float y[]; };\n"
"void main(){\n"
"  uint o=gl_GlobalInvocationID.x; uint S=gl_GlobalInvocationID.y;\n"
"  if(o>=uint(O_DIM) || S>=uint(S_DIM)) return;\n"
"  float acc=0.0f;\n"
"  uint rb=uint((I_DIM+1)/2);\n"
"  for(uint i=0u;i<uint(I_DIM);i++){\n"
"    uint byte=w[o*rb + (i>>1u)];\n"
"    int nib = (i&1u)!=0u ? int(byte>>4u) : int(byte&0xFu);\n"
"    int wq = nib-8;\n"
"    acc += x[S*uint(I_DIM) + i] * float(wq);\n"
"  }\n"
"  y[S*uint(O_DIM) + o] = acc * sc[o];\n"
"}\n";

static const char* S_SILU =
"#version 450\n"
"layout(local_size_x=64) in;\n"
"layout(std430,binding=0) buffer Gbuf { float g[]; };\n"
"layout(std430,binding=1) readonly buffer Ubuf { float u[]; };\n"
"void main(){\n"
"  uint i=gl_GlobalInvocationID.x; if(i>=uint(N_DIM)) return;\n"
"  float v=g[i]; g[i]= v/(1.0f+exp(-v)) * u[i];\n"
"}\n";

/* ---- embed constants via simple string substitution at compile time ---- */
/* We bake I/O/S as specialization-free literal by patching the GLSL string.  */
static char* patch_dims(const char* src, int I, int O, int S){
    /* replace I_DIM/O_DIM/S_DIM tokens */
    size_t cap = strlen(src)+64;
    char* out = malloc(cap);
    const char* p = src; int j=0;
    char ibuf[16], obuf[16], sbuf[16];
    snprintf(ibuf,sizeof ibuf,"%d",I); snprintf(obuf,sizeof obuf,"%d",O); snprintf(sbuf,sizeof sbuf,"%d",S);
    while(*p){
        if(!strncmp(p,"I_DIM",5)){ memcpy(out+j,ibuf,strlen(ibuf)); j+=strlen(ibuf); p+=5; }
        else if(!strncmp(p,"O_DIM",5)){ memcpy(out+j,obuf,strlen(obuf)); j+=strlen(obuf); p+=5; }
        else if(!strncmp(p,"S_DIM",5)){ memcpy(out+j,sbuf,strlen(sbuf)); j+=strlen(sbuf); p+=5; }
        else if(!strncmp(p,"N_DIM",5)){ memcpy(out+j,ibuf,strlen(ibuf)); j+=strlen(ibuf); p+=5; } /* silu uses N_DIM=I */
        else { out[j++]=*p++; }
    }
    out[j]=0; return out;
}

static VkShaderModule load_module(const char* glsl){
    static int counter=0;
    char tmpl[256]; snprintf(tmpl,sizeof tmpl,"/tmp/coli_vk_%d_%d.spv",(int)getpid(),counter++);
    char glslpath[300]; snprintf(glslpath,sizeof glslpath,"%s.comp",tmpl);
    FILE* f=fopen(glslpath,"w"); if(!f){ fprintf(stderr,"[VK] fopen glsl\n"); return NULL; } fputs(glsl,f); fclose(f);
    char cmd[512]; snprintf(cmd,sizeof cmd,"glslangValidator -V %s -o %s >/dev/null 2>&1",glslpath,tmpl);
    if(system(cmd)!=0){ fprintf(stderr,"[VK] glslang failed for %s\n",glslpath); return NULL; }
    FILE* sp=fopen(tmpl,"rb"); if(!sp){ fprintf(stderr,"[VK] spv missing\n"); return NULL; }
    fseek(sp,0,SEEK_END); long sz=ftell(sp); fseek(sp,0,SEEK_SET);
    char* bin=malloc(sz); if(fread(bin,1,sz,sp)!=(size_t)sz){ fprintf(stderr,"[VK] spv read\n"); return NULL; } fclose(sp);
    VkShaderModule m; VkShaderModuleCreateInfo ci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sz,.pCode=(void*)bin};
    if(vkCreateShaderModule(g_dev,&ci,NULL,&m)!=VK_SUCCESS){ fprintf(stderr,"[VK] shaderModule\n"); return NULL; }
    free(bin); return m;
}

/* ---- helpers ---- */
static uint32_t host_visible_type(){
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(g_pd,&mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++)
        if(mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) return i;
    return 0;
}
static VkBuffer make_buf(size_t bytes, VkDeviceMemory* mem){
    VkBuffer buf; VkBufferCreateInfo bci={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=bytes,
        .usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    if(vkCreateBuffer(g_dev,&bci,NULL,&buf)!=VK_SUCCESS) return NULL;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev,buf,&mr);
    VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=mr.size,.memoryTypeIndex=host_visible_type()};
    if(vkAllocateMemory(g_dev,&ai,NULL,mem)!=VK_SUCCESS){ vkDestroyBuffer(g_dev,buf,NULL); return NULL; }
    vkBindBufferMemory(g_dev,buf,*mem,0);
    return buf;
}
static VkShaderModule g_mod_matmul=NULL, g_mod_silu=NULL;

/* ---- shader module cache (avoid recompiling glslang per call) ---- */
#define VK_CACHE_MAX 256
typedef struct { char key[32]; VkShaderModule mod; } VkModCache;
static VkModCache g_matmul_cache[VK_CACHE_MAX]; static int g_matmul_n=0;
static VkModCache g_silu_cache[VK_CACHE_MAX];   static int g_silu_n=0;

static VkShaderModule cache_get(VkModCache* c, int* n, const char* key){
    for(int i=0;i<*n;i++) if(!strcmp(c[i].key,key)) return c[i].mod;
    return NULL;
}
static VkShaderModule cache_put(VkModCache* c, int* n, const char* key, VkShaderModule mod){
    if(*n<VK_CACHE_MAX){ strncpy(c[*n].key,key,sizeof c[*n].key-1); c[*n].key[sizeof c[*n].key-1]=0; c[*n].mod=mod; (*n)++; }
    return mod;
}
static VkShaderModule get_matmul_module(int I,int O,int S){
    char key[32]; snprintf(key,sizeof key,"%d_%d_%d",I,O,S);
    VkShaderModule m=cache_get(g_matmul_cache,&g_matmul_n,key);
    if(m) return m;
    char* glsl=patch_dims(S_MATMUL,I,O,S);
    m=load_module(glsl); free(glsl);
    if(m) return cache_put(g_matmul_cache,&g_matmul_n,key,m);
    return NULL;
}
static VkShaderModule get_silu_module(int N){
    char key[32]; snprintf(key,sizeof key,"%d",N);
    VkShaderModule m=cache_get(g_silu_cache,&g_silu_n,key);
    if(m) return m;
    char* glsl=patch_dims(S_SILU,N,0,0);
    m=load_module(glsl); free(glsl);
    if(m) return cache_put(g_silu_cache,&g_silu_n,key,m);
    return NULL;
}

/* dispatch a compute shader (single bind set of 4 storage buffers) */
static int dispatch4(VkShaderModule mod, VkBuffer b0,VkBuffer b1,VkBuffer b2,VkBuffer b3,
                     uint32_t x,uint32_t y, int I,int O,int S){
    VkDescriptorSetLayoutBinding bnd[4]={
        {0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {3,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0}};
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(g_dev,&(VkDescriptorSetLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=4,.pBindings=bnd},NULL,&dsl);
    VkPipelineLayout pl; vkCreatePipelineLayout(g_dev,&(VkPipelineLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl},NULL,&pl);
    VkPipeline ppl; VkPipelineShaderStageCreateInfo ssi={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=mod,.pName="main"};
    vkCreateComputePipelines(g_dev,0,1,&(VkComputePipelineCreateInfo){.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage=ssi,.layout=pl},NULL,&ppl);
    VkDescriptorPool dp; vkCreateDescriptorPool(g_dev,&(VkDescriptorPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}},NULL,&dp);
    VkDescriptorSet ds; vkAllocateDescriptorSets(g_dev,&(VkDescriptorSetAllocateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp,.descriptorSetCount=1,.pSetLayouts=&dsl},&ds);
    VkDescriptorBufferInfo bi[4]={{b0,0,VK_WHOLE_SIZE},{b1,0,VK_WHOLE_SIZE},{b2,0,VK_WHOLE_SIZE},{b3,0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wds[4];
    for(int i=0;i<4;i++) wds[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[i]};
    vkUpdateDescriptorSets(g_dev,4,wds,0,NULL);
    VkCommandBuffer cb; vkAllocateCommandBuffers(g_dev,&(VkCommandBufferAllocateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=g_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1},&cb);
    vkBeginCommandBuffer(cb,&(VkCommandBufferBeginInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO});
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,ppl);
    vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,0);
    vkCmdDispatch(cb,(x+63)/64,(y>0?y:1),1);
    vkEndCommandBuffer(cb);
    vkResetFences(g_dev,1,&g_fence);
    vkQueueSubmit(g_queue,1,&(VkSubmitInfo){.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb},g_fence);
    vkWaitForFences(g_dev,1,&g_fence,VK_TRUE,~0ULL);
    vkDestroyDescriptorPool(g_dev,dp,NULL); vkDestroyPipeline(g_dev,ppl,NULL);
    vkDestroyPipelineLayout(g_dev,pl,NULL); vkDestroyDescriptorSetLayout(g_dev,dsl,NULL);
    vkFreeCommandBuffers(g_dev,g_pool,1,&cb);
    return 1;
}

/* dispatch the silu shader (2 buffers: gbuf read_write at 0, ubuf readonly at 1) */
static int dispatch2(VkShaderModule mod, VkBuffer gbuf, VkBuffer ubuf, int N){
    VkDescriptorSetLayoutBinding bnd[2]={
        {0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0},
        {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,0}};
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(g_dev,&(VkDescriptorSetLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=2,.pBindings=bnd},NULL,&dsl);
    VkPipelineLayout pl; vkCreatePipelineLayout(g_dev,&(VkPipelineLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl},NULL,&pl);
    VkPipeline ppl; VkPipelineShaderStageCreateInfo ssi={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=mod,.pName="main"};
    vkCreateComputePipelines(g_dev,0,1,&(VkComputePipelineCreateInfo){.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.stage=ssi,.layout=pl},NULL,&ppl);
    VkDescriptorPool dp; vkCreateDescriptorPool(g_dev,&(VkDescriptorPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}},NULL,&dp);
    VkDescriptorSet ds; vkAllocateDescriptorSets(g_dev,&(VkDescriptorSetAllocateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp,.descriptorSetCount=1,.pSetLayouts=&dsl},&ds);
    VkDescriptorBufferInfo bi[2]={{gbuf,0,VK_WHOLE_SIZE},{ubuf,0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wds[2]={ {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[0]},
                                  {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=1,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[1]} };
    vkUpdateDescriptorSets(g_dev,2,wds,0,NULL);
    VkCommandBuffer cb; vkAllocateCommandBuffers(g_dev,&(VkCommandBufferAllocateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=g_pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1},&cb);
    vkBeginCommandBuffer(cb,&(VkCommandBufferBeginInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO});
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,ppl); vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,0);
    vkCmdDispatch(cb,((uint32_t)N+63)/64,1,1);
    vkEndCommandBuffer(cb); vkResetFences(g_dev,1,&g_fence);
    vkQueueSubmit(g_queue,1,&(VkSubmitInfo){.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cb},g_fence);
    vkWaitForFences(g_dev,1,&g_fence,VK_TRUE,~0ULL);
    vkDestroyDescriptorPool(g_dev,dp,NULL); vkDestroyPipeline(g_dev,ppl,NULL); vkDestroyPipelineLayout(g_dev,pl,NULL); vkDestroyDescriptorSetLayout(g_dev,dsl,NULL); vkFreeCommandBuffers(g_dev,g_pool,1,&cb);
    return 1;
}

/* ==================== ABI ==================== */

int coli_cuda_init(const int *devices, int count){
    if(g_enabled) return 1;
    VkApplicationInfo ai; memset(&ai,0,sizeof ai); ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici; memset(&ici,0,sizeof ici); ici.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo=&ai;
    if(vkCreateInstance(&ici,NULL,&g_inst)!=VK_SUCCESS){ fprintf(stderr,"[VK] instance\n"); return 0; }
    uint32_t ndev=0; vkEnumeratePhysicalDevices(g_inst,&ndev,NULL);
    if(ndev==0){ fprintf(stderr,"[VK] no devices\n"); return 0; }
    VkPhysicalDevice* pds=malloc(ndev*sizeof(VkPhysicalDevice)); vkEnumeratePhysicalDevices(g_inst,&ndev,pds);
    /* pick: prefer the device asked for, else first discrete/integrated */
    g_pd = (count>0)? NULL : pds[0];
    if(count>0){ for(uint32_t i=0;i<ndev;i++){ VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i],&p);
        if((int)p.deviceID==devices[0]) g_pd=pds[i]; } if(!g_pd) g_pd=pds[0]; }
    g_devices[0]=0; g_ndev=1; (void)devices;(void)count;
    /* queue */
    uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd,&nq,NULL);
    VkQueueFamilyProperties* qp=malloc(nq*sizeof*qp); vkGetPhysicalDeviceQueueFamilyProperties(g_pd,&nq,qp);
    g_qf=0; for(uint32_t i=0;i<nq;i++) if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){ g_qf=i; break; }
    free(qp);
    float qp_=1.0f; VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=g_qf,.queueCount=1,.pQueuePriorities=&qp_};
    if(vkCreateDevice(g_pd,&(VkDeviceCreateInfo){.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci},NULL,&g_dev)!=VK_SUCCESS){ fprintf(stderr,"[VK] device\n"); return 0; }
    vkGetDeviceQueue(g_dev,g_qf,0,&g_queue);
    vkCreateCommandPool(g_dev,&(VkCommandPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=g_qf},NULL,&g_pool);
    vkCreateFence(g_dev,&(VkFenceCreateInfo){.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},NULL,&g_fence);
    /* shaders are compiled lazily + cached per shape (get_matmul_module/get_silu_module) */
    g_enabled=1; free(pds);
    fprintf(stderr,"[VK] backend ready on Vulkan device\n");
    return 1;
}

void coli_cuda_shutdown(void){
    if(!g_enabled) return;
    /* destroy cached modules */
    for(int i=0;i<g_matmul_n;i++) vkDestroyShaderModule(g_dev,g_matmul_cache[i].mod,NULL);
    for(int i=0;i<g_silu_n;i++) vkDestroyShaderModule(g_dev,g_silu_cache[i].mod,NULL);
    vkDestroyFence(g_dev,g_fence,NULL); vkDestroyCommandPool(g_dev,g_pool,NULL);
    vkDestroyDevice(g_dev,NULL); vkDestroyInstance(g_inst,NULL);
    g_enabled=0; g_inst=NULL; g_dev=NULL;
}

int coli_cuda_device_count(void){ return g_enabled?1:0; }
int coli_cuda_device_at(int index){ return index==0?0:-1; }
int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes){
    if(!g_enabled) return 0;
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(g_pd,&mp);
    VkDeviceSize biggest=0; for(uint32_t i=0;i<mp.memoryHeapCount;i++) if(mp.memoryHeaps[i].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) biggest=mp.memoryHeaps[i].size;
    if(total_bytes)*total_bytes=(size_t)biggest; if(free_bytes)*free_bytes=(size_t)biggest; /* unified mem: report all */
    (void)device; return 1;
}
void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes){
    (void)device; if(tensor_count)*tensor_count=0; if(tensor_bytes)*tensor_bytes=0;
}
void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows, double *h2d_ms, double *kernel_ms, double *d2h_ms){
    if(calls)*calls=g_calls; if(experts)*experts=g_experts; if(rows)*rows=g_rows;
    if(h2d_ms)*h2d_ms=g_h2d_ms; if(kernel_ms)*kernel_ms=g_kernel_ms; if(d2h_ms)*d2h_ms=g_d2h_ms;
}

int coli_cuda_tensor_upload(ColiCudaTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int device){
    if(!g_enabled) return 0;
    if(!*tensor){
        ColiCudaTensor* t=calloc(1,sizeof*t);
        t->device=device; t->fmt=fmt; t->I=I; t->O=O;
        size_t wbytes = fmt==0 ? (size_t)O*I*sizeof(float)
                      : fmt==1 ? (size_t)O*I*sizeof(int8_t)
                      : fmt==2 ? (size_t)O*((I+1)/2)
                      : (size_t)O*((I+3)/4);
        t->wbytes=wbytes; t->sbytes=(size_t)O*sizeof(float);
        t->wbuf=make_buf(wbytes,&t->wmem); t->sbuf=make_buf(t->sbytes,&t->smem);
        if(!t->wbuf||!t->sbuf){ fprintf(stderr,"[VK] tensor alloc\n"); return 0; }
        *tensor=t;
    }
    ColiCudaTensor* t=*tensor;
    void* p;
    vkMapMemory(g_dev,t->wmem,0,t->wbytes,0,&p); memcpy(p,weights,t->wbytes); vkUnmapMemory(g_dev,t->wmem);
    vkMapMemory(g_dev,t->smem,0,t->sbytes,0,&p); memcpy(p,scales,t->sbytes); vkUnmapMemory(g_dev,t->smem);
    return 1;
}

/* int4 matmul: y[S,O] = x[S,I] @ W[O,I]^T, W int4 (2 nibbles/byte, signed), scale per row. */
int coli_cuda_matmul(ColiCudaTensor **tensor, float *y, const float *x,
                     const void *weights, const float *scales, int fmt, int S, int I, int O, int device){
    if(fmt!=2) return 0;                 /* only int4 accelerated; else CPU */
    if(S<1) return 0;
    if(!coli_cuda_tensor_upload(tensor,weights,scales,fmt,I,O,device)) return 0;
    ColiCudaTensor* t=*tensor;
    size_t xb=(size_t)S*I*sizeof(float), yb=(size_t)S*O*sizeof(float);
    VkBuffer xb_buf, yb_buf; VkDeviceMemory xm, ym;
    xb_buf=make_buf(xb,&xm); yb_buf=make_buf(yb,&ym);
    if(!xb_buf||!yb_buf) return 0;
    void* p;
    vkMapMemory(g_dev,xm,0,xb,0,&p); memcpy(p,x,xb); vkUnmapMemory(g_dev,xm);
    g_calls++;
    dispatch4(get_matmul_module(I,O,S), t->wbuf, t->sbuf, xb_buf, yb_buf, (uint32_t)O, (uint32_t)S, I,O,S);
    vkMapMemory(g_dev,ym,0,yb,0,&p); memcpy(y,p,yb); vkUnmapMemory(g_dev,ym);
    vkDestroyBuffer(g_dev,xb_buf,NULL); vkFreeMemory(g_dev,xm,NULL);
    vkDestroyBuffer(g_dev,yb_buf,NULL); vkFreeMemory(g_dev,ym,NULL);
    return 1;
}

int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S){
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=2||up->fmt!=2||down->fmt!=2) return 0;
    if(gate->device!=up->device||gate->device!=down->device) return 0;
    int D=gate->I, I=gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float), yb=(size_t)S*D*sizeof(float);
    VkBuffer xbuf, gbuf, ubuf, ybuf; VkDeviceMemory xm,gm,um,ym;
    xbuf=make_buf(xb,&xm); gbuf=make_buf(ib,&gm); ubuf=make_buf(ib,&um); ybuf=make_buf(yb,&ym);
    if(!xbuf||!gbuf||!ubuf||!ybuf) return 0;
    void* p; vkMapMemory(g_dev,xm,0,xb,0,&p); memcpy(p,x,xb); vkUnmapMemory(g_dev,xm);
    g_experts++; g_rows+=S;
    /* gate = x @ Wg^T ; up = x @ Wu^T */
    VkShaderModule m_gate=get_matmul_module(D,I,S);
    VkShaderModule m_up  =get_matmul_module(D,I,S);
    if(!m_gate||!m_up) return 0;
    dispatch4(m_gate, gate->wbuf, gate->sbuf, xbuf, gbuf, (uint32_t)I, (uint32_t)S, D,I,S);
    dispatch4(m_up,   up->wbuf,   up->sbuf,   xbuf, ubuf, (uint32_t)I, (uint32_t)S, D,I,S);
    /* silu(gate)*up -> gbuf */
    VkShaderModule m_silu=get_silu_module((int)((size_t)S*(size_t)I));
    if(!m_silu) return 0;
    dispatch2(m_silu, gbuf, ubuf, (int)((size_t)S*(size_t)I));
    /* down = silu_gate @ Wd^T -> y */
    VkShaderModule m_down=get_matmul_module(I,D,S);
    if(!m_down) return 0;
    dispatch4(m_down, down->wbuf, down->sbuf, gbuf, ybuf, (uint32_t)D, (uint32_t)S, I,D,S);
    vkMapMemory(g_dev,ym,0,yb,0,&p); memcpy(y,p,yb); vkUnmapMemory(g_dev,ym);
    vkDestroyBuffer(g_dev,xbuf,NULL); vkFreeMemory(g_dev,xm,NULL);
    vkDestroyBuffer(g_dev,gbuf,NULL); vkFreeMemory(g_dev,gm,NULL);
    vkDestroyBuffer(g_dev,ubuf,NULL); vkFreeMemory(g_dev,um,NULL);
    vkDestroyBuffer(g_dev,ybuf,NULL); vkFreeMemory(g_dev,ym,NULL);
    return 1;
}

int coli_cuda_expert_group(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups, ColiCudaTensor *const *downs, const int *rows, int count, float *y, const float *x){
    (void)gates;(void)ups;(void)downs;(void)rows;(void)count;(void)y;(void)x;
    return 0;  /* glm.c fans out to per-expert expert_mlp, which we implement */
}

void coli_cuda_tensor_free(ColiCudaTensor *tensor){
    if(!tensor) return;
    if(tensor->wbuf){ vkDestroyBuffer(g_dev,tensor->wbuf,NULL); vkFreeMemory(g_dev,tensor->wmem,NULL); }
    if(tensor->sbuf){ vkDestroyBuffer(g_dev,tensor->sbuf,NULL); vkFreeMemory(g_dev,tensor->smem,NULL); }
    free(tensor);
}
size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor){ return tensor? tensor->wbytes+tensor->sbytes : 0; }
int coli_cuda_tensor_device(const ColiCudaTensor *tensor){ return tensor? tensor->device : -1; }
int coli_cuda_tensor_update(ColiCudaTensor *tensor, const void *weights, const float *scales){
    if(!tensor) return 0;
    void* p; vkMapMemory(g_dev,tensor->wmem,0,tensor->wbytes,0,&p); memcpy(p,weights,tensor->wbytes); vkUnmapMemory(g_dev,tensor->wmem);
    vkMapMemory(g_dev,tensor->smem,0,tensor->sbytes,0,&p); memcpy(p,scales,tensor->sbytes); vkUnmapMemory(g_dev,tensor->smem);
    return 1;
}

/* ---- pipe_* / attention: not implemented -> engine falls back to CPU ---- */
float *coli_cuda_pipe_scratch(int d,int s,size_t b){(void)d;(void)s;(void)b;return NULL;}
void *coli_cuda_pipe_alloc(int d,size_t b){(void)d;(void)b;return NULL;}
void coli_cuda_pipe_free(int d,void*p){(void)d;(void)p;}
int coli_cuda_pipe_upload(int d,void*dst,const void*src,size_t b){(void)d;(void)dst;(void)src;(void)b;return 0;}
int coli_cuda_pipe_download(int d,const void*src,void*dst,size_t b){(void)d;(void)src;(void)dst;(void)b;return 0;}
int coli_cuda_pipe_rmsnorm(int d,float*y,const float*x,const float*w,int S,int D,float e){(void)d;(void)y;(void)x;(void)w;(void)S;(void)D;(void)e;return 0;}
int coli_cuda_pipe_rope(int d,float*v,const int*pos,int r,int st,int off,int R,int h,float th){(void)d;(void)v;(void)pos;(void)r;(void)st;(void)off;(void)R;(void)h;(void)th;return 0;}
int coli_cuda_pipe_silu_mul(int d,float*g,const float*u,size_t n){(void)d;(void)g;(void)u;(void)n;return 0;}
int coli_cuda_pipe_add(int d,float*x,const float*t,size_t n){(void)d;(void)x;(void)t;(void)n;return 0;}
int coli_cuda_pipe_rows_add(int d,float*x,const float*p,const int*r,int n,int D){(void)d;(void)x;(void)p;(void)r;(void)n;(void)D;return 0;}
int coli_cuda_pipe_gemm(ColiCudaTensor*t,float*y,const float*x,int S){(void)t;(void)y;(void)x;(void)S;return 0;}
int coli_cuda_pipe_rmsnorm_s(int d,float*y,const float*x,const float*w,int S,int D,float e,int xs,int ys){(void)d;(void)y;(void)x;(void)w;(void)S;(void)D;(void)e;(void)xs;(void)ys;return 0;}
int coli_cuda_pipe_rope_base(int d,float*v,int pb,int r,int st,int off,int R,int h,float th){(void)d;(void)v;(void)pb;(void)r;(void)st;(void)off;(void)R;(void)h;(void)th;return 0;}
int coli_cuda_pipe_copy2d(int d,float*dst,int dp,const float*src,int sp,int w,int h){(void)d;(void)dst;(void)dp;(void)src;(void)sp;(void)w;(void)h;return 0;}
int coli_cuda_pipe_peer_copy(int dd,float*dst,int sd,const float*src,size_t b){(void)dd;(void)dst;(void)sd;(void)src;(void)b;return 0;}
int coli_cuda_pipe_sync(int d){(void)d;return 0;}
int coli_cuda_attention_absorb(ColiCudaTensor*kv,float*c,const float*q,const float*l,const float*r,int H,int Q,int R,int V,int K,int T,float s){(void)kv;(void)c;(void)q;(void)l;(void)r;(void)H;(void)Q;(void)R;(void)V;(void)K;(void)T;(void)s;return 0;}
int coli_cuda_attention_absorb_batch(ColiCudaTensor*kv,float*c,const float*q,const float*l,const float*r,int S,int H,int Q,int R,int V,int K,int T,float s){(void)kv;(void)c;(void)q;(void)l;(void)r;(void)S;(void)H;(void)Q;(void)R;(void)V;(void)K;(void)T;(void)s;return 0;}
int coli_cuda_attention_project_batch(ColiCudaTensor*kv,ColiCudaTensor*o,float*out,const float*q,const float*l,const float*r,int S,int H,int Q,int R,int V,int K,int T,float s){(void)kv;(void)o;(void)out;(void)q;(void)l;(void)r;(void)S;(void)H;(void)Q;(void)R;(void)V;(void)K;(void)T;(void)s;return 0;}
int coli_cuda_attention_project_batch_dev(ColiCudaTensor*kv,ColiCudaTensor*o,float*out,const float*qd,const float*ld,const float*rd,int S,int H,int Q,int R,int V,int K,int T,float s){(void)kv;(void)o;(void)out;(void)qd;(void)ld;(void)rd;(void)S;(void)H;(void)Q;(void)R;(void)V;(void)K;(void)T;(void)s;return 0;}
int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor*kvb,float*cd,const float*qd,const float*ld,const float*rd,int S,int H,int Q,int R,int V,int K,int T,float s){(void)kvb;(void)cd;(void)qd;(void)ld;(void)rd;(void)S;(void)H;(void)Q;(void)R;(void)V;(void)K;(void)T;(void)s;return 0;}
int coli_cuda_attention_absorb_kvdev(ColiCudaTensor*kv,float*c,const float*q,const float*ld,const float*rd,int H,int Q,int R,int V,int K,int T,float s){(void)kv;(void)c;(void)q;(void)ld;(void)rd;(void)H;(void)Q;(void)R;(void)V;(void)K;(void)T;(void)s;return 0;}
int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor*kv,ColiCudaTensor*o,float*od,const float*qd,const float*ld,const float*rd,int S,int H,int Q,int R,int V,int K,int T,float s){(void)kv;(void)o;(void)od;(void)qd;(void)ld;(void)rd;(void)S;(void)H;(void)Q;(void)R;(void)V;(void)K;(void)T;(void)s;return 0;}
int coli_cuda_shared_mlp_w4a16(ColiCudaTensor*g,ColiCudaTensor*u,ColiCudaTensor*d,float*y,const float*x,int S){(void)g;(void)u;(void)d;(void)y;(void)x;(void)S;return 0;}
