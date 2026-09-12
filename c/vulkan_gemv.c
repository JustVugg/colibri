/* vulkan_gemv.c -- Vulkan compute backend for qwen36 MoE expert GEMVs.
 * See vulkan_gemv.h for the design. Builds standalone with -D VG_SELFTEST for a
 * CPU-vs-GPU correctness check; linked into qwen36.c / qwen36_serve.c otherwise.
 *
 * Vulkan is loaded DYNAMICALLY (dlopen/LoadLibrary "vulkan-1") so the default
 * binary keeps zero compile-time dependency on Vulkan and silently falls back to
 * CPU when the loader or a compute device is absent. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#define VK_LIB "vulkan-1.dll"
static void* dl_open(const char*n){ return LoadLibraryA(n); }
static void* dl_sym(void*l,const char*n){ return (void*)GetProcAddress((HMODULE)l,n); }
static void  dl_close(void*l){ FreeLibrary((HMODULE)l); }
#else
#include <dlfcn.h>
#define VK_LIB "libvulkan.so"
static void* dl_open(const char*n){ return dlopen(n,RTLD_NOW); }
static void* dl_sym(void*l,const char*n){ return dlsym(l,n); }
static void  dl_close(void*l){ dlclose(l); }
#endif

#include "vulkan_gemv.h"
#include "vulkan_core.h"
#include "vulkan_gemv_spv.h"
#include "vulkan_gemv_idp_spv.h"
#include "vulkan_gemv_int4_spv.h"

/* ---------- loaded Vulkan function pointers ---------- */
#define VK_FNS \
  X(vkCreateInstance) \
  X(vkEnumeratePhysicalDevices) \
  X(vkGetPhysicalDeviceQueueFamilyProperties) \
  X(vkGetPhysicalDeviceMemoryProperties) \
  X(vkGetPhysicalDeviceProperties) \
  X(vkGetPhysicalDeviceFeatures2) \
  X(vkDestroyInstance) \
  X(vkCreateDevice) \
  X(vkDestroyDevice) \
  X(vkGetDeviceQueue) \
  X(vkCreateBuffer) \
  X(vkDestroyBuffer) \
  X(vkGetBufferMemoryRequirements) \
  X(vkAllocateMemory) \
  X(vkFreeMemory) \
  X(vkBindBufferMemory) \
  X(vkMapMemory) \
  X(vkUnmapMemory) \
  X(vkFlushMappedMemoryRanges) \
  X(vkInvalidateMappedMemoryRanges) \
  X(vkCreateShaderModule) \
  X(vkDestroyShaderModule) \
  X(vkCreateDescriptorSetLayout) \
  X(vkDestroyDescriptorSetLayout) \
  X(vkCreateDescriptorPool) \
  X(vkDestroyDescriptorPool) \
  X(vkAllocateDescriptorSets) \
  X(vkUpdateDescriptorSets) \
  X(vkCreatePipelineLayout) \
  X(vkDestroyPipelineLayout) \
  X(vkCreateComputePipelines) \
  X(vkDestroyPipeline) \
  X(vkCreateCommandPool) \
  X(vkDestroyCommandPool) \
  X(vkAllocateCommandBuffers) \
  X(vkFreeCommandBuffers) \
  X(vkResetCommandBuffer) \
  X(vkBeginCommandBuffer) \
  X(vkEndCommandBuffer) \
  X(vkCmdBindPipeline) \
  X(vkCmdBindDescriptorSets) \
  X(vkCmdDispatch) \
  X(vkQueueSubmit) \
  X(vkQueueWaitIdle)

#define X(name) static PFN_##name g_##name = NULL;
VK_FNS
#undef X

static void* g_lib = NULL;
static int g_vg_ok = 0;

/* ---------- static sizes ---------- */
static int g_hidden=0, g_inter=0, g_cap=0, g_topk=0, g_nlayers=0, g_nslots=0;
static uint64_t g_tick = 0;

/* ---------- Vulkan objects ---------- */
static VkInstance       g_inst=VK_NULL_HANDLE;
static VkDevice         g_dev =VK_NULL_HANDLE;
static VkPhysicalDevice g_pd  =VK_NULL_HANDLE;
static VkQueue          g_q   =VK_NULL_HANDLE;
static uint32_t         g_qf  =~0u;
static VkShaderModule   g_sm  =VK_NULL_HANDLE;   /* float-path shader (fallback) */
static VkShaderModule   g_sm_idp=VK_NULL_HANDLE;  /* OpSDotKHR int8 shader */
static VkDescriptorSetLayout g_dsl=VK_NULL_HANDLE;
static VkDescriptorPool g_dpool=VK_NULL_HANDLE;
static VkDescriptorSet  g_ds  =VK_NULL_HANDLE;
static VkPipelineLayout g_pl  =VK_NULL_HANDLE;
static VkPipeline       g_pipe=VK_NULL_HANDLE;   /* float-path pipeline */
static VkPipeline       g_pipe_idp=VK_NULL_HANDLE;/* int8 dot-product pipeline */
static VkShaderModule   g_sm_int4=VK_NULL_HANDLE; /* int4 unpack + float GEMV shader */
static VkPipeline       g_pipe_int4=VK_NULL_HANDLE;/* int4 pipeline */
static int              g_use_idp=0;             /* 1 if IDP active */
static int              g_use_int4=0;            /* 1 if int4 path active */
static int              g_weight_bits=8;         /* 4 or 8 */
static int              g_dbg=0;                 /* set by MOE_DBG env */
static uint32_t         g_vendor_id=0;          /* GPU vendor (0x1002=AMD,0x10DE=NVIDIA,0x8086=Intel) */
static uint32_t         g_driver_ver=0;         /* GPU driver version (vendor-encoded) */
static VkCommandPool    g_cpool=VK_NULL_HANDLE;
static VkCommandBuffer  g_cmd =VK_NULL_HANDLE;

/* persistent buffers (host-visible + coherent) */
typedef struct { VkBuffer buf; VkDeviceMemory mem; void *ptr; VkDeviceSize size; } Buf;
static Buf g_w, g_s, g_x, g_y, g_meta;

/* GPU slot table (mirror of CPU LRU) */
typedef struct {
    int layer, eid, valid;
    uint64_t used;
    uint32_t woff_g, woff_u, woff_d;   /* uint offsets into g_w */
    uint32_t soff_g, soff_u, soff_d;   /* float offsets into g_s */
} GSlot;
static GSlot *g_slot = NULL;

/* per-slot byte/float layout (precomputed) */
static uint32_t g_slot_wbytes = 0;     /* bytes per slot in g_w  (3 * D * Ih) */
static uint32_t g_slot_sfloats = 0;    /* floats per slot in g_s (3 * D) */

#define CHECK(r,msg) do{ if((r)!=VK_SUCCESS){ fprintf(stderr,"[vg] %s failed (code %d)\n", msg, (int)(r)); goto fail; } }while(0)
static uint32_t align_up(uint32_t x, uint32_t a){ return (x + a - 1) & ~(a - 1); }

static VkResult vg_create_buf(VkDeviceSize size, VkBufferUsageFlags usage, Buf *b){
    VkBufferCreateInfo bi; memset(&bi,0,sizeof bi);
    bi.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size=size; bi.usage=usage; bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    VkResult r=g_vkCreateBuffer(g_dev,&bi,NULL,&b->buf);
    if(r!=VK_SUCCESS) return r;
    VkMemoryRequirements req; g_vkGetBufferMemoryRequirements(g_dev,b->buf,&req);
    VkPhysicalDeviceMemoryProperties mp; g_vkGetPhysicalDeviceMemoryProperties(g_pd,&mp);
    uint32_t mi=~0u;
    for(uint32_t i=0;i<mp.memoryTypeCount;i++){
        if((req.memoryTypeBits&(1u<<i)) &&
           (mp.memoryTypes[i].propertyFlags &
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
           ==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){
            mi=i; break;
        }
    }
    if(mi==~0u){ fprintf(stderr,"[vg] no host-visible coherent memory type\n"); return VK_ERROR_UNKNOWN; }
    VkMemoryAllocateInfo ai; memset(&ai,0,sizeof ai);
    ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize=req.size; ai.memoryTypeIndex=mi;
    r=g_vkAllocateMemory(g_dev,&ai,NULL,&b->mem);
    if(r!=VK_SUCCESS) return r;
    r=g_vkBindBufferMemory(g_dev,b->buf,b->mem,0);
    if(r!=VK_SUCCESS) return r;
    b->size=req.size;
    return g_vkMapMemory(g_dev,b->mem,0,req.size,0,&b->ptr);
}

static void vg_flush(Buf *b, VkDeviceSize off, VkDeviceSize len){
    if(len==0) return;
    VkMappedMemoryRange r; memset(&r,0,sizeof r);
    r.sType=VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE; r.memory=b->mem; r.offset=off; r.size=len;
    g_vkFlushMappedMemoryRanges(g_dev,1,&r);
}
static void vg_invalidate(Buf *b, VkDeviceSize off, VkDeviceSize len){
    if(len==0) return;
    VkMappedMemoryRange r; memset(&r,0,sizeof r);
    r.sType=VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE; r.memory=b->mem; r.offset=off; r.size=len;
    g_vkInvalidateMappedMemoryRanges(g_dev,1,&r);
}

static int vg_load_lib(void){
    g_lib=dl_open(VK_LIB);
    if(!g_lib){ fprintf(stderr,"[vg] cannot load %s\n", VK_LIB); return -1; }
#define X(name) \
    g_##name=(PFN_##name)dl_sym(g_lib,#name); \
    if(!g_##name){ fprintf(stderr,"[vg] missing %s\n", #name); return -1; }
    VK_FNS
#undef X
    return 0;
}

int vg_init(const vg_cfg *cfg){
    if(g_vg_ok) return 0;
    if(vg_load_lib()!=0) return -1;

    g_hidden=cfg->hidden; g_inter=cfg->inter; g_cap=cfg->cap;
    if(getenv("MOE_DBG")) g_dbg=1;
    g_topk=cfg->topk; g_nlayers=cfg->n_layers;
    g_weight_bits = (cfg->weight_bits==4) ? 4 : 8;   /* 0/other -> 8 */
    g_nslots=g_nlayers*g_cap;
    if(g_nslots<=0 || g_hidden<=0 || g_inter<=0) return -1;

    /* int4 weights are packed 2 nibbles/byte -> half the byte footprint of int8.
     * int8 = 1 byte/weight; int4 = 1/2 byte/weight.  3 matrices (g,u,d). */
    g_slot_wbytes = 3u * (uint32_t)g_hidden * (uint32_t)g_inter
                    * (g_weight_bits==4 ? 1u : 2u) / 2u;
    g_slot_wbytes = align_up(g_slot_wbytes, 4u);
    g_slot_sfloats = 3u * (uint32_t)g_hidden;

    /* instance — request 1.1 so vkGetPhysicalDeviceFeatures2 can report
     * 1.1+ extension features (shaderIntegerDotProduct for OpSDotKHR).
     * Fall back to 1.0 (float path only) if 1.1 is unavailable. */
    VkApplicationInfo ai; memset(&ai,0,sizeof ai);
    ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO; ai.apiVersion=VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci; memset(&ci,0,sizeof ci);
    ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ci.pApplicationInfo=&ai;
    VkResult ir = g_vkCreateInstance(&ci,NULL,&g_inst);
    if(ir!=VK_SUCCESS){
        ai.apiVersion=VK_API_VERSION_1_0;
        CHECK(g_vkCreateInstance(&ci,NULL,&g_inst),"vkCreateInstance");
    }

    uint32_t ndev=0; g_vkEnumeratePhysicalDevices(g_inst,&ndev,NULL);
    if(ndev==0){ fprintf(stderr,"[vg] no physical devices\n"); goto fail; }
    VkPhysicalDevice *pds=malloc(ndev*sizeof(VkPhysicalDevice));
    g_vkEnumeratePhysicalDevices(g_inst,&ndev,pds);
    g_pd=pds[0];
    /* Capture vendor/driver version so we can blacklist drivers known to crash
     * on OpSDotKHR pipeline compilation (e.g. AMD 0x800184). A segfault there
     * cannot be caught, so we must avoid invoking the compiler on them. */
    VkPhysicalDeviceProperties pdprops; memset(&pdprops,0,sizeof pdprops);
    g_vkGetPhysicalDeviceProperties(g_pd,&pdprops);
    g_vendor_id = pdprops.vendorID; g_driver_ver = pdprops.driverVersion;
    uint32_t nf=0; g_vkGetPhysicalDeviceQueueFamilyProperties(g_pd,&nf,NULL);
    VkQueueFamilyProperties *qfp=malloc(nf*sizeof(VkQueueFamilyProperties));
    g_vkGetPhysicalDeviceQueueFamilyProperties(g_pd,&nf,qfp);
    for(uint32_t i=0;i<nf;i++){ if(qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT){ g_qf=i; break; } }
    free(qfp); free(pds);
    if(g_qf==~0u){ fprintf(stderr,"[vg] no compute-capable queue family\n"); goto fail; }

    float qpri=1.0f;
    VkDeviceQueueCreateInfo dq; memset(&dq,0,sizeof dq);
    dq.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; dq.queueFamilyIndex=g_qf;
    dq.queueCount=1; dq.pQueuePriorities=&qpri;

    /* Query + enable VK_KHR_shader_integer_dot_product so we can use OpSDotKHR,
     * AND VK_KHR_shader_float16_int8 so the int8 (v4i8) types in the IDP shader
     * are legal. Both fall back to the float path if the device lacks them. */
    VkPhysicalDeviceFeatures2 f2; memset(&f2,0,sizeof f2);
    f2.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDeviceShaderIntegerDotProductFeatures dpf; memset(&dpf,0,sizeof dpf);
    dpf.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    VkPhysicalDeviceShaderFloat16Int8Features f16i8; memset(&f16i8,0,sizeof f16i8);
    f16i8.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    f2.pNext=&f16i8; f16i8.pNext=&dpf;
    /* NOTE: this trimmed vulkan_core.h typedefs the fn as returning void,
     * so we don't capture its VkResult. Output is written into dpf/f2 via
     * the pointer; a failed call leaves fields zeroed -> idp off (safe). */
    g_vkGetPhysicalDeviceFeatures2(g_pd,&f2);
    int idp_supported = (int)dpf.shaderIntegerDotProduct;
    int int8_supported = (int)f16i8.shaderInt8;
    dpf.shaderIntegerDotProduct = idp_supported ? VK_TRUE : VK_FALSE;
    f16i8.shaderInt8 = int8_supported ? VK_TRUE : VK_FALSE;

    VkDeviceCreateInfo dci; memset(&dci,0,sizeof dci);
    dci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; dci.queueCreateInfoCount=1;
    dci.pQueueCreateInfos=&dq; dci.pNext=&f2;
    CHECK(g_vkCreateDevice(g_pd,&dci,NULL,&g_dev),"vkCreateDevice");
    g_vkGetDeviceQueue(g_dev,g_qf,0,&g_q);

    /* shader module from embedded SPIR-V */
    VkShaderModuleCreateInfo smi; memset(&smi,0,sizeof smi);
    smi.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize=GEMV_SPV_LEN; smi.pCode=(const uint32_t*)GEMV_SPV;
    CHECK(g_vkCreateShaderModule(g_dev,&smi,NULL,&g_sm),"vkCreateShaderModule");

    /* descriptor set: 5 storage buffers (meta,x,w,scale,y) */
    VkDescriptorSetLayoutBinding bnd[5];
    for(int b=0;b<5;b++){ memset(&bnd[b],0,sizeof bnd[b]); bnd[b].binding=b;
        bnd[b].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bnd[b].descriptorCount=1;
        bnd[b].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dlci; memset(&dlci,0,sizeof dlci);
    dlci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; dlci.bindingCount=5; dlci.pBindings=bnd;
    CHECK(g_vkCreateDescriptorSetLayout(g_dev,&dlci,NULL,&g_dsl),"desc layout");

    VkDescriptorPoolSize dps; dps.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; dps.descriptorCount=5;
    VkDescriptorPoolCreateInfo dpci; memset(&dpci,0,sizeof dpci);
    dpci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; dpci.maxSets=1;
    dpci.poolSizeCount=1; dpci.pPoolSizes=&dps;
    CHECK(g_vkCreateDescriptorPool(g_dev,&dpci,NULL,&g_dpool),"desc pool");

    VkDescriptorSetAllocateInfo dsai; memset(&dsai,0,sizeof dsai);
    dsai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; dsai.descriptorPool=g_dpool;
    dsai.descriptorSetCount=1; dsai.pSetLayouts=&g_dsl;
    CHECK(g_vkAllocateDescriptorSets(g_dev,&dsai,&g_ds),"desc alloc");

    /* NOTE: descriptor buffers (g_w/g_s/g_x/g_y/g_meta) are bound AFTER they
     * are created below (see "bind descriptor buffers" block) — binding here
     * would attach VK_NULL_HANDLE since the buffers don't exist yet. */

    /* pipeline */
    VkPipelineShaderStageCreateInfo stage; memset(&stage,0,sizeof stage);
    stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; stage.module=g_sm; stage.pName="main";
    VkPipelineLayoutCreateInfo plci; memset(&plci,0,sizeof plci);
    plci.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; plci.setLayoutCount=1; plci.pSetLayouts=&g_dsl;
    CHECK(g_vkCreatePipelineLayout(g_dev,&plci,NULL,&g_pl),"pipeline layout");
    VkComputePipelineCreateInfo cpci; memset(&cpci,0,sizeof cpci);
    cpci.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; cpci.stage=stage; cpci.layout=g_pl;
    CHECK(g_vkCreateComputePipelines(g_dev,VK_NULL_HANDLE,1,&cpci,NULL,&g_pipe),"compute pipeline");

    /* int4 path (Shader-only, unpack nibbles + float GEMV). Created only when
     * the model weights are int4. Needs no special features, so it works on
     * every Vulkan driver -- including AMD 0x800184 whose int8/dot-product
     * compiler path segfaults. */
    if(g_weight_bits==4){
        VkShaderModuleCreateInfo i4smi; memset(&i4smi,0,sizeof i4smi);
        i4smi.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        i4smi.codeSize=GEMV_INT4_SPV_LEN; i4smi.pCode=(const uint32_t*)GEMV_INT4_SPV;
        VkResult i4r = g_vkCreateShaderModule(g_dev,&i4smi,NULL,&g_sm_int4);
        if(i4r==VK_SUCCESS){
            VkPipelineShaderStageCreateInfo i4stg; memset(&i4stg,0,sizeof i4stg);
            i4stg.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            i4stg.stage=VK_SHADER_STAGE_COMPUTE_BIT; i4stg.module=g_sm_int4; i4stg.pName="main";
            VkComputePipelineCreateInfo i4cpi; memset(&i4cpi,0,sizeof i4cpi);
            i4cpi.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; i4cpi.stage=i4stg; i4cpi.layout=g_pl;
            VkResult i4pr = g_vkCreateComputePipelines(g_dev,VK_NULL_HANDLE,1,&i4cpi,NULL,&g_pipe_int4);
            if(i4pr==VK_SUCCESS){ g_use_int4=1; }
            else {
#ifdef VG_SELFTEST
                fprintf(stderr,"[vg] int4 pipeline create FAILED (code %d)\n", (int)i4pr);
#endif
                if(g_sm_int4){ g_vkDestroyShaderModule(g_dev,g_sm_int4,NULL); g_sm_int4=VK_NULL_HANDLE; }
            }
        } else {
#ifdef VG_SELFTEST
            fprintf(stderr,"[vg] int4 shader module create FAILED (code %d)\n", (int)i4r);
#endif
        }
    }

    /* integer dot-product path (OpSDotKHR) when the device supports it, the
     * int8 types it needs (shaderInt8) are available, AND the driver is
     * known-good.  AMD driver 0x800184 (Radeon 780M, early RDNA3) crashes the
     * pipeline compiler on ANY shader that computes with 8-bit integers (both
     * plain int8 ALU and OpSDotKHR) — a driver bug. That segfault can't be
     * caught in-process, so we avoid invoking the compiler on it and fall
     * back to the verified float path.  Override with env COLIBRI_IDP:
     * 0 = force off, 1 = force on (accepts the crash risk on known-bad
     * drivers). */
    int idp_blacklisted = (g_vendor_id==0x1002u && g_driver_ver==0x800184u);
    const char *idp_env = getenv("COLIBRI_IDP");
    int idp_force_on  = (idp_env && idp_env[0]=='1');
    int idp_force_off = (idp_env && idp_env[0]=='0');
    int want_idp = 0;
    if(g_weight_bits!=4 && idp_supported && int8_supported && !idp_force_off){
        if(idp_force_on) want_idp = 1;
        else if(!idp_blacklisted) want_idp = 1;
    }
#ifdef VG_SELFTEST
    fprintf(stderr,"[vg] idp_supported=%d vendor=0x%X driver=0x%X blacklisted=%d\n",
            idp_supported, g_vendor_id, g_driver_ver, idp_blacklisted);
#endif
    if(!idp_supported){
        fprintf(stderr,"[vg] OpSDotKHR unsupported by device -> float path\n");
    } else if(idp_force_off){
        fprintf(stderr,"[vg] COLIBRI_IDP=0 -> int8 dot-product disabled (float path)\n");
    } else if(idp_blacklisted && !idp_force_on){
        fprintf(stderr,"[vg] int8 dot-product (OpSDotKHR) DISABLED: AMD driver 0x%X has a "
                "compiler bug that crashes on OpSDotKHR. Update the AMD graphics driver to "
                "enable GPU int8 GEMV. Using float path.\n", g_driver_ver);
    }
    if(want_idp){
#ifdef VG_SELFTEST
        fprintf(stderr,"[vg] creating IDP shader module...\n");
#endif
        VkShaderModuleCreateInfo ismi; memset(&ismi,0,sizeof ismi);
        ismi.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ismi.codeSize=GEMV_IDP_SPV_LEN; ismi.pCode=(const uint32_t*)GEMV_IDP_SPV;
        VkResult smr = g_vkCreateShaderModule(g_dev,&ismi,NULL,&g_sm_idp);
        if(smr==VK_SUCCESS){
#ifdef VG_SELFTEST
            fprintf(stderr,"[vg] IDP shader module OK; creating pipeline...\n");
#endif
            VkPipelineShaderStageCreateInfo istg; memset(&istg,0,sizeof istg);
            istg.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            istg.stage=VK_SHADER_STAGE_COMPUTE_BIT; istg.module=g_sm_idp; istg.pName="main";
            VkComputePipelineCreateInfo icpi; memset(&icpi,0,sizeof icpi);
            icpi.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; icpi.stage=istg; icpi.layout=g_pl;
            VkResult pr = g_vkCreateComputePipelines(g_dev,VK_NULL_HANDLE,1,&icpi,NULL,&g_pipe_idp);
            if(pr==VK_SUCCESS){
                g_use_idp=1;
#ifdef VG_SELFTEST
                fprintf(stderr,"[vg] IDP pipeline OK (g_use_idp=1)\n");
#endif
            } else {
#ifdef VG_SELFTEST
                fprintf(stderr,"[vg] IDP pipeline create FAILED (code %d)\n", (int)pr);
#endif
                if(g_sm_idp){ g_vkDestroyShaderModule(g_dev,g_sm_idp,NULL); g_sm_idp=VK_NULL_HANDLE; }
            }
        } else {
#ifdef VG_SELFTEST
            fprintf(stderr,"[vg] IDP shader module create FAILED (code %d)\n", (int)smr);
#endif
        }
    }

    /* command pool + buffer */
    VkCommandPoolCreateInfo cp2; memset(&cp2,0,sizeof cp2);
    cp2.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cp2.queueFamilyIndex=g_qf;
    CHECK(g_vkCreateCommandPool(g_dev,&cp2,NULL,&g_cpool),"cmd pool");
    VkCommandBufferAllocateInfo cbai; memset(&cbai,0,sizeof cbai);
    cbai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cbai.commandPool=g_cpool;
    cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    CHECK(g_vkAllocateCommandBuffers(g_dev,&cbai,&g_cmd),"cmd alloc");

    /* persistent buffers (host-visible coherent).
     *  - g_w: cached int8 weights, nslots * slot_wbytes
     *  - g_s: cached scales (f32), nslots * slot_sfloats
     *  - g_x: transient inputs (xs + per-expert gact)
     *  - g_y: transient outputs
     *  - g_meta: transient per-matrix metadata */
    {
        VkDeviceSize wsz=(VkDeviceSize)g_nslots*g_slot_wbytes;
        /* FIX (kreuzzelg): slot byte offsets are computed as uint32_t in the host
         * upload path and handed to the shader as 32-bit word offsets, so any
         * weights pool >= 4GB wraps/aliases (layers 32-39 alias layers 0-7 at
         * cache>=256). Refuse to enable Vulkan and fall back to the CPU MoE path
         * instead of silently producing wrong results. 64-bit shader offsets are a
         * follow-up. */
        if (wsz > 0xFFFFFFFFULL) {
            fprintf(stderr, "[vg] weights pool %.1f GB exceeds the 4GB Vulkan 32-bit "
                    "offset limit; falling back to CPU MoE path (64-bit offsets: follow-up).\n",
                    (double)wsz/(1024.0*1024.0*1024.0));
            goto fail;
        }
        if(wsz==0) wsz=16;
        CHECK(vg_create_buf(wsz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, &g_w),"buf w");
        VkDeviceSize ssz=(VkDeviceSize)g_nslots*g_slot_sfloats*sizeof(float);
        if(ssz==0) ssz=16;
        CHECK(vg_create_buf(ssz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, &g_s),"buf s");
        VkDeviceSize xsz=(VkDeviceSize)(g_hidden + g_topk*g_inter + 16)*sizeof(float);
        CHECK(vg_create_buf(xsz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &g_x),"buf x");
        VkDeviceSize ysz=(VkDeviceSize)(g_topk*g_hidden + 16)*sizeof(float);
        CHECK(vg_create_buf(ysz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &g_y),"buf y");
        VkDeviceSize msz=(VkDeviceSize)(1 + 6*2*g_topk + 16)*sizeof(uint32_t);
        CHECK(vg_create_buf(msz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT, &g_meta),"buf meta");
    }

    /* bind descriptor buffers (now that they exist) */
    {
        VkBuffer bufs[5]={g_meta.buf,g_x.buf,g_w.buf,g_s.buf,g_y.buf};
        VkDescriptorBufferInfo dbi[5];
        VkWriteDescriptorSet wds[5];
        for(int b=0;b<5;b++){ dbi[b].buffer=bufs[b]; dbi[b].offset=0; dbi[b].range=VK_WHOLE_SIZE;
            memset(&wds[b],0,sizeof wds[b]); wds[b].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wds[b].dstSet=g_ds; wds[b].dstBinding=b; wds[b].descriptorCount=1;
            wds[b].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[b].pBufferInfo=&dbi[b]; }
        g_vkUpdateDescriptorSets(g_dev,5,wds,0,NULL);
    }

    g_slot=calloc((size_t)g_nslots, sizeof(GSlot));
    if(!g_slot){ fprintf(stderr,"[vg] OOM slot table\n"); goto fail; }

    g_vg_ok=1;
    fprintf(stderr,"[vg] Vulkan GEMV backend ready: %d layers x %d experts, weights %.1f MB%s\n",
            g_nlayers, g_nslots, (double)((uint64_t)g_nslots*g_slot_wbytes)/(1024.0*1024.0),
            g_use_int4 ? " [int4 unpack+float GEMV ACTIVE]"
            : g_use_idp ? " [int8 dot-product OpSDotKHR ACTIVE]" : " [float path]");
    return 0;

fail:
    vg_shutdown();
    return -1;
}

int vg_ready(void){ return g_vg_ok; }
int vg_use_int4(void){ return g_use_int4; }

void vg_shutdown(void){
    if(g_slot){ free(g_slot); g_slot=NULL; }
    if(g_cmd)  g_vkFreeCommandBuffers(g_dev,g_cpool,1,&g_cmd);
    if(g_cpool)g_vkDestroyCommandPool(g_dev,g_cpool,NULL);
    if(g_pipe) g_vkDestroyPipeline(g_dev,g_pipe,NULL);
    if(g_pipe_idp) g_vkDestroyPipeline(g_dev,g_pipe_idp,NULL);
    if(g_pipe_int4) g_vkDestroyPipeline(g_dev,g_pipe_int4,NULL);
    if(g_sm_idp) g_vkDestroyShaderModule(g_dev,g_sm_idp,NULL);
    if(g_sm_int4) g_vkDestroyShaderModule(g_dev,g_sm_int4,NULL);
    if(g_pl)   g_vkDestroyPipelineLayout(g_dev,g_pl,NULL);
    if(g_dpool)g_vkDestroyDescriptorPool(g_dev,g_dpool,NULL);
    if(g_dsl)  g_vkDestroyDescriptorSetLayout(g_dev,g_dsl,NULL);
    if(g_sm)   g_vkDestroyShaderModule(g_dev,g_sm,NULL);
    if(g_y.buf){ g_vkDestroyBuffer(g_dev,g_y.buf,NULL); g_vkFreeMemory(g_dev,g_y.mem,NULL); }
    if(g_x.buf){ g_vkDestroyBuffer(g_dev,g_x.buf,NULL); g_vkFreeMemory(g_dev,g_x.mem,NULL); }
    if(g_s.buf){ g_vkDestroyBuffer(g_dev,g_s.buf,NULL); g_vkFreeMemory(g_dev,g_s.mem,NULL); }
    if(g_w.buf){ g_vkDestroyBuffer(g_dev,g_w.buf,NULL); g_vkFreeMemory(g_dev,g_w.mem,NULL); }
    if(g_meta.buf){ g_vkDestroyBuffer(g_dev,g_meta.buf,NULL); g_vkFreeMemory(g_dev,g_meta.mem,NULL); }
    if(g_dev)  g_vkDestroyDevice(g_dev,NULL);
    if(g_inst) g_vkDestroyInstance(g_inst,NULL);
    if(g_lib)  dl_close(g_lib);
    g_lib=NULL; g_vg_ok=0; g_dev=VK_NULL_HANDLE; g_inst=VK_NULL_HANDLE;
    memset(&g_w,0,sizeof g_w); memset(&g_s,0,sizeof g_s);
    memset(&g_x,0,sizeof g_x); memset(&g_y,0,sizeof g_y); memset(&g_meta,0,sizeof g_meta);
}

/* Upload one expert's weights/scales into GPU slot (layer*cap + li). */
void vg_expert_loaded(int layer, int eid, int li,
                      const int8_t *g, const int8_t *u, const int8_t *d,
                      const float *gs, const float *us, const float *ds){
    if(!g_vg_ok) return;
    int idx = layer*g_cap + li;
    if(idx<0 || idx>=g_nslots) return;
    GSlot *s=&g_slot[idx];
    uint32_t wbase = (uint32_t)idx * g_slot_wbytes;          /* bytes */
    uint32_t sbase = (uint32_t)idx * g_slot_sfloats;          /* floats */
    uint32_t D=(uint32_t)g_hidden, Ih=(uint32_t)g_inter;
    uint32_t gbytes = D*Ih*sizeof(int8_t);
    /* weights (raw int8 bytes; shader unpacks 4-per-uint) */
    memcpy((uint8_t*)g_w.ptr + wbase,            g,  gbytes);
    memcpy((uint8_t*)g_w.ptr + wbase + D*Ih,     u,  gbytes);
    memcpy((uint8_t*)g_w.ptr + wbase + 2*D*Ih,   d,  gbytes);
    /* scales (f32) */
    memcpy((float*)g_s.ptr + sbase,          gs, Ih*sizeof(float));
    memcpy((float*)g_s.ptr + sbase + Ih,     us, Ih*sizeof(float));
    memcpy((float*)g_s.ptr + sbase + 2*Ih,   ds, D *sizeof(float));
    vg_flush(&g_w, wbase, 3u*gbytes);
    vg_flush(&g_s, (VkDeviceSize)sbase*sizeof(float), (VkDeviceSize)(2u*Ih + D)*sizeof(float));

    s->woff_g = wbase/4u;        s->woff_u = (wbase + D*Ih)/4u;      s->woff_d = (wbase + 2*D*Ih)/4u;
    s->soff_g = sbase;           s->soff_u = sbase + Ih;             s->soff_d = sbase + 2*Ih;
    s->layer=layer; s->eid=eid; s->valid=1; s->used=++g_tick;
}

/* Called from the engine's forward thread, once per (layer, slot) before a
 * dispatch. Mirrors the CPU LRU: if the GPU slot already holds this eid we just
 * refresh its LRU stamp; otherwise we upload the int8 weights + scales. */
void vg_expert_ensure(int layer, int li, int eid,
                      const int8_t *g, const int8_t *u, const int8_t *d,
                      const float *gs, const float *us, const float *ds){
    if(!g_vg_ok) return;
    int idx = layer*g_cap + li;
    if(idx<0 || idx>=g_nslots) return;
    GSlot *s=&g_slot[idx];
    if(s->valid && s->eid==eid){ s->used=++g_tick; return; }   /* already cached */
    vg_expert_loaded(layer, eid, li, g, u, d, gs, us, ds);
}

/* int4 variant: weights packed 2 nibbles/byte. Scales/activation semantics
 * identical to the int8 path. Only valid when cfg.weight_bits==4. */
void vg_expert_loaded_int4(int layer, int eid, int li,
                           const uint8_t *g, const uint8_t *u, const uint8_t *d,
                           const float *gs, const float *us, const float *ds){
    if(!g_vg_ok) return;
    int idx = layer*g_cap + li;
    if(idx<0 || idx>=g_nslots) return;
    GSlot *s=&g_slot[idx];
    uint32_t wbase = (uint32_t)idx * g_slot_wbytes;          /* bytes */
    uint32_t sbase = (uint32_t)idx * g_slot_sfloats;          /* floats */
    uint32_t D=(uint32_t)g_hidden, Ih=(uint32_t)g_inter;
    uint32_t gbytes = D*Ih/2u;                                /* int4: half of int8 */
    memcpy((uint8_t*)g_w.ptr + wbase,            g,  gbytes);
    memcpy((uint8_t*)g_w.ptr + wbase + D*Ih/2u,  u,  gbytes);
    memcpy((uint8_t*)g_w.ptr + wbase + D*Ih,     d,  gbytes);
    /* scales (f32) -- same layout as int8 */
    memcpy((float*)g_s.ptr + sbase,          gs, Ih*sizeof(float));
    memcpy((float*)g_s.ptr + sbase + Ih,     us, Ih*sizeof(float));
    memcpy((float*)g_s.ptr + sbase + 2*Ih,   ds, D *sizeof(float));
    vg_flush(&g_w, wbase, 3u*gbytes);
    vg_flush(&g_s, (VkDeviceSize)sbase*sizeof(float), (VkDeviceSize)(2u*Ih + D)*sizeof(float));

    s->woff_g = wbase/4u;          s->woff_u = (wbase + D*Ih/2u)/4u;  s->woff_d = (wbase + D*Ih)/4u;
    s->soff_g = sbase;             s->soff_u = sbase + Ih;           s->soff_d = sbase + 2*Ih;
    s->layer=layer; s->eid=eid; s->valid=1; s->used=++g_tick;
}

void vg_expert_ensure_int4(int layer, int li, int eid,
                           const uint8_t *g, const uint8_t *u, const uint8_t *d,
                           const float *gs, const float *us, const float *ds){
    if(!g_vg_ok) return;
    int idx = layer*g_cap + li;
    if(idx<0 || idx>=g_nslots) return;
    GSlot *s=&g_slot[idx];
    if(s->valid && s->eid==eid){ s->used=++g_tick; return; }
    vg_expert_loaded_int4(layer, eid, li, g, u, d, gs, us, ds);
}

/* Quantize a float vector to packed int8 (4 int8/uint32, LE) with a symmetric
 * per-vector scale (absmax/127). Returns the scale. Used by the integer
 * dot-product path (activations become int8 so OpSDotKHR can be used). */
static float vg_quantize_pack(const float *x, int n, uint32_t *dst_packed){
    float amax=0.0f;
    for(int i=0;i<n;i++){ float a=x[i]<0?-x[i]:x[i]; if(a>amax) amax=a; }
    float s = amax>1e-9f ? amax/127.0f : 1.0f;
    int np=(n+3)/4;
    for(int g=0;g<np;g++){
        uint32_t w=0;
        for(int j=0;j<4;j++){
            int idx=g*4+j; int8_t v=0;
            if(idx<n){ int q=(int)(x[idx]/s + (x[idx]>=0?0.5f:-0.5f));
                       if(q>127)q=127; if(q<-127)q=-127; v=(int8_t)q; }
            w |= ((uint32_t)(unsigned char)v) << (j*8);
        }
        dst_packed[g]=w;
    }
    return s;
}
static uint32_t vg_f2u(float f){ uint32_t u; memcpy(&u,&f,4); return u; }
static int vg_dispatch(uint32_t nmat, uint32_t stride, const uint32_t *meta, uint32_t total){
    uint32_t groups=(total+63u)/64u;
    uint32_t nwords=1u+stride*nmat;
    /* meta buffer */
    memcpy(g_meta.ptr, meta, (size_t)nwords*sizeof(uint32_t));
    vg_flush(&g_meta, 0, (VkDeviceSize)nwords*sizeof(uint32_t));

    VkPipeline pipe = g_use_int4 ? g_pipe_int4 : (g_use_idp ? g_pipe_idp : g_pipe);
    VkCommandBufferBeginInfo bbi; memset(&bbi,0,sizeof bbi);
    bbi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if(g_vkResetCommandBuffer(g_cmd,0)!=VK_SUCCESS){ fprintf(stderr,"[vg-dbg] ResetCmdBuffer FAIL\n"); return -1; }
    if(g_vkBeginCommandBuffer(g_cmd,&bbi)!=VK_SUCCESS){ fprintf(stderr,"[vg-dbg] BeginCmdBuffer FAIL\n"); return -1; }
    g_vkCmdBindPipeline(g_cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    g_vkCmdBindDescriptorSets(g_cmd,VK_PIPELINE_BIND_POINT_COMPUTE,g_pl,0,1,&g_ds,0,NULL);
    g_vkCmdDispatch(g_cmd,groups,1,1);
    if(g_vkEndCommandBuffer(g_cmd)!=VK_SUCCESS){ fprintf(stderr,"[vg-dbg] EndCmdBuffer FAIL\n"); return -1; }
    VkSubmitInfo si; memset(&si,0,sizeof si);
    si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount=1; si.pCommandBuffers=&g_cmd;
    if(g_vkQueueSubmit(g_q,1,&si,VK_NULL_HANDLE)!=VK_SUCCESS){ fprintf(stderr,"[vg-dbg] QueueSubmit FAIL\n"); return -1; }
    if(g_vkQueueWaitIdle(g_q)!=VK_SUCCESS){ fprintf(stderr,"[vg-dbg] QueueWaitIdle FAIL\n"); return -1; }
    return 0;
}

void vg_moe_run(int layer, int K, const int *handles, const float *val,
                const float *xs, float *out){
    if(!g_vg_ok) return;
    const uint32_t D=(uint32_t)g_hidden, Ih=(uint32_t)g_inter;
    uint32_t *meta=malloc((size_t)(1u+6u*2u*(uint32_t)K)*sizeof(uint32_t));
    if(!meta) return;
    float *y=(float*)g_y.ptr;
    uint32_t *xq=(uint32_t*)g_x.ptr;   /* IDP: packed int8; float path aliases as float */

    if(g_use_idp){
        /* ---- IDP path: activations quantized to packed int8, OpSDotKHR ---- */
        /* Phase 1: gate + up, shared activation xs */
        float s1 = vg_quantize_pack(xs, (int)D, xq);          /* at uint offset 0 */
        vg_flush(&g_x, 0, (VkDeviceSize)(D/4u)*4u);
        uint32_t nm1=(uint32_t)(2*K);
        meta[0]=nm1;
        for(int k=0;k<K;k++){
            GSlot *s=&g_slot[handles[k]];
            uint32_t m=(uint32_t)(2*k);
            meta[1+6*m+0]=0;        meta[1+6*m+1]=s->woff_g; meta[1+6*m+2]=s->soff_g;
            meta[1+6*m+3]=D/4u;      meta[1+6*m+4]=Ih;        meta[1+6*m+5]=vg_f2u(s1);
            uint32_t m2=(uint32_t)(2*k+1);
            meta[1+6*m2+0]=0;        meta[1+6*m2+1]=s->woff_u; meta[1+6*m2+2]=s->soff_u;
            meta[1+6*m2+3]=D/4u;     meta[1+6*m2+4]=Ih;       meta[1+6*m2+5]=vg_f2u(s1);
        }
        uint32_t total1=(uint32_t)(2*K)*Ih;
        if(vg_dispatch(nm1,6,meta,total1)!=0){ free(meta); return; }
        vg_invalidate(&g_y,0,(VkDeviceSize)total1*sizeof(float));
        /* read back gate/up, compute gact = silu(g)*u per expert */
        float *gact=malloc((size_t)K*Ih*sizeof(float));
        for(int k=0;k<K;k++){
            const float *gk=y+(size_t)(2*k)*Ih;
            const float *uk=y+(size_t)(2*k+1)*Ih;
            float *ak=gact+(size_t)k*Ih;
            for(uint32_t i=0;i<Ih;i++){ float gv=gk[i]; ak[i]=gv/(1.0f+expf(-gv))*uk[i]; }
        }
        /* Phase 2: down, per-expert activation (own scale) */
        uint32_t nm2=(uint32_t)K;
        meta[0]=nm2;
        uint32_t xoff_k=0;
        float *sks=malloc((size_t)K*sizeof(float));
        for(int k=0;k<K;k++){
            float sk=vg_quantize_pack(gact+(size_t)k*Ih, (int)Ih, xq + (size_t)k*(Ih/4u));
            sks[k]=sk;
            GSlot *s=&g_slot[handles[k]];
            uint32_t m=(uint32_t)k;
            meta[1+6*m+0]=xoff_k;    meta[1+6*m+1]=s->woff_d; meta[1+6*m+2]=s->soff_d;
            meta[1+6*m+3]=Ih/4u;     meta[1+6*m+4]=D;         meta[1+6*m+5]=vg_f2u(sk);
            xoff_k += Ih/4u;
        }
        vg_flush(&g_x,0,(VkDeviceSize)xoff_k*4u);
        uint32_t total2=(uint32_t)K*D;
        if(vg_dispatch(nm2,6,meta,total2)!=0){ free(meta); free(gact); free(sks); return; }
        vg_invalidate(&g_y,0,(VkDeviceSize)total2*sizeof(float));
        for(int k=0;k<K;k++){
            const float *hk=y+(size_t)k*D; float w=val[k];
            for(uint32_t d=0;d<D;d++) out[d]+= w*hk[d];
        }
        free(meta); free(gact); free(sks);
        return;
    }

    /* ---- int4 path: unpack nibbles + float GEMV (Shader-only, AMD-safe) ---- */
    if(g_use_int4){
        float *x=(float*)g_x.ptr;
        memcpy(x, xs, D*sizeof(float));
        vg_flush(&g_x, 0, (VkDeviceSize)D*sizeof(float));
        uint32_t nm1=(uint32_t)(2*K);
        meta[0]=nm1;
        for(int k=0;k<K;k++){
            GSlot *s=&g_slot[handles[k]];
            uint32_t m=(uint32_t)(2*k);
            meta[1+5*m+0]=0;            meta[1+5*m+1]=s->woff_g; meta[1+5*m+2]=s->soff_g; meta[1+5*m+3]=D/8u; meta[1+5*m+4]=Ih;
            uint32_t m2=(uint32_t)(2*k+1);
            meta[1+5*m2+0]=0;           meta[1+5*m2+1]=s->woff_u; meta[1+5*m2+2]=s->soff_u; meta[1+5*m2+3]=D/8u; meta[1+5*m2+4]=Ih;
        }
        uint32_t total1=(uint32_t)(2*K)*Ih;
        if(vg_dispatch(nm1,5,meta,total1)!=0){ free(meta); return; }
        vg_invalidate(&g_y,0,(VkDeviceSize)total1*sizeof(float));
        if(g_dbg){
            static int pc=0;
            if(pc++<1){
            fprintf(stderr,"[vg-dbg] Phase1 gate[0..3]=%.4f %.4f %.4f %.4f  up[0..3]=%.4f %.4f %.4f %.4f\n",
                y[0],y[1],y[2],y[3], y[Ih],y[Ih+1],y[Ih+2],y[Ih+3]);
            }
        }
        for(int k=0;k<K;k++){
            const float *gk = y + (size_t)(2*k)*Ih;
            const float *uk = y + (size_t)(2*k+1)*Ih;
            float *ak = x + D + (size_t)k*Ih;
            for(uint32_t i=0;i<Ih;i++){
                float gv=gk[i]; float a=gv/(1.0f+expf(-gv)); ak[i]=a*uk[i];
            }
        }
        vg_flush(&g_x, (VkDeviceSize)D*sizeof(float), (VkDeviceSize)((uint32_t)K*Ih)*sizeof(float));
        uint32_t nm2=(uint32_t)K;
        meta[0]=nm2;
        for(int k=0;k<K;k++){
            GSlot *s=&g_slot[handles[k]];
            uint32_t m=(uint32_t)k;
            meta[1+5*m+0]=D + (uint32_t)k*Ih; meta[1+5*m+1]=s->woff_d; meta[1+5*m+2]=s->soff_d;
            meta[1+5*m+3]=Ih/8u; meta[1+5*m+4]=D;
        }
        uint32_t total2=(uint32_t)K*D;
        if(vg_dispatch(nm2,5,meta,total2)!=0){ free(meta); return; }
        vg_invalidate(&g_y,0,(VkDeviceSize)total2*sizeof(float));
        for(int k=0;k<K;k++){
            const float *hk = y + (size_t)k*D; float w=val[k];
            for(uint32_t d=0;d<D;d++) out[d]+= w*hk[d];
        }
        free(meta);
        return;
    }

    /* ---- float path (fallback) ---- */
    float *x=(float*)g_x.ptr;
    memcpy(x, xs, D*sizeof(float));
    vg_flush(&g_x, 0, (VkDeviceSize)D*sizeof(float));
    uint32_t nm1=(uint32_t)(2*K);
    meta[0]=nm1;
    for(int k=0;k<K;k++){
        GSlot *s=&g_slot[handles[k]];
        uint32_t m=(uint32_t)(2*k);
        meta[1+5*m+0]=0;            meta[1+5*m+1]=s->woff_g; meta[1+5*m+2]=s->soff_g; meta[1+5*m+3]=D; meta[1+5*m+4]=Ih;
        meta[1+5*m+5]=0;            meta[1+5*m+6]=s->woff_u; meta[1+5*m+7]=s->soff_u; meta[1+5*m+8]=D; meta[1+5*m+9]=Ih;
    }
    uint32_t total1=(uint32_t)(2*K)*Ih;
    if(vg_dispatch(nm1,5,meta,total1)!=0){ free(meta); return; }
    vg_invalidate(&g_y, 0, (VkDeviceSize)total1*sizeof(float));
    for(int k=0;k<K;k++){
        const float *gk = y + (size_t)(2*k)*Ih;
        const float *uk = y + (size_t)(2*k+1)*Ih;
        float *ak = x + D + (size_t)k*Ih;
        for(uint32_t i=0;i<Ih;i++){
            float gv=gk[i]; float a=gv/(1.0f+expf(-gv)); ak[i]=a*uk[i];
        }
    }
    vg_flush(&g_x, (VkDeviceSize)D*sizeof(float), (VkDeviceSize)((uint32_t)K*Ih)*sizeof(float));
    uint32_t nm2=(uint32_t)K;
    meta[0]=nm2;
    for(int k=0;k<K;k++){
        GSlot *s=&g_slot[handles[k]];
        uint32_t m=(uint32_t)k;
        meta[1+5*m+0]=D + (uint32_t)k*Ih; meta[1+5*m+1]=s->woff_d; meta[1+5*m+2]=s->soff_d;
        meta[1+5*m+3]=Ih; meta[1+5*m+4]=D;
    }
    uint32_t total2=(uint32_t)K*D;
    if(vg_dispatch(nm2,5,meta,total2)!=0){ free(meta); return; }
    vg_invalidate(&g_y, 0, (VkDeviceSize)total2*sizeof(float));
    for(int k=0;k<K;k++){
        const float *hk = y + (size_t)k*D;
        float w=val[k];
        for(uint32_t d=0;d<D;d++) out[d]+= w*hk[d];
    }
    free(meta);
}

/* ===================================================================== */
#ifdef VG_SELFTEST
/* CPU reference (mirrors qwen36.c matmul_q, no NEON) */
static void ref_matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O){
    for(int o=0;o<O;o++){
        const int8_t *w=q+(int64_t)o*I; double acc=0;
        for(int i=0;i<I;i++) acc+=(double)x[i]*(double)w[i];
        y[o]=(float)(acc*scale[o]);
    }
}
static float silu(float z){ return z/(1.0f+expf(-z)); }
/* symmetric int8 quantization scale (matches vg_quantize_pack) */
static float ref_qscale(const float *x, int n){
    float amax=0; for(int i=0;i<n;i++){ float a=x[i]<0?-x[i]:x[i]; if(a>amax)amax=a; }
    return amax>1e-9f ? amax/127.0f : 1.0f;
}
/* CPU int8-activation matmul (mirrors the IDP GPU path exactly) */
static void ref_mq_i8(float *y, const float *x, float xs, const int8_t *q, const float *wscale, int I, int O){
    for(int o=0;o<O;o++){
        const int8_t *w=q+(int64_t)o*I; long acc=0;
        for(int i=0;i<I;i++){ int vi=(int)(x[i]/xs + (x[i]>=0?0.5f:-0.5f)); if(vi>127)vi=127; if(vi<-127)vi=-127; acc+=(long)vi*(long)w[i]; }
        y[o]=wscale[o]*xs*(float)acc;
    }
}

int main(void){
    srand(20260724);
    int wbits = 4;   /* default: verify the int4 unpack+float path (AMD-safe) */
    const char *wbenv=getenv("COLIBRI_WBITS");
    if(wbenv) wbits = atoi(wbenv);
    /* Real-model dimensions (D=4096,Ih=512,K=8,cap=8) can be forced via env
     * (TD/TIH/TK/TCAP/TNL) to reproduce the "self-test passes, real model
     * outputs zero" bug without loading the 35B container. */
    int D_test = getenv("TD")   ? atoi(getenv("TD"))   : 96;
    int Ih_test= getenv("TIH")  ? atoi(getenv("TIH"))  : 40;
    int K_test = getenv("TK")   ? atoi(getenv("TK"))   : 4;
    int C_test = getenv("TCAP") ? atoi(getenv("TCAP")) : 6;
    int NL_test= getenv("TNL")  ? atoi(getenv("TNL"))  : 2;
    vg_cfg cfg; cfg.n_layers=NL_test; cfg.hidden=D_test; cfg.inter=Ih_test; cfg.cap=C_test; cfg.topk=K_test; cfg.weight_bits=wbits;
    if(vg_init(&cfg)!=0){ fprintf(stderr,"SELFTEST: vg_init failed (no GPU?) -> cannot verify\n"); return 2; }

    int D=cfg.hidden, Ih=cfg.inter, K=cfg.topk;
    /* build K experts (use layers 0,1) with random integer weights in [-8,7]
     * -- valid as both int4 and int8, so the same values feed both paths. */
    int nE = 2*cfg.cap;
    int nwb = Ih*D;                 /* weights per matrix (gate/up/down) */
    int8_t  *Eg=malloc((size_t)nE*nwb), *Eu=malloc((size_t)nE*nwb), *Ed=malloc((size_t)nE*nwb);
    uint8_t *Eg4=malloc((size_t)nE*nwb/2), *Eu4=malloc((size_t)nE*nwb/2), *Ed4=malloc((size_t)nE*nwb/2);
    float   *Egs=malloc((size_t)nE*Ih*4), *Eus=malloc((size_t)nE*Ih*4), *Eds=malloc((size_t)nE*D*4);
    for(int e=0;e<nE;e++){
        for(int j=0;j<nwb;j++){
            int8_t v=(int8_t)((rand()%16)-8);   /* signed int4 [-8,7] */
            Eg[e*nwb+j]=v; Eu[e*nwb+j]=v; Ed[e*nwb+j]=v;
        }
        /* pack 2 int4 per byte: even index -> low nibble, odd -> high nibble (LE) */
        for(int j=0;j<nwb;j+=2){
            uint8_t lo,hi;
            lo=(uint8_t)(Eg[e*nwb+j]&0xF);   hi=(uint8_t)(Eg[e*nwb+j+1]&0xF); Eg4[e*(nwb/2)+j/2]=(hi<<4)|lo;
            lo=(uint8_t)(Eu[e*nwb+j]&0xF);   hi=(uint8_t)(Eu[e*nwb+j+1]&0xF); Eu4[e*(nwb/2)+j/2]=(hi<<4)|lo;
            lo=(uint8_t)(Ed[e*nwb+j]&0xF);   hi=(uint8_t)(Ed[e*nwb+j+1]&0xF); Ed4[e*(nwb/2)+j/2]=(hi<<4)|lo;
        }
        for(int j=0;j<Ih;j++){ Egs[e*Ih+j]=0.01f+((float)rand()/RAND_MAX)*0.09f; Eus[e*Ih+j]=Egs[e*Ih+j]; }
        for(int j=0;j<D;j++)  Eds[e*D+j]=0.01f+((float)rand()/RAND_MAX)*0.09f;
    }

    float *xs=malloc(D*sizeof(float));
    for(int i=0;i<D;i++) xs[i]=((float)rand()/RAND_MAX)*2.f-1.f;
    float val[K]; float sv=0; for(int k=0;k<K;k++){ val[k]=((float)rand()/RAND_MAX)+0.1f; sv+=val[k]; } for(int k=0;k<K;k++) val[k]/=sv;

    /* upload experts (li = e % cap, layer = e / cap) */
    int handles[K];
    for(int k=0;k<K;k++){
        int e=k; int layer=e/cfg.cap, li=e%cfg.cap;
        if(wbits==4)
            vg_expert_loaded_int4(layer, e, li, Eg4+e*(nwb/2), Eu4+e*(nwb/2), Ed4+e*(nwb/2),
                                   Egs+e*Ih, Eus+e*Ih, Eds+e*D);
        else
            vg_expert_loaded(layer, e, li, Eg+e*nwb, Eu+e*nwb, Ed+e*nwb,
                             Egs+e*Ih, Eus+e*Ih, Eds+e*D);
        handles[k]=layer*cfg.cap+li;
    }

    /* GPU run */
    float *out_gpu=calloc(D,sizeof(float));
    vg_moe_run(0, K, handles, val, xs, out_gpu);

    /* CPU references */
    float *g=malloc(Ih*sizeof(float)), *u=malloc(Ih*sizeof(float)), *hh=malloc(D*sizeof(float));
    float *out_cpu=calloc(D,sizeof(float));       /* float activation reference */
    float *out_cpu_i8=calloc(D,sizeof(float));     /* int8 activation reference (matches IDP) */
    float s1 = ref_qscale(xs, D);                   /* shared activation scale for gate+up */
    for(int k=0;k<K;k++){
        int e=k;
        /* float path reference */
        ref_matmul_q(g, xs, Eg+e*Ih*D, Egs+e*Ih, D, Ih);
        ref_matmul_q(u, xs, Eu+e*Ih*D, Eus+e*Ih, D, Ih);
        for(int i=0;i<Ih;i++) g[i]=silu(g[i])*u[i];
        ref_matmul_q(hh, g, Ed+e*Ih*D, Eds+e*D, Ih, D);
        for(int d=0;d<D;d++) out_cpu[d]+=val[k]*hh[d];
        /* int8-activation reference (mirrors IDP: gate/up share s1, down per-expert) */
        ref_mq_i8(g, xs, s1, Eg+e*Ih*D, Egs+e*Ih, D, Ih);
        ref_mq_i8(u, xs, s1, Eu+e*Ih*D, Eus+e*Ih, D, Ih);
        for(int i=0;i<Ih;i++) g[i]=silu(g[i])*u[i];
        float sk = ref_qscale(g, Ih);
        ref_mq_i8(hh, g, sk, Ed+e*Ih*D, Eds+e*D, Ih, D);
        for(int d=0;d<D;d++) out_cpu_i8[d]+=val[k]*hh[d];
    }

    /* choose the reference that matches the active GPU path */
    float *out_ref = g_use_idp ? out_cpu_i8 : out_cpu;
    double dot=0,n1=0,n2=0;
    for(int d=0;d<D;d++){ dot+=(double)out_ref[d]*out_gpu[d]; n1+=(double)out_ref[d]*out_ref[d]; n2+=(double)out_gpu[d]*out_gpu[d]; }
    double cos = dot/sqrt(n1*n2);
    printf("weight_bits=%d  int4_active=%d  idp_active=%d\n", wbits, g_use_int4, g_use_idp);
    printf("CPU[0..3] = %.5f %.5f %.5f %.5f\n", out_ref[0],out_ref[1],out_ref[2],out_ref[3]);
    printf("GPU[0..3] = %.5f %.5f %.5f %.5f\n", out_gpu[0],out_gpu[1],out_gpu[2],out_gpu[3]);
    printf("cosine(GPU, %s-ref) = %.6f\n", g_use_idp?"int8-act":(g_use_int4?"int4":"float"), cos);

    /* second run (cached experts) should still match */
    float *out_gpu2=calloc(D,sizeof(float));
    vg_moe_run(0, K, handles, val, xs, out_gpu2);
    double dot2=0,n22=0; for(int d=0;d<D;d++){ dot2+=(double)out_ref[d]*out_gpu2[d]; n22+=(double)out_gpu2[d]*out_gpu2[d]; }
    double cos2=dot2/sqrt(n1*n22);
    printf("cosine(GPU cached, %s-ref) = %.6f\n", g_use_idp?"int8-act":(g_use_int4?"int4":"float"), cos2);

    int pass = (cos>0.999 && cos2>0.999);
    printf(pass?"RESULT: PASS (GPU batched MoE matches CPU)\n":"RESULT: FAIL\n");

    free(Eg);free(Eu);free(Ed);free(Eg4);free(Eu4);free(Ed4);free(Egs);free(Eus);free(Eds);free(xs);
    free(g);free(u);free(hh);free(out_cpu);free(out_gpu);free(out_gpu2);
    vg_shutdown();
    return pass?0:1;
}
#endif /* VG_SELFTEST */
