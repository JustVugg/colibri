 Diagnosis: Why VRAM is misreported under Vulkan                                                                                       
                                                                                                                                       
 The VRAM telemetry pipeline has two layers — the C engine emits HWINFO/TIERS lines, the Python server parses them into hwinfo/tiers   
 JSON, and the web client renders that JSON. Every layer is wired exclusively to the CUDA backend's data sources; the Vulkan backend   
 tracks the same information in separate globals that nothing reads.                                                                   
                                                                                                                                       
 Concretely, in c/telemetry.h:                                                                                                         
                                                                                                                                       
 - hwinfo_emit() reads g_cuda_ndev, coli_cuda_mem_info(), and m->gpu_expert_bytes — all under #ifdef COLI_CUDA. Under COLI_VULKAN,     
   ngpu stays 0, vram_total stays 0, and gpu_name stays "".                                                                            
 - tiers_emit() sets vram/vram_gb from m->gpu_expert_count/m->gpu_expert_bytes, again only under #ifdef COLI_CUDA. But the Vulkan tier 
   never writes those fields — it tracks residency in g_vk_reg_n / g_vk_reg_n2 and a local bytes accumulator inside vk_registry_fill() 
    (colibri.c:7604-7665), with no propagation to m->gpu_expert_*.                                                                     
 - emap_emit() marks a slot as VRAM-tier (tier=2) only when P[z].g.cuda is set — a CUDA-only struct field.                             
                                                                                                                                       
 Downstream consumers inherit the blank:                                                                                               
                                                                                                                                       
 - c/openai_server.py faithfully parses whatever HWINFO/TIERS the engine emits, so it ships zeros to the web client.                   
 - web/src/App.tsx then shows "0× GPU / 0 GB VRAM" and an all-RAM tier bar even when the Vulkan tier is live with hundreds of resident 
   experts.                                                                                                                            
 - c/tools/diag_harness.py and c/tools/efficiency.py parse only [CUDA] … GB VRAM and CUDA expert tier: … log lines — the Vulkan [VK] … 
    lines are invisible to them.                                                                                                       
                                                                                                                                       
 The Vulkan backend already has the needed primitives: coli_vk_mem_budget() / coli_vk_mem_budget2() report device-local heap           
 usage/budget via VK_EXT_memory_budget, coli_vk_tensor_bytes() reports per-tensor VRAM, and G.dev/G2.dev know their physical-device    
 properties. The work is purely plumbing.                                                                                              
                                                                                                                                       
 ────────────────────────────────────────────────────────────────────────────────                                                      
                                                                                                                                       
 Plan, ranked easiest → hardest                                                                                                        
                                                                                                                                       
 Each item is self-contained and can land independently; later items benefit from earlier ones but don't strictly require them.        
                                                                                                                                       
 ### 1. (Easiest) Engine: emit m->gpu_expert_* from the Vulkan tier                                                                    
                                                                                                                                       
 File: c/colibri.c, inside vk_registry_fill() (around line 7604-7685).                                                                 
                                                                                                                                       
 What: After each successful coli_vk_tensor_ensure for a dev0 expert, increment m->gpu_expert_count and add coli_vk_tensor_bytes() of  
 the three tensors to m->gpu_expert_bytes; do the same for dev2 into the same fields (or a paired gpu_expert_count2/bytes2 if you want 
 per-device fidelity — but merging is simpler and sufficient for tiers). On coli_vk_tensor_free/slot reuse (line 2151-2154),           
 decrement. On full tier teardown, zero them.                                                                                          
                                                                                                                                       
 Why easiest: It's a local edit in one function plus its mirror in the free path. m->gpu_expert_* already exists, is backend-neutral   
 storage, and is already read by tiers_emit(). Once written, TIERS vram/vram_gb starts working under Vulkan with zero telemetry-h      
 changes (the #ifdef COLI_CUDA guard in tiers_emit must be relaxed to #if defined(COLI_CUDA) || defined(COLI_VULKAN) — one-line        
 change).                                                                                                                              
                                                                                                                                       
 Risk: The dev2 case double-counts if you're not careful; decide whether gpu_expert_* is "dev0" or "all devices" and document it.      
 Recommend: all devices (matches how g_cuda_ndev multiplies).                                                                          
                                                                                                                                       
 ────────────────────────────────────────────────────────────────────────────────                                                      
                                                                                                                                       
 ### 2. (Easy) telemetry.h: make hwinfo_emit Vulkan-aware                                                                              
                                                                                                                                       
 File: c/telemetry.h, hwinfo_emit() (lines 96-110).                                                                                    
                                                                                                                                       
 What: Add a #ifdef COLI_VULKAN branch mirroring the CUDA one: set ngpu to the count of Vulkan devices brought up (g_vk_dev_count,     
 currently inferred from coli_vk_available() + coli_vk_dev2_available() — add a small counter if needed), set vram_total via           
 coli_vk_mem_budget() (budget field), and set gpu_name to e.g. "Vulkan device xN" or the p.deviceName from the [VK] ready line (cache  
 it in a static). Relax the guard to #if defined(COLI_CUDA) || defined(COLI_VULKAN).                                                   
                                                                                                                                       
 Why easy: coli_vk_mem_budget() already returns total device-local budget in GB; hw_probe already handles the CPU/RAM half. Just add   
 the parallel branch.                                                                                                                  
                                                                                                                                       
 Risk: VK_EXT_memory_budget may be absent on some drivers — coli_vk_mem_budget() returns 0 and vram_total falls back to 0. That's      
 acceptable and honest (it's "budget unavailable"), but consider a fallback to the heap size from vkGetPhysicalDeviceMemoryProperties  
 (non-EXT, always available) for the total field, reserving the budget/used split for the dynamic TIERS line.                          
                                                                                                                                       
 ────────────────────────────────────────────────────────────────────────────────                                                      
                                                                                                                                       
 ### 3. (Easy) telemetry.h: make emap_emit mark Vulkan-resident slots                                                                  
                                                                                                                                       
 File: c/telemetry.h, emap_emit() (lines 139-160).                                                                                     
                                                                                                                                       
 What: The tier = P[z].g.cuda ? 2 : 1 decision is CUDA-only. The Vulkan residency is tracked via vk_reg_at(layer,eid) / vk_reg_has()   
 (colibri.c:454-460). Expose a tiny coli_vk_reg_has(layer,eid) (or reuse the existing vk_reg_has via a declaration) and, in emap_emit, 
 set tier=2 when either the CUDA flag or the Vulkan registry reports residency. Relax the #ifdef COLI_CUDA to include COLI_VULKAN.     
                                                                                                                                       
 Why easy: One new predicate call per slot; the registry lookup already exists.                                                        
                                                                                                                                       
 Risk: vk_reg_at/vk_reg_has are static in colibri.c — either make them non-static and declare them in backend_vulkan.h (or a small     
 vulkan_telemetry.h), or add a public coli_vk_reg_has() wrapper. Prefer the wrapper to keep the registry's storage private.            
                                                                                                                                       
 ────────────────────────────────────────────────────────────────────────────────                                                      
                                                                                                                                       
 ### 4. (Medium) openai_server.py: no code change needed, but verify field stability                                                   
                                                                                                                                       
 File: c/openai_server.py (lines 1473-1501).                                                                                           
                                                                                                                                       
 What: The parser already reads vram_total_gb, vram, vram_gb generically — no change required once items 1-2 land. The only action is  
 a regression check: confirm the engine still emits HWINFO/TIERS with the same field count under COLI_VULKAN=1 builds, and that        
 vram_total_gb is no longer 0 when a Vulkan device is present.                                                                         
                                                                                                                                       
 Why medium and not "nothing": It's a verification/test task, not an edit — but if field semantics shift (e.g., you add vram_used_gb   
 or a backend tag), the parser and the TS types in web/src/lib/api.ts (lines 27-39) must be updated in lockstep. Decide the schema up  
 front in item 2.                                                                                                                      
                                                                                                                                       
 ────────────────────────────────────────────────────────────────────────────────                                                      
                                                                                                                                       
 ### 5. (Medium) Web client: distinguish the backend and label it                                                                      
                                                                                                                                       
 Files: web/src/lib/api.ts (types), web/src/App.tsx (line 262), web/src/i18n/*.ts.                                                     
                                                                                                                                       
 What: Today hwinfo.gpu is a free-form string and the GPU row says N× GPU / N GB VRAM. Add an optional hwinfo.backend?: "cuda" |       
 "vulkan" | "cpu" field (sourced from a new HWINFO field — see item 2) and surface it in the UI, e.g. Vulkan GPU / CUDA GPU, so a user 
 running Vulkan isn't told their VRAM is fine while seeing "0× GPU" (which is the current bug's user-facing symptom). Add i18n keys    
 (hwinfo.backendVulkan, etc.) in all four locale files.                                                                                
                                                                                                                                       
 Why medium: Pure additive TS + UI work, no engine coordination, but touches four locale files and the type contract. Independent of   
 items 1-3 at the code level, but only useful once they land.                                                                          
                                                                                                                                       
 Risk: Breaking the HealthResponse shape for older engine builds — make every new field optional with ?.                               
                                                                                                                                       
 ────────────────────────────────────────────────────────────────────────────────                                                      
                                                                                                                                       
 ### 6. (Medium) Diagnostic tools: parse [VK] log lines                                                                                
                                                                                                                                       
 Files: c/tools/diag_harness.py (lines 109-160, 351, 590), c/tools/efficiency.py (lines 74-90).                                        
                                                                                                                                       
 What: Add regexes mirroring the CUDA ones for the Vulkan lines that already exist in stderr:                                          
 - [VK] ready: <name>, … — device name (backend_vulkan.c:443)                                                                          
 - [VK] expert tier: <N> hot experts resident (<GB> GB VRAM … (colibri.c:7662)                                                         
 - [VK] dev2 ready: <name> … (colibri.c:1031)                                                                                          
 - [VK] dense preloaded: <N> tensors, <GB> GB VRAM … (colibri.c:7600)                                                                  
                                                                                                                                       
 Then efficiency.py's "GPU tier needs gpu_expert_count>0" gate (line 207) should accept the Vulkan resident count too, and             
 diag_harness.py's GPU summary section should print a Vulkan tier: row alongside the CUDA one.                                         
                                                                                                                                       
 Why medium: Mechanical regex+display work, but two files with distinct output formats and the efficiency report has a gating          
 predicate that must be generalized. No engine changes — the log lines already exist.                                                  
                                                                                                                                       
 Risk: The [VK] lines were written for humans, not machines; if field formats drift you'll need to update regexes. Consider adding a   
 machine-stable VKSTAT <N> <GB> stdout line in a follow-up (see item 8).
