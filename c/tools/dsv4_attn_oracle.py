#!/usr/bin/env python3
"""Official first-token layer-0 attention oracle (single-rank math)."""
import json, struct, sys, torch
from safetensors import safe_open

model, token, output=sys.argv[1],int(sys.argv[2]),sys.argv[3]
idx=json.load(open(model+"/model.safetensors.index.json"))["weight_map"]
def t(name):
    with safe_open(model+"/"+idx[name],framework="pt",device="cpu") as f:return f.get_tensor(name)
def rms(x,w):return (x.float()*torch.rsqrt(x.float().square().mean()+1e-6)*w.float()).bfloat16()
def deq(stem):
    w=t(stem+".weight").float();s=t(stem+".scale").float()
    return (w.unflatten(0,(s.shape[0],-1)).unflatten(2,(s.shape[1],-1))*s[:,None,:,None]).flatten(0,1).flatten(1,2)
def sim(x,block):
    z=x.float().view(-1,block);mx=z.abs().amax(1).clamp_min(1e-4)/448
    scale=torch.pow(2,torch.ceil(torch.log2(mx)))
    return ((z/scale[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale[:,None]).flatten().bfloat16()
def linear(stem,x):return torch.mv(deq(stem),sim(x,128).float()).bfloat16()

M,H=4,4096;x0=t("embed.weight")[token].repeat(M,1)
flat=x0.flatten().float();fn=t("layers.0.hc_attn_fn").float();scale=t("layers.0.hc_attn_scale").float();base=t("layers.0.hc_attn_base").float()
mix=torch.mv(fn,flat)*torch.rsqrt(flat.square().mean()+1e-6)
pre=torch.sigmoid(mix[:M]*scale[0]+base[:M])+1e-6
x=(pre[:,None]*x0.float()).sum(0).bfloat16();mhc=x.float();x=rms(x,t("layers.0.attn_norm.weight"))
kv=linear("layers.0.attn.wkv",x);kv=rms(kv,t("layers.0.attn.kv_norm.weight"))
kv=torch.cat([sim(kv[:-64],64),kv[-64:]])
qr=rms(linear("layers.0.attn.wq_a",x),t("layers.0.attn.q_norm.weight"))
q=linear("layers.0.attn.wq_b",qr).view(64,512)
q=(q.float()*torch.rsqrt(q.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()
score=torch.mv(q.float(),kv.float())/(512**.5)
keep=torch.sigmoid(score-t("layers.0.attn.attn_sink").float())
wa=deq("layers.0.attn.wo_a").bfloat16().view(8,1024,4096)
oa=torch.cat([torch.mv(wa[g].float(),(keep[g*8:g*8+8,None]*kv.float()).bfloat16().float().flatten()).bfloat16() for g in range(8)])
out=linear("layers.0.attn.wo_b",oa).float()
with open(output,"wb") as f:
    for value in (mhc,x.float(),sim(x,128).float(),kv.float(),oa.float(),out):
        f.write(struct.pack(f"<{value.numel()}f",*value.tolist()))
print(out[0].item())
