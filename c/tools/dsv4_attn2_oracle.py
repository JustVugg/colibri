#!/usr/bin/env python3
"""Official layer-0 position-1 attention oracle with a two-token KV cache."""
import json, math, struct, sys, torch
from safetensors import safe_open

model, first, second, output = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
idx = json.load(open(model + "/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(model + "/config.json"))
def t(name):
    with safe_open(model + "/" + idx[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)
def rms(x, w):
    return (x.float() * torch.rsqrt(x.float().square().mean() + 1e-6) * w.float()).bfloat16()
def sim(x, block):
    z=x.float().view(-1,block); scale=torch.pow(2,torch.ceil(torch.log2(z.abs().amax(1).clamp_min(1e-4)/448)))
    return ((z/scale[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale[:,None]).flatten().bfloat16()
def deq(stem):
    w=t(stem+".weight").float();s=t(stem+".scale").float()
    return (w.unflatten(0,(s.shape[0],-1)).unflatten(2,(s.shape[1],-1))*s[:,None,:,None]).flatten(0,1).flatten(1,2)
def linear(stem,x): return torch.mv(deq(stem),sim(x,128).float()).bfloat16()
def mhc(token):
    res=t("embed.weight")[token].repeat(4,1);flat=res.flatten().float()
    mix=torch.mv(t("layers.0.hc_attn_fn").float(),flat)*torch.rsqrt(flat.square().mean()+1e-6)
    sc=t("layers.0.hc_attn_scale").float();base=t("layers.0.hc_attn_base").float()
    pre=torch.sigmoid(mix[:4]*sc[0]+base[:4])+1e-6
    return (pre[:,None]*res.float()).sum(0).bfloat16()
rs=cfg["rope_scaling"]; dim=cfg["qk_rope_head_dim"]; base=cfg["rope_theta"]
freq=1/base**(torch.arange(0,dim,2,dtype=torch.float32)/dim)
def corr(rot): return dim*math.log(rs["original_max_position_embeddings"]/(rot*2*math.pi))/(2*math.log(base))
low=max(math.floor(corr(rs["beta_fast"])),0);high=min(math.ceil(corr(rs["beta_slow"])),dim-1)
ramp=((torch.arange(dim//2)-low)/(high-low)).clamp(0,1);smooth=1-ramp
freq=freq/rs["factor"]*(1-smooth)+freq*smooth
def rope(x,pos,inverse=False):
    shape=x.shape;y=x.float().view(*shape[:-1],-1,2);a=pos*freq*(-1 if inverse else 1);cs=torch.cos(a);sn=torch.sin(a)
    return torch.stack((y[...,0]*cs-y[...,1]*sn,y[...,0]*sn+y[...,1]*cs),-1).reshape(shape).bfloat16()
def qkv(token,pos):
    x=rms(mhc(token),t("layers.0.attn_norm.weight"))
    qr=rms(linear("layers.0.attn.wq_a",x),t("layers.0.attn.q_norm.weight"))
    q=linear("layers.0.attn.wq_b",qr).view(64,512)
    q=(q.float()*torch.rsqrt(q.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()
    q[:,-64:]=rope(q[:,-64:],pos)
    kv=rms(linear("layers.0.attn.wkv",x),t("layers.0.attn.kv_norm.weight"))
    kv[-64:]=rope(kv[-64:],pos);kv=torch.cat((sim(kv[:-64],64),kv[-64:]))
    return q,kv
_,kv0=qkv(first,0);q,kv1=qkv(second,1);cache=torch.stack((kv0,kv1))
score=torch.einsum("hd,td->ht",q.float(),cache.float())/(512**.5)
sink=t("layers.0.attn.attn_sink").float()[:,None]
prob=torch.softmax(torch.cat((score,sink),1),1)[:,:2]
o=torch.einsum("ht,td->hd",prob,cache.float()).bfloat16();o[:,-64:]=rope(o[:,-64:],1,True)
wa=deq("layers.0.attn.wo_a").bfloat16().view(8,1024,4096)
oa=torch.cat([torch.mv(wa[g].float(),o[g*8:g*8+8].flatten().float()).bfloat16() for g in range(8)])
out=linear("layers.0.attn.wo_b",oa).float()
with open(output,"wb") as f:f.write(struct.pack(f"<{out.numel()}f",*out.tolist()))
print(out[0].item())
