#!/usr/bin/env python3
"""Layer-0 mHC + MoE oracle, consuming the official attention trace."""
import json,struct,sys,torch
from safetensors import safe_open
model,token,attn_trace,output=sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4]
idx=json.load(open(model+"/model.safetensors.index.json"))["weight_map"]
def t(name):
    with safe_open(model+"/"+idx[name],framework="pt",device="cpu") as f:return f.get_tensor(name)
def rms(x,w):return (x.float()*torch.rsqrt(x.float().square().mean()+1e-6)*w.float()).bfloat16()
def sim(x,block):
    z=x.float().view(-1,block);scale=torch.pow(2,torch.ceil(torch.log2(z.abs().amax(1).clamp_min(1e-4)/448)))
    return ((z/scale[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale[:,None]).flatten().bfloat16()
def deq8(stem):
    w=t(stem+".weight").float();s=t(stem+".scale").float()
    return (w.unflatten(0,(s.shape[0],-1)).unflatten(2,(s.shape[1],-1))*s[:,None,:,None]).flatten(0,1).flatten(1,2)
def lin8(stem,x):return torch.mv(deq8(stem),sim(x,128).float()).bfloat16()
lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])
def deq4(stem):
    p=t(stem+".weight").view(torch.uint8);s=t(stem+".scale").float()
    lo=(p&15).long();hi=(p>>4).long();w=torch.stack((lut[lo],lut[hi]),-1).flatten(1)
    return (w.unflatten(1,(-1,32))*s[:,:,None]).flatten(1)
def lin4(stem,x):return torch.mv(deq4(stem),sim(x,128).float()).bfloat16()
def hc_pre(res,stem):
    M=4;flat=res.flatten().float();mix=torch.mv(t(stem+"_fn").float(),flat)*torch.rsqrt(flat.square().mean()+1e-6)
    scale=t(stem+"_scale").float();base=t(stem+"_base").float()
    pre=torch.sigmoid(mix[:M]*scale[0]+base[:M])+1e-6;post=torch.sigmoid(mix[M:2*M]*scale[1]+base[M:2*M])*2
    comb=torch.softmax((mix[2*M:]*scale[2]+base[2*M:]).view(M,M),-1)+1e-6
    comb=comb/(comb.sum(-2,keepdim=True)+1e-6)
    for _ in range(19):comb=comb/(comb.sum(-1,keepdim=True)+1e-6);comb=comb/(comb.sum(-2,keepdim=True)+1e-6)
    return (pre[:,None]*res.float()).sum(0).bfloat16(),post,comb
def hc_post(x,res,post,comb):return (post[:,None]*x.float()+(comb[:,:,None]*res[:,None,:].float()).sum(0)).bfloat16()

embed=t("embed.weight")[token];res=embed.repeat(4,1)
_,post,comb=hc_pre(res,"layers.0.hc_attn")
raw=open(attn_trace,"rb").read();attn=torch.tensor(struct.unpack(f"<{len(raw)//4}f",raw)[-4096:])
res=hc_post(attn,res,post,comb)
x,post,comb=hc_pre(res,"layers.0.hc_ffn");x=rms(x,t("layers.0.ffn_norm.weight"))
score=torch.nn.functional.softplus(torch.mv(t("layers.0.ffn.gate.weight").float(),x.float())).sqrt()
ids=t("layers.0.ffn.gate.tid2eid")[token].long();rw=score[ids];rw=rw/rw.sum()*1.5
moe=torch.zeros(4096)
for eid,weight in zip(ids.tolist(),rw):
    p=f"layers.0.ffn.experts.{eid}";g=lin4(p+".w1",x).float().clamp(max=10);u=lin4(p+".w3",x).float().clamp(-10,10)
    a=(torch.nn.functional.silu(g)*u*weight).bfloat16();moe+=lin4(p+".w2",a).float()
p="layers.0.ffn.shared_experts";g=lin8(p+".w1",x).float().clamp(max=10);u=lin8(p+".w3",x).float().clamp(-10,10)
moe=(moe+lin8(p+".w2",(torch.nn.functional.silu(g)*u).bfloat16()).float()).bfloat16()
res=hc_post(moe,res,post,comb).float().flatten()
with open(output,"wb") as f:f.write(struct.pack(f"<{res.numel()}f",*res.tolist()))
print(ids.tolist(),res[0].item())
