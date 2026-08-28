#!/usr/bin/env python3
"""First emitted ratio-4 attention compressor KV oracle."""
import json,math,struct,sys,torch
from safetensors import safe_open
model,layer,token,outp=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4];cfg=json.load(open(model+"/config.json"))
idx=json.load(open(model+"/model.safetensors.index.json"))["weight_map"]
def t(name):
    with safe_open(model+"/"+idx[name],framework="pt",device="cpu") as f:return f.get_tensor(name)
def rms(x,w):return (x.float()*torch.rsqrt(x.float().square().mean()+1e-6)*w.float()).bfloat16()
def sim(x,block):
    z=x.float().view(-1,block);sc=torch.pow(2,torch.ceil(torch.log2(z.abs().amax(1).clamp_min(1e-4)/448)))
    return ((z/sc[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).float()*sc[:,None]).flatten().bfloat16()
res=t("embed.weight")[token].repeat(4,1);flat=res.flatten().float();stem=f"layers.{layer}"
mix=torch.mv(t(stem+".hc_attn_fn").float(),flat)*torch.rsqrt(flat.square().mean()+1e-6)
sc=t(stem+".hc_attn_scale").float();base=t(stem+".hc_attn_base").float()
pre=torch.sigmoid(mix[:4]*sc[0]+base[:4])+1e-6
x=(pre[:,None]*res.float()).sum(0).bfloat16();x=rms(x,t(stem+".attn_norm.weight")).float()
p=stem+".attn.compressor";kv=torch.mv(t(p+".wkv.weight").float(),x);score=torch.mv(t(p+".wgate.weight").float(),x)
ape=t(p+".ape").float();values=torch.cat((kv[:512].repeat(4,1),kv[512:].repeat(4,1)))
scores=torch.cat((score[:512].repeat(4,1)+ape[:,:512],score[512:].repeat(4,1)+ape[:,512:]))
z=(values*scores.softmax(0)).sum(0);z=rms(z,t(p+".norm.weight"))
rs=cfg["rope_scaling"];dim=cfg["qk_rope_head_dim"];base=cfg["compress_rope_theta"]
freq=1/base**(torch.arange(0,dim,2,dtype=torch.float32)/dim)
def corr(rot):return dim*math.log(rs["original_max_position_embeddings"]/(rot*2*math.pi))/(2*math.log(base))
lo=max(math.floor(corr(rs["beta_fast"])),0);hi=min(math.ceil(corr(rs["beta_slow"])),dim-1)
ramp=((torch.arange(dim//2)-lo)/(hi-lo)).clamp(0,1);smooth=1-ramp;freq=freq/rs["factor"]*(1-smooth)+freq*smooth
y=z[-64:].float().view(-1,2);a=4*freq;z[-64:]=torch.stack((y[:,0]*a.cos()-y[:,1]*a.sin(),y[:,0]*a.sin()+y[:,1]*a.cos()),1).flatten().bfloat16()
z=torch.cat((sim(z[:-64],64),z[-64:])).float()
with open(outp,"wb") as f:f.write(struct.pack("<512f",*z.tolist()))
print(z[0].item())
