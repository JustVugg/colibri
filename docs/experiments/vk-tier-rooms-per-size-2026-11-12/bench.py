import json, time, urllib.request, sys, random
url, n = sys.argv[1], int(sys.argv[2])
random.seed(int(sys.argv[3]))
base = ("In distributed systems, consensus protocols like Raft and Paxos keep replicated logs consistent across failures. "
        "A mixture-of-experts transformer routes each token to a small subset of expert feed-forward networks, "
        "activating only a fraction of parameters while keeping total capacity high for better quality. ")
def call(text, mt):
    body = json.dumps({"model":"flint-colibri","messages":[{"role":"user","content":text}],"max_tokens":mt,"temperature":0}).encode()
    t0=time.time()
    r=json.load(urllib.request.urlopen(urllib.request.Request(url,body,{"Content-Type":"application/json"}),timeout=900))
    return r, time.time()-t0
call("hi", 16)   # warm-up
for i in range(n):
    nonce = " ".join(f"item{random.randint(0,10**9)}" for _ in range(40))
    text = nonce + "\n" + (base*110) + f"\nUnique {random.randint(0,10**9)}. In one short sentence, summarize the topic."
    r,w = call(text, 4)
    pt=(r.get("usage") or {}).get("prompt_tokens") or 0
    print(f"prefill {pt} tok {w:.2f}s {pt/w:.1f} tok/s", flush=True)
    r,w = call(f"Seed {random.randint(0,10**9)}. Write a 150-word explanation of how a heat pump works.", 128)
    u=r.get("usage") or {}
    print(f"decode {u.get('completion_tokens')} tok {w:.2f}s {u.get('completion_tokens',0)/w:.1f} tok/s (wall, incl. {u.get('prompt_tokens')} prompt tok)", flush=True)
