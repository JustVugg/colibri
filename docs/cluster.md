# Local cluster mode

The coordinator keeps token generation, routing, and KV state local while
disk-backed expert workers execute routed FFNs on other Macs. A layer's routed
batch-union is sent as one persistent TCP request, so a token does not incur one
round trip per expert.

Start the optional registration service:

```bash
./coli cluster coordinator --host 0.0.0.0 --port 8765
```

On each worker, with the same converted model available locally:

```bash
./coli cluster worker --model /nvme/glm52_i4 --port 9100 \
  --coordinator http://COORDINATOR:8765 --advertise-host WORKER_IP
```

Run the coordinator with discovery, or provide `--cluster-workers
HOST:PORT,...` for a static setup:

```bash
./coli serve --model /nvme/glm52_i4 \
  --cluster-coordinator http://127.0.0.1:8765
```

GLM-5.3 (`glm53`, including Flash) serves its streamed int4 experts the same
way. A GLM-5.3 worker loads only the expert table and cache, so the RAM the
dense weights would take goes to keeping its experts warm. Experts are owned by
a deterministic weighted hash, so each one always lands on the same worker.
Each worker measures its disk at startup and gets a share in proportion to it.
To set the shares yourself, pass one weight per listed worker with
`COLI_CLUSTER_WEIGHTS=3,2,1` on the coordinator, or `COLI_WORKER_WEIGHT` on a
worker. Requests for a layer go to every worker before any reply is read, so
their disks read at the same time, and while a worker computes one block of a
layer's experts it reads the next block into the other half of that layer's
cache slots, so its disk and its cores work at the same time too (a layer with
one slot runs its blocks in turn). Set `CLUSTER_WORKER_BIND` to a private
address to keep a worker off other interfaces. `GLM53_VERBOSE=1` on the
coordinator prints each worker's mean reply time at exit; `GLM53_VERBOSE=2`
prints every layer's per-worker reply time and the wait for the slowest, the
numbers to set the weights by.

A prefill chunk is the unit of disk reads on every worker: each chunk makes a
worker read a layer's experts again, so with workers the coordinator prefills
the whole prompt in one chunk, up to 1024 tokens. On two SATA-SSD workers that
took a 245-token prefill from 618 s to 454 s. Bigger chunks amortize the reads
further but put up to 16 KiB per token row in flight each way on every worker,
so the default stops there; `GLM53_PREFILL_CHUNK` still overrides, up to the
65536-row cap of one worker request (8192 tokens at top-8).

Nothing in a GLM-5.3 cluster waits forever. `GLM53_CLUSTER_TIMEOUT` (seconds,
default 120) bounds connecting to a worker, every message in flight, and a
worker's reply to a layer's request: a worker that does not answer in time
ends the coordinator with `expert worker HOST:PORT did not answer in N s`, and
a client that connects to a worker without completing the handshake in time
is dropped, so it cannot keep the port from the real coordinator. A
coordinator that has completed the handshake may pause between requests for
as long as it likes (a serve waiting for its next prompt); one that vanishes
without closing is noticed by TCP keepalive in about two deadlines. Raise the
deadline for a worker whose disk needs more than two minutes per prefill
chunk.

The transport is disabled unless workers are configured, so the existing
single-machine path remains unchanged. Dense-layer sharding and browser/WebGPU
workers are separate follow-up seams.

## Engines

The protocol (`COLIEX01`) is carried by two engines, and a worker serves the
family it was built for -- engines are never mixed on one coordinator:

- `colibri` (GLM): the routed experts of every layer, as above.
- `deepseek_v41` (DeepSeek V4.1 Flash): the backbone's fp4 routed experts. A
  worker holds one expert slot per layer and runs the same `matmul_mxfp4`
  kernel the coordinator would, so a row comes back as the bytes the local path
  produces. Routing, the shared expert, attention, the DSA indexer, the engram
  tables, the vision tower and the DSpark draft stages (their own, smaller
  expert set) stay on the coordinator.

```bash
./coli cluster worker --model /nvme/dsv41 --port 9100 --advertise-host WORKER_IP
./coli serve --model /nvme/dsv41 --cluster-workers WORKER_IP:9100
```

`c/tests/test_cluster_sharding.py` and `c/tests/test_dsv41_cluster_sharding.py`
are the token-exact gates: a local run and a delegated run of the same binary
must decode the same tokens, and the delegated one must have gone through the
worker.
