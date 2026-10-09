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

## Protocol v2

The wire protocol (`COLIEX01` in `c/colibri.c`) has two versions. v1, the
default, moves raw f32 rows and is what every coordinator speaks unless asked
otherwise, so a cluster of mixed engine builds keeps working. v2 adds an `act`
word to the header; a worker accepts both. The coordinator speaks v2 only when
it needs one of:

- **q8 activations**, `COLI_CLUSTER_ACT=q8`: every row crosses the wire as int8
  blocks of 32 behind one f32 scale, both directions -- about 3.5x fewer bytes,
  and not token-exact (the gate gives it a drift bound, not a seat).
- **The shared expert on the worker**, `CLUSTER_SHARED=1`: each layer's shared
  expert rides the layer's first routed request as one more item (every row,
  unweighted, in the request's `act`) to the worker `layer % n_workers`, which
  loads the three matrices once and keeps them (GLM-5.2 at 8-bit dense: 2.9 GB
  over all layers). With f32 rows the worker runs the same CPU sequence FASE E
  runs and the coordinator adds the rows at the same point, so neither a token
  nor a logits byte moves: `c/tests/test_cluster_sharding.py` holds that run to
  the local run's logits bytes. Off by default.

Whenever the coordinator will speak v2 it first sends every worker an empty v2
request, a hello. A worker built from an older engine closes on it instead of
answering, and the coordinator refuses that worker by name (`expert worker
HOST:PORT closed on the COLIEX01 v2 hello ... rebuild the worker`) before
generating anything. A worker refuses a version or activation format it does
not speak the same way, on its own stderr. The coordinator reports bytes sent
and received at exit.

The transport is disabled unless workers are configured, so the existing
single-machine path remains unchanged. Sharding the attention and dense
matrices across workers, and browser/WebGPU workers, are separate follow-up
seams.
