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

The transport is disabled unless workers are configured, so the existing
single-machine path remains unchanged. Dense-layer sharding and browser/WebGPU
workers are separate follow-up seams.
