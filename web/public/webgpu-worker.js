/* Standalone browser-side WebGPU expert worker.
 * The coordinator sends the same COLIEX01 batch used by native expert workers.
 * Weights are intentionally plain little-endian f32 in this first slice: this
 * is easy to validate across browsers, and a later exporter can add f16/int8
 * variants without changing the dispatch contract. */

const MAGIC = "COLIEX01";
const VERSION = 1;

function textMagic(bytes) {
  return new TextDecoder().decode(bytes);
}

function putU32(view, offset, value) {
  view.setUint32(offset, value >>> 0, false);
}

function getU32(view, offset) {
  return view.getUint32(offset, false);
}

function concat(parts) {
  const size = parts.reduce((total, part) => total + part.byteLength, 0);
  const out = new Uint8Array(size);
  let offset = 0;
  for (const part of parts) {
    out.set(new Uint8Array(part.buffer || part, part.byteOffset || 0, part.byteLength), offset);
    offset += part.byteLength;
  }
  return out;
}

export class WebGPUExpertWorker {
  constructor({ manifest, workerId, experts = ["*"], deviceType = "browser" }) {
    this.manifest = manifest;
    this.workerId = workerId;
    this.experts = experts;
    this.deviceType = deviceType;
    this.socket = null;
    this.device = null;
    this.cache = new Map();
    this.gatePipeline = null;
    this.downPipeline = null;
  }

  async connect(url) {
    if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter found");
    this.device = await adapter.requestDevice();
    this._createPipelines();
    this.socket = new WebSocket(url);
    this.socket.binaryType = "arraybuffer";
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", () => reject(new Error("WebGPU worker socket failed")), { once: true });
    });
    this.socket.send(JSON.stringify({
      role: "webgpu", node_id: this.workerId, host: "browser", port: 0,
      device_type: this.deviceType, precision: "f32", expert_ids: this.experts,
    }));
    this.socket.addEventListener("message", event => this._onMessage(event.data));
    return this;
  }

  _createPipelines() {
    const dims = `struct Dims { rows: u32, hidden: u32, intermediate: u32, pad: u32 };
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read> gate: array<f32>;
      @group(0) @binding(2) var<storage, read> up: array<f32>;
      @group(0) @binding(3) var<storage, read_write> hidden: array<f32>;
      @group(0) @binding(4) var<uniform> dims: Dims;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x; let total = dims.rows * dims.intermediate;
        if (index >= total) { return; }
        let row = index / dims.intermediate; let unit = index % dims.intermediate;
        var g = 0.0; var u = 0.0;
        for (var d = 0u; d < dims.hidden; d++) {
          let value = input[row * dims.hidden + d];
          g += gate[unit * dims.hidden + d] * value;
          u += up[unit * dims.hidden + d] * value;
        }
        let silu = g / (1.0 + exp(-g)); hidden[index] = silu * u;
      }`;
    this.gatePipeline = this.device.createComputePipeline({ layout: "auto", compute: {
      module: this.device.createShaderModule({ code: dims }), entryPoint: "main",
    }});
    const down = `struct Dims { rows: u32, hidden: u32, intermediate: u32, pad: u32 };
      @group(0) @binding(0) var<storage, read> hidden: array<f32>;
      @group(0) @binding(1) var<storage, read> down: array<f32>;
      @group(0) @binding(2) var<storage, read_write> output: array<f32>;
      @group(0) @binding(3) var<uniform> dims: Dims;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x; let total = dims.rows * dims.hidden;
        if (index >= total) { return; }
        let row = index / dims.hidden; let dim = index % dims.hidden;
        var sum = 0.0;
        for (var unit = 0u; unit < dims.intermediate; unit++) {
          sum += down[dim * dims.intermediate + unit] * hidden[row * dims.intermediate + unit];
        }
        output[index] = sum;
      }`;
    this.downPipeline = this.device.createComputePipeline({ layout: "auto", compute: {
      module: this.device.createShaderModule({ code: down }), entryPoint: "main",
    }});
  }

  async _loadExpert(layer, expertId) {
    const key = `${layer}:${expertId}`;
    if (this.cache.has(key)) return this.cache.get(key);
    const spec = this.manifest.experts[key];
    if (!spec) throw new Error(`manifest has no expert ${key}`);
    const [gate, up, down] = await Promise.all([spec.gate, spec.up, spec.down]
      .map(path => fetch(new URL(path, this.manifest.base_url || location.href))
        .then(response => { if (!response.ok) throw new Error(`cannot load ${path}`); return response.arrayBuffer(); })
        .then(buffer => new Float32Array(buffer))));
    const expert = { gate: this._weightBuffer(gate), up: this._weightBuffer(up), down: this._weightBuffer(down) };
    this.cache.set(key, expert);
    return expert;
  }

  _weightBuffer(values) {
    const buffer = this.device.createBuffer({ size: values.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, values);
    return buffer;
  }

  _buffer(bytes, usage) {
    const buffer = this.device.createBuffer({ size: Math.max(4, bytes.byteLength),
      usage: usage | GPUBufferUsage.COPY_SRC });
    this.device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  }

  async _forward(layer, expertId, rows, hidden, inputBytes) {
    const intermediate = this.manifest.intermediate;
    const expert = await this._loadExpert(layer, expertId);
    const input = this._buffer(inputBytes, GPUBufferUsage.STORAGE);
    const hiddenBuffer = this.device.createBuffer({ size: rows * intermediate * 4,
      usage: GPUBufferUsage.STORAGE });
    const output = this.device.createBuffer({ size: rows * hidden * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = this.device.createBuffer({ size: rows * hidden * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const uniform = this.device.createBuffer({ size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(uniform, 0, new Uint32Array([rows, hidden, intermediate, 0]));
    const gateGroup = this.device.createBindGroup({ layout: this.gatePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: expert.gate } },
      { binding: 2, resource: { buffer: expert.up } }, { binding: 3, resource: { buffer: hiddenBuffer } },
      { binding: 4, resource: { buffer: uniform } },
    ]});
    const downGroup = this.device.createBindGroup({ layout: this.downPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: hiddenBuffer } }, { binding: 1, resource: { buffer: expert.down } },
      { binding: 2, resource: { buffer: output } }, { binding: 3, resource: { buffer: uniform } },
    ]});
    const encoder = this.device.createCommandEncoder();
    const first = encoder.beginComputePass(); first.setPipeline(this.gatePipeline); first.setBindGroup(0, gateGroup);
    first.dispatchWorkgroups(Math.ceil(rows * intermediate / 64)); first.end();
    const second = encoder.beginComputePass(); second.setPipeline(this.downPipeline); second.setBindGroup(0, downGroup);
    second.dispatchWorkgroups(Math.ceil(rows * hidden / 64)); second.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, rows * hidden * 4);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = readback.getMappedRange().slice(0);
    readback.unmap();
    for (const buffer of [input, hiddenBuffer, output, readback, uniform]) buffer.destroy();
    return result;
  }

  async _onMessage(data) {
    try {
      const bytes = new Uint8Array(data);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      if (textMagic(bytes.subarray(0, 8)) !== MAGIC || getU32(view, 8) !== VERSION) throw new Error("bad request");
      const layer = getU32(view, 12), hidden = getU32(view, 16), count = getU32(view, 20);
      if (hidden !== this.manifest.hidden || count > 64) throw new Error("unsupported request shape");
      let offset = 24, outputs = [];
      for (let i = 0; i < count; i++) {
        const expertId = getU32(view, offset), rows = getU32(view, offset + 4); offset += 8;
        const size = rows * hidden * 4;
        const input = bytes.slice(offset, offset + size); offset += size;
        outputs.push({ expertId, rows, bytes: await this._forward(layer, expertId, rows, hidden, input) });
      }
      const parts = [new Uint8Array(20)]; const response = new DataView(parts[0].buffer);
      for (let i = 0; i < 8; i++) parts[0][i] = MAGIC.charCodeAt(i);
      putU32(response, 8, VERSION); putU32(response, 12, 0); putU32(response, 16, outputs.length);
      for (const output of outputs) { const header = new Uint8Array(8); const hv = new DataView(header.buffer);
        putU32(hv, 0, output.expertId); putU32(hv, 4, output.rows); parts.push(header, new Uint8Array(output.bytes)); }
      this.socket.send(concat(parts));
    } catch (error) {
      console.error("WebGPU expert request failed", error);
      const response = new Uint8Array(20); const view = new DataView(response.buffer);
      for (let i = 0; i < 8; i++) response[i] = MAGIC.charCodeAt(i);
      putU32(view, 8, VERSION); putU32(view, 12, 1); putU32(view, 16, 0);
      this.socket.send(response);
    }
  }
}
