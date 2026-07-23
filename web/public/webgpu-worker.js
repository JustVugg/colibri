/* Browser WebGPU expert worker. Weights are little-endian f32 in this first
 * format; the manifest keeps conversion explicit and auditable. */
const MAGIC="COLIEX01", VERSION=1;
const textMagic=bytes=>new TextDecoder().decode(bytes);
const putU32=(view,offset,value)=>view.setUint32(offset,value>>>0,false);
const getU32=(view,offset)=>view.getUint32(offset,false);
function concat(parts){const size=parts.reduce((n,p)=>n+p.byteLength,0),out=new Uint8Array(size);let at=0;
 for(const part of parts){out.set(new Uint8Array(part.buffer||part,part.byteOffset||0,part.byteLength),at);at+=part.byteLength}return out}

export class WebGPUExpertWorker {
 constructor({manifest,workerId,experts=["*"],deviceType="browser"}){this.manifest=manifest;this.workerId=workerId;this.experts=experts;this.deviceType=deviceType;this.cache=new Map()}
 async connect(url){
  if(!navigator.gpu)throw new Error("WebGPU is unavailable in this browser");
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error("no WebGPU adapter found");
  this.device=await adapter.requestDevice();this._createPipelines();this.socket=new WebSocket(url);this.socket.binaryType="arraybuffer";
  await new Promise((resolve,reject)=>{this.socket.addEventListener("open",resolve,{once:true});this.socket.addEventListener("error",()=>reject(new Error("WebGPU worker socket failed")),{once:true})});
  this.socket.send(JSON.stringify({role:"webgpu",node_id:this.workerId,host:"browser",port:0,device_type:this.deviceType,precision:"f32",expert_ids:this.experts}));
  this.socket.addEventListener("message",event=>this._onMessage(event.data));return this;
 }
 _createPipelines(){
  const common=`struct Dims{rows:u32,hidden:u32,intermediate:u32,pad:u32};`;
  const gate=`${common}@group(0)@binding(0)var<storage,read>input:array<f32>;@group(0)@binding(1)var<storage,read>gate:array<f32>;@group(0)@binding(2)var<storage,read>up:array<f32>;@group(0)@binding(3)var<storage,read_write>hidden:array<f32>;@group(0)@binding(4)var<uniform>dims:Dims;@compute@workgroup_size(64)fn main(@builtin(global_invocation_id)id:vec3<u32>){let i=id.x;let total=dims.rows*dims.intermediate;if(i>=total){return}let row=i/dims.intermediate;let unit=i%dims.intermediate;var g=0.0;var u=0.0;for(var d=0u;d<dims.hidden;d++){let v=input[row*dims.hidden+d];g+=gate[unit*dims.hidden+d]*v;u+=up[unit*dims.hidden+d]*v}hidden[i]=(g/(1.0+exp(-g)))*u}`;
  const down=`${common}@group(0)@binding(0)var<storage,read>hidden:array<f32>;@group(0)@binding(1)var<storage,read>down:array<f32>;@group(0)@binding(2)var<storage,read_write>output:array<f32>;@group(0)@binding(3)var<uniform>dims:Dims;@compute@workgroup_size(64)fn main(@builtin(global_invocation_id)id:vec3<u32>){let i=id.x;let total=dims.rows*dims.hidden;if(i>=total){return}let row=i/dims.hidden;let dim=i%dims.hidden;var sum=0.0;for(var unit=0u;unit<dims.intermediate;unit++){sum+=down[dim*dims.intermediate+unit]*hidden[row*dims.intermediate+unit]}output[i]=sum}`;
  const pipeline=code=>this.device.createComputePipeline({layout:"auto",compute:{module:this.device.createShaderModule({code}),entryPoint:"main"}});
  this.gatePipeline=pipeline(gate);this.downPipeline=pipeline(down);
 }
 async _loadExpert(layer,expertId){const key=`${layer}:${expertId}`;if(this.cache.has(key))return this.cache.get(key);const spec=this.manifest.experts[key];if(!spec)throw new Error(`manifest has no expert ${key}`);
  const values=await Promise.all([spec.gate,spec.up,spec.down].map(path=>fetch(new URL(path,this.manifest.base_url||location.href)).then(response=>{if(!response.ok)throw new Error(`cannot load ${path}`);return response.arrayBuffer()}).then(buffer=>new Float32Array(buffer))));
  const expert={gate:this._weightBuffer(values[0]),up:this._weightBuffer(values[1]),down:this._weightBuffer(values[2])};this.cache.set(key,expert);return expert;
 }
 _weightBuffer(values){const buffer=this.device.createBuffer({size:values.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(buffer,0,values);return buffer}
 _buffer(bytes,usage){const buffer=this.device.createBuffer({size:Math.max(4,bytes.byteLength),usage:usage|GPUBufferUsage.COPY_SRC});this.device.queue.writeBuffer(buffer,0,bytes);return buffer}
 async _forward(layer,expertId,rows,hidden,inputBytes){const intermediate=this.manifest.intermediate,expert=await this._loadExpert(layer,expertId);
  const input=this._buffer(inputBytes,GPUBufferUsage.STORAGE),hiddenBuffer=this.device.createBuffer({size:rows*intermediate*4,usage:GPUBufferUsage.STORAGE});
  const output=this.device.createBuffer({size:rows*hidden*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),readback=this.device.createBuffer({size:rows*hidden*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),uniform=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.device.queue.writeBuffer(uniform,0,new Uint32Array([rows,hidden,intermediate,0]));
  const gg=this.device.createBindGroup({layout:this.gatePipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:expert.gate}},{binding:2,resource:{buffer:expert.up}},{binding:3,resource:{buffer:hiddenBuffer}},{binding:4,resource:{buffer:uniform}}]});
  const dg=this.device.createBindGroup({layout:this.downPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:hiddenBuffer}},{binding:1,resource:{buffer:expert.down}},{binding:2,resource:{buffer:output}},{binding:3,resource:{buffer:uniform}}]});
  const encoder=this.device.createCommandEncoder(),first=encoder.beginComputePass();first.setPipeline(this.gatePipeline);first.setBindGroup(0,gg);first.dispatchWorkgroups(Math.ceil(rows*intermediate/64));first.end();
  const second=encoder.beginComputePass();second.setPipeline(this.downPipeline);second.setBindGroup(0,dg);second.dispatchWorkgroups(Math.ceil(rows*hidden/64));second.end();encoder.copyBufferToBuffer(output,0,readback,0,rows*hidden*4);this.device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);const result=readback.getMappedRange().slice(0);readback.unmap();[input,hiddenBuffer,output,readback,uniform].forEach(buffer=>buffer.destroy());return result;
 }
 async _onMessage(data){try{const bytes=new Uint8Array(data),view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);if(textMagic(bytes.subarray(0,8))!==MAGIC||getU32(view,8)!==VERSION)throw new Error("bad request");
  const layer=getU32(view,12),hidden=getU32(view,16),count=getU32(view,20);if(hidden!==this.manifest.hidden||count>64)throw new Error("unsupported request shape");let offset=24,outputs=[];
  for(let i=0;i<count;i++){const expertId=getU32(view,offset),rows=getU32(view,offset+4);offset+=8;const size=rows*hidden*4;if(rows<1||rows>65536||offset+size>bytes.byteLength)throw new Error("invalid activation shape");const input=bytes.slice(offset,offset+size);offset+=size;outputs.push({expertId,rows,bytes:await this._forward(layer,expertId,rows,hidden,input)})}
  const head=new Uint8Array(20),response=new DataView(head.buffer);for(let i=0;i<8;i++)head[i]=MAGIC.charCodeAt(i);putU32(response,8,VERSION);putU32(response,12,0);putU32(response,16,outputs.length);const parts=[head];
  for(const output of outputs){const item=new Uint8Array(8),view2=new DataView(item.buffer);putU32(view2,0,output.expertId);putU32(view2,4,output.rows);parts.push(item,new Uint8Array(output.bytes))}this.socket.send(concat(parts));
 }catch(error){console.error("WebGPU expert request failed",error);const response=new Uint8Array(20),view=new DataView(response.buffer);for(let i=0;i<8;i++)response[i]=MAGIC.charCodeAt(i);putU32(view,8,VERSION);putU32(view,12,1);putU32(view,16,0);this.socket.send(response)}}
}
