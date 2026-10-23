export interface Vec { x:number; y:number; z:number }
export interface Camera extends Vec { distance:number; yaw:number; pitch:number }
export interface Topic { key:string; center:number[]; count:number }
export interface AtlasEntry { affinity:Record<string,number>; top:string; label:string; entropy:number }
export interface CortexNode extends Vec, AtlasEntry { id:string; index:number; layer:number; expert:number; topic:string }
interface Projected extends Vec { depth:number; scale:number }
export interface ProjectedNode extends CortexNode { depth:number; scale:number }
export const initialCamera:Camera={x:0,y:0,z:0,distance:3.65,yaw:.14,pitch:.87};
const cortexVertices: (Vec & { shade: number })[] = [], cortexFaces: number[][] = [];
// Two folded hemispheres separated by a narrow fissure. This is a visual metaphor.
for(const side of [-1,1]){
 const offset=cortexVertices.length,rows=68,cols=46;
 for(let i=0;i<=rows;i++)for(let j=0;j<=cols;j++){
  const a=.025+(Math.PI-.05)*i/rows,b=-Math.PI/2+Math.PI*j/cols,sa=Math.sin(a),ca=Math.cos(a),sb=Math.sin(b),cb=Math.cos(b);
  const fold=.055*Math.sin(9*a+2.1*Math.sin(3*b))+.024*Math.sin(9*b+3*Math.cos(5*a));
  const x=side*(.058+(.98+fold)*sa*cb),y=((sb<0?.78:.56)+fold)*sa*sb-.035*ca,z=(1.10+fold)*ca;
  cortexVertices.push({x,y,z,shade:.5+.28*Math.sin(9*a+2.1*Math.sin(3*b))+.13*Math.sin(9*b+3*Math.cos(5*a))});
 }
 for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){
  const a=offset+i*(cols+1)+j;cortexFaces.push([a,a+1,a+cols+2,a+cols+1]);
 }
}

export function createCortex(canvas:HTMLCanvasElement, topics:Topic[], expertNodes:CortexNode[], onLabels:(nodes:ProjectedNode[])=>void) {
 const motion=matchMedia('(prefers-reduced-motion: reduce)');
 const brain:{cam:Camera;paused:boolean;time:number;level:number;topic:string|null;selected:number|null;projected:(ProjectedNode|null)[];shellCache?:{key:string;canvas:HTMLCanvasElement}}={cam:{...initialCamera},paused:motion.matches,time:0,level:0,topic:null,selected:null,projected:[]};
 let frame=0,last=0,lastLabels=0,disposed=false;
 let tween:{from:Camera;to:Camera;start:number}|null=null;
 function request(){if(!disposed && !document.hidden && !frame)frame=requestAnimationFrame(tick)}
 function tick(time:number){
   frame=0;if(disposed||document.hidden)return;
   if(!brain.paused)brain.time+=last?Math.min(40,time-last):0;last=time;
   if(tween){const t=Math.min(1,(time-tween.start)/1250),ease=1-Math.pow(1-t,3);for(const key of Object.keys(initialCamera) as (keyof Camera)[])brain.cam[key]=tween.from[key]+(tween.to[key]-tween.from[key])*ease;if(t>=1)tween=null}
   drawBrain();if(!brain.paused||tween)request();
 }
function drawBrain(){
 const box=canvas.getBoundingClientRect(),w=box.width,h=box.height;if(!w||!h)return;
 const dpr=Math.min(devicePixelRatio||1,2);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}
 const ctx=canvas.getContext('2d');if(!ctx)return;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 const cam=brain.cam,focal=Math.min(w*1.08,h*1.2),cx=w/2,cy=h*.43,cosY=Math.cos(cam.yaw),sinY=Math.sin(cam.yaw),cosP=Math.cos(cam.pitch),sinP=Math.sin(cam.pitch);
 function project(v: Vec){const dx=v.x-cam.x,dy=v.y-cam.y,dz=v.z-cam.z,x=dx*cosY+dz*sinY,z0=-dx*sinY+dz*cosY,y=dy*cosP-z0*sinP,z=dy*sinP+z0*cosP,depth=cam.distance-z;if(depth<.06)return null;const scale=focal/depth;return {x:cx+x*scale,y:cy+y*scale,z,depth,scale}}
 const shellOpacity=Math.max(0,Math.min(1,(cam.distance-1.05)/2.1));
 const glow=ctx.createRadialGradient(cx,cy,0,cx,cy,w*.6);glow.addColorStop(0,'rgba(91,148,114,.055)');glow.addColorStop(1,'rgba(91,148,114,0)');ctx.fillStyle=glow;ctx.fillRect(0,0,w,h);
 if(shellOpacity>.005){
  const cacheKey=[w,h,dpr,...Object.values(cam)].join(',');
  if(!brain.shellCache||brain.shellCache.key!==cacheKey){
   const surface=brain.shellCache?.canvas||document.createElement('canvas');if(surface.width!==canvas.width||surface.height!==canvas.height){surface.width=canvas.width;surface.height=canvas.height}const surfaceCtx=surface.getContext('2d');if(!surfaceCtx)return;surfaceCtx.setTransform(dpr,0,0,dpr,0,0);surfaceCtx.clearRect(0,0,w,h);

  const projected=cortexVertices.map(project),faces: {face:number[];points:Projected[];z:number}[]=[];
  for(const face of cortexFaces){const points=face.map(i=>projected[i]).filter((p):p is Projected=>p!==null);if(points.length!==face.length)continue;const z=points.reduce((sum,p)=>sum+p.z,0)/4;faces.push({face,points,z})}
  faces.sort((a,b)=>a.z-b.z);
  for(const f of faces){const shade=f.face.reduce((s,i)=>s+cortexVertices[i].shade,0)/4,near=Math.max(.15,Math.min(1,(f.z+1.25)/2.5)),lum=Math.round(30+shade*68+near*13);surfaceCtx.beginPath();f.points.forEach((p,i)=>i?surfaceCtx.lineTo(p.x,p.y):surfaceCtx.moveTo(p.x,p.y));surfaceCtx.closePath();surfaceCtx.fillStyle=`rgba(${lum*.8|0},${lum+16},${lum+3},${shellOpacity})`;surfaceCtx.fill();surfaceCtx.strokeStyle=`rgba(146,185,161,${(.012+shade*.025)*shellOpacity})`;surfaceCtx.lineWidth=.45;surfaceCtx.stroke()}
  // Fine cortical ridges and the central fissure make the outer form readable.
  for(let i=0;i<projected.length;i+=2){const p=projected[i];if(!p||p.z<-.2)continue;surfaceCtx.beginPath();surfaceCtx.arc(p.x,p.y,.6,0,Math.PI*2);surfaceCtx.fillStyle=`rgba(173,205,184,${.16*shellOpacity})`;surfaceCtx.fill()}
   brain.shellCache={key:cacheKey,canvas:surface};
  }
  ctx.drawImage(brain.shellCache!.canvas,0,0,w,h);
 }
 brain.projected=expertNodes.map(n=>{const p=project(n);return p?{...n,...p}:null});
 const topicFocused=brain.topic;
 // Illustrative paths connect nearby sample experts; they are not recorded routes.
 for(let i=0;i<expertNodes.length;i++){
  const a=brain.projected[i];if(!a)continue;
  const inTopic=a.topic===topicFocused,opacity=brain.level?(inTopic?.19:.018):.055;
  if(a.depth<.10)continue;
  for(const jump of [1,7]){const group=Math.floor(i/36)*36,b=brain.projected[group+(i%36+jump)%36];if(!b||b.depth<.10)continue;ctx.beginPath();ctx.moveTo(a.x,a.y);const mx=(a.x+b.x)/2,my=(a.y+b.y)/2+Math.sin(i)*Math.min(24,Math.abs(a.x-b.x)*.12);ctx.quadraticCurveTo(mx,my,b.x,b.y);ctx.strokeStyle=`rgba(151,206,170,${opacity})`;ctx.lineWidth=inTopic?.85:.55;ctx.stroke();if(inTopic&&jump===1&&i%4===0){const t=(brain.time*.00010+i*.13)%1,px=(1-t)*(1-t)*a.x+2*(1-t)*t*mx+t*t*b.x,py=(1-t)*(1-t)*a.y+2*(1-t)*t*my+t*t*b.y;ctx.beginPath();ctx.arc(px,py,1.6,0,Math.PI*2);ctx.fillStyle='rgba(192,231,204,.7)';ctx.fill()}}
 }
 const visible=brain.projected.filter((p):p is ProjectedNode=>p!==null).sort((a,b)=>a.z-b.z);
 for(const n of visible){if(n.x<-30||n.x>w+30||n.y<-30||n.y>h+30)continue;const active=n.topic===topicFocused,selected=n.index===brain.selected,size=Math.max(.7,Math.min(selected?6:4.1,(selected?.008:active?.0045:.0026)*n.scale)),alpha=brain.level?(active?.86:.1):.42;if(selected||active&&n.index%5===0){ctx.beginPath();ctx.arc(n.x,n.y,size*3.6,0,Math.PI*2);ctx.fillStyle=`rgba(159,216,183,${selected?.10:.035})`;ctx.fill()}ctx.beginPath();ctx.arc(n.x,n.y,size,0,Math.PI*2);ctx.fillStyle=`rgba(${selected?'220,242,227':active?'175,220,190':'128,177,146'},${alpha})`;ctx.fill();if(selected){ctx.beginPath();ctx.arc(n.x,n.y,size+4,0,Math.PI*2);ctx.strokeStyle='rgba(193,229,206,.55)';ctx.lineWidth=.7;ctx.stroke()}}
 if(brain.level===0){
  topics.forEach((topic,i)=>{
   const p=project({x:topic.center[0],y:topic.center[1],z:topic.center[2]});if(!p)return;const left=i<5,labelW=w<480?80:w<650?92:120,x=left?labelW:w-labelW,y=h*(.18+(i%5)*.105)+16;
   ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(left?x+12:x-12,y);ctx.lineTo(p.x,p.y);ctx.strokeStyle='rgba(164,198,177,.20)';ctx.lineWidth=.7;ctx.stroke();ctx.beginPath();ctx.arc(p.x,p.y,3,0,Math.PI*2);ctx.fillStyle='#99bca5';ctx.fill();
  });
 }

 const occupied: { x: number; y: number }[] = [];
 const labels = brain.projected.filter((n): n is ProjectedNode => !!n && n.topic === brain.topic && n.x > 40 && n.x < w-40 && n.y > 115 && n.y < h-170).sort((a,b)=>Number(b.index===brain.selected)-Number(a.index===brain.selected)||a.depth-b.depth).filter(n=>{
   if(occupied.length>=6 || occupied.some(p=>Math.abs(p.x-n.x)<90 && Math.abs(p.y-n.y)<38))return false;
   occupied.push(n);return true;
 });
 if (performance.now()-lastLabels>100 || brain.paused) { onLabels(brain.level && cam.distance<1.5 ? labels : []); lastLabels=performance.now(); }
}

 const resize=new ResizeObserver(request);resize.observe(canvas);
 const visibility=()=>{cancelAnimationFrame(frame);frame=0;last=0;if(!document.hidden)request()};
 const change=()=>{brain.paused=motion.matches;if(motion.matches&&tween){brain.cam={...tween.to};tween=null}request()};
 document.addEventListener('visibilitychange',visibility);motion.addEventListener('change',change);request();
 return {
   move(to:Camera,level:number,topic:string|null,selected:number|null){brain.level=level;brain.topic=topic;brain.selected=selected;if(motion.matches){brain.cam={...to};tween=null}else tween={from:{...brain.cam},to,start:performance.now()};lastLabels=0;request()},
   rotate(x:number,y:number){tween=null;brain.cam.yaw+=x;brain.cam.pitch=Math.max(-1.1,Math.min(1.1,brain.cam.pitch+y));request()},
   zoom(delta:number){tween=null;brain.cam.distance=Math.max(.25,Math.min(4.4,brain.cam.distance+delta));request()},
   pause(value:boolean){brain.paused=value;request()},
   closest(x:number,y:number){return brain.projected.filter((n):n is ProjectedNode=>!!n&&n.topic===brain.topic&&Math.hypot(n.x-x,n.y-y)<22).sort((a,b)=>Math.hypot(a.x-x,a.y-y)-Math.hypot(b.x-x,b.y-y))[0]},
   dispose(){disposed=true;cancelAnimationFrame(frame);resize.disconnect();document.removeEventListener('visibilitychange',visibility);motion.removeEventListener('change',change)}
 }
}
