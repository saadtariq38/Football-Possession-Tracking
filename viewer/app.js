'use strict';
const $=id=>document.getElementById(id), canvas=$('pitch'), ctx=canvas.getContext('2d');
const colors=['#fa8b75','#73b9ef'], cache=new Map(), pending=new Map();
let meta, frame=11185, target=11185, selected=null, playing=false, loading=false, revision=0, last=0, credit=0;
let geometry, drawn=[];
const finite=Number.isFinite;
function clock(f){const s=f/meta.fps;return `${String(Math.floor(s/60)).padStart(2,'0')}:${(s%60).toFixed(2).padStart(5,'0')}`;}
async function chunk(id){
 if(id<0||id>=Math.ceil(meta.frames/meta.chunkSize))return;
 if(cache.has(id)){const a=cache.get(id);cache.delete(id);cache.set(id,a);return a;}
 if(pending.has(id))return pending.get(id);
 const promise=(async()=>{const r=await fetch(`/data/${String(id).padStart(4,'0')}.bin`);if(!r.ok)throw Error('Tracking chunk unavailable');const a=new Float32Array(await r.arrayBuffer());const rows=Math.min(meta.chunkSize,meta.frames-id*meta.chunkSize);if(a.length!==rows*meta.stride)throw Error('Tracking chunk is incomplete');cache.set(id,a);while(cache.size>8)cache.delete(cache.keys().next().value);return a;})();
 pending.set(id,promise);try{return await promise;}finally{pending.delete(id);}
}
function row(f){if(f<0||f>=meta.frames)return null;const a=cache.get(Math.floor(f/meta.chunkSize));const start=(f%meta.chunkSize)*meta.stride;return a?a.subarray(start,start+meta.stride):null;}
function pause(){playing=false;$('play').textContent='Play';credit=0;}
async function show(value){
 target=Math.max(0,Math.min(meta.frames-1,Math.round(value)));const wanted=target, token=++revision;loading=true;$('status').textContent='Loading observations…';
 try{const id=Math.floor(wanted/meta.chunkSize);await chunk(id);if(wanted%meta.chunkSize<25&&wanted>=25)await chunk(id-1);if(token!==revision)return;frame=wanted;loading=false;last=performance.now();$('timeline').value=frame;$('frameInput').value=frame;render();if(frame%meta.chunkSize>meta.chunkSize-100)chunk(id+1).catch(()=>{});}
 catch(e){if(token!==revision)return;loading=false;pause();$('status').textContent=e.message;}
}
function render(){
 if(!meta)return;const data=row(frame);if(!data)return;
 const rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
 canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);
 const W=rect.width,H=rect.height,pad=28,scale=Math.min((W-2*pad)/meta.pitch[0],(H-2*pad)/meta.pitch[1]), pw=scale*meta.pitch[0],ph=scale*meta.pitch[1],ox=(W-pw)/2,oy=(H-ph)/2;
 geometry={ox,oy,pw,ph};const xy=(x,y)=>[ox+x*pw,oy+y*ph];
 ctx.fillStyle='#121e20';ctx.fillRect(0,0,W,H);ctx.fillStyle='#204b3b';ctx.fillRect(ox,oy,pw,ph);
 for(let i=0;i<10;i+=2){ctx.fillStyle='#244f3f';ctx.fillRect(ox+i*pw/10,oy,pw/10,ph);}
 ctx.strokeStyle='#b6cdb7';ctx.lineWidth=1.3;
 const line=(x1,y1,x2,y2)=>{ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();};
 const circle=(x,y,r)=>{ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.stroke();};
 ctx.strokeRect(ox,oy,pw,ph);line(ox+pw/2,oy,ox+pw/2,oy+ph);circle(ox+pw/2,oy+ph/2,9.15*scale);
 ctx.fillStyle='#b6cdb7';ctx.beginPath();ctx.arc(ox+pw/2,oy+ph/2,2,0,7);ctx.fill();
 for(const right of [false,true]){const x=right?ox+pw:ox,dir=right?-1:1;
  for(const [depth,width] of [[16.5,40.32],[5.5,18.32]])ctx.strokeRect(right?x-depth*scale:x,oy+(ph-width*scale)/2,depth*scale,width*scale);
  ctx.strokeRect(right?x:x-5,oy+ph/2-3.66*scale,5,7.32*scale);
  ctx.beginPath();const angle=Math.acos(5.5/9.15);ctx.arc(x+dir*11*scale,oy+ph/2,9.15*scale,right?Math.PI-angle:-angle,right?Math.PI+angle:angle);ctx.stroke();
  ctx.fillStyle='#b6cdb7';ctx.beginPath();ctx.arc(x+dir*11*scale,oy+ph/2,1.6,0,7);ctx.fill();
 }
 if($('trails').checked){for(let slot=0;slot<=meta.slots.length;slot++){ctx.strokeStyle=slot===meta.slots.length?'#fff4bd':colors[meta.slots[slot]];ctx.globalAlpha=.4;ctx.lineWidth=1.5;ctx.beginPath();let connected=false;for(let f=Math.max(0,frame-25);f<=frame;f++){const old=row(f),j=slot*3;if(!old||!finite(old[j])||!finite(old[j+1])){connected=false;continue;}const [x,y]=xy(old[j],old[j+1]);if(connected)ctx.lineTo(x,y);else ctx.moveTo(x,y);connected=true;}ctx.stroke();}ctx.globalAlpha=1;}
 drawn=[];let counts=[0,0];
 for(let slot=0;slot<=meta.slots.length;slot++){const j=slot*3;if(!finite(data[j])||!finite(data[j+1]))continue;const ball=slot===meta.slots.length;if(!ball)counts[meta.slots[slot]]++;const [x,y]=xy(data[j],data[j+1]);drawn.push({slot,x,y});
  if(slot===selected){ctx.strokeStyle='#eff5de';ctx.lineWidth=1.5;circle(x,y,11);}
  ctx.beginPath();ctx.arc(x,y,ball?4.5:6,0,7);ctx.fillStyle=ball?'#fff4bd':colors[meta.slots[slot]];ctx.fill();ctx.strokeStyle='#102c23';ctx.lineWidth=1.5;ctx.stroke();
  if($('labels').checked&&!ball){ctx.font='10px system-ui';ctx.textAlign='center';ctx.fillStyle='#f2f5e8';ctx.fillText(slot+1,x,y-11);}
 }
 const b=meta.slots.length*3, visible=finite(data[b])&&finite(data[b+1]);
 $('clock').textContent=clock(frame);$('frameText').textContent=frame.toLocaleString();$('rawText').textContent=(frame+1).toLocaleString();$('count').textContent=`${counts[0]+counts[1]} (${counts[0]} H / ${counts[1]} A)`;$('ballState').textContent=visible?'Observed':'Not observed';$('status').textContent=loading?'Loading observations…':visible?'● Ball observed':'○ Ball not observed';
 if(selected!==null){const j=selected*3,ball=selected===meta.slots.length;const title=ball?'Ball':`Track ${selected+1} · ${meta.teams[meta.slots[selected]]}`;const position=finite(data[j])&&finite(data[j+1]);const fmt=v=>finite(v)?v.toFixed(2):'Unavailable';$('details').replaceChildren();const heading=document.createElement('strong');heading.textContent=title;$('details').append(heading);const p=document.createElement('p');p.className='muted';p.textContent=position?`Pitch position: ${fmt(data[j]*meta.pitch[0])} m, ${fmt(data[j+1]*meta.pitch[1])} m. Speed: ${finite(data[j+2])?fmt(data[j+2])+' m/s':'unavailable'}.`:'Position not observed in this frame.';$('details').append(p);}
}
canvas.addEventListener('click',e=>{const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=null,distance=16;for(const p of drawn){const d=Math.hypot(p.x-x,p.y-y);if(d<distance){best=p.slot;distance=d;}}selected=best;if(best===null)$('details').innerHTML='<p class="muted">Click a player or the ball to inspect its observation.</p>';render();});
$('play').onclick=()=>{if(!meta)return;if(playing)pause();else{playing=true;$('play').textContent='Pause';last=performance.now();if(frame===meta.frames-1)show(0);}};
function step(d){if(!meta)return;pause();show(target+d);}
$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('timeline').oninput=e=>{if(!meta)return;pause();show(Number(e.target.value));};
$('jump').onsubmit=e=>{e.preventDefault();if(!meta)return;pause();show(Number($('frameInput').value));};
$('labels').onchange=render;$('trails').onchange=render;
window.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click();}if(e.code==='ArrowRight'||e.code==='ArrowLeft'){e.preventDefault();step(e.code==='ArrowRight'?1:-1);}});
new ResizeObserver(()=>render()).observe(canvas);
function tick(now){if(playing&&meta&&!loading){credit+=(now-last)*meta.fps*Number($('speed').value)/1000;if(credit>=1){const advance=Math.floor(credit);credit-=advance;show(frame+advance);if(target===meta.frames-1)pause();}}last=now;requestAnimationFrame(tick);}requestAnimationFrame(tick);
(async()=>{try{const r=await fetch('/data/manifest.json');if(!r.ok)throw Error('Manifest unavailable');meta=await r.json();$('timeline').max=meta.frames-1;$('frameInput').max=meta.frames-1;$('rangeText').textContent=`${meta.frames.toLocaleString()} FRAMES · ${clock(meta.frames-1)} VIDEO TIME`;await show(frame);}catch(e){$('status').textContent=e.message;}})();
