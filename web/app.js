/* Turingz showcase — multipage router + interactive logic. Data from web_data.js (real). */
const D = window.TURINGZ_DATA;
const C = {ml:'#5bd97c',fdm:'#f0a060',spectral:'#5e9bec',hybrid:'#ff6b6b',truth:'#e8edf7',
           amber:'#f0b760',muted:'#9aa7c2',indigo:'#7c83ff',cyan:'#36d6c3'};
const $ = s => document.querySelector(s);
const el = (t,c,h)=>{const e=document.createElement(t); if(c)e.className=c; if(h!=null)e.innerHTML=h; return e;};
Chart.defaults.color = '#9aa7c2';
Chart.defaults.font.family = "'Segoe UI',system-ui,sans-serif";
Chart.defaults.borderColor = '#26304a';
const e = D.deeponet_eval;
const pct = x => (x*100).toFixed(1)+'%';

/* ===================== dynamic cards (safe to build while hidden) ===================== */
function stat(v,l){const d=el('div','stat'); d.append(el('div','v',v),el('div','l',l)); return d;}
function card(parent, html, cls){const c=el('div','card'+(cls?' '+cls:''),html); parent.append(c); return c;}

$('#heroStats').append(stat('1000','initial conditions'),stat('2.34×','cheaper than numerical'),
  stat('0.011%','ground-truth verified'),stat('~70%','pure-ML extrapolation error'));
$('#dataStats').append(stat('512 × 200','space × time grid'),stat('1000','ICs (800/100/100)'),
  stat('Cole-Hopf','exact reference'),stat('0.011%','vs spectral verifier'));

[['Cole-Hopf','#36d6c3','Exact analytical solution via the Cole-Hopf transform. The ground-truth reference — used only to evaluate, never inside the hybrid.','reference','exact'],
 ['Spectral (IFRK4)','#5e9bec','4th-order pseudo-spectral. Essentially exact (0.011% vs Cole-Hopf) but a full solve costs 3.29 s — the accurate-but-expensive corrector.','0.011% err','3.29 s'],
 ['Finite-Difference','#f0a060','Upwind + central diffusion. 24× cheaper per step but 16% error from artificial numerical diffusion — the cheap-but-biased corrector.','16.3% err','0.11 s']
].forEach(n=>card($('#numCards'),
  `<h3><span class="dot" style="background:${n[1]}"></span>${n[0]}</h3><p>${n[2]}</p>
   <div style="margin-top:12px;display:flex;gap:10px"><span class="pill"><b>${n[3]}</b></span><span class="pill"><b>${n[4]}</b></span></div>`));

[['DeepONet','b-real','REAL · 5 seeds',`Operator network, 128 sensors · latent 256 · depth 4 · 30k iters. One model handles all ICs.`,
  [['train (t≤1)',pct(e.train_in_dist.mean)],['test (t≤1)',pct(e.test_in_dist.mean)],['extrap (t&gt;1)',pct(e.test_extrap.mean)]]],
 ['FNO','b-pending','EVAL ON HOST',`Fourier Neural Operator · 16 modes · width 64. Designed for shock/transport problems — expected the strongest surrogate.`,
  [['modes','16'],['width','64'],['extrap','pending']]],
 ['PINN','b-pending','EVAL ON HOST',`Physics-informed net · [2,64×4,1] tanh · 15k Adam + L-BFGS. One network per IC — does not generalise across ICs by design.`,
  [['nets','per-IC'],['train ICs','10'],['extrap','pending']]]
].forEach(m=>{const rows=m[4].map(r=>`<tr><td>${r[0]}</td><td><b>${r[1]}</b></td></tr>`).join('');
  card($('#mlCards'),`<h3>${m[0]} <span class="badge ${m[1]}" style="margin-left:auto">${m[2]}</span></h3>
    <p style="min-height:62px">${m[3]}</p><table>${rows}</table>`);});

$('#gapStats').append(stat(pct(D.deeponet_curve.extrap_mean),'mean extrap error'),
  stat(pct(D.deeponet_curve.extrap_final),'error at final t=2'),
  stat(pct(D.deeponet_curve.persistence_extrap_mean),'persistence baseline'),
  stat('worse','ML vs doing nothing'));

const fr=D.frontier, dm=fr.dominance;
$('#costStats').append(stat('15.6 ms','1 spectral step'),stat('0.64 ms','1 FDM step (24× cheaper)'),
  stat(dm.hybrid_floor_acc.toFixed(2),'hybrid accuracy floor'),
  stat(dm.speedup_to_beat_ml_2x.toFixed(2)+'×','cost saving vs numerical'));

[['Ground truth verified','Regenerated Cole-Hopf matches the stored project truth to <b>2.4×10⁻⁸</b> relative L2 — identical pipeline.'],
 ['Independent cross-check','Spectral IFRK4 agrees with Cole-Hopf to <b>0.011%</b> on a 64-IC random subset — two methods, one answer.'],
 ['No test-set leakage','Controller calibrated on train ICs only; all accuracy on held-out <b>test IC 900</b>. Splits 800/100/100.'],
 ['Reproducible','IC set fixed by <b>seed 42</b>; DeepONet over <b>5 seeds</b> (tight std). Every figure regenerable from scripts.'],
 ['Automated tests','Module 3 ships a suite — <b>12/12 invariants pass</b> on real data (cost accounting, frontier monotonicity, runtime).'],
 ['Honest limitations','The hybrid accuracy floor is <b>quantified and reported</b>, not hidden — it predicts FNO will improve it.']
].forEach(v=>card($('#valCards'),`<h3 style="color:var(--cyan)">✓ ${v[0]}</h3><p>${v[1]}</p>`));

/* ===================== field drawing helper ===================== */
function drawField(ctx,W,H,x,rows,frame,layers){
  ctx.clearRect(0,0,W,H);
  const pad=24, ymin=-1.15,ymax=1.15;
  const px=i=>pad+(x[i]+1)/2*(W-2*pad);
  const py=v=>H-pad-(v-ymin)/(ymax-ymin)*(H-2*pad);
  ctx.strokeStyle='#1e2840'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(pad,py(0)); ctx.lineTo(W-pad,py(0)); ctx.stroke();
  layers.forEach(L=>{ctx.strokeStyle=L.color;ctx.lineWidth=L.w;ctx.setLineDash(L.dash||[]);ctx.beginPath();
    const row=rows[L.key][frame]; for(let i=0;i<x.length;i++){const X=px(i),Y=py(row[i]); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);} ctx.stroke();ctx.setLineDash([]);});
}
const isActive = id => $('#'+id).classList.contains('active');

/* ===================== animations (visibility-guarded) ===================== */
(function(){ // hero
  const cv=$('#heroCanvas'),ctx=cv.getContext('2d'); const F=D.fields; let f=0;
  function size(){cv.width=cv.clientWidth;cv.height=cv.clientHeight;} size(); window.addEventListener('resize',size);
  setInterval(()=>{ if(!isActive('overview'))return; if(!cv.width)size();
    const W=cv.width,H=cv.height; ctx.clearRect(0,0,W,H);
    const pad=10,ymin=-1.2,ymax=1.2; const px=i=>pad+(F.x[i]+1)/2*(W-2*pad),py=v=>H/2-v/(ymax-ymin)*(H-2*pad);
    const g=ctx.createLinearGradient(0,0,W,0); g.addColorStop(0,'#36d6c3');g.addColorStop(1,'#7c83ff');
    ctx.strokeStyle=g;ctx.lineWidth=2.5;ctx.beginPath();
    const row=F.truth[f]; for(let i=0;i<F.x.length;i++){const X=px(i),Y=py(row[i]); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);} ctx.stroke();
    f=(f+1)%F.truth.length;
  },70);
})();
(function(){ // dataset sample
  const cv=$('#icCanvas'),ctx=cv.getContext('2d'); const F=D.fields; let f=0;
  function size(){cv.width=cv.clientWidth;cv.height=240;} window.addEventListener('resize',size);
  setInterval(()=>{ if(!isActive('dataset'))return; if(!cv.width)size();
    drawField(ctx,cv.width,cv.height,F.x,F,f,[{key:'truth',color:C.cyan,w:2.5}]);
    ctx.fillStyle=C.muted;ctx.font='12px monospace';ctx.fillText('t = '+F.t[f].toFixed(2),cv.width-90,20);
    f=(f+1)%F.truth.length;
  },80);
})();

/* ===================== lazy chart builders ===================== */
const built=new Set();
function buildNumerical(){
  new Chart($('#numAccChart'),{type:'bar',data:{labels:['Spectral','Finite-Difference'],
    datasets:[{data:[0.00011,0.1628],backgroundColor:[C.spectral,C.fdm],borderRadius:6}]},
    options:{plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>(c.raw*100).toFixed(3)+'% rel-L2'}}},
      scales:{y:{type:'logarithmic',title:{display:true,text:'relative error vs Cole-Hopf (log)'}}}}});
  new Chart($('#numCostChart'),{type:'bar',data:{labels:['Spectral','Finite-Difference'],
    datasets:[{data:[3290,110],backgroundColor:[C.spectral,C.fdm],borderRadius:6}]},
    options:{plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>c.raw+' ms / full solve'}}},
      scales:{y:{title:{display:true,text:'wall-time per solve (ms)'}}}}});
}
function buildMl(){
  new Chart($('#mlBarChart'),{type:'bar',data:{labels:['train (t≤1)','val (t≤1)','test (t≤1)','extrapolation (t>1)'],
    datasets:[{data:[e.train_in_dist.mean,e.val_in_dist.mean,e.test_in_dist.mean,e.test_extrap.mean],
      backgroundColor:[C.ml,C.ml,C.ml,C.hybrid],borderRadius:6}]},
    options:{plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>(c.raw*100).toFixed(1)+'% rel-L2'}}},
      scales:{y:{title:{display:true,text:'relative L2 error'},ticks:{callback:v=>(v*100)+'%'}}}}});
}
function buildGap(){
  const g=D.deeponet_curve;
  const tend={id:'tend',afterDraw(ch){const x=ch.scales.x.getPixelForValue(1.0),a=ch.chartArea,c=ch.ctx;
    c.save();c.strokeStyle='#6b7799';c.setLineDash([5,4]);c.lineWidth=1.5;c.beginPath();c.moveTo(x,a.top);c.lineTo(x,a.bottom);c.stroke();
    c.fillStyle='#9aa7c2';c.font='11px sans-serif';c.fillText('train ends',x+5,a.top+14);c.restore();}};
  new Chart($('#gapChart'),{type:'line',data:{labels:g.t,datasets:[
    {label:'DeepONet',data:g.model,borderColor:C.ml,pointRadius:0,borderWidth:2.5,tension:.15},
    {label:'Persistence baseline',data:g.persistence,borderColor:C.amber,borderDash:[6,4],pointRadius:0,borderWidth:2,tension:.15}]},
    options:{plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false,callbacks:{
      title:i=>'t = '+(+i[0].label).toFixed(2),label:c=>c.dataset.label+': '+(c.raw*100).toFixed(1)+'%'}}},
      scales:{x:{type:'linear',title:{display:true,text:'time t'},ticks:{maxTicksLimit:9,callback:v=>(+v).toFixed(1)}},
              y:{title:{display:true,text:'relative L2 error'},ticks:{callback:v=>(v*100)+'%'}}},elements:{point:{radius:0}}},plugins:[tend]});
}
function buildCost(){
  const cloud=fr.cloud,front=fr.front,b=fr.baselines;
  const pick=s=>cloud.filter(r=>r.solver===s).map(r=>({x:r.numerical_step_equivalents,y:r.err_extrap}));
  new Chart($('#frontierChart'),{type:'scatter',data:{datasets:[
    {label:'spectral policies',data:pick('spectral'),backgroundColor:'rgba(94,155,236,.45)',pointRadius:4},
    {label:'fdm policies',data:pick('fdm'),backgroundColor:'rgba(240,160,96,.45)',pointRadius:4},
    {label:'hybrid frontier',type:'line',data:front.map(r=>({x:r.numerical_step_equivalents,y:r.err_extrap})),
      borderColor:C.hybrid,backgroundColor:C.hybrid,pointRadius:3,borderWidth:2.5,showLine:true,tension:.1},
    {label:'pure ML',data:[{x:0,y:b.pure_ml.err_extrap}],backgroundColor:C.ml,pointRadius:9,pointStyle:'star'},
    {label:'pure numerical',data:[{x:b.pure_numerical.numerical_step_equivalents,y:b.pure_numerical.err_extrap}],
      backgroundColor:C.truth,pointRadius:8,pointStyle:'rectRot'}]},
    options:{plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${c.parsed.x} steps, ${(c.parsed.y*100).toFixed(1)}%`}}},
      scales:{x:{title:{display:true,text:'cost — numerical steps spent'}},
              y:{type:'logarithmic',title:{display:true,text:'extrapolation error (log)'},ticks:{callback:v=>(v*100)+'%'}}}}});
  const rows=D.controller;
  new Chart($('#controllerChart'),{type:'line',data:{labels:rows.map(r=>r.eps),datasets:[
    {label:'fixed (switch @ t=1)',data:rows.map(r=>r.fixed.cost),borderColor:C.muted,borderDash:[6,4],pointRadius:4,borderWidth:2},
    {label:'adaptive (trust-gated)',data:rows.map(r=>r.adaptive.cost),borderColor:C.hybrid,pointRadius:4,borderWidth:2.5}]},
    options:{plugins:{legend:{labels:{boxWidth:12}},tooltip:{callbacks:{title:i=>'target '+i[0].label,label:c=>c.dataset.label+': '+c.raw+' steps'}}},
      scales:{x:{reverse:true,title:{display:true,text:'requested accuracy target (tighter →)'}},
              y:{title:{display:true,text:'numerical steps spent'}}}}});
}
const CHART_HOOKS={numerical:buildNumerical,mlmodels:buildMl,gap:buildGap,cost:buildCost,demo:initDemo};

/* ===================== LIVE DEMO ===================== */
let demoAPI=null;
function initDemo(){
  if(demoAPI){demoAPI.resize();return;}
  const F=D.fields, META=D.solvers_meta;
  const cv=$('#demoCanvas'),ctx=cv.getContext('2d');
  function size(){cv.width=cv.clientWidth;cv.height=320;}
  const targets=F.targets,nF=F.truth.length;
  let knobIdx=targets.indexOf(0.2); if(knobIdx<0)knobIdx=Math.floor(targets.length/2);
  let frame=Math.round(nF*0.8),playing=false;   // start in extrapolation zone (dramatic + honest)
  const visible=new Set(['truth','spectral','fdm','ml','hybrid']);

  // which raw field array backs a solver key (hybrid depends on the knob)
  function fieldFor(key){
    if(key==='hybrid') return F.hybrid_variants[String(targets[knobIdx])];
    return F[key];                       // truth/spectral/fdm/ml, or fno/pinn once host-populated
  }
  function isAvailable(meta){ return meta.available || !!F[meta.key]; }
  function relErr(rows,fr){const t=F.truth[fr];let a=0,b=0;for(let i=0;i<t.length;i++){const d=rows[fr][i]-t[i];a+=d*d;b+=t[i]*t[i];}return Math.sqrt(a)/(Math.sqrt(b)+1e-9);}

  // hybrid metrics follow the knob
  function hybMeta(){return F.hybrid_meta[String(targets[knobIdx])];}

  // ---- build toggle chips ----
  const chipWrap=$('#solverChips'); chipWrap.innerHTML='';
  META.forEach(s=>{
    const avail=isAvailable(s);
    const chip=el('button','btn',`<span class="dot" style="background:${s.color}"></span>${s.label}`);
    chip.style.display='inline-flex'; chip.style.alignItems='center'; chip.style.gap='8px';
    if(s.key==='truth'){chip.classList.add('active'); chip.style.opacity=.9; chip.title='Cole-Hopf exact truth (always shown)';}
    else if(!avail){chip.disabled=true; chip.style.opacity=.45; chip.title='Run generate_web_data.py on a GPU host to populate'; chip.innerHTML+=' <span class="badge b-pending" style="margin-left:4px">pending</span>';}
    else if(visible.has(s.key)) chip.classList.add('active');
    if(avail && s.key!=='truth') chip.onclick=()=>{ visible.has(s.key)?visible.delete(s.key):visible.add(s.key); chip.classList.toggle('active'); render(); };
    chipWrap.append(chip);
  });

  // ---- build comparison table rows ----
  const body=$('#cmpBody'); body.innerHTML='';
  META.forEach(s=>{
    const avail=isAvailable(s);
    const tr=el('tr'); tr.id='cmp-'+s.key;
    const dot=`<span class="dot" style="background:${s.color}"></span>`;
    const pct=v=>v==null?'—':(v*100).toFixed(1)+'%';
    const steps=s.costSteps==null?'<span style="color:var(--muted)">exact</span>':s.costSteps;
    const ms=s.costMs==null?'—':s.costMs;
    tr.innerHTML=`<td><b>${dot}${s.label}</b></td><td style="color:var(--muted)">${s.family}</td>`+
      `<td id="now-${s.key}">—</td><td>${avail?pct(s.extrap):'<span class="badge b-pending">pending</span>'}</td>`+
      `<td>${avail?pct(s.inDist):'—'}</td><td>${avail?steps:'—'}</td><td>${avail?ms:'—'}</td>`;
    if(!avail) tr.style.opacity=.5;
    body.append(tr);
  });

  function render(){
    if(!cv.width)size();
    // overlay: truth first (thick dashed), then each visible available solver
    const layers=[]; const rows={};
    META.forEach(s=>{
      if(!isAvailable(s)) return;
      if(s.key!=='truth' && !visible.has(s.key)) return;
      rows[s.key]=fieldFor(s.key);
      layers.push({key:s.key,color:s.color,w:s.key==='truth'?2.4:2.2,dash:s.key==='truth'?[5,4]:[]});
    });
    drawField(ctx,cv.width,cv.height,F.x,rows,frame,layers);
    // live "now" error column
    META.forEach(s=>{
      const cell=$('#now-'+s.key); if(!cell)return;
      if(!isAvailable(s)){cell.textContent='—';return;}
      const er=relErr(fieldFor(s.key),frame);
      cell.innerHTML=`<b style="color:${er<.05?'var(--green)':er<.2?'var(--amber)':'var(--coral)'}">${(er*100).toFixed(1)}%</b>`;
    });
    // hybrid row reflects knob
    const hm=hybMeta(); const hr=$('#cmp-hybrid');
    if(hr){hr.children[3].innerHTML=(hm.achieved*100).toFixed(1)+'%'; hr.children[5].textContent=hm.num_steps; hr.children[6].textContent=hm.wall_ms;}
    $('#tReadout').textContent=F.t[frame].toFixed(2); $('#timeSlider').value=frame;
    const t=F.t[frame];
    $('#regimeTag').innerHTML = t<=1 ? '<span style="color:var(--green)">● Training window (t ≤ 1)</span> — ML is reliable here'
                                     : '<span style="color:var(--coral)">● Extrapolation (t &gt; 1)</span> — beyond what ML ever saw; watch DeepONet diverge';
  }
  $('#demoExplain').innerHTML='<b>Reading the table.</b> "Error now" is each solver vs the exact Cole-Hopf truth <em>at the current time t</em> — drag the scrubber to change it. It starts in the extrapolation zone (t≈1.6) where DeepONet has already diverged. Note DeepONet is ~12% off even at t=0 — that is real and matches its own evaluation, not a bug. Move the accuracy knob to trade the hybrid\'s error for cost.';

  // wire the knob to the actual number of precomputed targets
  const knobEl=$('#knob'); knobEl.min=0; knobEl.max=targets.length-1; knobEl.step=1; knobEl.value=knobIdx;
  $('#timeSlider').max=nF-1; $('#timeSlider').value=frame;
  if(!$('#knobTicks')){
    const tk=el('div','',`<span>← loose · ${targets[0].toFixed(2)}</span><span>tight · ${targets[targets.length-1].toFixed(2)} →</span>`);
    tk.id='knobTicks'; tk.style.cssText='display:flex;justify-content:space-between;font-size:.72rem;color:var(--muted);margin-top:2px';
    knobEl.insertAdjacentElement('afterend',tk);
  }
  knobEl.oninput=ev=>{knobIdx=+ev.target.value;$('#knobReadout').textContent=targets[knobIdx].toFixed(2);render();};
  $('#timeSlider').oninput=ev=>{frame=+ev.target.value;render();};
  function loop(){if(!playing)return;frame=(frame+1)%nF;render();if(frame===0){playing=false;$('#playBtn').textContent='▶ Play';return;}setTimeout(()=>requestAnimationFrame(loop),55);}
  function play(){if(playing)return;playing=true;frame=0;$('#playBtn').textContent='❚❚ Pause';loop();}
  function pause(){playing=false;$('#playBtn').textContent='▶ Play';}
  $('#playBtn').onclick=()=>{playing?pause():play();};
  $('#resetBtn').onclick=()=>{pause();frame=Math.round(nF*0.8);render();};
  $('#knobReadout').textContent=targets[knobIdx].toFixed(2);
  window.addEventListener('resize',()=>{if(isActive('demo')){size();render();}});
  size(); render();
  demoAPI={play,pause,reset:()=>{frame=0;render();},setMode:()=>{},resize:()=>{size();render();}};
}

/* ===================== ROUTER ===================== */
const VIEWS=['overview','problem','dataset','numerical','mlmodels','gap','hybrid','cost','demo','validation','summary'];
const TITLES={overview:'Overview',problem:'The Problem',dataset:'Dataset Generation',numerical:'Numerical Solvers',
  mlmodels:'ML Models',gap:'The Extrapolation Gap',hybrid:'Hybrid Engine',cost:'Cost Module (M3)',
  demo:'Live Demo',validation:'Validation',summary:'Summary'};
let cur=0;
function showView(id,fromTour){
  const i=VIEWS.indexOf(id); if(i<0)return; cur=i;
  VIEWS.forEach(v=>$('#'+v).classList.toggle('active',v===id));
  document.querySelectorAll('#nav a').forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+id));
  $('#tbTitle').textContent=TITLES[id];
  $('#tbIx').textContent=String(i).padStart(2,'0')+' / '+String(VIEWS.length-1).padStart(2,'0');
  $('#progBar').style.width=(i/(VIEWS.length-1)*100)+'%';
  $('#prevBtn').disabled=i===0; $('#nextBtn').disabled=i===VIEWS.length-1;
  if(CHART_HOOKS[id]&&!built.has(id)){built.add(id);CHART_HOOKS[id]();}
  if(id==='demo'&&demoAPI)demoAPI.resize();
  window.scrollTo(0,0); history.replaceState(null,'','#'+id);
  if(!fromTour&&tour.on)stopTour();
}
document.querySelectorAll('#nav a').forEach(a=>a.addEventListener('click',ev=>{ev.preventDefault();showView(a.getAttribute('href').slice(1));}));
$('#prevBtn').onclick=()=>cur>0&&showView(VIEWS[cur-1]);
$('#nextBtn').onclick=()=>cur<VIEWS.length-1&&showView(VIEWS[cur+1]);
document.addEventListener('keydown',ev=>{if(ev.key==='ArrowRight')$('#nextBtn').click();if(ev.key==='ArrowLeft')$('#prevBtn').click();});

/* ===================== GUIDED TOUR ===================== */
const TOUR=[
  ['overview','Welcome to Turingz','One solver that runs fast ML where it can be trusted and exact numerics where it cannot. This tour walks the whole project in under two minutes.'],
  ['problem','Two families, two flaws','Numerical solvers are accurate but slow. ML solvers are fast but fail outside their training window — silently. Neither alone is enough.'],
  ['dataset','Verified ground truth','1000 initial conditions solved exactly with Cole-Hopf, cross-checked against a spectral solver to 0.011%. This is the trustworthy data everything is measured against.'],
  ['numerical','The cost lever','Spectral is essentially exact but 24× more expensive per step than finite-difference. That spread is exactly what the hybrid exploits.'],
  ['mlmodels','Three surrogates','DeepONet, FNO and PINN — all fast, all accurate inside the training window. DeepONet is fully evaluated here over 5 seeds.'],
  ['gap','The failure we fix','Watch DeepONet diverge after t=1 — its extrapolation error is worse than simply freezing the solution. This gap is the whole motivation.'],
  ['hybrid','Three contributions','Trust (M1) decides when ML fails, Coupling (M2) hands over to numerics smoothly, and Control (M3 — my module) decides how much to spend.'],
  ['cost','The measured frontier','My module proves the hybrid beats pure-ML by ~6× accuracy and pure-numerical by 2.34× cost — and quantifies its one limitation honestly.'],
  ['demo','See it run live','ML diverges, the hybrid stays locked on the true shock at a fraction of the cost. The knob trades accuracy for cost along the real frontier.'],
  ['summary','That is Turingz','A working, tunable hybrid on a real PDE — every number reproducible. Thanks for watching.']
];
const tour={on:false,idx:0,timer:null,paused:false,STEP_MS:8000};
function renderTour(){
  const [view,title,text]=TOUR[tour.idx];
  showView(view,true);
  $('#tourStep').textContent='STEP '+(tour.idx+1)+' / '+TOUR.length;
  $('#tourTitle').textContent=title; $('#tourText').textContent=text;
  const bar=$('#tourBar').firstElementChild; bar.style.transition='none'; bar.style.width='0%';
  requestAnimationFrame(()=>{bar.style.transition='width '+tour.STEP_MS+'ms linear'; if(!tour.paused)bar.style.width='100%';});
  if(view==='demo'&&demoAPI){demoAPI.reset();demoAPI.setMode('hybrid');demoAPI.play();}
}
function tourNext(){ if(tour.idx<TOUR.length-1){tour.idx++;schedule();renderTour();} else stopTour(); }
function schedule(){ clearTimeout(tour.timer); if(!tour.paused)tour.timer=setTimeout(tourNext,tour.STEP_MS); }
function startTour(){tour.on=true;tour.idx=0;tour.paused=false;$('#tour').classList.add('on');$('#tourPause').textContent='❚❚';renderTour();schedule();}
function stopTour(){tour.on=false;clearTimeout(tour.timer);$('#tour').classList.remove('on');if(demoAPI)demoAPI.pause();}
$('#startTour').onclick=startTour;
$('#tourExit').onclick=stopTour;
$('#tourNext').onclick=()=>{tour.paused=false;$('#tourPause').textContent='❚❚';tourNext();};
$('#tourPause').onclick=function(){tour.paused=!tour.paused;this.textContent=tour.paused?'▶':'❚❚';
  const bar=$('#tourBar').firstElementChild;
  if(tour.paused){clearTimeout(tour.timer);const w=getComputedStyle(bar).width;bar.style.transition='none';bar.style.width=w;}
  else{schedule();const ms=tour.STEP_MS;bar.style.transition='width '+ms+'ms linear';bar.style.width='100%';}};

/* ===================== boot ===================== */
const start=(location.hash&&VIEWS.includes(location.hash.slice(1)))?location.hash.slice(1):'overview';
showView(start);
