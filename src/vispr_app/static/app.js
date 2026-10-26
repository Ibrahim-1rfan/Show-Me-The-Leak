'use strict';
const $ = id => document.getElementById(id);
const state = {token:'', attributes:[], settings:null, queue:[], jobs:[], limit:5, selected:null, busy:false, saving:Promise.resolve(), dirty:0};
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function notice(text='') {$('notice').textContent=text;}
async function api(path, options={}) {
  options.headers = {...options.headers, 'X-VISPR-Token':state.token};
  const response = await fetch(path, options);
  if (!response.ok) {let body; try {body=await response.json();} catch {} throw new Error(typeof body?.detail==='string' ? body.detail : `Request failed (${response.status}). Check your settings and local backend.`);}
  return response.json();
}
function currentSettings() {
  return {...state.settings, mode:$('mode').value, base_url:$('endpoint').value, model:$('model').value.trim(), margin:Number($('margin').value)};
}
function save() {
  state.settings=currentSettings();
  const snapshot=JSON.stringify(state.settings); const revision=++state.dirty;
  $('saved').textContent='Saving preferences…';
  state.saving=state.saving.catch(()=>{}).then(async()=>{
    await api('/api/settings',{method:'PUT',headers:{'Content-Type':'application/json'},body:snapshot});
    if(revision===state.dirty) $('saved').textContent='Saved locally · Applies to the next batch';
  });
  state.saving.catch(error=>{notice(error.message);$('saved').textContent='Not saved — retry before processing';});
  return state.saving;
}
function renderAttributes() {
  const enabled=new Set(state.settings.enabled), query=$('search').value.toLowerCase(), filter=$('filter').value;
  $('enabled-count').innerHTML=`${enabled.size} <small>/ 28</small>`;
  const items=state.attributes.filter(a=>(a.name+' '+a.attribute_id).toLowerCase().includes(query) && (filter==='all'||enabled.has(a.attribute_id)===(filter==='enabled')));
  let group='';
  $('attributes').innerHTML=items.map(a=>{
    let heading=''; if(group!==a.category){group=a.category;heading=`<div class="attribute-group">${escapeHTML(group)}</div>`;}
    return `${heading}<div class="attribute"><div title="${escapeHTML(a.reason || a.description)}"><strong>${escapeHTML(a.name)}</strong><code>${escapeHTML(a.attribute_id)}</code></div><div class="switch-wrap"><button class="switch" role="switch" aria-checked="${enabled.has(a.attribute_id)}" aria-label="Redact ${escapeHTML(a.name)}" data-attribute="${escapeHTML(a.attribute_id)}"></button><small>${enabled.has(a.attribute_id)?'ON · Redact':'OFF · Allow'}</small></div></div>`;
  }).join('') || '<p class="muted">No matching attributes.</p>';
}
function updateControls() {
  const running=state.busy||state.jobs.some(j=>['queued','processing'].includes(j.status));
  $('process').disabled=running||!state.queue.length;
  $('files').disabled=running; $('cancel').hidden=!running; $('clear').disabled=running;
  $('queue-count').textContent=`${state.queue.length} / ${state.limit}`;
}
function renderQueue() {
  $('queue').innerHTML=state.queue.map((q,i)=>`<div class="queue-row"><img src="${q.url}" alt="Upload preview"><div><strong>${escapeHTML(q.file.name)}</strong><small>${(q.file.size/1024/1024).toFixed(2)} MB · Queued</small></div><button data-remove="${i}" aria-label="Remove ${escapeHTML(q.file.name)}">×</button></div>`).join('');
  updateControls();
}
async function addFiles(files) {
  if(state.busy||state.jobs.some(j=>['queued','processing'].includes(j.status))) return;
  const errors=[];
  for(const file of files) {
    if(state.queue.length>=state.limit){errors.push(`Choose at most ${state.limit} images per batch.`);break;}
    if(!['image/png','image/jpeg','image/webp'].includes(file.type)){errors.push(`${file.name}: use PNG, JPEG or WebP.`);continue;}
    if(file.size>20*1024*1024){errors.push(`${file.name}: exceeds 20 MB.`);continue;}
    const url=URL.createObjectURL(file);
    try {const image=new Image();image.src=url;await image.decode();if(image.width*image.height>20000000)throw new Error();state.queue.push({file,url});}
    catch {URL.revokeObjectURL(url);errors.push(`${file.name}: unreadable or exceeds 20 megapixels.`);}
  }
  notice(errors.join(' '));renderQueue();$('files').value='';
}
function modeChanged(){const lm=$('mode').value==='lmstudio';$('endpoint-field').hidden=!lm;$('model-label').textContent=lm?'Loaded model ID':'Local checkpoint directory';$('model').placeholder=lm?'Check connection to discover models':'D:\\Models\\Qwen3-VL';}
async function checkModel(){
  $('check').disabled=true;$('connection-label').textContent='Checking…';
  try {await save();const result=await api('/api/models/check',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(currentSettings())});
    $('model-dot').classList.toggle('ready',result.ready);$('connection-label').textContent=result.ready?'Vision model loaded':result.reachable?'Server / files reachable':'Unavailable';
    $('connection-message').textContent=result.message;
    $('models').innerHTML=result.models.filter(m=>m.vision&&m.loaded).map(m=>`<option value="${escapeHTML(m.id)}"></option>`).join('');
  }catch(error){notice(error.message);$('connection-label').textContent='Unavailable';}finally{$('check').disabled=false;}
}
function renderJobs(){
  updateControls();
  const active=state.jobs.filter(j=>['queued','processing'].includes(j.status));
  const done=state.jobs.filter(j=>!['queued','processing'].includes(j.status));
  $('progress').textContent=active.length?`${done.length} / ${state.jobs.length} finished · ${active.find(j=>j.status==='processing')?.stage||'Queued'}${active.some(j=>j.cancel_requested)?' · Cancellation requested':''}`:'';
  $('zip').hidden=!state.jobs.some(j=>j.status==='completed');
  $('empty-results').hidden=!!state.jobs.length;
  $('result-tabs').innerHTML=state.jobs.map(j=>`<button data-job="${j.id}" class="${state.selected===j.id?'selected':''}">${escapeHTML(j.filename)} · ${j.status}</button>`).join('');
  if(!state.selected&&state.jobs.length)state.selected=state.jobs[0].id;
  renderResult();
}
let resultSignature='';
function renderResult(force=false){
  const job=state.jobs.find(j=>j.id===state.selected), signature=JSON.stringify(job);
  if(!force&&signature===resultSignature)return;resultSignature=signature;
  if(!job){$('result-content').innerHTML='';return;}
  if(job.status!=='completed'){
    $('result-content').innerHTML=`<div class="warning"><strong>${escapeHTML(job.stage)}</strong><p>${escapeHTML(job.error||'')}</p>${job.status==='processing'?'<p>Inference runs locally. Exact progress is unavailable. You can request cancellation at the next safe boundary.</p>':''}</div>`;return;
  }
  const groups={};job.detections.forEach(d=>{const g=groups[d.name]??={count:0,redacted:0,invalid:0};g.count++;g.redacted+=d.redacted?1:0;g.invalid+=d.status!=='valid'?1:0;});
  const boxes=job.detections.filter(d=>d.box).map(d=>`<div class="box" title="${escapeHTML(d.name)}" style="left:${100*d.box[0]/job.width}%;top:${100*d.box[1]/job.height}%;width:${100*(d.box[2]-d.box[0])/job.width}%;height:${100*(d.box[3]-d.box[1])/job.height}%"></div>`).join('');
  $('result-content').innerHTML=`<div class="result-meta">${escapeHTML(job.settings.mode)} · ${escapeHTML(job.settings.model)} · ${job.seconds??'—'} s · ${job.width} × ${job.height} px</div><div class="result-toolbar"><label><input id="overlay-toggle" type="checkbox" checked>Prediction boxes</label><button id="compare-toggle" class="compare-toggle">Show output only</button></div><div class="comparison"><div id="original-pane"><div class="pane-label">ORIGINAL · MODEL PREDICTIONS</div><div class="image-pane"><img src="${job.original}" alt="Original image with optional prediction overlays">${boxes}</div></div><div><div class="pane-label">OUTPUT · PIXEL REDACTION</div><div class="image-pane"><img src="${job.output}" alt="Redacted image; review before sharing"></div></div></div>${job.warnings.length?`<div class="warning"><strong>${job.incomplete?'Incomplete localization — review required':'Prediction warnings'}</strong>${job.warnings.map(w=>`<p>${escapeHTML(w)}</p>`).join('')}</div>`:''}<h3>${job.detections.length} predicted instance${job.detections.length===1?'':'s'}</h3>${Object.entries(groups).map(([name,g])=>`<div class="detection"><span>${escapeHTML(name)} · ${g.count}</span><span>${g.redacted} masked · ${g.count-g.redacted-g.invalid} allowed${g.invalid?` · ${g.invalid} unlocalized`:''}</span></div>`).join('')||'<p class="warning">No detections returned. This does not establish that the image is free of PII.</p>'}<details><summary class="small">Inspect individual detections</summary>${job.detections.map(d=>`<div class="detection"><span>${d.id} · ${escapeHTML(d.attribute_id||'Unknown')}<br>Confidence: ${d.confidence===null?'not provided':d.confidence.toFixed(2)}</span><span>${d.status==='invalid'?'Invalid / unlocalized':d.redacted?'Masked':'Allowed'}<br>${d.box?d.box.join(', '):'No valid box'}</span></div>`).join('')}</details><div class="download-row"><span class="small">Mask pixels verified · Human review required</span><a class="button-link" href="${job.output}?download=true">Download PNG ↓</a></div>`;
  $('overlay-toggle').onchange=e=>document.querySelectorAll('.box').forEach(b=>b.hidden=!e.target.checked);
  $('compare-toggle').onclick=e=>{const pane=$('original-pane');pane.hidden=!pane.hidden;pane.parentElement.style.gridTemplateColumns=pane.hidden?'1fr':'';e.target.textContent=pane.hidden?'Show side by side':'Show output only';};
}
async function processImages(){
  if(state.busy)return;state.busy=true;updateControls();notice();
  try{await save();const form=new FormData();state.queue.forEach(q=>form.append('files',q.file));await api('/api/jobs',{method:'POST',body:form});state.queue.forEach(q=>URL.revokeObjectURL(q.url));state.queue=[];renderQueue();await poll();}
  catch(error){notice(error.message);}finally{state.busy=false;updateControls();}
}
async function poll(){const data=await api('/api/state');state.jobs=data.jobs;renderJobs();}
async function init(){
  try{const data=await api('/api/state');Object.assign(state,{token:data.token,attributes:data.attributes,settings:data.settings,jobs:data.jobs,limit:data.batch_limit});
    $('mode').value=state.settings.mode;$('endpoint').value=state.settings.base_url;$('model').value=state.settings.model;$('margin').value=state.settings.margin;
    $('upload-hint').textContent=`PNG, JPG or WebP · Up to ${state.limit} images · 20 MB each`;
    modeChanged();renderAttributes();renderQueue();renderJobs();$('saved').textContent='Preferences loaded · Saved locally';
  }catch(error){notice('Local backend unavailable. Start the application and reload this page.');return;}
  $('files').onchange=e=>addFiles(e.target.files);
  $('dropzone').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('files').click();}};
  for(const name of ['dragover','dragenter'])$('dropzone').addEventListener(name,e=>{e.preventDefault();$('dropzone').classList.add('drag');});
  $('dropzone').ondragleave=()=>$('dropzone').classList.remove('drag');
  $('dropzone').ondrop=e=>{e.preventDefault();$('dropzone').classList.remove('drag');addFiles(e.dataTransfer.files);};
  $('queue').onclick=e=>{const button=e.target.closest('[data-remove]');if(button&&!state.busy){const [removed]=state.queue.splice(Number(button.dataset.remove),1);URL.revokeObjectURL(removed.url);renderQueue();}};
  $('attributes').onclick=e=>{const button=e.target.closest('[data-attribute]');if(!button)return;const enabled=new Set(state.settings.enabled),id=button.dataset.attribute;enabled.has(id)?enabled.delete(id):enabled.add(id);state.settings.enabled=[...enabled];renderAttributes();save();};
  for(const [id,on] of [['enable-all',true],['disable-all',false],['reset',true]])$(id).onclick=()=>{state.settings.enabled=on?state.attributes.map(a=>a.attribute_id):[];renderAttributes();save();};
  $('search').oninput=renderAttributes;$('filter').onchange=renderAttributes;
  for(const id of ['mode','endpoint','model','margin'])$(id).onchange=()=>{modeChanged();$('connection-label').textContent='Configuration changed — check again';$('model-dot').classList.remove('ready');save();};
  $('check').onclick=checkModel;$('process').onclick=processImages;
  $('cancel').onclick=async()=>{try{await api('/api/jobs/cancel',{method:'POST'});await poll();}catch(e){notice(e.message);}};
  $('clear').onclick=async()=>{try{await api('/api/session',{method:'DELETE'});state.selected=null;state.queue.forEach(q=>URL.revokeObjectURL(q.url));state.queue=[];renderQueue();await poll();notice('Session images and outputs removed. Preferences retained.');}catch(e){notice(e.message);}};
  $('result-tabs').onclick=e=>{const b=e.target.closest('[data-job]');if(b){state.selected=b.dataset.job;renderJobs();renderResult(true);}};
  async function tick(){try{await poll();}catch{notice('Connection to the local backend was lost. Check the application terminal; queued images remain in this browser.');}setTimeout(tick,1200);}setTimeout(tick,1200);
}
init();
