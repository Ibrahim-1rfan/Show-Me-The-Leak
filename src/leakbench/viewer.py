"""Colab results browser backed by selected run files; no model execution."""
import html,json,io
from pathlib import Path
from .common import read_json
from .dataset import records
from .progress import active_execution

def discover_runs(project,category=None):
    project=Path(project);bases=[project/'results']
    bases.extend(p for pattern in ('vlm_*','ocr_*','results_ocr') for p in project.glob(pattern) if p.is_dir())
    drive=Path('/content/drive/MyDrive/Leak_Benchmark_Results/benchmark_v2/results')
    if drive.is_dir():bases.append(drive)
    found={}
    for base in bases:
        for p in base.rglob('run_config.json'):
            if 'source' in p.parts or 'attempts' in p.parts:continue
            try:config=read_json(p)
            except (ValueError,OSError):continue
            if category and config.get('category')!=category:continue
            if not config.get('settings',{}).get('models'):continue
            found[str(p.parent.resolve())]=config
    return sorted(found.items(),key=lambda item:Path(item[0]).name,reverse=True)

def image_details(project,run,model,record):
    from .worker import output_dir,valid_completed
    run=Path(run);config=read_json(run/'run_config.json')
    if model not in config['settings']['models']:raise ValueError('Model not in selected run')
    folder=output_dir(run,model,record);_,execution=active_execution(run)
    intended=(execution or config).get('intended_image_ids')
    prediction={};error=None
    if (folder/'prediction.json').exists():
        try:prediction=read_json(folder/'prediction.json')
        except (ValueError,OSError) as exc:error=str(exc)
    failures=[]
    for path in (run/model).glob('*failure.json'):
        try:failures.append({'file':path.name,**read_json(path)})
        except (ValueError,OSError):failures.append({'file':path.name,'error':'Unreadable failure record'})
    original=next(iter(sorted(folder.glob('original.*'))),None)
    source='saved original'
    if original is None and Path(record['image_path']).exists():original=Path(record['image_path']);source='dataset preview (not a processed output)'
    raw=None
    if (folder/'raw_predictions.json').exists():
        try:raw=read_json(folder/'raw_predictions.json')
        except (ValueError,OSError) as exc:raw={'error':str(exc)}
    return {'run_id':run.name,'model_key':model,'image_id':record['id'],'filename':record['filename'],
        'category':record['category'],'intended':None if intended is None else record['id'] in intended,
        'folder':str(folder),'prediction':prediction,'raw':raw,'read_error':error,'model_failures':failures,
        'intact_complete_prediction':valid_completed(folder,config['protocol_id'],record,model),
        'artifacts':{'Original':original,'Predicted regions':folder/'predicted.png','Redacted':folder/'redacted.png'},
        'original_source':source,'mask_path':folder/'redaction_mask.png'}

def show_results_viewer(project,category=None,initial_run=None):
    import ipywidgets as w
    from IPython.display import display,clear_output
    from PIL import Image
    project=Path(project);cohort=records(project/'data/vispr_eval_50')
    runs=discover_runs(project,category)
    if initial_run and (Path(initial_run)/'run_config.json').exists() and str(Path(initial_run).resolve()) not in dict(runs):
        runs.insert(0,(str(Path(initial_run).resolve()),read_json(Path(initial_run)/'run_config.json')))
    if not runs:
        display(w.HTML('No saved runs found. Run configuration/execution first, or mount Drive and restore a run.'))
        return None
    options=[(f'{Path(path).name} | {cfg.get("category")} | {cfg.get("mode","legacy: see run notes")} | '+
              ('Drive' if '/drive/' in path.replace('\\','/') else 'local'),path) for path,cfg in runs]
    run_select=w.Dropdown(options=options,description='Run:',layout=w.Layout(width='95%'))
    if initial_run and str(Path(initial_run).resolve()) in dict(runs):run_select.value=str(Path(initial_run).resolve())
    model_select=w.Dropdown(description='Model:');kind=w.Dropdown(options=[('All images','all'),('Documents','documents'),('General','images')],description='Type:')
    image_select=w.Dropdown(description='Image:',layout=w.Layout(width='95%'))
    previous=w.Button(description='Previous');next_button=w.Button(description='Next')
    jump=w.BoundedIntText(value=1,min=1,max=50,description='Jump to #')
    panel=w.Output();busy={'value':False};shown=[]
    def render(*_):
        if busy['value'] or not image_select.value or not model_select.value:return
        r=next(r for r in cohort if r['id']==image_select.value)
        detail=image_details(project,run_select.value,model_select.value,r)
        pred=detail['prediction'];validation=pred.get('redaction_validation',{})
        with panel:
            clear_output(wait=True)
            title=f'{r["filename"]} → VISPR {r["id"]} | {model_select.value} | run {detail["run_id"]}'
            display(w.HTML('<h3>'+html.escape(title)+'</h3><p>'+html.escape(detail['folder'])+'</p>'))
            if detail['intended'] is False:message='Not requested in this execution. The original below is a dataset preview.'
            elif not pred:message='No saved prediction. See model errors below; this image may be pending or blocked.'
            else:message=f'Prediction: {pred.get("status")}; intact complete artifacts: {detail["intact_complete_prediction"]}; mask checks: {validation.get("technical_pass",False)}; privacy verified: {validation.get("privacy_verified",False)}. Inspect remaining misses.'
            display(w.HTML('<p>'+html.escape(message)+'</p>'))
            cards=[]
            for label,path in detail['artifacts'].items():
                title=label+' — '+detail['original_source'] if label=='Original' else label
                children=[w.HTML('<b>'+html.escape(title)+'</b>')]
                if path and path.is_file():
                    try:
                        with Image.open(path) as im:
                            im=im.convert('RGB');im.thumbnail((640,540));buf=io.BytesIO();im.save(buf,format='PNG')
                        children.append(w.Image(value=buf.getvalue(),format='png',layout=w.Layout(max_width='100%')))
                    except (OSError,ValueError) as exc:children.append(w.HTML('Unreadable image: '+html.escape(str(exc))))
                else:children.append(w.HTML('Missing artifact. '+html.escape(pred.get('error') or pred.get('redaction_status') or 'No image processing output was saved.')))
                cards.append(w.VBox(children,layout=w.Layout(width='32%',min_width='250px')))
            display(w.Box(cards,layout=w.Layout(display='flex',flex_flow='row wrap')))
            evidence={k:detail[k] for k in ('intended','read_error','model_failures','mask_path')}
            evidence.update(validation=validation,prediction=pred,mask_location=str(detail['mask_path']))
            panes=[w.Textarea(value=json.dumps(value,indent=2,default=str),disabled=True,layout=w.Layout(width='100%',height='300px'))
                   for value in (evidence,detail['raw'])]
            mask=detail['mask_path']
            mask_panel=w.VBox([w.HTML('White pixels are masked; black pixels are retained.'),
                w.Image(value=mask.read_bytes(),format='png',layout=w.Layout(max_width='640px'))]) if mask.is_file() else w.HTML('No saved mask for this image.')
            tabs=w.Tab(children=[*panes,mask_panel]);tabs.set_title(0,'Detections, masks and errors');tabs.set_title(1,'Raw model output');tabs.set_title(2,'Saved mask');display(tabs)
    def update_images(*_):
        busy['value']=True
        shown[:]=[r for r in cohort if kind.value=='all' or r['category']==kind.value]
        image_select.options=[(f'{i}: {r["filename"]} → {r["id"]}',r['id']) for i,r in enumerate(shown,1)]
        image_select.value=shown[0]['id'] if shown else None
        jump.max=max(1,len(shown));jump.value=1
        busy['value']=False;render()
    def update_run(*_):
        busy['value']=True
        model_select.options=read_json(Path(run_select.value)/'run_config.json')['settings']['models']
        model_select.value=model_select.options[0]
        busy['value']=False;update_images()
    def navigate(delta):
        if not shown:return
        i=next(i for i,r in enumerate(shown) if r['id']==image_select.value)
        jump.value=max(1,min(len(shown),i+delta+1));image_select.value=shown[jump.value-1]['id']
    def jump_to(change):
        if not busy['value'] and shown:image_select.value=shown[change['new']-1]['id']
    def sync_index(change):
        if not busy['value'] and shown:
            busy['value']=True;jump.value=next(i for i,r in enumerate(shown,1) if r['id']==change['new']);busy['value']=False
        render()
    run_select.observe(update_run,names='value');kind.observe(update_images,names='value')
    model_select.observe(render,names='value');image_select.observe(sync_index,names='value');jump.observe(jump_to,names='value')
    previous.on_click(lambda _:navigate(-1));next_button.on_click(lambda _:navigate(1))
    display(w.VBox([w.HTML('Select a run and model, then browse. Indices are convenience positions; the original VISPR ID is always shown.'),
        run_select,w.HBox([model_select,kind]),image_select,w.HBox([previous,next_button,jump]),panel]))
    update_run()
    return {'run':run_select,'model':model_select,'category':kind,'image':image_select,
            'previous':previous,'next':next_button,'jump':jump,'output':panel}
