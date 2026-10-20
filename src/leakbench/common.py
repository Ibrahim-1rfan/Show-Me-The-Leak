"""Atomic state, credential-safe progress, and observable stages."""
import hashlib, json, os, re, threading, time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def fingerprint(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)
def clean(text):
    text=str(text)
    for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):
        value=os.environ.get(key)
        if value: text=text.replace(value,'[REDACTED]')
    return re.sub(r'hf_[A-Za-z0-9]{15,}', '[REDACTED]',text)
def log(message, log_path=None, **fields):
    row={'time':now(),'message':clean(message),**fields}
    encoded=clean(json.dumps(row,ensure_ascii=False))
    print(encoded,flush=True)
    if log_path:
        with Path(log_path).open('a',encoding='utf-8') as f:f.write(encoded+'\n')

@contextmanager
def stage(name, state_path=None, **details):
    started=time.monotonic();started_at=now();stop=threading.Event()
    def emit():
        state={'stage':name,'started_at':started_at,'elapsed_seconds':round(time.monotonic()-started,1),**details}
        if state_path:write_json(state_path,state)
        log(name,**state)
    emit()
    def beat():
        while not stop.wait(10):emit()
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    try:yield
    finally:
        stop.set();thread.join(timeout=1)
        log(name+' finished',elapsed_seconds=round(time.monotonic()-started,2),**details)
