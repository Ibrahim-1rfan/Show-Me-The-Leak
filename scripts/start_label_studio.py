"""Start the isolated local annotation server; never creates users or annotations."""
import json,os,socket,subprocess,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    python=ROOT/'.venv-annotation/Scripts/python.exe'
    if not python.exists():raise SystemExit('Install the separate annotation environment first; see GUIDE.md.')
    package=ROOT/'manual_annotation_dataset';data=ROOT/'data/label_studio'
    if not (package/'manifests/master.json').exists():raise SystemExit('Prepared annotation package is missing.')
    data.mkdir(parents=True,exist_ok=True)
    url='http://127.0.0.1:8080'
    with socket.socket() as s:
        busy=s.connect_ex(('127.0.0.1',8080))==0
    if busy:
        if (data/'server.json').exists():
            try:
                with urllib.request.urlopen(url+'/health',timeout=3) as r:
                    if r.status==200:print('Existing local server: '+url);return
            except Exception:pass
        raise SystemExit('Port 8080 is occupied. Inspect the existing service before starting another server.')
    env=os.environ.copy();env.update(LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED='true',LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=str(package),LABEL_STUDIO_BASE_DATA_DIR=str(data),COLLECT_ANALYTICS='false',PYTHONUTF8='1')
    command=[str(python),'-m','label_studio.server','start','--internal-host','127.0.0.1','--port','8080','--data-dir',str(data),'--no-browser']
    with (data/'server.log').open('a',encoding='utf-8') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    (data/'server.json').write_text(json.dumps({'pid':proc.pid,'url':url,'data_directory':str(data)},indent=2)+'\n')
    for _ in range(60):
        if proc.poll() is not None:raise SystemExit('Server exited. Inspect data/label_studio/server.log.')
        try:
            with urllib.request.urlopen(url+'/health',timeout=1) as response:
                if response.status==200:print('Label Studio is ready: '+url);return
        except Exception:pass
        time.sleep(1)
    raise SystemExit('Server is still starting; check its log and reopen '+url+'. Do not reinstall or delete its database.')

if __name__=='__main__':main()
