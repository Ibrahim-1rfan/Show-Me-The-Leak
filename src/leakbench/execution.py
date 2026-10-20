"""Visible setup and subprocess supervision with bounded waits."""
import os, queue, subprocess, sys, threading, time, signal
from pathlib import Path
from .common import clean, log, read_json, stage
from .environment import install_commands, probe_environment, PROFILES

def run_command(command, label, timeout=3600, log_path=None, state_path=None, image_timeout=600, stage_timeouts=None):
    log(label+' starting')
    process=subprocess.Popen([str(x) for x in command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,encoding='utf-8',errors='replace',bufsize=1,
        env={**os.environ,'PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false'},
        start_new_session=os.name!='nt')
    lines=queue.Queue();reader_done=threading.Event()
    def reader():
        try:
            for line in process.stdout: lines.put(clean(line.rstrip()))
        finally:reader_done.set()
    reader_thread=threading.Thread(target=reader,daemon=True);reader_thread.start()
    started=last=time.monotonic()
    try:
        while process.poll() is None or not reader_done.is_set() or not lines.empty():
            try:
                line=lines.get(timeout=.25);print(line,flush=True)
                if log_path:
                    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
                    with Path(log_path).open('a',encoding='utf-8') as f:f.write(line+'\n')
            except queue.Empty:pass
            elapsed=time.monotonic()-started
            if process.poll() is None and elapsed>timeout:raise TimeoutError(label+' exceeded total stage timeout')
            state=None
            if state_path and Path(state_path).exists():
                try:state=read_json(state_path)
                except (ValueError,OSError):pass
            if state and state.get('stage')=='image' and state.get('elapsed_seconds',0)>image_timeout:
                raise TimeoutError(f"Image {state.get('image_id')} exceeded {image_timeout}s; partial outputs retained")
            if state and state.get('stage') in (stage_timeouts or {}):
                limit=stage_timeouts[state['stage']]
                if state.get('elapsed_seconds',0)>limit:
                    raise TimeoutError(f"{state['stage']} exceeded {limit}s; inspect the stage log before retrying")
            if time.monotonic()-last>=10:
                log(label+' running',elapsed_seconds=round(elapsed),state=state);last=time.monotonic()
        if process.returncode:raise RuntimeError(f'{label} failed with exit code {process.returncode}; see captured stage log')
    finally:
        if process.poll() is None:
            if os.name!='nt':os.killpg(process.pid,signal.SIGTERM)
            else:process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name!='nt':os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
                process.wait()
        reader_thread.join(timeout=2)
        process.stdout.close()
    log(label+' completed',elapsed_seconds=round(time.monotonic()-started,2))

def setup_profile(profile,env_root,log_dir):
    if sys.platform!='linux':raise RuntimeError('Model execution requires a Linux Colab GPU runtime.')
    python,constraints,commands=install_commands(profile,env_root)
    if python.exists():
        try:
            with stage('Verify existing '+profile+' environment'):
                metadata=probe_environment(python,profile)
            log('Reusing verified environment',profile=profile,environment=metadata)
            return python
        except RuntimeError as exc:log('Environment needs repair',reason=clean(exc))
    logfile=Path(log_dir)/f'setup_{profile}.log'
    run_command([sys.executable,'-m','pip','install','uv==0.8.22'],'Install environment manager',600,logfile)
    for i,command in enumerate(commands):
        if i==2:
            spec=PROFILES[profile]
            constraints.write_text(f"torch=={spec['torch']}\ntorchvision=={spec['torchvision']}\n"+'\n'.join(x for x in spec['packages'] if '==' in x)+'\n')
        run_command(command,f'{profile} setup stage {i+1}/{len(commands)}',3600,logfile)
    metadata=probe_environment(python,profile);log('GPU environment verified',**metadata)
    return python
