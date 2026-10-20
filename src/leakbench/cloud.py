"""Resume checksum-verified copies to the existing mounted Google Drive root."""
import os, shutil
from pathlib import Path
from .common import sha, log, write_json
DRIVE_ROOT=Path('/content/drive/MyDrive/Leak_Benchmark_Results')

def verified_copy_tree(source,destination):
    from tqdm.auto import tqdm
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source==destination or source in destination.parents or destination in source.parents:raise ValueError('Backup paths must be separate non-nested folders')
    if not source.is_dir():raise FileNotFoundError('Backup source does not exist: '+str(source))
    files=[p for p in sorted(source.rglob('*')) if p.is_file() and not p.name.endswith(('.tmp','.uploading'))
           and p.name!='upload_verification.json'
           and '__pycache__' not in p.parts]
    checks=[]
    for path in tqdm(files,desc='Cloud copy and verify',unit='file'):
        relative=path.relative_to(source);target=destination/relative;expected=sha(path)
        if target.exists() and sha(target)==expected:
            checks.append({'path':relative.as_posix(),'sha256':expected,'status':'already_verified'});continue
        target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.uploading')
        shutil.copyfile(path,temp)
        if sha(temp)!=expected:raise IOError('Backup checksum mismatch: '+str(relative))
        os.replace(temp,target)
        if sha(target)!=expected:raise IOError('Backup verification failed: '+str(relative))
        checks.append({'path':relative.as_posix(),'sha256':expected,'status':'copied_and_verified'})
    write_json(destination/'upload_verification.json',{'files':checks,'verified_count':len(checks)})
    log('Drive filesystem backup verified',files=len(checks),destination=str(destination),
        verification_scope='Read-back hashes through mounted Drive; remote server durability is not independently queried')
    return checks

def backup(project,category,run_id):
    project=Path(project)
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Mount Google Drive with drive.mount before backup.')
    verified_copy_tree(project/'data/vispr_eval_50',DRIVE_ROOT/'benchmark_v2/data/vispr_eval_50')
    return verified_copy_tree(project/'results'/category/run_id,DRIVE_ROOT/'benchmark_v2/results'/category/run_id)
