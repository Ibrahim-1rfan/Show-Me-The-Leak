"""Package a reporting-only repair for an already running Colab notebook."""
import base64
import hashlib
import io
from pathlib import Path
import textwrap
import zipfile


def build_recovery(root):
    root=Path(root)
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name in ('metrics.py','reporting.py','orchestrate.py','progress.py'):
            bundle.writestr(name,(root/'src/leakbench'/name).read_bytes())
    payload=base64.b64encode(memory.getvalue()).decode()
    checksum=hashlib.sha256(memory.getvalue()).hexdigest()
    source=textwrap.dedent('''\
        """Upload to the EXISTING Colab session, then run:
        %run -i /content/repair_colab_reporting.py

        Repairs reporting only. No installs, downloads, model loads or inference.
        Uses existing PROJECT/run_dir variables (or the run from the reported log).
        Keeps predictions, run configuration and original source snapshot unchanged.
        REPORT_REPAIR_SAVE_CLOUD=False can disable the final Drive backup explicitly.
        """
        from pathlib import Path
        import base64, hashlib, importlib, io, json, sys, zipfile
        from datetime import datetime, timezone
        PROJECT=Path(globals().get('PROJECT','/content/Show-Me-The-Leak-v2'))
        run_dir=Path(globals().get('run_dir',PROJECT/'results/vlm/20261020T230625225013Z'))
        if not (run_dir/'run_config.json').is_file():
            raise FileNotFoundError('Saved run not found. Use this in the original runtime, or restore your run from Drive and set run_dir. Do not rerun inference to recover reports.')
        if not list(run_dir.glob('*/*/*/prediction.json')):
            raise FileNotFoundError('No saved predictions in run_dir. Select the existing result folder before applying this repair.')
        repair_sources=PROJECT/'src/leakbench'
        repair_names={'metrics.py','reporting.py','orchestrate.py','progress.py'}
        if not all((repair_sources/name).is_file() for name in repair_names-{"progress.py"}):
            raise FileNotFoundError('Existing notebook source is missing. Restore the notebook project before repairing reports.')
        PAYLOAD=__PAYLOAD__
        repair_data=base64.b64decode(PAYLOAD)
        if hashlib.sha256(repair_data).hexdigest()!=__CHECKSUM__:
            raise ValueError('Repair payload checksum mismatch')
        repair_stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        repair_dir=run_dir/'reporting_repairs'/repair_stamp
        repair_dir.mkdir(parents=True)
        repair_record={'repair':'partial_status_reporting_20261021','applied_at':repair_stamp,
            'scope':'reporting and orchestration only; inference predictions, original run_config and source snapshot unchanged',
            'status':'applying','files':{}}
        with zipfile.ZipFile(io.BytesIO(repair_data)) as repair_bundle:
            if set(repair_bundle.namelist())!=repair_names:
                raise ValueError('Unexpected repair files')
            for repair_name in sorted(repair_names):
                repair_target=repair_sources/repair_name
                repair_before=repair_target.read_bytes() if repair_target.exists() else b'';repair_after=repair_bundle.read(repair_name)
                compile(repair_after,str(repair_target),'exec')
                (repair_dir/('before_'+repair_name)).write_bytes(repair_before)
                (repair_dir/('after_'+repair_name)).write_bytes(repair_after)
                repair_record['files'][repair_name]={
                    'before_sha256':hashlib.sha256(repair_before).hexdigest(),
                    'after_sha256':hashlib.sha256(repair_after).hexdigest()}
                repair_target.write_bytes(repair_after)
        (repair_dir/'repair.json').write_text(json.dumps(repair_record,indent=2),encoding='utf-8')
        if str(PROJECT/'src') not in sys.path:sys.path.insert(0,str(PROJECT/'src'))
        importlib.invalidate_caches()
        # Execute verified source directly, avoiding stale bytecode in a live kernel.
        for repair_name in ('progress','metrics','reporting','orchestrate'):
            repair_module=importlib.import_module('leakbench.'+repair_name)
            repair_path=repair_sources/(repair_name+'.py')
            repair_module.__file__=str(repair_path)
            exec(compile(repair_path.read_bytes(),str(repair_path),'exec'),repair_module.__dict__)
        from leakbench.common import read_json,write_json,clean
        from leakbench.reporting import report
        from leakbench.orchestrate import execute_category
        from leakbench.cloud import backup
        repair_config=read_json(run_dir/'run_config.json')
        try:
            report_dir=report(PROJECT,run_dir,repair_config['settings']['models'])
            repair_record['status']='reports_regenerated'
            repair_record['report_dir']=str(report_dir)
            # Keep old misleading orchestration_failure.json files as historical evidence.
            repair_record['historical_reporting_failures']=[p.relative_to(run_dir).as_posix()
                for p in run_dir.glob('*/orchestration_failure.json')
                if read_json(p).get('error')=='status must be ok or error']
            print('Reports rebuilt from existing predictions:',report_dir)
            print('Partial results still need review. No model was rerun.')
        except Exception as repair_error:
            repair_record.update(status='report_failed',error=clean(repair_error))
            raise
        finally:
            write_json(repair_dir/'repair.json',repair_record)
            if globals().get('REPORT_REPAIR_SAVE_CLOUD',True):
                try:backup(PROJECT,repair_config['category'],run_dir.name)
                except Exception as repair_backup_error:
                    repair_record['backup_error']=clean(repair_backup_error)
                    write_json(repair_dir/'repair.json',repair_record)
                    print('Reports are local; Drive backup failed. Retry the backup cell:',clean(repair_backup_error))
        print('Open reports/summary.json or report.html; use the notebook download cell to export this run.')
        ''').replace('__PAYLOAD__',repr(payload)).replace('__CHECKSUM__',repr(checksum))
    target=root/'dist/repair_colab_reporting.py'
    target.parent.mkdir(exist_ok=True)
    target.write_text(source,encoding='utf-8')
    return target


if __name__=='__main__':
    print(build_recovery(Path(__file__).resolve().parents[1]))
