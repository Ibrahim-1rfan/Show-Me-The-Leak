"""Bounded, visible retry for transient model-download failures; retain HF cache."""
import time
from .common import log,clean

def snapshot_with_retry(download=None,*,attempts=3,sleep=time.sleep,**kwargs):
    if download is None:
        from huggingface_hub import snapshot_download
        download=snapshot_download
    kwargs.setdefault('max_workers',2)
    for attempt in range(1,attempts+1):
        try:return download(**kwargs)
        except Exception as exc:
            message=clean(exc)
            status=getattr(getattr(exc,'response',None),'status_code',None)
            transient=status in (429,500,502,503,504) or any(t in message.lower() for t in ('429','rate limit','timed out','connection reset','503','502'))
            if not transient or attempt==attempts:raise
            delay=min(60,15*2**(attempt-1))
            log('Transient model-download failure; keeping cache and retrying',attempt=attempt,
                attempts=attempts,delay_seconds=delay,error=message)
            for _ in range(delay//5):sleep(5)
