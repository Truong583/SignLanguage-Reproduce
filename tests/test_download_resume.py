import io
import os
from pathlib import Path
from types import SimpleNamespace
import urllib.error
import pytest
from scripts import fetch_phoenix as fetch


class Response(io.BytesIO):
    def __init__(self,body,status=200,headers=None,timeout_after_body=False):
        super().__init__(body); self.status=status
        self.headers=headers or {'Content-Length':str(len(body))}
        self.timeout_after_body=timeout_after_body
    def read(self,n=-1):
        if self.timeout_after_body and self.tell()==len(self.getvalue()): raise TimeoutError('read timed out')
        return super().read(n)


@pytest.mark.parametrize('first_timeout',[True,False])
def test_retry_resumes_written_bytes_without_duplicate_payload(tmp_path,monkeypatch,first_timeout):
    target=tmp_path/'archive'; requests=[]; delays=[]
    responses=[Response(b'abc',headers={'Content-Length':'6'},timeout_after_body=first_timeout),
               Response(b'def',206,{'Content-Range':'bytes 3-5/6','Content-Length':'3'})]
    def open_(request,timeout):
        requests.append(request.get_header('Range')); assert timeout==180
        return responses.pop(0)
    monkeypatch.setattr(fetch.urllib.request,'urlopen',open_)
    monkeypatch.setattr(fetch.time,'sleep',delays.append)
    fetch.download_resumable('https://example.invalid/archive',target,attempts=2)
    assert target.read_bytes()==b'abcdef' and not (tmp_path/'archive.part').exists()
    assert requests==[None,'bytes=3-'] and delays==[5]


def test_ignored_range_restarts_instead_of_appending(tmp_path,monkeypatch):
    target=tmp_path/'archive'; (tmp_path/'archive.part').write_bytes(b'old-prefix')
    monkeypatch.setattr(fetch.urllib.request,'urlopen',lambda *a,**k:Response(b'new-body'))
    fetch.download_resumable('https://example.invalid/archive',target)
    assert target.read_bytes()==b'new-body'


def test_misaligned_range_retains_partial_without_appending(tmp_path,monkeypatch):
    target=tmp_path/'archive'; partial=tmp_path/'archive.part'; partial.write_bytes(b'abc')
    monkeypatch.setattr(fetch.urllib.request,'urlopen',lambda *a,**k:Response(b'def',206,{'Content-Range':'bytes 2-4/6'}))
    with pytest.raises(ValueError,match='Invalid Content-Range'):
        fetch.download_resumable('https://example.invalid/archive',target)
    assert partial.read_bytes()==b'abc' and not target.exists()


def test_timeout_retries_are_bounded_and_keep_partial(tmp_path,monkeypatch):
    target=tmp_path/'archive'; partial=tmp_path/'archive.part'; partial.write_bytes(b'abc'); calls=[]
    def fail(request,timeout): calls.append(request); raise TimeoutError('network down')
    monkeypatch.setattr(fetch.urllib.request,'urlopen',fail)
    monkeypatch.setattr(fetch.time,'sleep',lambda seconds:None)
    with pytest.raises(TimeoutError): fetch.download_resumable('https://example.invalid/archive',target,attempts=2)
    assert len(calls)==2 and partial.read_bytes()==b'abc' and not target.exists()


def test_permanent_http_error_does_not_retry(tmp_path,monkeypatch):
    calls=[]
    def fail(request,timeout):
        calls.append(request); raise urllib.error.HTTPError(request.full_url,404,'missing',{},None)
    monkeypatch.setattr(fetch.urllib.request,'urlopen',fail)
    with pytest.raises(urllib.error.HTTPError): fetch.download_resumable('https://example.invalid/archive',tmp_path/'archive')
    assert len(calls)==1


@pytest.mark.parametrize('free_gib,accepted',[(130,True),(110,False),(9,False)])
def test_preparation_budget_credits_existing_partial_and_retains_real_free_guard(tmp_path,monkeypatch,free_gib,accepted):
    archive=tmp_path/'archive'; partial=tmp_path/'archive.part'; partial.write_bytes(b'cached')
    original=Path.stat
    def stat_(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        if path==partial:
            fields=list(value); fields[6]=39*1024**3; return os.stat_result(fields)
        return value
    monkeypatch.setattr(Path,'stat',stat_)
    monkeypatch.setattr(fetch.shutil,'disk_usage',lambda path:SimpleNamespace(free=free_gib*1024**3))
    if accepted: fetch.check_prepare_space(tmp_path/'phoenix14t',archive)
    else:
        with pytest.raises(RuntimeError,match='Not enough preparation space'):
            fetch.check_prepare_space(tmp_path/'phoenix14t',archive)
