"""Fetch and safely extract the official PHOENIX14T release when absent."""
from pathlib import Path, PurePosixPath
import os
import shutil
import tarfile
import time
import re
import http.client
import urllib.error
import urllib.request

URL="https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"

class IncompleteDownload(IOError): pass


def download_resumable(url: str, archive: Path, attempts=8) -> None:
    archive=Path(archive)
    if attempts<1: raise ValueError('attempts must be positive')
    for attempt in range(attempts):
        try:
            _download_once(url,archive)
            return
        except (TimeoutError,ConnectionError,urllib.error.URLError,http.client.IncompleteRead,
                http.client.RemoteDisconnected,IncompleteDownload) as exc:
            if isinstance(exc,urllib.error.HTTPError) and exc.code not in {408,429,500,502,503,504}: raise
            if attempt+1==attempts:
                print('Download interrupted after bounded retries. Partial file retained; rerun to resume.',flush=True)
                raise
            delay=min(5*2**attempt,60)
            print(f'Download connection interrupted ({type(exc).__name__}); retry {attempt+2}/{attempts} in {delay}s, keeping partial data.',flush=True)
            time.sleep(delay)


def _download_once(url: str, archive: Path) -> None:
    archive.parent.mkdir(parents=True,exist_ok=True)
    partial=archive.with_name(archive.name+".part")
    offset=partial.stat().st_size if partial.exists() else 0
    request=urllib.request.Request(url,headers={"User-Agent":"MixSignGraph-reproduction/1.0",**({"Range":f"bytes={offset}-"} if offset else {})})
    print(f'Dataset download: starting at {offset/1024**3:.2f} GiB (cached partial).',flush=True)
    try: response=urllib.request.urlopen(request,timeout=180)
    except urllib.error.HTTPError as exc:
        match=re.fullmatch(r'bytes \*/(\d+)',exc.headers.get('Content-Range','')) if exc.code==416 else None
        if match and offset==int(match[1]) and offset>0:
            exc.close(); partial.replace(archive); return
        raise
    with response:
        if response.status==200:
            if offset: print('Server did not accept Range; restarting this archive download.',flush=True)
            offset=0; mode='wb'
            length=response.headers.get('Content-Length')
            total=int(length) if length and length.isdigit() else None
        elif response.status==206:
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
            if not match or int(match[1])!=offset or not offset<=int(match[2])<int(match[3]):
                raise ValueError('Invalid Content-Range; cached partial retained without appending a mismatched response.')
            total=int(match[3]); mode='ab'
        else: raise ValueError(f'Unexpected download HTTP status: {response.status}')
        last_report=offset; last_time=time.monotonic()
        with partial.open(mode) as out:
            while True:
                chunk=response.read(1024*1024)
                if not chunk: break
                out.write(chunk)
                current=out.tell()
                if current-last_report>=1024**3 or time.monotonic()-last_time>=30:
                    size=f"/{total/1024**3:.1f} GiB" if total else ""
                    print(f"Dataset download: {current/1024**3:.1f} GiB{size}",flush=True)
                    last_report=current
                    last_time=time.monotonic()
            out.flush(); os.fsync(out.fileno())
    if total and partial.stat().st_size!=total:
        raise IncompleteDownload(f'Incomplete download: got {partial.stat().st_size} of {total} bytes')
    partial.replace(archive)


def check_prepare_space(data_root,archive,minimum_gib=150):
    data_root=Path(data_root); archive=Path(archive)
    free=shutil.disk_usage(data_root.parent).free
    cached=archive if archive.is_file() else archive.with_name(archive.name+'.part')
    credit=cached.stat().st_size if cached.is_file() and not cached.is_symlink() and cached.stat().st_dev==data_root.parent.stat().st_dev else 0
    if free<10*1024**3 or free+credit<minimum_gib*1024**3:
        raise RuntimeError(f'Not enough preparation space: {free/1024**3:.1f} GiB free, {credit/1024**3:.1f} GiB archive already stored; reserve {minimum_gib} GiB total before preparation.')

def valid_layout(root: Path) -> bool:
    manual=root/"annotations"/"manual"
    frames=root/"features"/"fullFrame-210x260px"
    return (manual.is_dir() and frames.is_dir() and
            all((manual/f"PHOENIX-2014-T.{split}.corpus.csv").is_file()
                and (frames/split).is_dir() for split in ("train","dev","test")))

def safe_extract(source, staging):
    def members():
        count=0
        for member in source:
            name=PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
                raise ValueError(f"Unsupported archive entry: {member.name} (type={member.type!r})")
            count+=1
            if count%100000==0: print(f"Dataset extraction: {count:,} entries",flush=True)
            yield member
    # data_filter rejects device nodes and links escaping the extraction root,
    # including writes through previously extracted symlinks. Internal links are valid.
    source.extractall(staging,members=members(),filter="data")

def ensure_phoenix(data_root: Path, archive: Path) -> Path:
    data_root=Path(data_root).resolve(); archive=Path(archive).resolve()
    if valid_layout(data_root):
        print(f"PHOENIX14T already prepared: {data_root}",flush=True)
        return data_root
    if not archive.is_file():
        print("Downloading official PHOENIX14T archive (~39 GB); this happens only when the cached archive is absent.",flush=True)
        download_resumable(URL,archive)
    if archive.stat().st_size>1024**3:
        free=shutil.disk_usage(data_root.parent).free
        reserve=int(archive.stat().st_size*1.5)+10*1024**3
        if free<reserve:
            raise RuntimeError(f"Not enough scratch space for extraction: {free/1024**3:.1f} GiB free; reserve at least {reserve/1024**3:.1f} GiB beyond the cached archive.")
    staging=data_root.parent/(data_root.name+".extracting")
    if staging.exists(): shutil.rmtree(staging)
    staging.mkdir(parents=True,exist_ok=True)
    print("Extracting PHOENIX14T; this can take a while on the first run.",flush=True)
    with tarfile.open(archive,"r|gz") as source:
        safe_extract(source,staging)
    manuals=list(staging.rglob("PHOENIX-2014-T.train.corpus.csv"))
    if not manuals:
        raise FileNotFoundError("Archive extracted but PHOENIX-2014-T.train.corpus.csv was not found")
    release_root=manuals[0].parent.parent.parent
    expected=release_root/"features"/"fullFrame-210x260px"
    if not expected.is_dir():
        raise FileNotFoundError(f"Missing frames directory in archive: {expected}")
    data_root.mkdir(parents=True,exist_ok=True)
    for name in ("annotations","features"):
        source_path=release_root/name
        if source_path.exists():
            dest=data_root/name
            if dest.exists(): shutil.rmtree(dest)
            shutil.move(str(source_path),str(dest))
    shutil.rmtree(staging,ignore_errors=True)
    if not valid_layout(data_root):
        raise RuntimeError(f"PHOENIX14T extraction is incomplete under {data_root}")
    print(f"PHOENIX14T is ready: {data_root}",flush=True)
    return data_root

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--archive",type=Path,required=True)
    args=parser.parse_args()
    ensure_phoenix(args.data_root,args.archive)
