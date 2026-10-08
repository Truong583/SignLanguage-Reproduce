"""Fetch and safely extract the official PHOENIX14T release when absent."""
from pathlib import Path, PurePosixPath
import os
import shutil
import tarfile
import time
import urllib.request

URL="https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"

def download_resumable(url: str, archive: Path) -> None:
    archive.parent.mkdir(parents=True,exist_ok=True)
    partial=archive.with_name(archive.name+".part")
    offset=partial.stat().st_size if partial.exists() else 0
    request=urllib.request.Request(url,headers={"User-Agent":"MixSignGraph-reproduction/1.0",**({"Range":f"bytes={offset}-"} if offset else {})})
    try:
        response=urllib.request.urlopen(request,timeout=90)
        if offset and response.status!=206:
            response.close(); partial.unlink(missing_ok=True); offset=0
            request=urllib.request.Request(url,headers={"User-Agent":"MixSignGraph-reproduction/1.0"})
            response=urllib.request.urlopen(request,timeout=90)
        total=response.headers.get("Content-Length")
        total=int(total)+offset if total and total.isdigit() else None
        mode="ab" if offset else "wb"
        last_report=offset
        with response,partial.open(mode) as out:
            while True:
                chunk=response.read(8*1024*1024)
                if not chunk: break
                out.write(chunk)
                current=out.tell()
                if current-last_report>=1024**3:
                    size=f"/{total/1024**3:.1f} GiB" if total else ""
                    print(f"Dataset download: {current/1024**3:.1f} GiB{size}",flush=True)
                    last_report=current
            out.flush(); os.fsync(out.fileno())
        if total and partial.stat().st_size!=total:
            raise IOError(f"Incomplete download: got {partial.stat().st_size} of {total} bytes; rerun to resume")
        partial.replace(archive)
    except Exception:
        print(f"Download interrupted. Partial file retained at {partial}; rerun to resume.",flush=True)
        raise

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
