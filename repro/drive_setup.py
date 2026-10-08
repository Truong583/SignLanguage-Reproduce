"""Stream the official PH14T archive into lossless per-video ZIPs on Drive.

Only one compressed video is buffered locally. A dataset commit is written only
after every annotated example is matched, with no missing or extra videos.
Restarting preparation re-reads the tar stream and reuses verified uploaded clips.
"""
import csv
import hashlib
import io
import json
from pathlib import PurePosixPath, Path
import re
import tarfile
import tempfile
import zipfile
import requests
from .persistence import DriveStore, digest, safe_run_id

SOURCE='https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz'
FOLDER='application/vnd.google-apps.folder'


def children(service,parent):
    # Parent IDs are validated before interpolation, never user-provided query text.
    if not re.fullmatch(r'[A-Za-z0-9_-]+',parent): raise ValueError('Invalid Drive ID')
    result=[]
    token=None
    while True:
        page=service.files().list(q=f"'{parent}' in parents and trashed=false",pageSize=1000,
            fields='nextPageToken,files(id,name,mimeType,size,md5Checksum,appProperties)',pageToken=token).execute(num_retries=5)
        result.extend(page.get('files',[]))
        token=page.get('nextPageToken')
        if not token: return result


def named(service,parent,name,folder=False,create=False):
    found=[f for f in children(service,parent) if f['name']==name]
    if len(found)>1: raise ValueError(f'Duplicate Drive names: {name}; resolve duplicates first')
    if found:
        if folder!=(found[0]['mimeType']==FOLDER): raise ValueError(f'Wrong Drive item type: {name}')
        return found[0]['id']
    if not create: return None
    if not folder: raise ValueError('Only folders can be created here')
    return service.files().create(body={'name':name,'mimeType':FOLDER,'parents':[parent]},fields='id').execute(num_retries=5)['id']


def upload_bytes(service,parent,name,value,properties=None):
    from googleapiclient.http import MediaIoBaseUpload
    request=service.files().create(body={'name':name,'parents':[parent],'appProperties':properties or {}},
        media_body=MediaIoBaseUpload(io.BytesIO(value),mimetype='application/octet-stream',resumable=True,chunksize=8*1024**2),
        fields='id,md5Checksum')
    result=None
    while result is None: _,result=request.next_chunk(num_retries=5)
    if result.get('md5Checksum')!=hashlib.md5(value).hexdigest(): raise IOError('Drive upload integrity check failed')
    return result['id']


def load_json(store,ident):
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/'metadata.json'
        store.get(ident,path)
        return json.loads(path.read_text(encoding='utf-8'))


def ingest_stream(stream,service,input_folder):
    """Dependency-injected stream/API for offline tests; no extraction to disk."""
    from scripts.prepare_data import cleaners
    normalizer=cleaners()['clean_phoenix_2014_trans']
    existing={}
    for item in children(service,input_folder):
        if item['name'] in existing: raise ValueError('Duplicate input filenames; resolve before preparation')
        existing[item['name']]=item
    clips={}
    annotations={}
    current=None
    memory=None
    archive=None
    frame_names=set()

    def finish():
        nonlocal archive,memory
        if current is None: return
        archive.close()
        value=memory.getvalue()
        sha=hashlib.sha256(value).hexdigest()
        filename=hashlib.sha256(current.encode()).hexdigest()+'.zip'
        old=existing.get(filename)
        if old:
            if old.get('md5Checksum')!=hashlib.md5(value).hexdigest() or int(old.get('size',-1))!=len(value):
                raise ValueError('Previously uploaded video differs; use a new Drive input folder')
            ident=old['id']
        else:
            ident=upload_bytes(service,input_folder,filename,value,{'clip_sha256':sha})
        clips[current]={'clip_id':ident,'clip_sha256':sha,'clip_bytes':len(value),'frame_count':len(frame_names)}
        if len(clips)%25==0: print(f'Drive: prepared {len(clips)} complete videos',flush=True)
        memory.close()

    try:
        with tarfile.open(fileobj=stream,mode='r|gz') as tar:
            for member in tar:
                path=PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe archive member')
                if not member.isfile():
                    if not member.isdir(): raise ValueError('Links and special files are not allowed')
                    continue
                annotation=re.fullmatch(r'PHOENIX-2014-T\.(train|dev|test)\.corpus\.csv',path.name)
                if annotation:
                    if member.size>32*1024**2: raise ValueError('Unexpected annotation size')
                    split=annotation[1]
                    if split in annotations: raise ValueError('Duplicate annotations')
                    annotations[split]=tar.extractfile(member).read().decode('utf-8-sig')
                    continue
                parts=path.parts
                if 'fullFrame-210x260px' not in parts or path.suffix.lower() not in {'.png','.jpg','.jpeg'}: continue
                i=parts.index('fullFrame-210x260px')
                if len(parts)<i+4 or parts[i+1] not in {'train','dev','test'}: raise ValueError('Unknown frame layout')
                key='/'.join(parts[i+1:i+3])
                if key!=current:
                    finish()
                    if key in clips: raise ValueError('Non-contiguous video in source tar; preparation stopped, no dataset committed')
                    current=key
                    memory=io.BytesIO()
                    archive=zipfile.ZipFile(memory,'w',compression=zipfile.ZIP_STORED)
                    frame_names=set()
                if path.name in frame_names: raise ValueError('Duplicate frame name')
                if member.size>16*1024**2 or memory.tell()+member.size>512*1024**2:
                    raise ValueError('One video exceeds the 512 MiB preparation buffer limit')
                frame_names.add(path.name)
                info=zipfile.ZipInfo(path.name,date_time=(1980,1,1,0,0,0))
                archive.writestr(info,tar.extractfile(member).read())
        finish()
    finally:
        if archive: archive.close()
        if memory and not memory.closed: memory.close()
    prepared={}
    matched=set()
    ids=set()
    for split in ('train','dev','test'):
        if split not in annotations: raise ValueError(f'Missing {split} annotations')
        rows=[]
        for row in csv.DictReader(io.StringIO(annotations[split]),delimiter='|'):
            name=row['name']
            key=f'{split}/{name}'
            if key not in clips or name in ids: raise ValueError(f'Missing video or duplicate sample: {key}')
            gloss=row.get('orth') or row.get('annotation') or row.get('gloss')
            if not gloss: raise ValueError('Missing gloss label')
            ids.add(name)
            matched.add(key)
            rows.append({'id':name,'gloss':normalizer(gloss).lower(),
                         'text':(row.get('translation') or row.get('text') or '').strip().lower(),**clips[key]})
        if not rows: raise ValueError(f'Empty {split}')
        prepared[split]=rows
    if matched!=set(clips): raise ValueError('Source has unannotated videos; refusing to silently drop samples')
    return {'schema':1,'dataset':'phoenix14t','source':SOURCE,'splits':prepared,
            'storage':'original frame bytes in per-video ZIP; no image re-encoding'}


def prepare(root_id,allow_download=True):
    store=DriveStore(root_id)
    service=store.service
    info=service.files().get(fileId=root_id,fields='mimeType,capabilities(canAddChildren)').execute(num_retries=5)
    if info['mimeType']!=FOLDER or not info.get('capabilities',{}).get('canAddChildren'):
        raise ValueError('DRIVE_ROOT_ID must be a writable Drive folder')
    folder=named(service,root_id,'input',folder=True,create=True)
    ready=named(service,folder,'dataset-v1.json')
    if ready: return load_json(store,ready)
    if not allow_download: raise FileNotFoundError('No prepared dataset on Drive; enable PREPARE_DATA_ONCE')
    print('Streaming official 39 GB download to Drive, one video at a time. No full local dataset copy.',flush=True)
    with requests.get(SOURCE,stream=True,timeout=(30,180)) as response:
        response.raise_for_status()
        result=ingest_stream(response.raw,service,folder)
    upload_bytes(service,folder,'dataset-v1.json',json.dumps(result,ensure_ascii=False).encode('utf-8'))
    return result
