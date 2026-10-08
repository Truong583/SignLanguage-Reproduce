"""Versioned checkpoint publication: payload first, commit descriptor last.

Never overwrite the last good remote generation. Verify downloaded SHA-256;
fall back to the previous committed generation after corruption/interruption.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid


def digest(path, algorithm='sha256'):
    h=hashlib.new(algorithm)
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''): h.update(data)
    return h.hexdigest()


def safe_run_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',value):
        raise ValueError('run_id: 1..80 ASCII letters/digits/_/-; no spaces or paths')
    return value


class LocalStore:
    def __init__(self,folder):
        self.folder=Path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
    def put(self,path,name):
        dest=self.folder/name
        tmp=dest.with_name(dest.name+'.uploading')
        with open(path,'rb') as source,open(tmp,'wb') as out:
            shutil.copyfileobj(source,out,8*1024*1024)
            out.flush()
            os.fsync(out.fileno())
        if digest(path)!=digest(tmp): raise IOError('Persistent copy checksum mismatch')
        os.replace(tmp,dest)
        return name
    def get(self,ident,path):
        if Path(ident).name!=ident: raise ValueError('Invalid persistent filename')
        shutil.copyfile(self.folder/ident,path)
    def commits(self,kind):
        return sorted([p.name for p in self.folder.glob(f'{kind}-*.json')],reverse=True)
    def remove(self,ident):
        if re.fullmatch(r'(last|best)-[0-9]+-[a-f0-9]+\.(pt|json)',ident):
            (self.folder/ident).unlink(missing_ok=True)


class DriveStore:
    def __init__(self,folder,credentials_file=None,service=None):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',folder): raise ValueError('Invalid Drive folder ID')
        if service is None:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            credentials_file=credentials_file or os.environ.get('SIGN_DRIVE_CREDENTIALS')
            if not credentials_file: raise ValueError('Missing private Drive credentials file')
            creds=Credentials.from_authorized_user_file(credentials_file)
            service=build('drive','v3',credentials=creds,cache_discovery=False)
        self.service,self.folder=service,folder
    def put(self,path,name):
        from googleapiclient.http import MediaFileUpload
        media=MediaFileUpload(str(path),resumable=True,chunksize=8*1024*1024)
        request=self.service.files().create(body={'name':name,'parents':[self.folder],
              'appProperties':{'signlanguage_checkpoint':'v1'}},media_body=media,fields='id,md5Checksum')
        result=None
        while result is None: _,result=request.next_chunk(num_retries=5)
        if result.get('md5Checksum')!=digest(path,'md5'): raise IOError('Drive server checksum mismatch')
        return result['id']
    def get(self,ident,path):
        from googleapiclient.http import MediaIoBaseDownload
        with open(path,'wb') as out:
            request=self.service.files().get_media(fileId=ident)
            download=MediaIoBaseDownload(out,request,chunksize=8*1024*1024)
            done=False
            while not done: _,done=download.next_chunk(num_retries=5)
    def commits(self,kind):
        results=[]
        token=None
        while True:
            page=self.service.files().list(q=f"'{self.folder}' in parents and trashed=false and appProperties has {{ key='signlanguage_checkpoint' and value='v1' }}",
                fields='nextPageToken,files(id,name)',pageToken=token,pageSize=1000).execute(num_retries=5)
            results += [r for r in page.get('files',[]) if re.fullmatch(kind+r'-[0-9]+-[a-f0-9]+\.json',r['name'])]
            token=page.get('nextPageToken')
            if not token: break
        return [r['id'] for r in sorted(results,key=lambda r:r['name'],reverse=True)]
    def remove(self,ident):
        info=self.service.files().get(fileId=ident,fields='name,parents,appProperties').execute(num_retries=5)
        if self.folder not in info.get('parents',[]) or info.get('appProperties',{}).get('signlanguage_checkpoint')!='v1':
            raise ValueError('Refusing to delete a file not owned by this checkpoint store')
        if not re.fullmatch(r'(last|best)-[0-9]+-[a-f0-9]+\.(pt|json)',info['name']):
            raise ValueError('Refusing to delete non-checkpoint file')
        self.service.files().delete(fileId=ident).execute(num_retries=5)


def make_store(cfg):
    if not cfg: return None
    if cfg['backend']=='local': return LocalStore(Path(cfg['root'])/safe_run_id(cfg['run_id']))
    if cfg['backend']=='gdrive': return DriveStore(cfg['folder_id'])
    raise ValueError('Unknown persistence backend')


def publish(store,path,kind='last',keep=2):
    if keep<2: raise ValueError('Keep at least two recoverable generations')
    path=Path(path)
    name=f'{kind}-{time.time_ns():020d}-{uuid.uuid4().hex}'
    payload=store.put(path,name+'.pt')
    descriptor={'schema':1,'payload':payload,'sha256':digest(path),'bytes':path.stat().st_size}
    desc=path.parent/(name+'.json')
    try:
        desc.write_text(json.dumps(descriptor),encoding='utf-8')
        commit=store.put(desc,name+'.json')
    finally: desc.unlink(missing_ok=True)
    # Deletion happens only after a successfully published new commit. Failures
    # during cleanup never invalidate the committed checkpoint.
    for old in store.commits(kind)[keep:]:
        try:
            store.get(old,desc)
            previous=json.loads(desc.read_text())
            store.remove(old)
            store.remove(previous['payload'])
        except Exception:
            print('Checkpoint committed; an older generation could not be cleaned up.',flush=True)
        finally: desc.unlink(missing_ok=True)
    print(f'PERSISTED {kind}: {descriptor["sha256"][:12]}',flush=True)
    return commit


def restore(store,destination,kind='last'):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    commits=store.commits(kind)
    if not commits: return None
    desc=destination.with_suffix('.restore.json')
    tmp=destination.with_suffix('.restore.tmp')
    for ident in commits:
        try:
            store.get(ident,desc)
            data=json.loads(desc.read_text())
            store.get(data['payload'],tmp)
            if tmp.stat().st_size!=data['bytes'] or digest(tmp)!=data['sha256']:
                raise IOError('Corrupt checkpoint')
            os.replace(tmp,destination)
            return destination
        except Exception:
            print('Could not restore one generation; trying previous committed generation.',flush=True)
        finally:
            desc.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)
    raise IOError('Persistent checkpoints exist, but none can be restored. Training must not restart from scratch.')
