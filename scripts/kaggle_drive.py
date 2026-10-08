"""Small Drive API helper used by the Kaggle notebook; credentials stay in memory."""
from pathlib import Path
import json
import os
import shutil
import tempfile

DRIVE_SCOPE='https://www.googleapis.com/auth/drive'

def service_from_json(value):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    info=json.loads(value)
    credentials=Credentials.from_authorized_user_info(info,scopes=[DRIVE_SCOPE])
    if not credentials.valid:
        credentials.refresh(Request())
    return build('drive','v3',credentials=credentials,cache_discovery=False)


def download_file(service,file_id,destination):
    from googleapiclient.http import MediaIoBaseDownload
    import io
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.download-tmp')
    request=service.files().get_media(fileId=file_id)
    last_report=0
    with temporary.open('wb') as stream:
        downloader=MediaIoBaseDownload(stream,request,chunksize=16*1024*1024)
        done=False
        while not done:
            status,done=downloader.next_chunk()
            if status:
                progress=int(status.progress()*100)
                if progress>=last_report+10:
                    print(f'Drive download {destination.name}: {progress}%',flush=True)
                    last_report=progress
    os.replace(temporary,destination)
    return destination


def find_child(service,parent_id,name,mime_type=None):
    escaped=name.replace("'","\\'")
    query=f"'{parent_id}' in parents and name='{escaped}' and trashed=false"
    if mime_type: query+=f" and mimeType='{mime_type}'"
    result=service.files().list(q=query,fields='files(id,name,mimeType)',pageSize=100,
                                supportsAllDrives=True,includeItemsFromAllDrives=True).execute()
    return result.get('files',[])


def ensure_folder(service,parent_id,name):
    matches=find_child(service,parent_id,name,'application/vnd.google-apps.folder')
    if len(matches)>1: raise ValueError(f'Multiple Drive folders named {name}; remove ambiguity')
    if matches: return matches[0]['id']
    item=service.files().create(body={'name':name,'mimeType':'application/vnd.google-apps.folder',
                                      'parents':[parent_id]},fields='id,name',supportsAllDrives=True).execute()
    return item['id']


def upsert_file(service,parent_id,source,name=None):
    from googleapiclient.http import MediaFileUpload
    source=Path(source)
    name=name or source.name
    matches=find_child(service,parent_id,name)
    if len(matches)>1: raise ValueError(f'Multiple Drive files named {name}; remove ambiguity')
    metadata={'name':name,'mimeType':'application/octet-stream'}
    media=MediaFileUpload(str(source),mimetype='application/octet-stream',resumable=True,chunksize=16*1024*1024)
    if matches:
        return service.files().update(fileId=matches[0]['id'],media_body=media,
                                      supportsAllDrives=True,fields='id,name,size,modifiedTime').execute()
    metadata['parents']=[parent_id]
    return service.files().create(body=metadata,media_body=media,supportsAllDrives=True,
                                  fields='id,name,size,modifiedTime').execute()


def sync_checkpoint_directory(service,local_dir,drive_folder_id):
    local_dir=Path(local_dir)
    names=('last.pt','best.pt','run_metadata.json','vocab.json','history.jsonl',
           'best_dev_metrics.json','best_dev_predictions.json','test_metrics.json',
           'test_predictions.json','notebook-config.yaml','notebook-training.log')
    results=[]
    for name in names:
        source=local_dir/name
        if source.is_file():
            results.append(upsert_file(service,drive_folder_id,source))
    return results
