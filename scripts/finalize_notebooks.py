"""Apply runtime wiring and validate generated notebook artifacts."""
from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]/"notebooks"
colab_path=root/"MixSignGraph_Colab.ipynb"
colab=json.loads(colab_path.read_text(encoding="utf-8"))
config="".join(colab["cells"][4]["source"])
config=config.replace('PROJECT_ZIP=Path("/content/drive/MyDrive/SignLanguage-Reproduce-portable.zip")', 'SHARED_DRIVE_ROOT=Path("/content/drive/MyDrive/SignLanguage-Reproduction")\nPROJECT_ZIP=SHARED_DRIVE_ROOT/"SignLanguage-Reproduce-portable.zip"')
config=config.replace('DATA_ROOT=Path("/content/drive/MyDrive/SignLanguage-Data/phoenix14t")', 'DATA_ROOT=Path("/content/phoenix14t")\nPHOENIX_ARCHIVE=SHARED_DRIVE_ROOT/"phoenix-2014-T.v3.tar.gz"')
config=config.replace('DATA_ROOT=SHARED_DRIVE_ROOT/"data/phoenix14t"', 'DATA_ROOT=Path("/content/phoenix14t")\nPHOENIX_ARCHIVE=SHARED_DRIVE_ROOT/"phoenix-2014-T.v3.tar.gz"')
config=config.replace('MANIFEST_ROOT=Path("/content/drive/MyDrive/SignLanguage-Manifests/phoenix14t")', 'MANIFEST_ROOT=SHARED_DRIVE_ROOT/"manifests/phoenix14t"')
config=config.replace('DRIVE_RUNS_ROOT=Path("/content/drive/MyDrive/SignLanguage-Runs")', 'DRIVE_RUNS_ROOT=SHARED_DRIVE_ROOT/"runs"')
colab["cells"][4]["source"]=config.splitlines(keepends=True)
colab["cells"][1]["source"]=["## Chuẩn bị Drive một lần\n", "Đưa `SignLanguage-Reproduce-portable.zip` vào `MyDrive/SignLanguage-Reproduction/`. Notebook tự tải và lưu archive PHOENIX14T vào cùng thư mục Drive nếu chưa có; bạn không cần tự tải hoặc giải nén dữ liệu. Cần Internet và khoảng 39 GB trống trên Drive cho archive.\n"]
colab["cells"][0]["source"]=["# MixSignGraph tái dựng — Google Colab\n", "\n", "Bật GPU và Internet một lần trong Runtime settings, rồi chọn **Runtime → Run all**. Notebook tự lấy mã từ ZIP trong Drive, tự tải/giải nén PHOENIX14T nếu chưa có, rồi huấn luyện. Checkpoint và kết quả được lưu trong cùng thư mục Drive; chạy lại sẽ tiếp tục từ checkpoint mới nhất.\n"]
fetch_cell={"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["import subprocess, sys\n", "subprocess.run([sys.executable, 'scripts/fetch_phoenix.py', '--data-root', str(DATA_ROOT), '--archive', str(PHOENIX_ARCHIVE)], cwd=project_root, check=True)\n"]}
colab["cells"].insert(8,fetch_cell)
colab["cells"][10]["source"]=["## Dữ liệu\n", "Notebook tự tải archive PHOENIX14T từ máy chủ RWTH vào Drive nếu chưa có, rồi giải nén vào ổ tạm Colab để tránh tạo hàng trăm nghìn file nhỏ trực tiếp trên Drive. Archive trên Drive giúp lần chạy sau không phải tải lại từ nguồn. Cần khoảng 39 GB trống trên Drive và đủ dung lượng runtime để giải nén.\n"]
colab_path.write_text(json.dumps(colab,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
path=root/"MixSignGraph_Kaggle.ipynb"
nb=json.loads(path.read_text(encoding="utf-8"))
cells=nb["cells"]
config="".join(cells[3]["source"])
config=config.replace('DRIVE_PROJECT_FILE_ID="PASTE_PROJECT_ZIP_FILE_ID"\nDRIVE_DATA_FILE_ID="PASTE_DATA_ZIP_FILE_ID"\nDATA_ARCHIVE_NAME="phoenix14t.zip"', 'DATA_ARCHIVE_NAME="phoenix-2014-T.v3.tar.gz"')
config=config.replace('DATA_ROOT=WORKSPACE/"dataset"', 'DATA_ROOT=Path("/tmp/phoenix14t")')
config=config.replace('if "PASTE_" in DRIVE_WORKSPACE_FOLDER_ID+DRIVE_PROJECT_FILE_ID+DRIVE_DATA_FILE_ID:\n    raise ValueError("Fill the 3 Google Drive IDs")', 'if "PASTE_" in DRIVE_WORKSPACE_FOLDER_ID:\n    raise ValueError("Fill the shared Google Drive folder ID")')
cells[3]["source"]=config.splitlines(keepends=True)
cells[5]["source"]=["%pip install -q google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2"]
bootstrap="".join(cells[6]["source"])
bootstrap=bootstrap.replace('project_drive_file=drive_service.files().get(fileId=DRIVE_PROJECT_FILE_ID,fields="id,name,mimeType").execute()', '''project_matches=drive_service.files().list(q=f"'{DRIVE_WORKSPACE_FOLDER_ID}' in parents and name='SignLanguage-Reproduce-portable.zip' and trashed=false",fields="files(id,name,mimeType)",pageSize=10,supportsAllDrives=True,includeItemsFromAllDrives=True).execute().get("files",[])
if len(project_matches)!=1: raise FileNotFoundError("Put exactly one SignLanguage-Reproduce-portable.zip in the shared Drive folder")
project_drive_file=project_matches[0]
DRIVE_PROJECT_FILE_ID=project_drive_file["id"]''')
cells[6]["source"]=bootstrap.splitlines(keepends=True)
cells[0]["source"]=["# MixSignGraph tái dựng — Kaggle + Google Drive\n", "\n", "Bật GPU và Internet trong Kaggle, tạo Kaggle Secret theo hướng dẫn một lần, điền ID thư mục Drive chung rồi chọn **Run All**. Notebook tự tải mã và dữ liệu nếu chưa có; checkpoint/kết quả lưu Drive và tiếp tục ở lần chạy sau.\n"]
cells[1]["source"]=["## Chuẩn bị một lần\n", "Đặt `SignLanguage-Reproduce-portable.zip` vào thư mục Drive chung. Tạo OAuth Kaggle Secret một lần theo `notebooks/DRIVE_SETUP.md`; sau đó notebook tự tìm ZIP mã và tự tải/cache dữ liệu PHOENIX14T.\n"]
cells[2]["source"]=["## Cấu hình\n", "Chỉ cần điền ID thư mục Drive `SignLanguage-Reproduction`. Mặc định notebook chạy PHOENIX14T CSLR.\n"]
setup="""import zipfile, shutil, subprocess, sys
sys.path.insert(0,str(PROJECT_ROOT))
from scripts.kaggle_drive import download_file,ensure_folder,find_child,upsert_file
subprocess.run([sys.executable,"scripts/verify_bundle.py"],cwd=PROJECT_ROOT,check=True)
DRIVE_RUNS_FOLDER_ID=ensure_folder(drive_service,DRIVE_WORKSPACE_FOLDER_ID,"runs")
matches=find_child(drive_service,DRIVE_RUNS_FOLDER_ID,RUN_ID,"application/vnd.google-apps.folder")
if len(matches)>1: raise ValueError("Duplicate Drive run folders")
DRIVE_RUN_FOLDER_ID=matches[0]["id"] if matches else ensure_folder(drive_service,DRIVE_RUNS_FOLDER_ID,RUN_ID)
LOCAL_RUN_DIR.mkdir(parents=True,exist_ok=True)
for name in ("last.pt","best.pt","run_metadata.json","vocab.json","history.jsonl","best_dev_metrics.json","best_dev_predictions.json"):
    matches=find_child(drive_service,DRIVE_RUN_FOLDER_ID,name)
    if len(matches)>1: raise ValueError(f"Duplicate Drive file: {name}")
    if matches: download_file(drive_service,matches[0]["id"],LOCAL_RUN_DIR/name)
if PHASE in ("slt","gfslt"):
    source_phase="cslr" if PHASE=="slt" else "tctc"
    source_run_id=f"{DATASET}_{source_phase}_seed{SEED}"
    source_runs=find_child(drive_service,DRIVE_RUNS_FOLDER_ID,source_run_id,"application/vnd.google-apps.folder")
    if source_runs:
        source_checkpoints=find_child(drive_service,source_runs[0]["id"],"best.pt")
        if source_checkpoints: download_file(drive_service,source_checkpoints[0]["id"],LOCAL_RUN_DIR/"initialization.pt")
archive_matches=find_child(drive_service,DRIVE_WORKSPACE_FOLDER_ID,DATA_ARCHIVE_NAME)
archive_path=Path("/tmp")/DATA_ARCHIVE_NAME
if archive_matches:
    if len(archive_matches)>1: raise ValueError("Duplicate PHOENIX archives in shared Drive folder")
    print("Loading the cached PHOENIX archive from Drive.")
    download_file(drive_service,archive_matches[0]["id"],archive_path)
else:
    from scripts.fetch_phoenix import download_resumable,URL
    print("The Drive cache is empty; downloading the official PHOENIX archive.")
    download_resumable(URL,archive_path)
    quota=drive_service.about().get(fields="storageQuota").execute().get("storageQuota",{})
    limit=int(quota.get("limit") or 0); usage=int(quota.get("usage") or 0)
    available=limit-usage if limit else None
    if available is None or available>=archive_path.stat().st_size:
        try:
            upsert_file(drive_service,DRIVE_WORKSPACE_FOLDER_ID,archive_path,DATA_ARCHIVE_NAME)
            print("Cached the archive in the shared Drive folder for later sessions.")
        except Exception as exc:
            print(f"Could not cache archive to Drive ({exc}); continuing this session with the local copy.")
    else:
        print("Drive has less free space than the archive; continuing locally. Kaggle may need to download it again next session.")
subprocess.run([sys.executable,"scripts/fetch_phoenix.py","--data-root",str(DATA_ROOT),"--archive",str(archive_path)],cwd=PROJECT_ROOT,check=True)
print("Project, PHOENIX data, and prior checkpoints are ready.")"""
cells[7]["source"]=setup.splitlines(keepends=True)
cells[14]["source"]=["## Dữ liệu và checkpoint\n", "Notebook tự tìm archive PHOENIX14T trong Drive; nếu chưa có, nó tải từ RWTH, cố gắng lưu archive vào Drive, rồi giải nén ở ổ tạm Kaggle. Kaggle giới hạn dung lượng output được lưu; dữ liệu lớn được giữ trong ổ tạm, còn checkpoint và kết quả được đồng bộ lên Drive. Cần Internet, dung lượng scratch đủ cho dữ liệu đã giải nén, và khoảng 39 GB trống trên Drive để lưu archive dùng chung.\n"]
# Kaggle writes locally first, then atomically uploads complete epoch outputs to Drive.
train="".join(cells[15]["source"]).replace("OUTPUT_RUN_DIR","LOCAL_RUN_DIR")
train=train.replace('shutil.copy2(LOCAL_RUN_DIR/"notebook-config.yaml",LOCAL_RUN_DIR/"notebook-config.yaml")', 'pass  # Kaggle outputs are uploaded directly to the shared Drive folder below.')
train=train.replace('if source.is_file(): shutil.copy2(source,LOCAL_RUN_DIR/name)', 'if source.is_file() and source.resolve()!= (LOCAL_RUN_DIR/name).resolve(): shutil.copy2(source,LOCAL_RUN_DIR/name)')
train=train.replace('source_run=LOCAL_RUN_DIR.parent/f"{DATASET}_{source_phase}_seed{SEED}"/"best.pt"\n    if source_run.is_file():\n        source_local=LOCAL_RUN_DIR/"initialization.pt"\n        shutil.copy2(source_run,source_local)\n        args += ["--initialize-from",str(source_local)]\n    else:', 'source_local=LOCAL_RUN_DIR/"initialization.pt"\n    if source_local.is_file():\n        args += ["--initialize-from",str(source_local)]\n    else:')
cells[15]["source"]=train.splitlines(keepends=True)
# Fetch assets before doctor validates the runtime config.
cells[10],cells[12]=cells[12],cells[10]
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")

# Notebook code cells should be syntactically valid after removing IPython magics.
for file in (root/"MixSignGraph_Colab.ipynb",path):
    notebook=json.loads(file.read_text(encoding="utf-8"))
    for index,cell in enumerate(notebook["cells"]):
        if cell["cell_type"]!="code": continue
        source="".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        compile(source,f"{file.name}:cell{index}","exec")
    print(f"Notebook JSON and code cells valid: {file.name}")

from upgrade_suite_notebooks import upgrade
upgrade()
