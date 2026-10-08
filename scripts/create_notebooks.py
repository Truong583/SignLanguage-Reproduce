"""Create separate Colab and Kaggle notebooks from editable plain text cells."""
from pathlib import Path
import json
import textwrap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'notebooks'
OUT.mkdir(exist_ok=True)

def md(source):
    return {'cell_type':'markdown','metadata':{},'source':textwrap.dedent(source).strip().splitlines(keepends=True)}

def code(source):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
            'source':textwrap.dedent(source).strip().splitlines(keepends=True)}

def save(name,cells,metadata):
    notebook={'cells':cells,'metadata':metadata,
              'nbformat':4,'nbformat_minor':5}
    (OUT/name).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')

COMMON_CONFIG=r'''
from pathlib import Path
import json, os, shutil, subprocess, sys, time
import yaml

DATASET = "phoenix14t"  # phoenix14t, phoenix14, csl_daily, how2sign, openasl
PHASE = "cslr"          # cslr, tctc, slt, gfslt
SEED = 0
GLOBAL_BATCH = 6
MICRO_BATCH = 1
WORKERS = 2
EPOCHS = 50
RESUME_LAST = True
EVALUATE_TEST_AT_END = True

RUN_ID = f"{DATASET}_{PHASE}_seed{SEED}"
CONFIG_NAME = f"{DATASET}_{PHASE}.yaml"
'''

SETUP_CODE=r'''
from pathlib import Path
import hashlib, zipfile, shutil, subprocess, sys

if not PROJECT_ZIP.is_file():
    raise FileNotFoundError(f"Upload the project ZIP here first: {PROJECT_ZIP}")
project_root = WORKSPACE / "SignLanguage-Reproduce"
if not project_root.exists():
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PROJECT_ZIP) as archive:
        for entry in archive.infolist():
            target=(WORKSPACE / entry.filename).resolve()
            if not target.is_relative_to(WORKSPACE.resolve()):
                raise ValueError(f"Unsafe ZIP path: {entry.filename}")
        archive.extractall(WORKSPACE)
if not (project_root / "scripts/launch.py").is_file():
    raise FileNotFoundError("Project ZIP has an unexpected layout")
subprocess.run([sys.executable, "scripts/verify_bundle.py"],cwd=project_root,check=True)
print("Project checksum verified:", project_root)
'''

KAGGLE_DRIVE_CODE=r'''
from kaggle_secrets import UserSecretsClient
from pathlib import Path
import json, os, sys, zipfile, subprocess
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

try:
    oauth_json=UserSecretsClient().get_secret("GOOGLE_DRIVE_OAUTH_JSON")
except Exception as exc:
    raise RuntimeError("Add your private Kaggle Secret GOOGLE_DRIVE_OAUTH_JSON first; see notebooks/DRIVE_SETUP.md") from exc

credentials=Credentials.from_authorized_user_info(json.loads(oauth_json),scopes=["https://www.googleapis.com/auth/drive"])
if not credentials.valid: credentials.refresh(Request())
drive_service=build("drive","v3",credentials=credentials,cache_discovery=False)
del oauth_json,credentials
def drive_download(file_id,destination):
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_name(destination.name+".download-tmp")
    with temp.open("wb") as stream:
        downloader=MediaIoBaseDownload(stream,drive_service.files().get_media(fileId=file_id),chunksize=16*1024*1024)
        done=False
        while not done: _,done=downloader.next_chunk()
    os.replace(temp,destination)
WORKSPACE.mkdir(parents=True,exist_ok=True)
project_drive_file=drive_service.files().get(fileId=DRIVE_PROJECT_FILE_ID,fields="id,name,mimeType").execute()
if project_drive_file["mimeType"]!="application/zip" and not project_drive_file["name"].lower().endswith(".zip"):
    raise ValueError("The Drive project file must be the portable ZIP")
PROJECT_ZIP=WORKSPACE/project_drive_file["name"]
drive_download(DRIVE_PROJECT_FILE_ID,PROJECT_ZIP)
with zipfile.ZipFile(PROJECT_ZIP) as archive:
    for entry in archive.infolist():
        if not (WORKSPACE/entry.filename).resolve().is_relative_to(WORKSPACE.resolve()): raise ValueError("Unsafe project ZIP path")
    archive.extractall(WORKSPACE)
PROJECT_ROOT=WORKSPACE/"SignLanguage-Reproduce"
if not (PROJECT_ROOT/"scripts/launch.py").is_file(): raise FileNotFoundError("Unexpected project ZIP layout")
sys.path.insert(0,str(PROJECT_ROOT))
subprocess.run([sys.executable,"scripts/verify_bundle.py"],cwd=PROJECT_ROOT,check=True)
from scripts.kaggle_drive import download_file,ensure_folder,find_child,upsert_file
print("Project ZIP downloaded from the shared Drive folder.")
'''

DATA_CODE=r'''
from pathlib import Path
import shutil, subprocess, sys

data_root=Path(DATA_ROOT).expanduser().resolve()
manifest_dir=Path(MANIFEST_ROOT).expanduser().resolve()
if DATASET in ("phoenix14t","phoenix14"):
    annotation_dir=data_root / "annotations" / "manual"
    subprocess.run([sys.executable,"scripts/prepare_data.py","--dataset",DATASET,
        "--annotations",str(annotation_dir),"--data-root",str(data_root),
        "--output",str(manifest_dir),"--overwrite"],cwd=project_root,check=True)
else:
    source_manifests=data_root / "manifests"
    if not all((source_manifests/f"{split}.jsonl").is_file() for split in ("train","dev","test")):
        raise FileNotFoundError(f"Create train/dev/test JSONL manifests in {source_manifests}")
    manifest_dir.mkdir(parents=True,exist_ok=True)
    for split in ("train","dev","test"):
        shutil.copy2(source_manifests/f"{split}.jsonl",manifest_dir/f"{split}.jsonl")

if PHASE in ("tctc","gfslt"):
    pseudo_dir=manifest_dir.parent/(manifest_dir.name+"_tctc")
    if not all('pseudo_gloss' in line for split in ("train","dev","test")
        for line in [__import__('json').loads(s) for s in (manifest_dir/f"{split}.jsonl").read_text(encoding='utf-8').splitlines() if s.strip()]):
        raise ValueError("TCTC needs pseudo_gloss in each manifest. Run prepare_tctc.py with an explicitly selected local lemmatizer first.")
    pseudo_dir.mkdir(parents=True,exist_ok=True)
    for split in ("train","dev","test"):
        shutil.copy2(manifest_dir/f"{split}.jsonl",pseudo_dir/f"{split}.jsonl")
    manifest_dir=pseudo_dir

if DATASET in ("how2sign","openasl"):
    FEATURES_ROOT=data_root
else:
    FEATURES_ROOT=data_root

config_template=project_root/"configs_repro"/CONFIG_NAME
if not config_template.is_file():
    raise FileNotFoundError(f"No shipped config: {config_template}")
cfg=yaml.safe_load(config_template.read_text(encoding="utf-8"))
cfg["data_root"]=str(FEATURES_ROOT)
for split in ("train","dev","test"):
    cfg[split]=str(manifest_dir/f"{split}.jsonl")
cfg["output"]=str(LOCAL_RUN_DIR)
cfg["global_batch"]=GLOBAL_BATCH
cfg["micro_batch"]=MICRO_BATCH
cfg["workers"]=WORKERS
cfg["epochs"]=EPOCHS
if DATASET in ("phoenix14t","phoenix14"):
    cfg["resnet_weights"]=str(project_root/"assets/resnet18-f37072fd.pth")
if PHASE in ("slt","gfslt"):
    cfg["mbart_path"]=str(project_root/"assets/mbart-large-cc25")
LOCAL_RUN_DIR.mkdir(parents=True,exist_ok=True)
CONFIG_PATH=LOCAL_RUN_DIR/"notebook-config.yaml"
CONFIG_PATH.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
subprocess.run([sys.executable,"scripts/doctor.py","--config",str(CONFIG_PATH)],cwd=project_root,check=True)
print("Ready:",CONFIG_PATH)
'''

ASSET_CODE=r'''
from pathlib import Path
import subprocess, sys

if DATASET in ("phoenix14t","phoenix14") and not (project_root/"assets/resnet18-f37072fd.pth").is_file():
    subprocess.run([sys.executable,"scripts/fetch_assets.py","--resnet","--output",
                    str(project_root/"assets")],cwd=project_root,check=True)
if PHASE in ("slt","gfslt") and not (project_root/"assets/mbart-large-cc25/config.json").is_file():
    subprocess.run([sys.executable,"scripts/fetch_assets.py","--mbart",
        "--mbart-revision","f417e5563320b2cc8aabe4329d986b238809067f",
        "--output",str(project_root/"assets")],cwd=project_root,check=True)
'''

TRAIN_CODE=r'''
import os, shutil, subprocess, sys, time
from pathlib import Path

DRIVE_RUN_DIR.mkdir(parents=True,exist_ok=True)
last_local=LOCAL_RUN_DIR/"last.pt"
best_local=LOCAL_RUN_DIR/"best.pt"
if DRIVE_RUN_DIR.joinpath("last.pt").is_file() and RESUME_LAST and not last_local.exists():
    shutil.copy2(DRIVE_RUN_DIR/"last.pt",last_local)
    for name in ("best.pt","run_metadata.json","vocab.json","history.jsonl","best_dev_metrics.json","best_dev_predictions.json"):
        source=DRIVE_RUN_DIR/name
        if source.is_file(): shutil.copy2(source,LOCAL_RUN_DIR/name)

env=os.environ.copy()
env["HF_HUB_OFFLINE"]="1"
env["TRANSFORMERS_OFFLINE"]="1"
env["PYTHONUNBUFFERED"]="1"
args=[sys.executable,"scripts/launch.py","--config",str(CONFIG_PATH),"--micro-batch",str(MICRO_BATCH),"--workers",str(WORKERS)]
if RESUME_LAST and last_local.is_file():
    args += ["--resume",str(last_local)]
elif PHASE in ("slt","gfslt"):
    source_phase="cslr" if PHASE=="slt" else "tctc"
    source_run=DRIVE_RUN_DIR.parent/f"{DATASET}_{source_phase}_seed{SEED}"/"best.pt"
    if source_run.is_file():
        source_local=LOCAL_RUN_DIR/"initialization.pt"
        shutil.copy2(source_run,source_local)
        args += ["--initialize-from",str(source_local)]
    else:
        print(f"No warm-start checkpoint at {source_run}; this phase starts from scratch.")

log_path=DRIVE_RUN_DIR/"notebook-training.log"
def sync_outputs():
    for name in ("last.pt","best.pt","run_metadata.json","vocab.json","history.jsonl","best_dev_metrics.json","best_dev_predictions.json"):
        source=LOCAL_RUN_DIR/name
        if source.is_file():
            staged=DRIVE_RUN_DIR/(name+".sync-tmp")
            shutil.copy2(source,staged)
            os.replace(staged,DRIVE_RUN_DIR/name)
    if (LOCAL_RUN_DIR/"notebook-config.yaml").is_file():
        shutil.copy2(LOCAL_RUN_DIR/"notebook-config.yaml",DRIVE_RUN_DIR/"notebook-config.yaml")
    if "drive_service" in globals() and "DRIVE_RUN_FOLDER_ID" in globals():
        from scripts.kaggle_drive import sync_checkpoint_directory
        sync_checkpoint_directory(drive_service,LOCAL_RUN_DIR,DRIVE_RUN_FOLDER_ID)

process=None
with log_path.open("a",encoding="utf-8") as log:
    try:
        process=subprocess.Popen(args,cwd=project_root,env=env,stdout=log,stderr=subprocess.STDOUT)
        last_stamp=None
        while process.poll() is None:
            if last_local.is_file():
                stamp=(last_local.stat().st_size,last_local.stat().st_mtime_ns)
                if stamp!=last_stamp:
                    time.sleep(3)
                    if last_local.is_file():
                        current=(last_local.stat().st_size,last_local.stat().st_mtime_ns)
                        if current==stamp:
                            sync_outputs()
                            last_stamp=stamp
                            print("Drive checkpoint synchronized:",time.strftime("%H:%M:%S"),stamp[0],"bytes")
            time.sleep(25)
        sync_outputs()
        if process.returncode!=0:
            tail=log_path.read_text(encoding="utf-8",errors="replace").splitlines()[-60:]
            print("\n".join(tail))
            raise RuntimeError(f"Training stopped with exit code {process.returncode}; last complete epoch is saved when present.")
        print("Training complete. Latest checkpoint and logs are on Drive.")
    except KeyboardInterrupt:
        print("Stopping after saving the last completed epoch...")
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=90)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        sync_outputs()
        print("Saved checkpoint on Drive. Re-run this cell to resume.")

if EVALUATE_TEST_AT_END and best_local.is_file():
    result=subprocess.run([sys.executable,"scripts/launch.py","--config",str(CONFIG_PATH),
        "--mode","eval","--split","test","--checkpoint",str(best_local)],
        cwd=project_root,env=env,check=True)
    sync_outputs()
    for name in ("test_metrics.json","test_predictions.json"):
        source=LOCAL_RUN_DIR/name
        if source.is_file(): shutil.copy2(source,DRIVE_RUN_DIR/name)
'''

colab=[
    md('''# MixSignGraph tái dựng — Google Colab\n\n**Chạy nhanh:** mở notebook trong Colab, bật GPU ở Runtime → Change runtime type → GPU, sửa các đường dẫn ở ô Cấu hình, rồi chọn **Runtime → Run all**. Colab sẽ xin quyền Google Drive lần đầu. Checkpoint được sao lưu lên Drive sau mỗi epoch hoàn tất; chạy lại notebook sẽ tiếp tục từ `last.pt`.\n\nDự án ZIP và dữ liệu cần được chuẩn bị trong Drive trước. PHOENIX/CSL không được notebook tự tải vì giấy phép/quyền truy cập.'''),
    md('''## Chuẩn bị Drive một lần\n\n1. Tải `SignLanguage-Reproduce-portable.zip` từ nơi bạn lưu dự án lên `MyDrive/`.\n2. Đặt dữ liệu theo cấu trúc bên dưới, hoặc chỉnh các biến đường dẫn. Không chia sẻ công khai dataset có hạn chế.'''),
    code('''from google.colab import drive\ndrive.mount('/content/drive')'''),
    md('''## Cấu hình — chỉ sửa ô này\n\nMặc định là PHOENIX14T CSLR. Với CSL-Daily/How2Sign/OpenASL, chuẩn bị đúng JSONL/feature như hướng dẫn. `slt` cần checkpoint CSLR; `gfslt` cần checkpoint TCTC.'''),
    code('''from pathlib import Path\n\nPROJECT_ZIP=Path("/content/drive/MyDrive/SignLanguage-Reproduce-portable.zip")\nWORKSPACE=Path("/content")\nDATA_ROOT=Path("/content/drive/MyDrive/SignLanguage-Data/phoenix14t")\nMANIFEST_ROOT=Path("/content/drive/MyDrive/SignLanguage-Manifests/phoenix14t")\nDRIVE_RUNS_ROOT=Path("/content/drive/MyDrive/SignLanguage-Runs")\nDATASET="phoenix14t"  # phoenix14t, phoenix14, csl_daily, how2sign, openasl\nPHASE="cslr"           # cslr, tctc, slt, gfslt\nSEED=0\nGLOBAL_BATCH=6\nMICRO_BATCH=1\nWORKERS=2\nEPOCHS=50\nRESUME_LAST=True\nEVALUATE_TEST_AT_END=True\nRUN_ID=f"{DATASET}_{PHASE}_seed{SEED}"\nLOCAL_RUN_DIR=Path("/content/SignLanguage-Reproduce/runs")/RUN_ID\nDRIVE_RUN_DIR=DRIVE_RUNS_ROOT/RUN_ID\nCONFIG_NAME=f"{DATASET}_{PHASE}.yaml"\nif GLOBAL_BATCH%(MICRO_BATCH)!=0: raise ValueError("global batch must divide by micro-batch")'''),
    md('''## Cài mã và thư viện\n\nNotebook chỉ cài thư viện vào runtime tạm của Colab. Nó không sửa máy tính cá nhân.'''),
    code(SETUP_CODE),
    code('''%pip install -q -r /content/SignLanguage-Reproduce/requirements-repro.txt'''),
    code(ASSET_CODE),
    md('''## Dữ liệu\n\nPHOENIX14T mặc định cần `SignLanguage-Data/phoenix14t/features/fullFrame-210x260px/{train,dev,test}/...` và CSV ở `annotations/manual/`. Checkpoint và manifest lưu trên Drive; đọc ảnh trực tiếp từ Drive có thể chậm. Nếu dữ liệu nằm chỗ khác, sửa `DATA_ROOT`.'''),
    code(DATA_CODE),
    md('''## Trọng số và khởi chạy\n\nTải ResNet ImageNet/mBART cần bật internet cho phiên Colab. mBART chỉ tải cho `slt` hoặc `gfslt`. Quá trình lưu checkpoint theo epoch; nếu dừng giữa epoch thì notebook tiếp tục từ epoch hoàn tất gần nhất. Đừng xóa `SignLanguage-Runs/<RUN_ID>/last.pt` nếu muốn tiếp tục.'''),
    code(TRAIN_CODE),
    md('''## Lưu ý về kết quả\n\n`best.pt` được chọn bằng tập dev; đánh giá test được chạy sau huấn luyện. Ghi nhận đây là kết quả của bản tái dựng, chưa được xác nhận bằng dữ liệu/checkpoint của tác giả. Nếu Colab ngắt, mở notebook lại, mount cùng Drive, giữ nguyên cấu hình và chạy lại ô khởi chạy.''')
]

kaggle=[
    md('''# MixSignGraph tái dựng — Kaggle + Google Drive\n\nNotebook đọc ZIP mã, tải gói dữ liệu, lưu checkpoint và kết quả vào **cùng thư mục Drive**. Bật **GPU** và **Internet** trong Kaggle, điền hai Drive file IDs và folder ID, cấp OAuth một lần theo `notebooks/DRIVE_SETUP.md`, rồi chạy **Run All**. Checkpoint được đồng bộ lên Drive sau mỗi epoch hoàn tất; chạy notebook lại sẽ tiếp tục từ checkpoint mới nhất.\n\nKaggle không gắn Drive như Colab; notebook dùng Google Drive API, nên bước OAuth + Kaggle Secret là bắt buộc cho Drive riêng tư.'''),
    md('''## Chuẩn bị Drive/Kaggle một lần\n\n1. Đặt `SignLanguage-Reproduce-portable.zip` và một ZIP dữ liệu đã chuẩn bị trong cùng folder Drive riêng tư. ZIP dữ liệu phải giải nén ra thư mục chứa frame/annotation như mục 5 trong hướng dẫn chính.\n2. Chạy helper OAuth một lần theo `notebooks/DRIVE_SETUP.md`, lưu JSON làm Kaggle Secret `GOOGLE_DRIVE_OAUTH_JSON`.\n3. Copy ID của folder Drive, ZIP mã và ZIP dữ liệu vào ô cấu hình. Không đưa client secret/token vào notebook hoặc output.'''),
    md('''## Cấu hình\n\nMặc định dùng đường dẫn Dataset mẫu `/kaggle/input/phoenix14t`. Thay bằng slug Kaggle Dataset thật của bạn. `CHECKPOINT_INPUT` để trống ở lần đầu; khi tiếp tục, trỏ tới `last.pt` trong Notebook Output đã thêm làm Input.'''),
    code('''from pathlib import Path\n\nWORKSPACE=Path("/kaggle/working")\nDRIVE_WORKSPACE_FOLDER_ID="PASTE_SHARED_DRIVE_FOLDER_ID"\nDRIVE_PROJECT_FILE_ID="PASTE_PROJECT_ZIP_FILE_ID"\nDRIVE_DATA_FILE_ID="PASTE_DATA_ZIP_FILE_ID"\nDATA_ARCHIVE_NAME="phoenix14t.zip"\nDATA_ROOT=WORKSPACE/"dataset"\nMANIFEST_ROOT=WORKSPACE/"manifests/phoenix14t"\nOUTPUT_ROOT=WORKSPACE/"SignLanguage-Runs"\nDATASET="phoenix14t"  # phoenix14t, phoenix14, csl_daily, how2sign, openasl\nPHASE="cslr"           # cslr, tctc, slt, gfslt\nSEED=0\nGLOBAL_BATCH=6\nMICRO_BATCH=1\nWORKERS=2\nEPOCHS=50\nRESUME_LAST=True\nEVALUATE_TEST_AT_END=True\nRUN_ID=f"{DATASET}_{PHASE}_seed{SEED}"\nPROJECT_ROOT=WORKSPACE/"SignLanguage-Reproduce"\nLOCAL_RUN_DIR=OUTPUT_ROOT/RUN_ID\nCONFIG_NAME=f"{DATASET}_{PHASE}.yaml"\nif "PASTE_" in DRIVE_WORKSPACE_FOLDER_ID+DRIVE_PROJECT_FILE_ID+DRIVE_DATA_FILE_ID:\n    raise ValueError("Fill the 3 Google Drive IDs")\nif GLOBAL_BATCH%MICRO_BATCH: raise ValueError("global batch must divide by micro-batch")'''),
    md('''## Kiểm tra và cài thư viện\n\nDữ liệu Input chỉ đọc; checkpoint ghi trong `/kaggle/working` và được giữ trong Output khi tạo Version. Không đặt checkpoint vào `/kaggle/input`.'''),
    code('''%pip install -q google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2\nfrom pathlib import Path\nimport sys\nPROJECT_ROOT=WORKSPACE/"SignLanguage-Reproduce"\nif not PROJECT_ROOT.exists():\n    # A small bootstrap copy contains the Drive API helper before the main project ZIP is downloaded.\n    raise RuntimeError("Set up the project ZIP in Drive and run the Drive setup cell below.")'''),
    code(KAGGLE_DRIVE_CODE),
    code('''import zipfile, shutil, subprocess, sys\nif not PROJECT_ROOT.exists():\n    WORKSPACE.mkdir(parents=True,exist_ok=True)\n    with zipfile.ZipFile(PROJECT_ZIP) as archive:\n        for entry in archive.infolist():\n            target=(WORKSPACE/entry.filename).resolve()\n            if not target.is_relative_to(WORKSPACE.resolve()): raise ValueError("Unsafe project ZIP path")\n        archive.extractall(WORKSPACE)\nif not (PROJECT_ROOT/"scripts/launch.py").is_file(): raise FileNotFoundError("Unexpected project ZIP layout")\nsys.path.insert(0,str(PROJECT_ROOT))\nfrom scripts.kaggle_drive import download_file,ensure_folder,find_child,upsert_file\nsubprocess.run([sys.executable,"scripts/verify_bundle.py"],cwd=PROJECT_ROOT,check=True)\nDRIVE_RUNS_FOLDER_ID=ensure_folder(drive_service,DRIVE_WORKSPACE_FOLDER_ID,"SignLanguage-Runs")\nrun_match=find_child(drive_service,DRIVE_RUNS_FOLDER_ID,RUN_ID,"application/vnd.google-apps.folder")\nif len(run_match)>1: raise ValueError("Duplicate Drive run folders; resolve them first")\nDRIVE_RUN_FOLDER_ID=run_match[0]["id"] if run_match else ensure_folder(drive_service,DRIVE_RUNS_FOLDER_ID,RUN_ID)\narchive_path=WORKSPACE/DATA_ARCHIVE_NAME\ndownload_file(drive_service,DRIVE_DATA_FILE_ID,archive_path)\ndata_extract=WORKSPACE/"dataset"\nif not data_extract.exists():\n    with zipfile.ZipFile(archive_path) as archive:\n        for entry in archive.infolist():\n            target=(data_extract/entry.filename).resolve()\n            if not target.is_relative_to(data_extract.resolve()): raise ValueError("Unsafe dataset ZIP path")\n        archive.extractall(data_extract)\nprint("Project/data loaded; checkpoints will sync to the same Drive folder.")\n'''),
    code('''%pip install -q -r /kaggle/working/SignLanguage-Reproduce/requirements-repro.txt'''),
    md('''## Chuẩn bị manifest và cấu hình'''),
    code(DATA_CODE.replace('LOCAL_RUN_DIR','LOCAL_RUN_DIR')),
    md('''## Tải trọng số nếu cần\n\nMã sẽ tải ResNet/mBART vào vùng làm việc Kaggle. Cần bật Internet. Notebook có thể tải lại asset khi phiên Kaggle được tạo mới.'''),
    code(ASSET_CODE),
    md('''## Checkpoint dùng chung trên Drive\n\nNotebook tìm `last.pt` và `best.pt` trong thư mục run trên cùng Drive. Khi bắt đầu sẽ tải checkpoint mới nhất; cuối mỗi epoch sẽ tải lên checkpoint mới.'''),
    md('''## Huấn luyện, lưu và tiếp tục\n\nNotebook đồng bộ checkpoint sau mỗi epoch hoàn tất trong `/kaggle/working/SignLanguage-Runs/<RUN_ID>`. Nếu dừng phiên, chọn **Save Version** để xuất Output. Ở phiên mới, thêm Output cũ làm Input, đặt `CHECKPOINT_INPUT`, giữ nguyên dataset/config và chạy lại.'''),
    code(TRAIN_CODE.replace('DRIVE_RUN_DIR','OUTPUT_RUN_DIR').replace('DRIVE_RUN_DIR.mkdir','OUTPUT_RUN_DIR.mkdir')
         .replace('DRIVE_RUN_DIR.joinpath','OUTPUT_RUN_DIR.joinpath')
         .replace('DRIVE_RUN_DIR/','OUTPUT_RUN_DIR/')
         .replace('DRIVE_RUN_DIR.parent','OUTPUT_RUN_DIR.parent')
         .replace('log_path=OUTPUT_RUN_DIR/"notebook-training.log"','log_path=OUTPUT_RUN_DIR/"notebook-training.log"')
         .replace('source_run=OUTPUT_RUN_DIR.parent', 'source_run=OUTPUT_RUN_DIR.parent')),
    md('''## Drive chung và checkpoint\n\nCả dữ liệu, source ZIP, `last.pt`, `best.pt`, log và kết quả đều nằm dưới folder Drive bạn đã chỉ định. Dataset ZIP sẽ tải lại vào `/kaggle/working` khi Kaggle tạo runtime mới; cần đủ dung lượng trống. Nếu dataset lớn hơn dung lượng hoặc quota/API Drive không cho phép tải, dùng Kaggle Dataset làm cache đầu vào rồi điều chỉnh notebook. Không chia sẻ Kaggle Secret.''')
]

save('MixSignGraph_Colab.ipynb',colab,{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.10'},
    'colab':{'name':'MixSignGraph_Colab.ipynb','provenance':[],'gpuType':'T4','accelerator':'GPU'}
})
save('MixSignGraph_Kaggle.ipynb',kaggle,{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},
    'kaggle':{'accelerator':'GPU','isInternetOn':True,'show_gpu_on_off':True}
})
print(f'Created notebooks in {OUT}')
