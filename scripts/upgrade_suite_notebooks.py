"""Apply common campaign cells after create_notebooks.py and finalize_notebooks.py."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def source(cell, text): cell['source']=text.strip().splitlines(keepends=True)


PREPARE='''import subprocess, sys
from pathlib import Path
manifest_dir=Path(MANIFEST_ROOT)
subprocess.run([sys.executable,"scripts/prepare_data.py","--dataset","phoenix14t",
    "--annotations",str(DATA_ROOT/"annotations/manual"),"--data-root",str(DATA_ROOT),
    "--output",str(manifest_dir),"--overwrite"],cwd=project_root,check=True)
for kind in ("swin_t","pvig_tiny"):
    subprocess.run([sys.executable,"scripts/fetch_backbones.py","--backbone",kind],cwd=project_root,check=True)
import torch, torchvision
if not torch.cuda.is_available(): raise RuntimeError("Enable GPU before running the experiment suite")
print("GPU:",[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("torch:",torch.__version__,"torchvision:",torchvision.__version__)
'''

TRAIN='''import os, subprocess, sys, tempfile, time
from pathlib import Path
env=os.environ.copy()
env["HF_HUB_OFFLINE"]="1"; env["TRANSFORMERS_OFFLINE"]="1"; env["PYTHONUNBUFFERED"]="1"
suite_output=project_root/"runs"/CAMPAIGN_NAME
args=[sys.executable,"scripts/run_suite.py","--data-root",str(DATA_ROOT),
      "--manifests",str(MANIFEST_ROOT),"--output",str(suite_output),
      "--minutes",str(max(3,SESSION_MINUTES-(time.monotonic()-NOTEBOOK_START)/60)),"--epochs",str(EPOCHS),"--workers",str(WORKERS)]
if ONLY_EXPERIMENTS: args += ["--only",ONLY_EXPERIMENTS]
credentials_file=None
try:
    if "drive_service" in globals():
        from kaggle_secrets import UserSecretsClient
        fd,name=tempfile.mkstemp(prefix="signlanguage-private-",suffix=".json",dir="/tmp")
        credentials_file=Path(name)
        with os.fdopen(fd,"w") as f:
            f.write(UserSecretsClient().get_secret("GOOGLE_DRIVE_OAUTH_JSON"))
        os.chmod(credentials_file,0o600)
        env["SIGN_DRIVE_CREDENTIALS"]=str(credentials_file)
        args += ["--drive-folder",DRIVE_WORKSPACE_FOLDER_ID]
    else:
        args += ["--persist-root",str(SHARED_DRIVE_ROOT/"runs"/CAMPAIGN_NAME)]
    process=subprocess.Popen(args,cwd=project_root,env=env)
    try:
        code=process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try: process.wait(timeout=120)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        raise
    if code: raise RuntimeError("Campaign stopped with an error; inspect output. Saved checkpoints remain available.")
    print("Session finished. Run All again to resume pending experiments; see suite_summary.csv on Drive.")
finally:
    if credentials_file: credentials_file.unlink(missing_ok=True)
'''


def upgrade():
    for platform in ('Colab','Kaggle'):
        path=ROOT/f'notebooks/MixSignGraph_{platform}.ipynb'
        nb=json.loads(path.read_text(encoding='utf-8')); cells=nb['cells']
        ci=4 if platform=='Colab' else 3
        source(cells[0],f'# PHOENIX14T–CSLR: 68 lượt chạy — {platform}\n\nRun All chuẩn bị dữ liệu chung rồi chạy 67 cấu hình định lượng và một baseline cho so sánh gloss định tính. Mỗi thí nghiệm có checkpoint riêng trên Drive. Chạy lại tiếp tục thí nghiệm dở, bỏ qua thí nghiệm hoàn tất. Đây là tái dựng theo paper, có giả định công khai trong `docs/PHOENIX14T_SUITE.md`; chưa chứng minh đạt số paper.')
        common='''
DATASET="phoenix14t"
PHASE="cslr"
WORKERS=2
EPOCHS=50
SESSION_MINUTES=600  # entire notebook session, including preparation
ONLY_EXPERIMENTS=""  # empty: all 68; e.g. "main,modules_none"
CAMPAIGN_NAME="phoenix14t_cslr_suite_seed0"
NOTEBOOK_START=time.monotonic()
'''
        if platform=='Colab':
            config='''from pathlib import Path
import time
SHARED_DRIVE_ROOT=Path("/content/drive/MyDrive/SignLanguage-Reproduction")
PROJECT_ZIP=SHARED_DRIVE_ROOT/"SignLanguage-Reproduce-portable.zip"
WORKSPACE=Path("/content")
DATA_ROOT=Path("/content/phoenix14t")
PHOENIX_ARCHIVE=SHARED_DRIVE_ROOT/"phoenix-2014-T.v3.tar.gz"
MANIFEST_ROOT=SHARED_DRIVE_ROOT/"manifests/phoenix14t"
'''+common
        else:
            config='''from pathlib import Path
import time
WORKSPACE=Path("/kaggle/working")
DRIVE_WORKSPACE_FOLDER_ID="1RpGbT-9SXIr21OVfVmRRn4K_XhRulCeL"
DATA_ARCHIVE_NAME="phoenix-2014-T.v3.tar.gz"
DATA_ROOT=Path("/tmp/phoenix14t")
MANIFEST_ROOT=WORKSPACE/"manifests/phoenix14t"
PROJECT_ROOT=WORKSPACE/"SignLanguage-Reproduce"
project_root=PROJECT_ROOT
'''+common
        source(cells[ci],config)
        if platform=='Colab':
            source(cells[3],'## Cấu hình\n\nMặc định chạy đủ 68 lượt PHOENIX14T–CSLR. Giữ EPOCHS=50; SESSION_MINUTES giới hạn mỗi phiên. Nếu đổi epoch, đổi CAMPAIGN_NAME để giữ tách biệt với chiến dịch 50 epoch.')
            # Always extract updated source, even in an already-running session.
            bootstrap=''.join(cells[6]['source']).replace('if not project_root.exists():','if True:  # refresh source from the current Drive ZIP')
            source(cells[6],bootstrap)
            source(cells[7],'%pip install -q -r /content/SignLanguage-Reproduce/requirements-notebook-cslr.txt')
            source(cells[11],PREPARE)
            source(cells[13],TRAIN)
            source(cells[12],'## Hàng đợi thí nghiệm\n\nKết quả ở Drive `runs/phoenix14t_cslr_suite_seed0/`, mỗi cấu hình một thư mục. Checkpoint lưu sau mỗi 200 optimizer updates và khi kết thúc epoch/dừng phiên có kiểm soát. Phiên bị nền tảng ngắt đột ngột sẽ tiếp tục từ checkpoint đã commit gần nhất.')
            source(cells[14],'## Kết quả\n\nĐọc `suite_summary.csv`: pending là chưa hoàn tất, measured reconstruction là số đo thật. Không có số đo thì không tự điền số paper. Phiên mới có thể phải giải nén lại archive đã cache trên Drive; không cần tải lại từ RWTH.')
        else:
            source(cells[2],'## Cấu hình\n\nĐã điền ID thư mục Drive bạn cung cấp. Mặc định chạy đủ 68 lượt PHOENIX14T–CSLR qua nhiều phiên có checkpoint.')
            source(cells[8],'%pip install -q -r /kaggle/working/SignLanguage-Reproduce/requirements-notebook-cslr.txt')
            setup=''.join(cells[7]['source'])
            if 'matches=find_child(drive_service,DRIVE_RUNS_FOLDER_ID,RUN_ID' in setup:
                start=setup.index('matches=find_child(drive_service,DRIVE_RUNS_FOLDER_ID,RUN_ID')
                end=setup.index('archive_matches=find_child')
                setup=setup[:start]+setup[end:]
            if 'if archive_path.is_file():' not in setup:
                setup=setup.replace('if archive_matches:', 'if archive_path.is_file():\n    print("Reusing the archive already downloaded in this Kaggle session.")\nelif archive_matches:')
            source(cells[7],setup)
            source(cells[12],PREPARE)
            source(cells[13],'## Checkpoint từng thí nghiệm\n\nHàng đợi tự tạo `runs/phoenix14t_cslr_suite_seed0/<experiment>/` trên cùng Drive. Mỗi checkpoint có checksum và ít nhất hai thế hệ để khôi phục nếu upload bị ngắt. Token chỉ nằm trong file tạm riêng của runtime, không lưu vào Output.')
            source(cells[15],TRAIN)
            source(cells[16],'## Tiếp tục\n\nChạy lại Run All: thí nghiệm hoàn tất được bỏ qua, thí nghiệm dở khôi phục checkpoint gần nhất. Mặc định không đổi batch/LR/epoch khi số GPU thay đổi. `suite_summary.csv` trên Drive tổng hợp số đo và độ lệch paper. Archive và checkpoint cần dung lượng Drive đủ; code dừng rõ nếu không lưu được checkpoint.')
        cells[:]=[cell for cell in cells if cell.get('metadata',{}).get('id')!='local-monitoring-note']
        cells.append({'cell_type':'markdown','metadata':{'id':'local-monitoring-note'},'source':[
            '## Bản cập nhật không CI\n',
            'Notebook vẫn dùng Run all và thư mục Drive chung; không cần thêm secret W&B. Khi code đổi, thay ZIP mới trong Drive trước phiên chạy mới. Không chạy Colab và Kaggle đồng thời vào cùng chiến dịch.\n',
            'Máy Linux của cô có luồng riêng: chạy `setup_machine.py` một lần, sau đó `run.py` để tự lấy bản sửa từ GitHub Private và gửi lỗi lên W&B. Xem `docs/MAY_CO_VA_LAPTOP.md`. Cả ba nơi dùng cùng catalog PHOENIX14T–CSLR; chưa có số đo thật để khẳng định tái lập paper.\n']})
        for i,cell in enumerate(cells):
            if cell['cell_type']=='code' and not ''.join(cell['source']).lstrip().startswith('%'):
                compile(''.join(cell['source']),f'{platform}:cell{i}','exec')
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        print(f'Upgraded suite notebook: {path.name}')


if __name__=='__main__': upgrade()
