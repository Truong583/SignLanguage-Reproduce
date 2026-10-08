# Hướng dẫn chạy và chuyển máy — MixSignGraph

## 1. Đọc trước khi chạy

Đây là **bộ mã tái dựng có kiểm thử**, dựa trên repository của tác giả ở commit `af5e8475d755b9d2e92c0142c8b7084651c3a4ee`, phục vụ làm lại thí nghiệm MixSignGraph, NeurIPS 2025.

**Chưa tái lập được kết quả paper.** Mã công bố thiếu đường chạy HSG hoàn chỉnh, thư mục tiền xử lý và các tài sản mBART đã rút gọn. Không thể suy ra chính xác toàn bộ thí nghiệm của tác giả chỉ bằng viết thêm code. Xem [báo cáo đối chiếu](docs/AUDIT.md) trước khi dùng kết quả để viết bài.

Đã bổ sung: HSG hai chiều; sửa đồ thị thời gian; CSLR, TCTC và Sign2Text; đọc dữ liệu RGB hoặc đặc trưng; huấn luyện một/nhiều GPU; tích lũy gradient; checkpoint; đánh giá; Docker; kiểm tra dữ liệu; cấu hình; checksum. Mã gốc vẫn nằm trong các thư mục ban đầu.

Chưa kèm: dataset thật, trọng số ResNet/mBART, mô hình lemmatization, đặc trưng I3D của tác giả. Các file trong `runs/integration-*` chỉ là dữ liệu giả và checkpoint kiểm thử, tuyệt đối không dùng làm kết quả nghiên cứu.

Mục tiêu thực tế đầu tiên: chạy **PHOENIX14T CSLR**, kiểm tra WER dev/test, sau đó mới mở rộng sang SLT/TCTC. Mốc tham khảo của paper: dev 16.7%, test 19.0% WER. Đạt/không đạt mốc phải dựa trên phép đo thật.

**Máy Linux của cô tự lấy bản sửa và gửi lỗi:** cấu hình một lần bằng `python3 setup_machine.py`, sau đó chạy `python3 run.py`; xem [hướng dẫn laptop và máy cô](docs/MAY_CO_VA_LAPTOP.md). Không có CI. Chạy một lượt không giám sát dùng `python run.py --once`. Mặc định là chiến dịch 68 lượt PHOENIX14T–CSLR: chuẩn bị dữ liệu/backbones, huấn luyện/resume, bỏ qua run hoàn tất, đánh giá và so sánh số paper. Local và hai notebook dùng chung catalog. Điều kiện máy đích ở [CHAY_MOT_LENH.md](CHAY_MOT_LENH.md); danh sách và giả định ở [PHOENIX14T_SUITE.md](docs/PHOENIX14T_SUITE.md). Muốn chỉ chạy một config, thêm `--single`.

## 2. Cấu trúc thư mục

```text
SignLanguage-Reproduce/
  HUONG_DAN_CHAY.md          tài liệu này
  Dockerfile                môi trường Linux GPU
  requirements-repro.txt    phiên bản phụ thuộc trực tiếp
  configs_repro/            cấu hình độc lập cho từng thí nghiệm
  repro/                    mã chạy bổ sung
  repro/vendor/             bản sao thành phần gốc đã thích nghi
  scripts/                  chuẩn bị dữ liệu, launcher, doctor, kiểm tra
  tests/                    kiểm thử tính đúng đắn
  docs/                     paper, báo cáo khác biệt, mốc kết quả
  reports/                  kết quả kiểm thử thực tế
  data/                     dữ liệu và manifest do bạn chuẩn bị
  assets/                   trọng số/tokenizer/mô hình NLP
  runs/                     log, checkpoint, dự đoán
  MixSignGraph/             mã tác giả, giữ nguyên
  SignGraph/ SLTpose/ ...   các thành phần còn lại của repository
```

Tất cả lệnh sau chạy từ thư mục `SignLanguage-Reproduce`, trong **Bash trên Linux**. Trên Windows, dùng Docker Desktop với backend WSL2 và chạy lệnh trong WSL; nên đặt dữ liệu trên filesystem Linux của WSL để đọc hàng triệu ảnh nhanh hơn. Windows Python tại máy chuẩn bị chỉ được dùng để kiểm thử, không phải môi trường đa GPU đích.

## 3. Điều kiện máy đích và cách bảo vệ môi trường

Máy cần có NVIDIA GPU, driver tương thích image CUDA, Docker đang hoạt động và NVIDIA Container Toolkit đã được quản trị viên cấu hình. Dự án **không cài lại driver, CUDA hệ thống, không khởi động lại Docker và không sửa môi trường Python của máy chủ**.

Kiểm tra trước:

```bash
nvidia-smi
docker version
docker info
df -h .
free -h
```

Nếu thiếu Docker/Container Toolkit, dùng hướng dẫn chính thức của [Docker](https://docs.docker.com/engine/install/) và [NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), để quản trị viên thiết lập. Không tự chạy các lệnh restart dịch vụ trên máy đang có công việc của người khác.

Script `scripts/container.sh` mặc định:

- GPU được chỉ định rõ bằng `GPU_IDS`; mặc định chỉ GPU 0.
- Giới hạn RAM 16 GB, CPU 4 lõi, shared memory 2 GB; tắt swap của container.
- Chạy bằng UID/GID người dùng, bỏ Linux capabilities, không privileged, không mount Docker socket.
- Mã nguồn, dữ liệu và assets chỉ đọc; chỉ `runs` được ghi. Chỉ bước chuẩn bị mới chủ động bật quyền ghi tương ứng.
- Network tắt khi huấn luyện; không truyền credential hoặc mount thư mục home của máy chủ.
- Container tự xóa khi kết thúc; checkpoint và log còn trong `runs`.

Docker không bảo đảm tuyệt đối không ảnh hưởng hiệu năng máy: GPU vẫn tiêu thụ VRAM/điện, dữ liệu/checkpoint vẫn dùng dung lượng đĩa. Hãy chọn GPU được phân bổ cho bạn, chừa RAM cho máy chủ và giám sát dung lượng. Không có giới hạn VRAM cứng trong wrapper này; OOM thì giảm micro-batch, không chiếm thêm GPU của người khác.

## 4. Build môi trường

```bash
docker build -t mixsigngraph-repro:local .
GPU_IDS=none bash scripts/container.sh python -m repro.smoke --device cpu
GPU_IDS=0 bash scripts/container.sh python -m repro.smoke --device cuda --rgb
```

Build có chạy bộ unit test và `pip check`. **Docker build chưa được chạy xác nhận tại máy chuẩn bị vì Docker daemon chưa hoạt động.** Hãy coi ba lệnh trên là bước nghiệm thu bắt buộc tại máy đích.

Image mặc định: PyTorch 2.5.1 / CUDA 12.4. Paper dùng PyTorch 1.11; bản đóng gói chọn môi trường mới hơn để dễ vận hành, nên đây là một khác biệt phải ghi vào báo cáo. GPU thế hệ mới không được image này hỗ trợ cần image CUDA/PyTorch tương ứng. Có thể thử profile mới hơn:

```bash
docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime -t mixsigngraph-repro:cu128 .
IMAGE=mixsigngraph-repro:cu128 GPU_IDS=0 bash scripts/container.sh python -m repro.smoke --device cuda --rgb
```

Không có cam kết mọi GPU đều hỗ trợ. Dùng `nvidia-smi` và smoke test để xác nhận, không tự cập nhật driver trên máy dùng chung. Các bản build khác nhau cần lưu riêng log và image.

## 5. Chuẩn bị dữ liệu PHOENIX14T

Lấy dataset theo hướng dẫn và điều kiện sử dụng tại [trang RWTH chính thức](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/). Không dùng nhầm PHOENIX14 với PHOENIX14T.

Giải nén để có dạng sau (giữ nguyên train/dev/test chính thức):

```text
data/phoenix14t/
  features/fullFrame-210x260px/train/<sample-name>/*.png
  features/fullFrame-210x260px/dev/<sample-name>/*.png
  features/fullFrame-210x260px/test/<sample-name>/*.png
  annotations/manual/PHOENIX-2014-T.train.corpus.csv
  annotations/manual/PHOENIX-2014-T.dev.corpus.csv
  annotations/manual/PHOENIX-2014-T.test.corpus.csv
```

Nếu gói tải có thêm tầng thư mục, đặt `data_root` trỏ đúng tầng chứa `features`. Không tự đổi tên mẫu, không trộn split. Trình chuẩn bị đọc cột `name`, `orth`/`annotation`/`gloss`, và `translation`/`text`; CSV phân cách `|`.

```bash
GPU_IDS=none DATA_MODE=rw bash scripts/container.sh \
  python scripts/prepare_data.py --dataset phoenix14t \
  --annotations data/phoenix14t/annotations/manual \
  --data-root data/phoenix14t --output data/manifests/phoenix14t
```

Nếu đường dẫn frame khác, thêm `--frames-template 'features/fullFrame-210x260px/{split}/{name}'` theo cấu trúc thật. Nếu tên CSV khác, dùng `--csv-template 'ten-file.{split}.csv'`. Chương trình dừng khi thiếu ảnh, đường dẫn sai hoặc trùng ID; không tự bỏ mẫu. Số mẫu dự kiến PHOENIX14T: train 7096, dev 519, test 642.

## 6. Chuẩn bị trọng số

Chỉ bước tải này cần mạng và quyền ghi `assets`:

```bash
GPU_IDS=none NETWORK=bridge ASSET_MODE=rw bash scripts/container.sh \
  env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python scripts/fetch_assets.py --resnet
```

File kết quả: `assets/resnet18-f37072fd.pth`; script kiểm tra prefix SHA-256 do PyTorch công bố và lưu hash đầy đủ. CSLR không cần mBART.

Cho SLT, mBART CC25 là một lựa chọn phù hợp mô tả paper nhưng **không phải bộ mBART đã rút gọn của tác giả**. Tải revision được ghim:

```bash
GPU_IDS=none NETWORK=bridge ASSET_MODE=rw bash scripts/container.sh \
  env HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python scripts/fetch_assets.py --mbart \
  --mbart-revision f417e5563320b2cc8aabe4329d986b238809067f
```

Bản mBART đầy đủ có thể cần nhiều VRAM hơn bản tác giả; 3×24 GB trong paper không tự động chứng minh bản này chạy vừa cùng phần cứng. Không tự thay tokenizer rút gọn vào model đầy đủ hoặc ngược lại.

## 7. Kiểm tra trước khi huấn luyện

```bash
GPU_IDS=0 bash scripts/container.sh \
  python scripts/doctor.py --config configs_repro/phoenix14t_cslr.yaml --full

GPU_IDS=0,1,2 bash scripts/container.sh \
  python scripts/launch.py --config configs_repro/phoenix14t_cslr.yaml --plan
```

`--full` mở mọi ảnh để phát hiện ảnh hỏng; có thể chậm. Doctor kiểm tra split, đường dẫn, số mẫu, nhãn không rỗng, độ dài CTC và token ngoài từ vựng train. Nó không chứng minh dataset đúng bản tác giả; vẫn cần đối chiếu nguồn, số mẫu và checksum.

Không tự cắt video/nhãn để tránh lỗi. Với nhãn CTC lặp liền nhau cần thêm bước blank; nếu dữ liệu không đủ bước thời gian, script báo lỗi. Temporal augmentation có thể làm video ngắn thêm; khi gặp lỗi phải kiểm tra quy trình tiền xử lý, không âm thầm bỏ nhãn.

## 8. Chạy với số GPU khác nhau

```bash
# 1 GPU
GPU_IDS=0 MEMORY=16g CPUS=4 bash scripts/container.sh \
  python scripts/launch.py --config configs_repro/phoenix14t_cslr.yaml

# 3 GPU, cấu hình gần bố trí 3×2 mẫu của repository hơn
GPU_IDS=0,1,2 MEMORY=32g CPUS=8 SHM_SIZE=4g bash scripts/container.sh \
  python scripts/launch.py --config configs_repro/phoenix14t_cslr.yaml --micro-batch 2
```

Chỉ chạy một trong các lệnh cho một `output`. Muốn chạy thí nghiệm khác, sao chép YAML và đổi `output`, `seed` rồi lưu rõ cấu hình.

Quy tắc: `global_batch = số_GPU × micro_batch × accumulation`. Mặc định global batch 6 suy từ config công bố 3 GPU × 2 mẫu, **không phải batch được paper ghi rõ**.

| GPU nhìn thấy | GPU tự chọn | Micro-batch mặc định | Accumulation | Batch hiệu dụng |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 6 | 6 |
| 2 | 2 | 1 | 3 | 6 |
| 3 | 3 | 1 | 2 | 6 |
| 4 | 3 | 1 | 2 | 6 |
| 8 | 6 | 1 | 1 | 6 |

Launcher chọn số GPU là ước của batch, không tự đổi learning rate hoặc batch để dùng hết GPU. `--gpus N` ép số lượng; cấu hình không chia hết sẽ báo lỗi. Số `GPU_IDS` là chỉ số trên host; trong container GPU được đánh số lại từ 0. Với 4/8 GPU, có thể giữ bố trí 3 GPU × 2 mẫu bằng `--gpus 3 --micro-batch 2`.

Giữ batch hiệu dụng **không làm các lần chạy tương đương tuyệt đối**: BatchNorm/SyncBatchNorm thấy micro-batch khác nhau, thứ tự số học và RNG khác nhau. Để so sánh sát hơn, ưu tiên cùng GPU count, micro-batch, image, seed và dataset. Mặc định an toàn về VRAM là micro-batch 1; không có cơ chế tự đoán VRAM vừa cho mọi video.

Mẹo tối ưu có kiểm soát: dùng SSD cho frame; bắt đầu 2 data workers mỗi GPU; giảm về 0 nếu RAM/shared memory thiếu; tăng dần micro-batch khi đã đo VRAM và vẫn chia hết global batch. `workers` được tính **mỗi GPU**, tổng CPU/RAM tăng theo số tiến trình. Không bật frame subsampling, hạ 224px hoặc đổi BF16 chỉ để nhanh rồi coi kết quả tương đương.

## 9. Checkpoint, tiếp tục và đánh giá

Trong `runs/phoenix14t_cslr_seed0` có `last.pt`, `best.pt`, `history.jsonl`, `run_metadata.json`, vocabulary, dự đoán và metric dev. Chỉ giữ last/best để tránh đầy ổ đĩa. Ghi checkpoint theo kiểu file tạm rồi thay thế; vẫn cần dung lượng cho file tạm.

Tiếp tục từ **cuối epoch đã lưu**:

```bash
GPU_IDS=0,1,2 MEMORY=32g CPUS=8 bash scripts/container.sh \
  python scripts/launch.py --config configs_repro/phoenix14t_cslr.yaml --micro-batch 2 \
  --resume runs/phoenix14t_cslr_seed0/last.pt
```

Được đổi số GPU miễn thỏa batch; RNG được phục hồi từng rank nếu count giữ nguyên. Nếu đổi count, khởi tạo lại RNG có ghi thông báo; không hứa bitwise reproducibility. Dừng giữa epoch thì chạy lại epoch chưa hoàn tất. Chỉ load checkpoint tự tạo hoặc từ nguồn bạn tin cậy, vì checkpoint có trạng thái Python optimizer/RNG.

Sau khi đã chọn `best.pt` bằng dev, đánh giá test một lần:

```bash
GPU_IDS=0 bash scripts/container.sh \
  python scripts/launch.py --config configs_repro/phoenix14t_cslr.yaml \
  --mode eval --split test --checkpoint runs/phoenix14t_cslr_seed0/best.pt

GPU_IDS=none bash scripts/container.sh python scripts/compare_results.py \
  --experiment phoenix14t_cslr \
  --dev runs/phoenix14t_cslr_seed0/best_dev_metrics.json \
  --test runs/phoenix14t_cslr_seed0/test_metrics.json
```

WER có riêng `conv` và `sequence`. Config chọn trước `sequence`; không lấy đầu tốt nhất dựa trên test. Decoder beam 10 không language model, triển khai Python để tránh phụ thuộc C++ ctcdecode; có thể chậm trên tập lớn. Beam 1 là greedy, chỉ dùng kiểm tra nhanh và phải ghi là thay đổi đánh giá.

## 10. Chạy SLT dựa trên gloss

Sau khi CSLR đã chạy ổn và có mBART local:

```bash
GPU_IDS=0,1,2 MEMORY=48g CPUS=8 bash scripts/container.sh \
  python scripts/launch.py --config configs_repro/phoenix14t_slt.yaml \
  --initialize-from runs/phoenix14t_cslr_seed0/best.pt
```

`--initialize-from` chỉ nạp trọng số phần nhận dạng rồi bắt đầu optimizer mới. `--resume` phục hồi cả optimizer/scheduler/scaler, chỉ dùng tiếp tục cùng thí nghiệm. Hãy đặt RAM theo máy thật, không sao chép 48g nếu máy không đủ RAM.

SLT dùng CE label smoothing 0.2, mBART đầy đủ, phép ghép feature + EOS + language token theo cấu trúc code gốc. Đây là bản tái dựng, khác bộ tokenizer/vocabulary pruning chưa công bố. Chưa kiểm chứng huấn luyện mBART thật tại máy chuẩn bị.

## 11. TCTC và gloss-free SLT

Chuẩn bị spaCy model tiếng Đức/Anh có lemmatizer trong `assets/nlp/`. Phiên bản pipeline là một phần của thí nghiệm: ghi rõ model/version/hash, không tự chọn mô hình mới rồi coi là recipe tác giả. Docker có spaCy nhưng **không kèm mô hình ngôn ngữ**.

```bash
GPU_IDS=none DATA_MODE=rw bash scripts/container.sh \
  python scripts/prepare_tctc.py --manifests data/manifests/phoenix14t \
  --output data/manifests/phoenix14t_tctc --spacy-model assets/nlp/de_core_news_sm

GPU_IDS=0,1,2 MEMORY=32g CPUS=8 bash scripts/container.sh \
  python scripts/launch.py --config configs_repro/phoenix14t_tctc.yaml

GPU_IDS=0,1,2 MEMORY=48g CPUS=8 bash scripts/container.sh \
  python scripts/launch.py --config configs_repro/phoenix14t_gfslt.yaml \
  --initialize-from runs/phoenix14t_tctc_seed0/best.pt
```

Vocab pseudo-gloss chỉ học từ train; dev/test token chưa thấy thành `<unk>` để tính CTC. Các nhãn thật vẫn có thể được giữ trong manifest cho kiểm toán, nhưng loss đọc `pseudo_gloss`, không đọc gloss thật. Không dùng checkpoint CSLR gloss thật cho nhánh gloss-free. Với tiếng Trung phải xác định quy trình phân từ/pseudo-gloss phù hợp; không coi mô hình spaCy Đức/Anh là thay thế được. Có thể đưa `pseudo_gloss` đã kiểm chứng vào manifest trực tiếp.

## 12. Các bộ dữ liệu khác và phần chưa thể xác nhận

PHOENIX14: dùng `prepare_data.py --dataset phoenix14`, chỉ định tên CSV/frames-template đúng bản dataset; config `phoenix14_cslr.yaml`.

CSL-Daily: chuẩn bị JSONL theo mẫu dưới đây rồi chạy `--dataset generic`. Không thay mẫu `S000005_P0004_T00` bằng video khác như code gốc. Nếu thiếu video, xử lý từ nguồn dữ liệu thay vì sửa nhãn âm thầm.

```json
{"id":"S000001_P0001_T00","frames":"sentence/frames_512x512/S000001_P0001_T00","gloss":"token1 token2","text":"câu dịch đã phân tách đúng quy trình"}
```

How2Sign/OpenASL: paper dùng **I3D features**, trong khi repo không cung cấp đủ pipeline/trọng số/split khớp. Config bổ sung nhận `.npy` float32 `[T,D]`, mặc định D=1024 là giả định phải xác nhận theo extractor. Không lấy video RGB chạy backbone rồi gọi là tái lập bảng I3D.

```json
{"id":"sample001","features":"features/sample001.npy","text":"the translated sentence","pseudo_gloss":"the translate sentence"}
```

Đặt ba file nguồn `train.jsonl`, `dev.jsonl`, `test.jsonl` rồi kiểm tra và xuất manifest:

```bash
GPU_IDS=none DATA_MODE=rw bash scripts/container.sh python scripts/prepare_data.py \
  --dataset generic --annotations data/source_manifests/how2sign \
  --data-root data/how2sign --output data/manifests/how2sign_tctc
```

Config có cho CSL-Daily CSLR/SLT/TCTC/GFSLT và How2Sign/OpenASL TCTC/GFSLT. PHOENIX14T–CSLR đã có chiến dịch ablation tái dựng, với giả định công khai. Chưa có pipeline được tác giả xác nhận cho Sign2Gloss2Text hoặc I3D extraction chính xác; xem tài sản còn thiếu trong `docs/AUDIT.md`.

## 13. Chuyển nguyên thư mục sang máy khác

Khi đã có dataset/weights, mang theo `data`, `assets`, `runs` nếu cần resume, code và cấu hình. Có thể bỏ các `runs/integration-*` khi tự đóng gói; chúng không phục vụ huấn luyện thật. Không cần mang môi trường Conda/Python của máy cũ.

Máy đích có mạng: copy cả thư mục, build image tại máy mới, chạy smoke và doctor trước training.

File `SignLanguage-Reproduce-portable.zip` bàn giao cùng dự án chỉ chứa code/tài liệu/cấu hình và các thư mục dữ liệu rỗng. Nó không chứa dữ liệu giả của kiểm thử, dataset thật, pretrained weights hay Docker image đã build. Giải nén ZIP là điểm bắt đầu sạch, sau đó làm các bước chuẩn bị ở trên.

Máy đích không có mạng: chuẩn bị image ở máy có mạng **cùng kiến trúc CPU Linux x86_64**, rồi xuất image:

```bash
docker image inspect mixsigngraph-repro:local > runs/docker-image-inspect.json
docker save -o mixsigngraph-repro-image.tar mixsigngraph-repro:local
python scripts/verify_bundle.py --write --include-data --manifest TRANSFER_SHA256.json
```

Copy thư mục và image tar sang máy đích (tar không nằm trong inventory; ghi riêng SHA-256 bằng `sha256sum mixsigngraph-repro-image.tar`). Tại máy mới:

```bash
sha256sum mixsigngraph-repro-image.tar
docker load -i mixsigngraph-repro-image.tar
GPU_IDS=none bash scripts/container.sh python scripts/verify_bundle.py --manifest TRANSFER_SHA256.json
GPU_IDS=0 bash scripts/container.sh python -m repro.smoke --device cuda --rgb
GPU_IDS=0 bash scripts/container.sh python scripts/doctor.py --config configs_repro/phoenix14t_cslr.yaml
```

`BUNDLE_SHA256.json` kèm theo chỉ kiểm tra code/tài liệu, không chứng thực dataset hoặc weights. Manifest `TRANSFER_SHA256.json` tự tạo mới kiểm tra cả tài sản đã chuẩn bị. Hash toàn bộ ảnh có thể mất nhiều thời gian; đây là bước một lần trước khi chuyển.

## 14. Xử lý lỗi thường gặp

| Lỗi | Xử lý |
|---|---|
| Docker daemon không chạy | Nhờ quản trị viên bật/cấu hình; không thể dùng Docker chỉ bằng copy thư mục |
| GPU không thấy/no kernel image | Kiểm tra driver và image CUDA/PyTorch hỗ trợ GPU, smoke test lại |
| CUDA out of memory | Micro-batch 1, workers thấp; SLT mBART đầy đủ có thể vẫn không vừa. Đổi phần cứng/recipe và ghi lại khác biệt |
| Shared memory/bus error | Giảm workers hoặc tăng `SHM_SIZE` trong giới hạn RAM máy |
| Permission denied ở runs | Đảm bảo thư mục thuộc user hiện tại; wrapper dùng UID/GID hiện tại, không dùng chmod 777 |
| Missing dataset/model | Chuẩn bị tài sản theo bước 5–6; training offline không tự tải |
| CTC alignment impossible | Kiểm tra frame count, annotation, augmentation; không cắt nhãn để chạy tiếp |
| Vocab/manifest mismatch khi resume | Dùng đúng dữ liệu/vocab cũ hoặc mở thí nghiệm mới; không bypass kiểm tra |
| Không dùng hết GPU | Batch 6 không chia hết số GPU; xem bảng ở bước 8 |
| WER khác paper | Kiểm tra `docs/AUDIT.md`, preprocessing, HSG, phiên bản, head/decoder, split; không sửa test để chọn model |

Dừng bằng Ctrl+C. Không dùng `docker system prune`, không xóa container/image/volume của người khác. Nếu muốn gỡ image này, xác minh đúng tag rồi dùng `docker image rm mixsigngraph-repro:local`; dữ liệu vẫn nằm trong thư mục dự án.

## 15. Ghi báo cáo kết quả

Lưu image ID, `run_metadata.json`, hashes dữ liệu/weights, seed, GPU model/count, micro/global batch, head giải mã, best epoch, dev/test predictions và metrics. Chạy nhiều seed để ước lượng độ biến thiên nếu có đủ tài nguyên; không chọn seed bằng test.

Gọi kết quả hiện tại là **kết quả bản tái dựng**, cho đến khi có đủ tài sản/recipe tác giả và đối chiếu được từng khác biệt. Báo cáo kiểm thử tại máy chuẩn bị: [reports/VALIDATION.md](reports/VALIDATION.md).
