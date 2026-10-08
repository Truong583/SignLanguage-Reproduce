# Báo cáo kiểm tra — cập nhật 2026-10-08

## Lượt mới từ xa cho cấu hình đơn

- Host chuyển tên campaign đã được duyệt vào cả chế độ --single. Container dùng thư mục `runs/<campaign>/<config-stem>` khi campaign khác mặc định; mặc định giữ nguyên vị trí output cũ. Nhờ vậy policy new_campaign cho sửa code tính toán không vô tình quay lại thư mục checkpoint của mô hình cũ.
- **41 passed, 1.61 giây** (`runs/pytest-remote-single-campaign`): kiểm tra truyền campaign qua Docker và chọn output mới ở pipeline đơn, cùng guards/checkpoint/update/live log. Không thay mô hình/loss hoặc chạy PHOENIX thật.

## Live log và phục hồi sau tắt máy

- Sửa đường dẫn live log: status ghi `runs/diagnostics/<id>`, còn mount observer đã là runs; bỏ tiền tố runs để đọc đúng console.log. Live file lọc key, tối đa32KiB, được thay atomic và gửi theo policy now. Tests kiểm tra lỗi mạng vẫn giữ file local và chặn đường dẫn ngoài diagnostics.
- Cấu hình đơn mặc định checkpoint mỗi200 optimizer updates, backend local giữ tối thiểu hai thế hệ last/best có checksum. Runner restore bản commit đáng tin trước train. Chỉ cho phép thay các trường cadence/storage khi nạp config cũ; recipe/code/data guards vẫn giữ. Checkpoint partial không được eval/đánh dấu COMPLETED; exit75 là session lưu dở để lần sau tiếp tục.
- `service.py install --single` cài unit riêng theo workspace, User là tài khoản thường, Group=docker, Requires Docker/mount, KillMode mixed, stop timeout180s. Có lệnh stop/status/logs; default giữ phạm vi main. Không cài service trên máy phát triển. Reuse image đã dựng giúp tiếp tục khi mạng mất nếu đủ data/weights. W&B session mới đọc lại history nguồn để tránh mất scalar chỉ được queued ở phiên cũ.
- **62 passed, 28.59 giây** (`runs/pytest-power-recovery-integrated`). Có integration CPU thật trên vài mẫu features giả: save giữa epoch, cố ý corrupt payload mới, restore thế hệ trước rồi train hết epoch. Không phải PHOENIX hoặc thử cắt điện thật. Sau chỉnh setup, nhóm kiểm tra chạy lại **31 passed, 1 deselected, 3.95 giây** (bỏ integration đã qua).
- Chưa nghiệm thu systemd, boot/shutdown thật trên máy cô hoặc file live upload online từ bản sửa. Không bảo đảm mọi sự cố mất điện/ổ đĩa; chỉ checkpoint commit nguyên vẹn được phục hồi, phần chưa lưu phải tính lại. Không gửi checkpoint/data/key lên W&B hoặc GitHub.

## Theo dõi cấu hình đơn trên W&B

- Observer đọc cả `runs/<experiment>/history.jsonl` của `--single` và `runs/<campaign>/<experiment>/history.jsonl` của suite. Trước sửa chỉ đọc kiểu suite, khiến cấu hình đơn thiếu scalar loss/WER trên W&B.
- **19 passed, 0.52 giây** (`runs/pytest-observer-single`): có kiểm tra hai dạng thư mục, cursor tránh ghi lặp và bỏ trường nhạy cảm. Không gửi dữ liệu lên tài khoản W&B người dùng trong kiểm tra; cần health.json của máy cô để xác nhận kết nối thật.

## Sửa timeout tải PHOENIX

- Log người dùng xác nhận GPU probe đã thành công, nhưng đọc HTTP archive PHOENIX bị TimeoutError trước training. Tải dở vẫn nằm trong data/*.part. Downloader thêm tối đa8 lượt thử, timeout180s, backoff5/10/20/40/60s; giữ byte đã ghi, kiểm tra Content-Range trước append, server trả200 thì tải lại thay vì ghi trùng. Incomplete download được giữ để thử tiếp; lỗi HTTP vĩnh viễn hoặc range lệch dừng.
- Giảm khối đọc1MiB, in tiến độ theo30s hoặc1GiB. Guard150GiB cho chuẩn bị được tính bằng free + archive/.part đã chiếm trên cùng filesystem; vẫn yêu cầu ít nhất10GiB trống và giữ guard riêng trước giải nén. Không coi dung lượng file đã cache là chỗ trống thật.
- **48 passed, 1.62 giây** (`runs/pytest-download-retry`): phản hồi mạng giả kiểm tra timeout sau ghi/truncated body, range bị bỏ qua/range lệch, giới hạn retry, HTTP404, budget giữ filepartial, cùng các tests Docker/update trước đó. Chưa tải archive thật hoặc nghiệm thu kết nối RWTH/Drive tại laptop; cần log máy cô để xác nhận.

## Sửa Docker build từ log máy cô

- Log máy Ubuntu 22.04 của người dùng xác nhận pip install hoàn tất nhưng pip check thất bại với `ninja 1.11.1.1 is not supported on this platform`; chưa bắt đầu training. Đây là lỗi kiểm tra wheel trong image nền, không phải thông báo mất mạng.
- Dockerfile cài lại wheel **ninja1.11.1.3** bằng `--only-binary=:all: --force-reinstall --no-deps` trước requirements; giữ nguyên pip check. Wheel x86_64 từ PyPI chính thức có hai Tag riêng `py3-none-manylinux_2_12_x86_64` và `py3-none-manylinux2010_x86_64`; metadata/tag đã được đọc xác nhận. Không sửa code tính toán, driver, PyTorch/CUDA hay bỏ kiểm tra phụ thuộc.
- Docker build của observer lưu log/diagnostic local. Lỗi được đánh dấu theo revision; không tự dựng lại cùng bản lỗi. Trong khi chờ vẫn kiểm tra GitHub; bản sửa mới có thể được kích hoạt. Sau khi observer hoạt động, diagnostic tồn đọng mới có thể gửi W&B. Không hứa gửi W&B khi image chưa dựng được.
- **39 passed, 1.90 giây** trong nhóm update/orchestration/extraction (`runs/pytest-ninja-build-recovery`). Thêm hai tình huống giả: lỗi build chờ update, và có bản sửa mới thì dựng thành công trước khi gọi trainer. Chưa build Linux image có bản sửa này tại laptop; máy cô cần chạy lại để xác nhận pip check và GPU probe thật.

## Repo Public và API key W&B mới

- Người dùng chủ động chuyển repository Public. Setup tự nhận diện Public, không hỏi/lưu GitHub token, còn updater bỏ header GitHub auth; W&B project vẫn được kiểm tra riêng tư. Bộ tests liên quan **37 passed, 2.35 giây** (`runs/pytest-public-repo`). Bao gồm Public/Private setup, khóa/snapshot, redaction key mới, mount và orchestration. Không có CI trên server.
- SDK W&B đổi từ0.18.7 sang **0.27.0**, dùng Settings `x_disable_stats`. Kiểm chứng bằng package cài vào thư mục tách biệt ngoài dự án: Settings chấp nhận định dạng key mới với key giả, import cùng protobuf **5.28.3**, init offline, log scalar, finish thành công. Không dùng key thật đã xuất hiện trong chat, không đăng nhập hoặc upload dữ liệu người dùng. Xác thực W&B online vẫn cần làm trên máy cô.
- Mã tính toán trong `repro/`, recipe và fingerprint checkpoint giữ nguyên. Thay SDK giám sát không xác nhận tương đương kết quả giữa các image; Docker lưu environment lock khi build.

## Luồng giám sát không CI

- Toàn bộ tests local: **131 passed, 111.16 giây** (`runs/pytest-no-ci-release`). Bao gồm dữ liệu giả/GPU local, core numerical checks, supervisor, runtime redaction, khóa và snapshot Git. Không chạy chiến dịch PHOENIX thật.
- Sau bổ sung bắt trạng thái worker bị tắt bất ngờ: nhóm kiểm tra orchestration/update/extraction chạy lại **34 passed, 1.28 giây** (`runs/pytest-supervision-final`).
- Bổ sung mountpoints data/assets/runs rỗng trong snapshot chỉ đọc để các Docker bind con có target sẵn; nhóm kiểm tra chạy lại **34 passed, 1.23 giây** (`runs/pytest-mountpoints-final`). Các thư mục này không chứa dữ liệu/checkpoint/secret trong mã nguồn.
- Supervisor được kiểm tra bằng subprocess/time giả: worker đã dừng trước khi kiểm tra update; cùng bản lỗi không chạy lại; giữ trạng thái qua restart; Docker observer không có GPU/data/weights/token GitHub/socket; project public/missing/schema không xác định bị từ chối trước telemetry init.
- Snapshot thực từ Git local được giải nén, đối chiếu checksum; kiểm tra rollback, atomic activation thất bại, bảo toàn dữ liệu/checkpoint, chặn path traversal/symlink. Byte nguồn được giữ qua Git bằng `.gitattributes` để không đổi checkpoint fingerprint giữa Windows/Linux.
- Không có `.github/workflows` CI. Docker build dùng `RUN_TESTS=0` mặc định; không chạy synthetic training mỗi cập nhật. Checksum, compile cú pháp, GPU probe, dữ liệu và checkpoint guards vẫn chạy.
- Đã kiểm tra source/metadata wheel W&B **0.18.7** đúng phiên bản pin: Settings và project access field; protobuf5.28.3 đáp ứng khoảng phụ thuộc. Chưa kiểm chứng W&B online, quyền/token thật hoặc alert trên tài khoản của người dùng. Các kiểm tra privacy/network dùng phản hồi giả; không thay thế nghiệm thu API thật.
- Hai notebook tái tạo từ source và compile code cells; giữ Drive chung/catalog/checkpoint, thêm hướng dẫn phân biệt notebook với supervisor Linux.
- Docker daemon local chưa hoạt động: chưa build/chạy image, chưa kiểm chứng Docker Linux, systemd user service hoặc NCCL trên máy cô. Không tuyên bố luồng đã triển khai trên máy cô hay đạt số paper.

## Release chiến dịch PHOENIX14T–CSLR

- `python -m pytest -q tests --basetemp runs/pytest-release-suite`: **102 passed, 158.04 giây**. Toàn bộ 68 cấu hình qua forward/backward backbone với dữ liệu giả, kiểm tra không có trainable parameter bị bỏ quên; bốn loại graph convolution qua RGB CTC loss/backward FP16 trên GPU local. Bao gồm kiểm tra công thức distance/SAGE/GATv2 bằng reference độc lập, completed vs partial checkpoint, identity portable, thư mục Drive chung, alignment của temporal augmentation, graph overlays và tar nội bộ/escape.
- `python scripts/create_notebooks.py` rồi `python scripts/finalize_notebooks.py`: PASS; hai notebook được tái tạo từ source, code cells compile, chứa hàng đợi 68 lượt và folder ID người dùng cung cấp. Các cell IPython `%pip` không chạy trong kiểm tra syntax.
- Swin-T và PyViG-Tiny ImageNet từ nguồn chính thức đã tải và nạp **strict** tại local. SHA256 Swin `704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b`; PyViG `06c49bda678b26a8cbc69c69e2762c3496eb4d9db8cacb5e88cdd517f56e4d4f`, được khóa trong downloader.
- Giải nén dùng stream một lượt, chấp nhận hardlink/symlink nội bộ qua data filter, chặn escape và special entries, đi lên đúng ba cấp từ annotations/manual. Unit archive có hardlinks đã qua; chưa giải nén archive thật tại local.
- Sau bộ kiểm thử trên, bổ sung reserve dung lượng trước giải nén/run mới và thứ tự publish best trước last để tránh last tham chiếu best chưa upload; integration train/resume/eval và targeted checkpoint tests được chạy lại.
- Đây là kiểm tra mã/khả năng tính toán trên dữ liệu giả, **không** là huấn luyện PHOENIX14T thật, chạy toàn bộ chiến dịch hoặc đạt số paper. Chưa build/chạy Docker vì daemon local không hoạt động; chưa nghiệm thu nhiều GPU/NCCL hoặc OAuth/Drive end-to-end cho chiến dịch mới.

## Bản chạy một lệnh

- `python run.py --plan`: PASS, in đầy đủ workflow mặc định PHOENIX14T–CSLR, không tải hoặc huấn luyện.
- `python -m pytest -q tests --basetemp runs/pytest-one-command-final`: **22 passed, 9.87 giây**. Thêm kiểm tra mount chỉ đọc, training tắt mạng, tiếp tục `last.pt`, chấm dev/test bằng cùng `best.pt`, từ chối thay config của run hiện có, giải nén đúng root của archive mẫu và hardlink nội bộ, chặn path/link thoát ra ngoài.
- Đã sửa đường dẫn root sau giải nén (CSV nằm trong `annotations/manual/` nên phải đi lên ba cấp). Bộ lọc chấp nhận liên kết nội bộ và dùng `tarfile` data filter để chặn escape. Chưa giải nén archive thật 39 GiB tại máy phát triển; kiểm tra bằng archive mẫu.
- Docker daemon của máy phát triển vẫn không hoạt động. Chưa build/chạy launcher end-to-end qua Docker, nhiều GPU thật, dữ liệu thật hoặc đạt metric paper. Các giới hạn nghiệm thu dưới đây vẫn áp dụng.

## Đã thực hiện

Môi trường local: Windows 11, Python 3.13.12, PyTorch 2.10.0, torchvision 0.26.0, CUDA runtime 12.8, NVIDIA RTX 3050 Laptop 6 GB. Không cài thư viện vào môi trường host và không sửa driver/Docker.

| Kiểm tra | Kết quả thực đo |
|---|---|
| `python -m pytest -q tests --basetemp runs/pytest-release` | 17 passed, 14.98 giây; gồm bản EdgeConv cuối và mBART nhỏ |
| HSG mapping hình chữ nhật và cạnh hai chiều | PASS |
| GCN sparse so với phép nhân adjacency dense độc lập, gradient | PASS |
| TSG chọn đúng cặp gần nhất, chỉ nối frame kề | PASS |
| HSG nhận gradient ở cả hai đầu high/low | PASS |
| EdgeConv sparse đối chiếu vòng lặp duyệt từng láng giềng, node cô lập | PASS |
| CTC beam so với tổng xác suất tất cả paths của ví dụ nhỏ | PASS |
| WER có substitution/deletion/insertion; BLEU/ROUGE đối chiếu câu trùng | PASS |
| Chia batch 1/2/3/4/8 GPU bằng mô phỏng toán học; sampler không trùng | PASS, chưa phải multi-GPU hardware run |
| `python -m repro.smoke --device cpu` | PASS, synthetic feature CTC backward; loss 5.1205801964 |
| `python -m repro.smoke --device cuda --rgb` sau cập nhật EdgeConv | PASS, RGB 224px, toàn backbone và hai HSG, loss 4.3832335472; peak allocated 731371520 bytes cho mẫu cực ngắn |
| `python scripts/integration_check.py --directory runs/integration-03` | PASS: train epoch1, last/best checkpoint, resume epoch2, đánh giá đủ 3/3 mẫu giả |
| Compile Python `repro`, `scripts`, `tests` | PASS |
| mBART ngẫu nhiên rất nhỏ + tokenizer stub: loss, gradient feature/model, EOS/language mask và beam generation | PASS; không kiểm tra tokenizer thật hoặc pretrained mBART |
| Git diff của file tracked từ tác giả | Không sửa file gốc |

VRAM smoke **không** là ước tính VRAM cho video thật, không gồm mBART và không chứng minh dataset dài chạy vừa 6 GB.

## Lỗi môi trường đã ghi nhận

- Unit test lần đầu: 14 passed, 1 setup error do Windows sandbox không đọc được thư mục temporary mặc định. Chạy lại với temporary trong `runs` được 15/15.
- Integration lần đầu với `CUDA_VISIBLE_DEVICES` rỗng gặp lỗi khởi tạo cuDNN ở bản PyTorch local. Thêm lựa chọn `device: cpu` cho integration và không ẩn thiết bị theo cách đó; pipeline CPU sau sửa đã qua.
- Thử hai tiến trình CPU qua torchrun: bản PyTorch Windows báo thiếu libuv khi rendezvous. Chưa xác nhận DDP runtime; mã chia sampler/effective batch có unit test, nhưng không thay thế bài test Linux. Log thử nghiệm ở `runs/integration-ddp-01` không phải một run thành công.
- Docker CLI có nhưng daemon không hoạt động. Chưa build image, chưa chạy `pip check` trong image, chưa kiểm chứng phụ thuộc đã pin hay NCCL trên máy đích. Dockerfile chạy pip check; mặc định không chạy unit tests. Tests chỉ bật khi chủ động dùng build arg RUN_TESTS=1.

## Chưa kiểm chứng / không được suy diễn thành đã hoàn thành

- Huấn luyện/đánh giá trên dataset thật, mọi chỉ số paper.
- Chạy nhiều GPU thật, Synced BatchNorm và thay world size trên phần cứng thật.
- mBART thật, SLT/TCTC toàn pipeline, NLP models và I3D extraction.
- Tương thích checkpoint tác giả, tokenizer đã prune, official evaluator.
- Hiệu năng tối ưu trên mọi loại GPU, tính tương đương số học với PyTorch1.11.

## Nghiệm thu tại máy đích

```bash
docker build -t mixsigngraph-repro:local .
GPU_IDS=none bash scripts/container.sh python -m repro.smoke --device cpu
GPU_IDS=0 bash scripts/container.sh python -m repro.smoke --device cuda --rgb
GPU_IDS=none bash scripts/container.sh python scripts/integration_check.py --directory runs/acceptance-single
GPU_IDS=none bash scripts/container.sh python scripts/integration_check.py --directory runs/acceptance-ddp --distributed
```

Sau đó chạy doctor, một thí nghiệm ngắn với dataset thật, kiểm tra log/VRAM/loss/checkpoint rồi mới chạy đủ epoch. Giữ kết quả nghiệm thu mới cùng hồ sơ nghiên cứu; không thay thế kết quả fail bằng tuyên bố đã tái lập paper.
