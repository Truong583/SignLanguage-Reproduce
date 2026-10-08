# Chạy trên máy khác bằng một lệnh

**Luồng laptop sửa code → máy Linux của cô tự lấy bản sửa, Docker train → W&B gửi lỗi:** đọc [MAY_CO_VA_LAPTOP.md](docs/MAY_CO_VA_LAPTOP.md). Cấu hình một lần bằng `python3 setup_machine.py`, sau đó dùng `python3 run.py`. Không có CI/GitHub Actions; checksum, cú pháp và kiểm tra checkpoint vẫn được giữ.

Chép **toàn bộ thư mục** `SignLanguage-Reproduce` sang máy đích, mở terminal trong thư mục đó rồi chạy:

```bash
python run.py --once
```

Phần dưới mô tả runner một lượt không giám sát từ xa (`--once`). Chế độ mặc định có giám sát dùng cùng pipeline và chờ bản sửa sau khi dừng. Trên Linux nếu tên Python là `python3`, dùng `python3`.

## Máy đích cần có gì trước?

- Python 3.10 trở lên; không cần cài các thư viện AI vào Python của máy.
- GPU NVIDIA, driver có sẵn, Docker đang chạy với Linux containers và có quyền truy cập GPU. Linux cần NVIDIA Container Toolkit; Windows cần Docker Desktop/WSL2 có hỗ trợ GPU. Script kiểm tra và dừng nếu thiếu, không tự sửa driver hoặc hệ thống.
- Internet cho lần dựng image/tải dữ liệu đầu tiên.
- Ít nhất 150 GiB trống trong ổ chứa thư mục dự án khi chưa có dữ liệu. Docker còn cần chỗ riêng cho image và cache build. Đây là ngưỡng kiểm tra bảo thủ, không bảo đảm cho mọi bản dữ liệu/ổ Docker.
- Docker có tối thiểu khoảng 12 GiB RAM; script dành 70% RAM Docker và tối đa 8 CPU, chừa tài nguyên cho máy. GPU phải đủ VRAM cho video thật; chưa xác nhận giới hạn VRAM của thí nghiệm này.

## Một lệnh đó làm gì?

1. Kiểm tra checksum mã nguồn, dựng Docker (lần sau dùng cache), kiểm tra GPU.
2. Tải archive PHOENIX14T chính thức nếu chưa có, giải nén, tạo train/dev/test manifest và tải ResNet ImageNet weights.
3. Kiểm tra dữ liệu, tải thêm Swin-T/PyViG ImageNet, lần lượt huấn luyện **68 cấu hình PHOENIX14T–CSLR**: seed 0, batch hiệu dụng 6, 50 epoch mỗi run. Danh sách và giả định ở `docs/PHOENIX14T_SUITE.md`.
4. Có `last.pt` thì tiếp tục run cũ. Dừng giữa epoch sẽ tiếp tục theo checkpoint mà trainer đã lưu; phần chưa lưu phải chạy lại. Không tạo checkpoint nếu dừng trong tải/chuẩn bị dữ liệu.
5. Mỗi run dùng cùng `best.pt` được chọn bằng dev để chấm dev/test. Run hoàn tất được bỏ qua khi chạy lại; xuất bảng tổng hợp, độ lệch paper, gloss định tính và hình graph.

Số GPU được chọn tự động để giữ nguyên batch hiệu dụng. Ví dụ: 1 GPU tích lũy 6 lần, 2 GPU tích lũy 3 lần, 3 GPU tích lũy 2 lần, 4 GPU dùng 3 GPU, 8 GPU dùng 6 GPU (micro-batch 1). Đổi số GPU không bảo đảm kết quả giống từng bit. Script không tự đổi LR/epoch/kiến trúc để chạy nhanh hơn.

## Kết quả ở đâu?

`runs/phoenix14t_cslr_suite_seed0/<experiment>/` chứa:

- `last.pt`, `best.pt`: checkpoint tiếp tục và checkpoint tốt nhất trên dev.
- `suite-config.yaml`, `run_metadata.json`, `vocab.json`, `history.jsonl`: cấu hình, môi trường, vocab, lịch sử huấn luyện.
- `dev_metrics.json`, `test_metrics.json`, các file predictions: số đo thật.
- `comparison.json`: số paper, số đo, độ lệch; giả định cũng lưu trong cấu hình.
- `COMPLETED.json`: chỉ xuất khi huấn luyện và đánh giá đều thành công.

Tắt rồi bật lại: chạy lại **cùng lệnh**. Khi chuyển máy và muốn tiếp tục, chép cả `data/`, `assets/`, `runs/` cùng mã nguồn. Local không đồng bộ Google Drive; hai notebook giữ cơ chế Drive riêng như trước.

## Phạm vi tái lập phải báo cáo đúng

Mặc định một lệnh chạy **chiến dịch PHOENIX14T–CSLR**: 67 cấu hình định lượng và một tham chiếu định tính. Root chiến dịch có `suite_summary.csv/json`, `qualitative.json` và hình graph. Các task/dataset khác nằm ngoài chiến dịch này. Repository tác giả còn thiếu recipe và HSG forward đầy đủ; đây là bản tái dựng đã kiểm thử, chưa có kết quả huấn luyện trên dữ liệu thật để khẳng định đạt số paper. Đọc `docs/AUDIT.md` và `docs/PHOENIX14T_SUITE.md` trước khi nộp kết luận.

Các config khác có thể dùng cùng runner, ví dụ:

```bash
python run.py --once --single --config configs_repro/phoenix14t_slt.yaml
```

SLT cần checkpoint CSLR/TCTC đúng phase chạy trước. Pipeline tự lấy mBART với revision đã khóa. Các dataset khác cần dữ liệu/manifests có quyền sử dụng đã chuẩn bị; runner dừng rõ khi thiếu, không tự chọn dữ liệu thay thế. Runner chưa bao phủ Sign2Gloss2Text hoặc extraction I3D đúng tác giả. Các ablation CSLR đã có mã chạy tái dựng và danh sách giả định, không gắn nhãn checkpoint chính thức tác giả.

## Tài nguyên và môi trường

Có thể đặt giới hạn rõ khi cần: `python run.py --memory-gb 24 --cpus 6`. GPU quá mới cho image mặc định thì cần image phù hợp, ví dụ cấu hình `--base-image`; môi trường mới phải được ghi nhận và kiểm chứng, không có lựa chọn image nào bảo đảm cho mọi GPU.

Docker chỉ ghi vào `data/`, `assets/`, `runs/`; root filesystem của container và mã nguồn mount chỉ đọc. Container không chạy privileged, không mount Docker socket hoặc thư mục khác của máy. Giai đoạn chuẩn bị có Internet; giai đoạn huấn luyện/đánh giá tắt mạng. Có giới hạn RAM, CPU, tiến trình và shared memory. GPU dùng chung vẫn có thể cạnh tranh tài nguyên với công việc khác; chọn máy/GPU được dành cho thí nghiệm.

Nếu CUDA OOM, không tự chạy lại với recipe khác: ghi lỗi, chọn GPU đủ VRAM. Nếu hết dung lượng/RAM hoặc thiếu Docker, script dừng và giữ những checkpoint/archive đã lưu. Script không xóa container/image khác hoặc đổi cấu hình Docker của máy.

Xem quy trình mà không tải/chạy: `python run.py --plan`.

## Kiểm chứng trước khi nhận kết quả

Launcher đã được kiểm tra cấu trúc lệnh Docker và orchestration bằng unit test. Máy phát triển hiện chưa có Docker daemon hoạt động nên chưa kiểm chứng end-to-end bằng Docker/GPU hoặc huấn luyện PHOENIX14T thật. Docker build mặc định không chạy tests. Máy đích vẫn kiểm tra GPU, dữ liệu, training và evaluation; chỉ coi thành công khi có `COMPLETED.json` cùng metrics thật.
