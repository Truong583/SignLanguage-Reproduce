# Phạm vi thí nghiệm PHOENIX14T–CSLR

Mục tiêu: chạy mã tái dựng theo paper để **đo**, đối chiếu và báo cáo kết quả trên một dataset/task được chọn. Không thay số đo bằng số paper. Không coi unit test là chứng cứ đã tái lập chỉ số.

## Cách chạy

Trên **máy đích**, giải nén thư mục, mở terminal trong thư mục đó và chạy `python run.py --once`. Đây là toàn bộ chiến dịch, không phải chỉ một model nữa. Muốn tự lấy bản sửa và gửi lỗi lên W&B, cấu hình một lần theo [MAY_CO_VA_LAPTOP.md](MAY_CO_VA_LAPTOP.md), rồi dùng `python run.py`. Điều kiện: Python 3.12+, Docker Linux containers, NVIDIA driver và Docker GPU support, Internet, đủ RAM/VRAM/dung lượng. Không cần chạy huấn luyện trên máy phát triển.

Colab/Kaggle: dùng hai notebook mới. Kaggle đã điền folder ID `1RpGbT-9SXIr21OVfVmRRn4K_XhRulCeL`. OAuth Secret vẫn như đã thiết lập. Colab dùng `MyDrive/SignLanguage-Reproduction/`; folder này phải chính là folder có ID trên để hai nền tảng dùng chung. Thư mục Drive là nguồn archive/ZIP và nơi lưu kết quả/checkpoint; runtime chỉ dùng ổ tạm để đọc ảnh nhanh.

## Danh sách

| Nguồn trong paper | Lượt chạy riêng | Nội dung |
|---|---:|---|
| Bảng 5 / bảng 1 đầy đủ | 1 | MixSignGraph chính, WER dev/test |
| Bảng 1, các dòng còn lại | 12 | Không graph; sáu module đơn; ba cặp cùng loại; hai bộ ba cùng stage |
| Bảng 13b, các thứ tự còn lại | 5 | Tổng cộng sáu thứ tự, một thứ tự đã là model chính |
| Bảng 13a | 8 | Dense/sparse LSG × sparse/dense TSG × fixed/dynamic HSG |
| Hình 6, các điểm ngoài mặc định | 26 | Lần lượt đổi K_l1/K_l2 trong 2–9; K_t1/K_t2 trong 7,21,35,49,63,77,91 |
| Bảng 14b | 2 | Cosine, Chebyshev; Euclidean đã là mặc định |
| Bảng 14c | 3 | GATv2, SAGE, GCN; EdgeConv đã là mặc định |
| Bảng 14a | 3 | Swin-T, PyViG-Tiny, Self-Attention |
| Bảng 14d | 3 | Một triplet graph tại patch 8,16,32 |
| Bảng 14e | 2 | Stage 8→16→32 và 4→8→16→32; 16→32 là model chính |
| Bảng 14f | 2 | DropEdge 15%, 30%; 0% đã là mặc định |
| Bảng 3, tham chiếu định tính | 1 | MultiSignGraph tái dựng bỏ hai HSG, so sánh gloss trên cùng mẫu test |
| **Tổng** | **68** | **67 cấu hình định lượng và một tham chiếu định tính** |

Catalog máy đọc và các YAML để kiểm tra ở `configs_repro/phoenix14t_suite/`. Trình chạy lấy catalog từ `repro/suite.py`; file cấu hình thực dùng luôn được lưu kèm từng run. Các kết quả của phương pháp thuộc paper khác trong bảng 5 là số tham chiếu, không phải yêu cầu chạy lại repository của mọi phương pháp được trích dẫn. Các task SLT/TCTC và dataset khác nằm ngoài chiến dịch này.

## Những chỗ tác giả thiếu, lựa chọn được công khai

Đọc thêm `docs/AUDIT.md`. Các lựa chọn dưới đây là giả định có thể chạy và kiểm thử, không phải xác nhận từ tác giả:

- HSG nối feature của hai scale liền nhau bằng projection, cạnh hai chiều theo vùng và merge stride2. Vị trí chính xác/initializer chưa được release.
- Giữ LSG EdgeConv của bản release cho reference, gồm fc1/fc2, grouped MLP, normalization và relative-position bias. Các khoảng cách khác giữ các lớp trước/sau đó. Euclidean trong LSG là bình phương khoảng cách trên feature đã normalize; TSG dùng khoảng cách Euclidean FP32. Paper không khóa toàn bộ chi tiết normalize/bias.
- LSG sparse chọn K cặp node toàn frame, bỏ self-pairs; TSG sparse chọn K cặp giữa hai frame kề. TSG dense chọn K đích cho mỗi nguồn rồi thêm cạnh đảo. HSG dynamic chọn K láng giềng trên node high/low sau projection. Các quy ước K và hướng cạnh của biến thể không được release đầy đủ.
- Bảng 13a nói dùng một triplet nhưng không ghi stage; chiến dịch chọn patch16. Không âm thầm coi cấu hình này giống model hai stage dù số paper của dòng đầu trùng baseline.
- GATv2 một head, LeakyReLU0.2, source/target projection riêng, self-loop; SAGE mean; GCN degree-normalized có self-loop; TSG/HSG EdgeConv Linear(2C,C)+ReLU. LSG giữ chiều C→2C→C khi thay graph convolution. Head count/hidden widths/activation tác giả không công bố đầy đủ.
- Swin-T và PyViG-Tiny dùng feature trunk từ ImageNet, nạp trọng số strict; paper không xác định variant PyViG. Bản compatible PyViG đối chiếu với mã Huawei commit `a328aa023d4e3e66cca690ecc54f77daf436b9ff`, checkpoint official `pvig_ti_78.5.pth.tar`. Giữ preprocessing RGB chung của chiến dịch, không tự chọn recipe riêng cho từng backbone.
- SA dùng 8 heads, giữ projection C→2C→C của LSG; thay TSG bằng attention trên cặp frame kề và HSG bằng attention trên node hai scale. Paper không mô tả đủ số layer/head/FFN của phép thay này.
- Các stage sớm dùng K của stage đầu; patch4 HSG nối conv1 với layer1 sau maxpool. Stem ResNet18 và trọng số ImageNet giữ nguyên; patch ở đây là stride của feature map. Không tạo checkpoint cho stem mới rồi gọi đó là pretrained thật.
- DropEdge chọn mask Bernoulli mỗi training forward. LSG mask messages sau edge BatchNorm, trước max aggregation; HSG bỏ edges trước graph convolution. Eval không drop. Lịch mask chính xác mỗi epoch/cách xử lý BN không được công bố.
- MultiSignGraph định tính được dựng bằng cách bỏ HSG. Không gán số WER của SignGraph vào run này khi paper không xác nhận hai model giống nhau.
- Recipe 50 epoch/batch6/LR/optimizer theo bản code đã audit; chưa có recipe chính thức riêng cho 68 lượt. Chỉ một seed0: không tự tạo mean/std hay error bars từ một lượt chạy.
- Temporal augmentation 0.8–1.2 được chặn ở độ dài tối thiểu hợp lệ cho CTC, tính cả gloss lặp kề nhau. Đây là guard tái dựng để augmentation không gây lỗi alignment cho clip ngắn; không cắt gloss hoặc bỏ sample. Base alignment không hợp lệ vẫn bị doctor từ chối.

## Checkpoint và nhiều phiên

Mỗi cấu hình có `last.pt`, `best.pt`, config, code hash, manifest hash, môi trường, log và dự đoán riêng. Model chọn bằng WER **dev**, sau đó cùng checkpoint chấm dev/test. Không chọn head/checkpoint dựa trên test.

Local lưu atomic vào `runs/phoenix14t_cslr_suite_seed0/<id>/`. Notebook xuất checkpoint sau mỗi 200 optimizer updates, cuối epoch và khi dừng có kiểm soát. Drive dùng payload + descriptor checksum và ít nhất hai thế hệ; upload bị ngắt không ghi đè thế hệ tốt trước đó. Tắt cưỡng bức runtime có thể mất phần tính từ checkpoint được commit gần nhất. Thay số GPU giữ batch hiệu dụng nhưng không bảo đảm bitwise identical.

Notebook giới hạn phiên 600 phút tính từ ô cấu hình, trừ thời gian chuẩn bị. Nếu chuẩn bị vượt ngân sách, chỉ dành thêm tối đa vài phút để ghi checkpoint, rồi dừng. Đánh giá cuối epoch có thể vượt phần thời gian còn lại; lúc platform ngắt sẽ tiếp tục từ checkpoint đã lưu. Không hứa Run All hoàn tất cả 68 lượt trong một phiên.

Chạy lại tự khôi phục run dở và bỏ qua run có marker hoàn tất đúng config/code/data. Nếu thay epoch, seed, code hoặc dữ liệu: tạo chiến dịch mới (`CAMPAIGN_NAME` ở notebook; `--output` khi gọi `scripts/run_suite.py`), không trộn kết quả.

## Đầu ra để báo cáo

- `suite_summary.csv/json`: paper WER/Del/Ins, measured WER/Del/Ins và delta; pending chưa có số đo. Hình6 không được đọc thành số giả từ vị trí pixel; điểm tham chiếu số của hình để null, sweep xuất số đo thật.
- `<id>/comparison.json`, `dev/test_metrics.json`, predictions, `history.jsonl`, `run_metadata.json`, `suite-config.yaml`, `identity.json`, `COMPLETED.json`.
- `qualitative.json`: reference và gloss dự đoán của model chính / tham chiếu MultiSignGraph. Tìm mẫu theo toàn bộ gloss reference trong bảng3; nếu không tìm thấy, báo rõ dùng mẫu đầu test.
- `graph_stage16.png`, `graph_stage32.png`, `graph_edges.json`: cạnh thực từ model chính, dựng hình theo nhóm LSG/TSG/HSG. Vẽ subset deterministic, không gán nhãn cạnh “quan trọng/nền” khi chưa có tiêu chí định lượng như tác giả. Đây là visualization tái dựng để đối chiếu hình5, không phải ảnh gốc paper.

## Tài nguyên và kiểm chứng

67 cấu hình định lượng thường cần nhiều lần thời lượng của một training run; stage4/SA/PyViG có thể cần VRAM lớn hơn reference. Chia query KNN và cặp temporal theo chunk để giảm bộ nhớ khoảng cách; giữ thuật toán chọn cạnh. Không tự cắt video, giảm resolution, đổi batch/LR hoặc bỏ cấu hình khi CUDA OOM. Run lỗi sẽ dừng rõ, checkpoint của run trước vẫn giữ.

Archive gần39 GiB cộng dữ liệu giải nén, Docker cache, checkpoint của 68 model có thể cần dung lượng lớn hơn nhiều so với một model. Drive không phải vô hạn; nếu publish checkpoint thất bại, dừng để tránh mất quá trình. Không tự xóa dataset hoặc checkpoint của người dùng.

Máy phát triển chỉ chạy kiểm thử/smoke trên dữ liệu giả và tải pretrained backbones để xác nhận nạp đúng. Huấn luyện/đánh giá dữ liệu thật, Docker/NCCL nhiều GPU và đạt số paper vẫn phải nghiệm thu tại máy đích. Sai khác recipe còn ở trên phải đi kèm báo cáo nộp cô.
