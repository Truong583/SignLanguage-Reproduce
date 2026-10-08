# Đối chiếu paper và repository

Đã đối chiếu thêm nhánh `master`: nhánh này thiếu nhiều thành phần hơn `main` tại các commit đã kiểm tra. Xem [báo cáo nhánh](BRANCH_COMPARISON.md). Không cần thay mã chạy hiện tại bằng master.

Ngày kiểm tra: 2026-10-06. Nguồn:

- GitHub: https://github.com/gswycf/SignLanguage/tree/af5e8475d755b9d2e92c0142c8b7084651c3a4ee
- Paper người dùng đưa: https://openreview.net/pdf?id=YjZYMHvlRs
- Bản công bố NeurIPS dùng để đọc khi OpenReview yêu cầu xác minh trình duyệt: https://papers.nips.cc/paper_files/paper/2025/file/cbc1ad2066f0afebbcea930c5688fc1f-Paper-Conference.pdf

## Kết luận về mức độ tái lập

Repository công bố không đủ để chứng minh tái lập chính xác toàn bộ paper. README của MixSignGraph ghi “The code is not prepared yet.” Đây là vấn đề tài sản/recipe gốc, không thể giải quyết bằng đoán code rồi gắn nhãn chính thức. Bộ bổ sung này là một triển khai tái dựng có nguồn gốc và kiểm thử; chưa có số đo dữ liệu thật.

## Các phát hiện trực tiếp từ mã

| Vị trí | Phát hiện | Cách xử lý trong bản bổ sung |
|---|---|---|
| `MixSignGraph/modules/resnet.py` | Backbone chỉ gọi LSG/TSG, không import/gọi MixGraph/HSG | Giữ gốc; thêm hai HSG ở backbone bản sao |
| `modules/gcn_lib/mixgraph.py` | Cạnh đảo lặp lại chiều gốc; index dựa high-all-frames trong khi feature được ghép frame-major | HSG mới chạy độc lập từng frame, cạnh high↔low đối xứng, kiểm thử chỉ số |
| `modules/gcn_lib/temgraph.py` | Similarity là số không dương nhưng phép threshold `<0.05` đổi gần như toàn bộ sang 100, làm top-k không còn chọn khoảng cách | Chọn top-k khoảng cách Euclidean nhỏ nhất trên toàn bộ cặp patch giữa hai frame kề |
| `dataset/dataloader_video1.py` | `__len__` trừ 1; tự cắt gloss; thay sample CSL-Daily; fallback video kế tiếp | Dataset mới không mất mẫu, không thay mẫu, không cắt nhãn; lỗi dữ liệu dừng rõ |
| `seq_scriptsT.py` | Cắt token cuối mọi prediction bằng `[:-1]`; chọn đầu conv/LSTM tốt hơn riêng trên từng split | Decoder mới không thêm EOS CTC; giữ trọn prediction; chọn head trước khi test |
| `main2.py` | Khởi tạo distributed/device chồng Accelerate, đường dẫn máy tác giả, ghi config nhiều rank | Launcher/DDP riêng, một rank ghi output, đường dẫn tương đối |
| `main2.py` | Loader trọng số bỏ qua mọi tên có translation, kể cả dùng cho eval/resume | Eval/resume load strict toàn model; pretrain transfer tách riêng |
| `slrt_network.py` | Có thể bỏ qua thành phần loss NaN/Inf | Dừng khi loss không hữu hạn; CTC FP32, kiểm tra độ dài và repeat |
| `seq_scriptsT.py` | Scheduler step ở đầu và cuối epoch | Một scheduler step cuối epoch |
| `modules/gcn_lib/pos_embed.py` | `np.float` không còn trên NumPy mới | Bản sao dùng `np.float64` |
| Config và tokenizer | Tham chiếu `../Dataprocessing`, gloss ids, mBART pruned/map ids không có trong repo | Manifest JSONL, vocab train-only, mBART đầy đủ; đánh dấu khác biệt |

## Những giả định không được coi là xác nhận của tác giả

1. **Vị trí HSG:** lấy feature layer2 (28×28,128) và sau graph layer3 (14×14,256) cho HSG1; lấy feature stage trước (14×14,256) và sau graph layer4 (7×7,512) cho HSG2. Ghép hai chiều, hạ nhánh high bằng conv stride 2 rồi cộng nhánh low. Công thức mapping vùng theo Eq.1, nhưng vị trí nối chính xác thiếu trong release.
2. **Graph convolution:** mặc định EdgeConv cho LSG/TSG/HSG theo kết luận Appendix A.6 trang25. TSG/HSG dùng MLP `Linear(2C,C)+ReLU` và max aggregation; chi tiết MLP này là giả định vì file release còn dùng GCNConv. `graph_conv: gcn` đổi TSG/HSG về GCN chuẩn hóa đối xứng có self-loop để đối chiếu với release; LSG giữ EdgeConv gốc. Vẫn cần checkpoint/kiến trúc thật để xác nhận.
3. **Thứ tự graph:** mặc định TSG→LSG→HSG theo Table13b và kết luận rõ ở Appendix A.6 trang24. Sơ đồ có minh họa LSG→TSG→HSG. `graph_order: lsg-tsg` cho phép đối chiếu nhưng không tự chọn bằng test.
4. **Backbone:** giữ ResNet18 và các hyperparameter k=3,4,49,49 trong code; global temporal K là số cặp patch/frame-pair, không phải K láng giềng/mỗi patch. HSG mới không tương thích trực tiếp checkpoint tác giả chưa công bố.
5. **Tiền xử lý:** RGB 256→224 crop, horizontal flip 0.5, temporal scale 0.8–1.2, ImageNet normalize; giữ ý tưởng code nhưng dùng PIL LANCZOS thay OpenCV resize. Thứ tự RNG không giống code gốc.
6. **Nhãn/vocab:** chuẩn hóa gloss PHOENIX bằng hai hàm gốc, chữ thường; text chỉ lowercase/strip và giữ dấu câu cho SLT. TCTC dùng lemmatizer local do người dùng chọn; tác giả chưa cung cấp phiên bản NLP. Vocab train-only có `<blank>` và `<unk>`; khác vocab pickle tác giả. OOV phải báo cáo.
7. **Loss:** giữ hai CTC + self-distillation với hệ số 1,1,25 từ YAML; bài mô tả CTC/CE không khóa đầy đủ recipe. CE mới dùng PyTorch label smoothing thay custom KL smoothing trên vocabulary pruned.
8. **Huấn luyện:** Adam LR 1e-4, WD 1e-4, 50 epoch, milestones20/35, gamma0.2 từ repository; global batch6 suy từ 3×2. Không coi các con số này là recipe chính thức cho cả năm dataset. Feature-I3D cùng global temporal head là lựa chọn tái dựng cần đối chiếu.
9. **Môi trường:** paper PyTorch1.11, Docker mặc định2.5.1/CUDA12.4; kiểm thử local2.10/CUDA12.8. Không có bản lock môi trường gốc đầy đủ. Docker ghi `pip freeze` sau build để đóng băng environment thực tế; tái sử dụng image đã export để tránh drift dependency bắc cầu.
10. **Đánh giá:** WER Levenshtein chuẩn cost1 thay bộ alignment weighted của repo; có thể khác phân rã S/D/I (và alignment tối ưu). BLEU/ROUGE dùng chính file upstream. Python prefix beam10 không LM chưa được so sánh với binary ctcdecode. Không cam kết metric chính thức RWTH tương đương hoàn toàn; cần đối chiếu script đánh giá chính thức trước khi công bố.
11. **Tài sản ngôn ngữ:** dùng mBART CC25 đầy đủ; repo mong bộ model/tokenizer đã prune. Số tham số/VRAM không bằng Table12. Không đoán các map ids hoặc gloss embedding vắng mặt.
12. **Mẫu ngắn và CTC:** bộ đọc RGB train của bản tái dựng đã giữ tối thiểu `4*(số token + số cặp token liền nhau trùng nhau)-3` frame bằng cách lặp frame có sẵn khi cần, để đầu ra K5/P2/K5/P2 đủ bước CTC. Đây là chính sách bổ sung đã có, không phải recipe được tác giả xác nhận; không cắt gloss như loader gốc. Từ v10 doctor báo số mẫu cần xử lý này thay vì chặn theo độ dài RGB gốc. Feature train không có temporal resampling vẫn bị chặn nếu CTC không thể căn chỉnh. Dev/test giữ nguyên frame/nhãn: inference tính WER, không tính CTC loss; mẫu có ít bước decode hơn reference được báo cáo nhưng không bỏ, không resample theo nhãn. Guard độ dài và loss hữu hạn trong model vẫn giữ nguyên.
13. **Bộ nhớ GPU/RAM:** v11 chuyển toàn bộ tensor lưu cho backward sang CPU; log máy cô sau đó ghi tiến trình nhận SIGKILL (signal 9) gần micro-step 361. Hết RAM trong giới hạn Docker 21 GiB là nguyên nhân nghi ngờ, chưa được chứng minh chỉ từ exit code. V12 dùng mặc định `activation_offload: cpu_checkpoint`: checkpoint không reentrant theo khối đặc trưng, giữ đầu vào ranh giới khối trên CPU và tính lại tensor trung gian khi backward. Recompute giữ RNG và dùng bản sao buffer BatchNorm để không cập nhật running statistics hai lần. Không đổi độ phân giải, frame, graph K, nhãn, batch hiệu dụng 6, precision FP16, dropout 0.3 hoặc optimizer. Loader mặc định không chạy worker nền và không pin/prefetch batch, nhằm giảm RAM đồng thời; giới hạn Docker 21 GiB vẫn giữ nguyên, không dùng ổ đĩa làm nơi lưu activation. Đánh đổi là tính lại và truyền CPU↔GPU nhiều hơn. CPU/eval không áp dụng checkpoint CUDA này. Không tuyên bố bitwise-equivalent, nhất là backward MaxPool3d CUDA FP16 không xác định, hoặc bảo đảm mọi clip đều vừa RAM/VRAM. EdgeConv dùng nhóm tối đa 16 đồ thị độc lập thống nhất cho train/eval và mọi chế độ lưu activation, không chia node/cạnh bên trong đồ thị hay BatchNorm. Công thức không đổi nhưng làm tròn CUDA có thể khác v11; đối chiếu công thức bằng float64 và đối chiếu storage modes riêng biệt. Code fingerprint thay đổi nên deployment v12 tạo campaign mới, giữ checkpoint cũ thay vì bỏ guard hash. Launcher ghi thêm bằng chứng cgroup/Docker khi tiến trình bị dừng; phải đọc bằng chứng đó trước khi kết luận OOM. Kết quả kiểm thử từng phiên bản nằm ở `reports/VALIDATION.md`; chưa có số đo dev/test hoàn tất trên máy cô.

## Phạm vi các đường chạy

| Thí nghiệm | Có mã chạy bổ sung | Còn cần |
|---|---|---|
| PHOENIX14/14T CSLR | RGB→MixGraph→TemporalConv→BiLSTM→CTC | Dataset, ResNet weights, xác nhận HSG/recipe/metric |
| CSL-Daily CSLR | Cùng pipeline, manifest chuẩn | Dataset có quyền truy cập, split/annotation chuẩn |
| PHOENIX14T/CSL-Daily Sign2Text | Pretrain CSLR rồi mBART | mBART/tokenizer thật; chạy kiểm chứng SLT |
| Gloss-free Sign2Text | TCTC từ text rồi SLT | NLP model/segmentation/pseudo labels đúng recipe |
| How2Sign/OpenASL | Nhận I3D `[T,D]`, TCTC/SLT | Đặc trưng, extractor, split/version đúng tác giả |
| Sign2Gloss2Text | Chưa có runner bổ sung | Gloss-to-text recipe/vocab và pipeline riêng |
| Ablation PHOENIX14T–CSLR | 68 lượt trong runner local và hai notebook; xem PHOENIX14T_SUITE.md | Nghiệm thu dữ liệu thật, recipe/hyperparameters tác giả chưa công bố |
| Chi phí FLOPs / các task khác | Chưa sweep đầy đủ | Quy ước đo, cấu hình và tài nguyên tác giả |

## Cần tác giả xác nhận/cung cấp để đi tiếp tới tái lập chính xác

- Backbone MixSignGraph thật có HSG trong forward và checkpoint tương ứng.
- Loại graph conv, vị trí HSG, thứ tự module và initializer chính xác.
- Cấu hình training từng dataset/task/ablation, seed, batch, precision, criterion selection.
- Thư mục Dataprocessing đầy đủ: annotation splits, vocab gloss/text/pseudo, embedding, map ids, model/tokenizer đã prune.
- I3D extractor, checkpoint, sampling/normalization, dimension, manifests How2Sign/OpenASL. Table11 và mô tả A.2 có số split khác nhau; không tự giả định hai bản giống nhau.
- Bộ chấm điểm, cách normalize gloss/text, beam search settings và các checkpoint đã cho ra bảng paper.

Đây là danh sách tài sản thiếu, không phải email đã gửi. Dự án không tự liên hệ tác giả.

## Nguyên tắc đọc kết quả

Unit test/smoke chứng minh các invariants và khả năng chạy trong môi trường kiểm tra. Chúng **không** chứng minh kiến trúc là bản chính xác của tác giả, không chứng minh đã tái lập số paper. Mọi result mới phải đi kèm config/hash/environment và danh sách khác biệt này.
