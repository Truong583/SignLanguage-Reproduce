# Nguồn mã và quyền sử dụng

- Mã gốc: https://github.com/gswycf/SignLanguage, commit `af5e8475d755b9d2e92c0142c8b7084651c3a4ee`.
- Các thư mục `MixSignGraph`, `SignGraph`, `SLTpose`, `SignMbart`, `doc` và `README.md` được giữ nguyên từ commit trên. Repository không cung cấp LICENSE cấp toàn dự án tại thời điểm kiểm tra. Không tự gán MIT/Apache cho toàn bộ bản sao này; cần hỏi tác giả nếu muốn phân phối/cấp phép lại ngoài phạm vi sử dụng được cho phép.
- `repro/vendor` là các bản sao có chỉnh sửa; `vendor_manifest.json` ghi nguồn và SHA-256. Giữ nguyên thông báo Huawei/Meta có trong các file graph/position embedding.
- Bộ tính BLEU đi kèm repository có thông báo Apache-2.0 của Amazon; bộ ROUGE ghi nguồn tf_seq2seq/sumy. Giữ nguyên các thông báo trong file gốc.
- Paper: Gan et al., *MixSignGraph: A Sign Sequence is Worth Mixed Graphs of Nodes*, NeurIPS 2025. Bản PDF được giữ để đối chiếu nghiên cứu, không coi là nội dung do dự án này sáng tác.
- Dataset, ImageNet ResNet, mBART, mô hình ngôn ngữ spaCy có nguồn và điều kiện sử dụng riêng; không được coi là đã kèm sẵn hoặc đã được cấp quyền bởi bản đóng gói này.

Phần bổ sung `repro/`, scripts, Docker và tài liệu là bản tái dựng phục vụ yêu cầu của người dùng; không phải bản phát hành chính thức của nhóm tác giả và không chứng minh các con số trong paper đã được tái lập.

Bổ sung backbone: `repro/ablation.py` có feature trunk PyViG-Tiny tương thích trọng số chính thức, đối chiếu module/shape theo Huawei Efficient-AI-Backbones, commit `a328aa023d4e3e66cca690ecc54f77daf436b9ff`, `vig_pytorch/pyramid_vig.py`. Không thêm giấy phép toàn dự án khi upstream không công bố LICENSE ở root. Swin-T dùng implementation torchvision và checkpoint ImageNet chính thức; URL và SHA-256 cả hai backbone được khóa trong `scripts/fetch_backbones.py`. Các trọng số tải vào assets không nằm trong ZIP source.
