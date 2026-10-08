# Kiểm tra nhánh master

## Kết luận

Giữ bộ chạy hiện tại dựa trên `main`; không có bản triển khai hoàn chỉnh hơn ở đầu nhánh `master` đã kiểm tra để thay thế. Không sửa thuật toán hoặc Docker chỉ vì đổi nhánh.

Hai phiên bản đối chiếu từ bản clone:

- `main`: `af5e8475d755b9d2e92c0142c8b7084651c3a4ee`.
- `master`: `3d0f767c06b99f156ff016261f639d80b8191a8f`.

## Bằng chứng

| Thành phần | master so với main |
|---|---|
| `MixSignGraph/modules/gcn_lib` và `gcn_lib_hgnn` | Không có ở đầu master, trong khi backbone vẫn import `gcn_lib` |
| `MixSignGraph/modules/loss.py` | Chỉ có dòng `to do`, không phải mã Python hợp lệ |
| `MixSignGraph/modules/translation.py` | Thiếu class XentLoss và phương thức forward của SLTModel; slrt_network vẫn import XentLoss |
| `MixSignGraph/modules/resnet.py` | Phần mô hình/forward giống main; khác biệt chỉ loại bỏ đoạn test cuối file, không thêm HSG |
| `MixSignGraph/README.md` | Vẫn ghi code chưa chuẩn bị xong |
| `SignGraph` | Không có khác biệt giữa hai đầu nhánh đã đối chiếu |
| Training, dataset loader, config của MixSignGraph | Không có bản sửa mới trong master so với main |

Lịch sử `master` cho thấy commit `131ba1d` ngày 2026-04-27 loại bỏ các file graph nói trên. Lịch sử file đã kiểm tra có `resnet_mix.cpython-38.pyc` và `resnet_mixP.cpython-38.pyc`, nhưng không có source `.py` tương ứng; không coi bytecode cache là bản source HSG hoàn chỉnh được xác nhận.

## Phạm vi kiểm tra và thay đổi

Đã đọc cây file, diff của hai nhánh, code backbone/translation/loss và lịch sử các file graph liên quan. Không thực thi mã master. Chỉ bổ sung báo cáo này và cập nhật checksum/ZIP; mã chạy bổ sung giữ nguyên, không cần lặp lại kiểm thử mô hình.

Trang nhánh đã truy cập: https://github.com/gswycf/SignLanguage/tree/master

Lần `git fetch` để xác nhận cập nhật mới nhất không thực hiện được: hệ thống duyệt tự động báo hết hạn mức, không phải kết luận hành động nguy hiểm. Vì vậy kết luận về code được giới hạn ở hai commit ghi trên, không khẳng định đã xác minh mọi thay đổi mới sau bản clone.
