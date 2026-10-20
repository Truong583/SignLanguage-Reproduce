# SLT dài trên máy cô

Luồng: laptop chuẩn bị bản sửa → GitHub chứa code và lệnh chạy ghim commit → dịch vụ `signlanguage-slt-worker.service` tải việc mới khi rảnh → Docker train → container riêng gửi log và metric tới W&B.

- Dữ liệu, mBART và checkpoint CSLR gắn chỉ đọc.
- Kết quả mới: `/mnt/annie/Truong_K17/SignLanguage-Reproduce/runs/phoenix14t_slt_cc25_cpuadam_v1`.
- `last.pt` phục hồi, `best.pt` chọn theo ROUGE trên dev; lịch sử ở `history.jsonl`. Ghi sau cập nhật đầu và mỗi 100 cập nhật, cùng cuối epoch. Chỉ giữ bản last/best, không tạo 50 checkpoint lớn.
- Train 50 epoch; Adam LR base 1e-4, mBART 1e-5, mapping 1e-3; giảm LR tại epoch20/35 với gamma0.2. Microbatch1 tích lũy6. Không freeze mBART. Cuối cùng đánh giá best trên test.
- Giới hạn Docker 21GiB RAM, 2CPU; GPU allocator85%. Adam moments trên CPU. Kiểm tra còn ít nhất30GiB trống trước khi chạy; mỗi lần lưu chừa10GiB sau dung lượng ghi tạm dự kiến.
- Dừng có xử lý tín hiệu sẽ cố lưu tại ranh giới cập nhật. Mất điện/kill/OOM chỉ đảm bảo dùng checkpoint đã lưu còn đọc được; phần chưa lưu phải chạy lại. Mất mạng không chặn GPU train, observer thử gửi log lại mỗi60giây.
- Dịch vụ không thay code đang chạy. Push code đơn thuần chưa phát lệnh chạy: cần manifest `automation/slt-job.json` với job_id mới, commit và SHA256. Lỗi không tự lặp vô hạn. Bản sửa cần giữ đúng cấu hình/trọng số tương thích; thay thuật toán không được tự gắn nhãn resume tương đương.
- Một lượt worker tối đa7ngày; engine chủ động lưu và kết thúc phiên sau6ngày nếu chưa xong. Có thể phát việc mới từ laptop để resume; không cần SSH. Reboot trong lúc chạy đánh dấu việc cũ interrupted, cần job_id mới để nhận lại. Không cam kết tự tiếp tục sau reboot với worker hiện tại.

Đây là Sign2Text tái dựng dùng mBART CC25 đầy đủ và CSLR epoch22 đã chốt. Tác giả dùng model/vocabulary đã prune chưa có tài sản tương ứng, nên không tuyên bố khớp toàn bộ paper. Xem `SLT_AUDIT_20261019.md`.
