# Log W&B, mất mạng và tắt máy

`live_log_tail.txt` là bản log gần nhất (tối đa32KiB) được lọc key trước khi gửi W&B. Trong project, chọn run `supervisor-...` mới nhất → Files → live_log_tail.txt. Metadata và requirements là thông tin môi trường, không phải tiến độ. Summary `status.json/stage` phân biệt prepare (chuẩn bị/tải dữ liệu) và train; trạng thái Running của run observer chỉ nói tiến trình giám sát, không chứng minh GPU đang train.

## Cập nhật một lần trên máy cô

Để nạp phần sửa log và chạy nền, dừng lệnh đang chạy bằng Ctrl+C rồi đợi dấu nhắc Terminal. Đang tải sẽ giữ `.part`; có thể mất tối đa phần khối đọc chưa được ghi. Sau đó:

```bash
cd /mnt/annie/Truong_K17/SignLanguage-Reproduce
git pull --ff-only
python3 update.py --automatic
python3 service.py install --single
```

Chỉ chạy dòng tiếp theo nếu dòng trước thành công. `install --single` tạo một system service riêng theo đường dẫn dự án. Sudo chỉ dùng để cài/bật service này; tiến trình thực tế chạy bằng tài khoản annie, Group=docker, không chạy Python/training bằng root. Service chờ Docker, phụ thuộc mount chứa dự án, tự khởi động khi Linux bật lại. Không phải nhập lại W&B key đã lưu. Không cài hoặc đổi driver. Không tự cài dịch vụ trên máy phát triển Windows.

Service mặc định chạy cấu hình chính, giống `run.py --single`. Chỉ chọn `install --suite` khi muốn cả68 lượt và đã đủ dung lượng. Không chạy thêm run.py trong Terminal khi service đã chạy; khóa workspace ngăn hai chiến dịch cùng ghi dữ liệu.

Xem tiến độ ngay trên máy cô:

```bash
python3 service.py logs
```

Ctrl+C ở cửa sổ xem logs chỉ đóng việc xem, không dừng service. Đóng Terminal, khóa màn hình hoặc đăng xuất vẫn giữ service. Máy phải bật, có điện; dịch vụ không thể chạy khi máy đã tắt. Không cần enable-linger vì đây là system service có User=annie, không phải user service cũ.

Xem tình trạng hoặc chủ động dừng trước khi tắt:

```bash
python3 service.py status
python3 service.py stop
```

Đợi status hiển thị inactive rồi dùng Shutdown của Ubuntu. Service được bật ở boot nên lần bật máy sau sẽ chạy lại. Muốn tạm giữ máy không tự train ở các lần boot sau thì `sudo systemctl disable <tên-unit>`; tên unit được in khi install và lưu ở `.updates/service.json`. Không xóa checkpoint hoặc `.part` để dừng chạy.

## Nếu cài service báo "bad unit file setting"

Bản v9 sửa `WorkingDirectory` thành đường dẫn scalar không bọc dấu ngoặc kép. `ExecStart` vẫn giữ cách quote từng đối số. Trước khi cài, helper chạy `systemd-analyze verify` bằng parser ngay trên máy đích; cấu hình sai sẽ không được cài đè. Bản sửa không thay code huấn luyện hoặc xóa file dữ liệu `.part`.

Nếu lỗi này xuất hiện sau khi Docker dựng thành công, lấy bản sửa rồi cài lại:

```bash
cd /mnt/annie/Truong_K17/SignLanguage-Reproduce &&
git pull --ff-only &&
python3 update.py --automatic &&
python3 service.py install --single &&
python3 service.py status
```

Chỉ đóng Terminal sau khi thấy service `active (running)`. Nếu vẫn lỗi, gửi kết quả `systemctl status signlanguage-8afccb21070c.service --no-pager -l` và `sudo journalctl -u signlanguage-8afccb21070c.service -n 50 --no-pager`. Không chạy thêm `run.py` khi service đã hoạt động.

## Nếu doctor báo Impossible CTC alignment sau khi tải đủ PHOENIX14T

Bản v10 sửa kiểm tra RGB train theo chính sách temporal resampling đã có trong bộ đọc dữ liệu, thay vì kiểm tra độ dài frame gốc. Không đổi nhãn, không bỏ mẫu, không thay kiến trúc/loss hoặc fingerprint code huấn luyện. Doctor in số mẫu train cần tăng độ dài và số mẫu dev/test bị giới hạn độ dài decode; dev/test không được tăng độ dài dựa trên nhãn. Feature train vẫn dừng khi căn chỉnh CTC bất khả thi.

Nếu service đang hoạt động và pipeline vừa dừng do lỗi này, không chạy thêm `run.py` hay tải lại dữ liệu. Sau khi bản sửa được publish, supervisor sẽ lấy bản mới khi kiểm tra GitHub, thường khoảng 5 phút một lần trong thời gian chờ. Log lượt mới phải có `PHOENIX14T already prepared` và báo cáo doctor trước khi train. Cần mạng để lấy bản sửa; `active (running)` của service chỉ chứng minh supervisor đang chạy, không chứng minh huấn luyện đã thành công.

## Khi mất mạng hoặc kết nối W&B bị ngắt

- Đang tải: timeout được thử lại tối đa8 lần, giữ `.part`. Server hỗ trợ Range thì tải tiếp; nếu không thì phải tải lại archive. Hết giới hạn thì chờ, không coi là lỗi mô hình. Gửi log nếu cần xử lý; không xóa `.part`.
- Đã đủ data, weights và image: train container vốn tắt mạng nên tiếp tục được. Host dùng lại image đã dựng; không gọi registry để dựng lại nếu tag đã có.
- W&B/kiểm tra GitHub tạm thời không truy cập được. Log/metrics gốc vẫn ở runs. Khi mạng trở lại observer thử kết nối lại, gửi log hiện tại và scan lịch sử. Khi observer được khởi động lại, history được đọc lại cho session mới để tránh bỏ sót metrics chỉ nằm trong bộ đệm offline của session cũ. Chỉ các scalar và log được gửi, không video/checkpoint/key.

## Khi máy mất điện hoặc bị tắt

- Tải dở: dữ liệu đã ghi trong `.part` nằm trên ổ; sau khi bật lại, tiếp tục nếu server hỗ trợ.
- Giải nén dở: có thể phải giải nén lại từ archive đã tải, không cần tải lại archive còn nguyên.
- Huấn luyện cấu hình đơn: checkpoint mặc định mỗi200 optimizer updates và cuối epoch; optimizer/scheduler/scaler/RNG/cursor nằm trong payload. Một backend local riêng giữ ít nhất hai thế hệ checkpoint có checksum. Khi bắt đầu lại, restore thử bản commit mới nhất; lỗi checksum thì thử thế hệ trước. Nếu mọi bản đều lỗi, dừng để xử lý, không tự train lại từ đầu.
- Chỉ những gì đã lưu thành công được phục hồi; phần sau checkpoint phải chạy lại. Mất điện khi đang ghi, hỏng filesystem/ổ đĩa vẫn có thể làm mất dữ liệu. Không thể bảo đảm phục hồi mọi sự cố phần cứng. Giữ backup checkpoint quan trọng khi đủ chỗ.
- Lượt đang chạy hoặc bị ngắt sẽ được tiếp tục khi khởi động lại. Lượt đã hoàn tất không được tự coi là một thí nghiệm mới. Lượt bị lỗi được ghi nhận sẽ chờ bản sửa, không tự chạy cùng lỗi vô hạn.

Chưa nghiệm thu systemd/khởi động lại/tắt điện thật trên máy cô. Kiểm tra status, log và checkpoint sau lần khởi động đầu tiên; không thử rút điện để kiểm tra tính năng này.
