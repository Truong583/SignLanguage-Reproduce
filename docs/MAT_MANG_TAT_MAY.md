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

## Khi train báo CUDA OOM hoặc bị dừng bởi signal 9

V10 gặp thiếu VRAM. V11 chuyển tensor trung gian sang RAM; log máy cô sau đó ghi SIGKILL (signal 9) gần micro-step 361. Đây là tiến trình bị buộc dừng, chưa đủ để kết luận hỏng máy hay chắc chắn hết RAM. RAM chạm giới hạn Docker 21 GiB là nguyên nhân nghi ngờ cần đối chiếu với cgroup/Docker.

V12 dùng `cpu_checkpoint`: lưu đầu vào ranh giới khối trên CPU, tính lại tensor trung gian từng khối khi backward, thay vì giữ toàn bộ chúng trong RAM. RNG được giữ cho recompute; buffer BatchNorm được sao chép để không cập nhật running statistics hai lần. WORKERS=0 và tắt pin/prefetch hạn chế số batch nằm trong RAM cùng lúc. Không đổi batch hiệu dụng, độ phân giải, nhãn, frame, dropout hoặc graph K; không tăng giới hạn RAM/CPU Docker và không dùng disk offload. Tốc độ có thể giảm. Chưa thể bảo đảm mọi mẫu đều vừa bộ nhớ; cần xác nhận bằng log lần chạy thật.

EdgeConv xử lý tối đa 16 đồ thị độc lập mỗi nhóm, dùng cùng quy tắc trong train/eval và mọi chế độ lưu activation. Mỗi đồ thị giữ đủ node/cạnh; không chia trục thời gian của TSG, không chia BatchNorm và không cắt video/nhãn. Cách tính theo nhóm giữ công thức toán học; so với phép nhân batched lớn của v11 có thể khác làm tròn CUDA, nên không tuyên bố bitwise giống checkpoint cũ. Số đo kiểm tra tổng hợp được ghi trong `reports/VALIDATION.md`; không thay thế nghiệm thu trên máy cô hay dữ liệu thật.

Nếu service đang chờ sau lỗi v11, supervisor lấy v12 sau khi bản sửa được publish và tự tạo campaign mới theo DEPLOYMENT_POLICY.json do fingerprint code đổi. Không cần bấm Stop run trên W&B hoặc chạy run.py thêm. Archive, dữ liệu giải nén, weights và checkpoint cũ được giữ. Log lượt mới phải có release `phoenix14t-cslr-bounded-activation-checkpoint-v12` và `Activation storage: cpu_checkpoint`. Nếu lại bị dừng, log có thêm dữ liệu cgroup/Docker để phân biệt giới hạn RAM với nguyên nhân khác; gửi phần cuối `live_log_tail.txt` và `diagnostic.json`. Không tự tăng giới hạn RAM để né lỗi.

Các run supervisor là phiên giám sát CPU. Trạng thái Running không chứng minh GPU đang train: kiểm tra `status.json/status`, `supervisor.json/outcome` và heartbeat mới nhất trong Summary. V11 thêm xử lý SIGTERM và SDK finish để kết thúc phiên observer khi cập nhật; phiên cũ v9/v10 có thể còn trạng thái lỗi thời. Không thể bảo đảm gửi trạng thái cuối nếu bị kill cứng hoặc mất mạng.

## Khi mất mạng hoặc kết nối W&B bị ngắt

- Đang tải: timeout được thử lại tối đa8 lần, giữ `.part`. Server hỗ trợ Range thì tải tiếp; nếu không thì phải tải lại archive. Hết giới hạn thì chờ, không coi là lỗi mô hình. Gửi log nếu cần xử lý; không xóa `.part`.
- Đã đủ data, weights và image: train container vốn tắt mạng nên tiếp tục được. Host dùng lại image đã dựng; không gọi registry để dựng lại nếu tag đã có.
- W&B/kiểm tra GitHub tạm thời không truy cập được. Log/metrics gốc vẫn ở runs. Khi mạng trở lại observer thử kết nối lại, gửi log hiện tại và scan lịch sử. Khi observer được khởi động lại, history được đọc lại cho session mới để tránh bỏ sót metrics chỉ nằm trong bộ đệm offline của session cũ. Chỉ các scalar và log được gửi, không video/checkpoint/key.

## Khi máy mất điện hoặc bị tắt

- Tải dở: dữ liệu đã ghi trong `.part` nằm trên ổ; sau khi bật lại, tiếp tục nếu server hỗ trợ.
- Giải nén dở: có thể phải giải nén lại từ archive đã tải, không cần tải lại archive còn nguyên.
- Từ v12, huấn luyện cấu hình đơn qua `run_local.py` lưu checkpoint mặc định mỗi 50 optimizer updates và cuối epoch. Với máy cô hiện tại (1 GPU, micro-batch 1, accumulation 6), 50 updates tương ứng 300 micro-step. Notebook/suite vẫn mặc định 200 updates để hạn chế thời gian upload checkpoint Drive. Optimizer/scheduler/scaler/RNG/cursor nằm trong payload. Một backend local riêng giữ ít nhất hai thế hệ checkpoint có checksum. Khi bắt đầu lại, restore thử bản commit mới nhất; lỗi checksum thì thử thế hệ trước. Nếu mọi bản đều lỗi, dừng để xử lý, không tự train lại từ đầu.
- Chỉ những gì đã lưu thành công được phục hồi; phần sau checkpoint phải chạy lại. Mất điện khi đang ghi, hỏng filesystem/ổ đĩa vẫn có thể làm mất dữ liệu. Không thể bảo đảm phục hồi mọi sự cố phần cứng. Giữ backup checkpoint quan trọng khi đủ chỗ.
- Lượt đang chạy hoặc bị ngắt sẽ được tiếp tục khi khởi động lại. Lượt đã hoàn tất không được tự coi là một thí nghiệm mới. Lượt bị lỗi được ghi nhận sẽ chờ bản sửa, không tự chạy cùng lỗi vô hạn.

Đã xác nhận service systemd chạy trên máy cô; chưa nghiệm thu khởi động lại/tắt điện thật. Kiểm tra status, log và checkpoint sau lần khởi động đầu tiên; không thử rút điện để kiểm tra tính năng này.


## V13: laptop đóng vẫn giữ lịch sử train/dev

Viewer trên laptop ở `D:\Chay_Paper\SignLanguage-Monitor` lưu điểm loss, checkpoint, tốc độ đã quan sát, tổng kết epoch và log đã nhận. Lịch sử là dữ liệu cục bộ trong `.monitor-cache`, không phải bộ nhớ tạm của tab trình duyệt. Bản cũ có cache loss được đọc tiếp, không xóa hay khởi tạo lại dữ liệu người dùng.

W&B history chứa loss train trung bình và WER dev sau mỗi epoch; viewer truy vấn lại lịch sử thật theo trang, kể cả laptop đã tắt. Các campaign khác nhau được chọn riêng trong bảng tổng kết, không nối số đo của chúng thành một thí nghiệm. Run observer mới vẫn có danh sách lịch sử riêng; khi tự chuyển run, có thể chọn run cũ để xem tiếp.

V13 bổ sung observer gửi các đoạn console đã lọc thông tin xác thực vào `console_archive/<diagnostic-id>/<byte-offset>.txt` và manifest `console_archive/index.json`. Mỗi poll đọc tối đa hai khối 256 KiB cho console đang hoạt động và hai khối cho một console cũ còn trong `runs/diagnostics`. Cursor theo byte gốc; log UTF-8 giữ đủ dòng hoàn chỉnh. Mất mạng không xóa log host; sau khi kết nối/khởi động lại observer sẽ gửi lại. Không đọc hoặc gửi video, weights, checkpoint, key. Chỉ gửi vào project đã được kiểm tra Private/Team/Restricted theo guard có sẵn. Kho log có thể tăng dung lượng W&B và spool tương ứng; không đặt giữ dữ liệu vô hạn ở dịch vụ ngoài mà không tính quota.

Khi observer V13 đã được nạp, laptop có thể tải các phần console phát sinh lúc nó tắt. Viewer có lựa chọn phiên log, lọc train/dev/epoch, xem phần cũ hơn và tải `.txt`. Console lịch sử còn trên ổ host cũng được tải dần; mất/hỏng/xóa log host thì không thể phục hồi từ code. Khi notebook/teacher dùng observer cũ chỉ gửi tail 32 KiB, không thể khôi phục toàn bộ log bị ghi đè từ W&B; viewer báo rõ phạm vi thu thập. V13 không bổ sung log từng video dev vào vòng đánh giá; bảng dev dùng kết quả tổng kết thật của epoch.

Bản này không sửa `repro/`, config, dữ liệu, checkpoint hoặc thuật toán. Training fingerprint và recipe được đối chiếu với V12 và không đổi. Supervisor tiếp tục pin source khi worker đang chạy; không tự nạp V13 giữa lượt train. Máy cô nhận bản mới theo lịch cập nhật đã có sau khi worker kết thúc/dừng. Không cần dừng lượt hiện tại chỉ để cập nhật giao diện laptop. Cài viewer mới không có nghĩa observer trên máy cô đã được đổi. Muốn đổi observer ngay giữa lượt cần thao tác riêng trên máy cô; không có SSH nên không khẳng định đã làm từ laptop.

Kiểm chứng: viewer 26 kiểm tra về persistence/readonly API, epoch backfill, cô lập run, giữ log và mất mạng; observer/background/supervision 43 kiểm tra qua. Truy vấn W&B chỉ đọc đã lấy được tổng kết epoch 1–7 của run người dùng; API key đọc từ vault không được in, ghi vào cache hoặc thay đổi. Các số đo này không phải kết quả cuối cùng của paper.
