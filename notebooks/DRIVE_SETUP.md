# Chạy notebook Colab/Kaggle

Mục tiêu là: sau một lần chuẩn bị Drive và quyền Kaggle, bật GPU/Internet rồi bấm **Run all**. Notebook tự lấy mã, tự tải PHOENIX14T nếu chưa có, giải nén và lưu checkpoint/kết quả vào Drive.

## Một lần chuẩn bị Google Drive

1. Tạo thư mục riêng tư `My Drive/SignLanguage-Reproduction`.
2. Tải `SignLanguage-Reproduce-portable.zip` trong thư mục dự án lên thư mục Drive đó.
3. Không cần tự tải PHOENIX14T hoặc tự tạo `phoenix14t.zip`. Colab/Kaggle tự tải archive chính thức `phoenix-2014-T.v3.tar.gz` từ [RWTH Aachen](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) khi archive chưa có trong thư mục Drive.

Archive gốc khoảng 39 GB, vì vậy Drive cần còn đủ dung lượng để lưu nó. Nếu Drive không đủ chỗ, notebook Kaggle vẫn cố tải và chạy trong phiên hiện tại, nhưng không cache archive; lần sau sẽ phải tải lại. Colab cần lưu archive lên Drive để dùng lại. Dữ liệu giải nén nằm ở ổ tạm của runtime, không nhân đôi hàng trăm nghìn ảnh trên Drive.

## Một lần cấp quyền Drive cho Kaggle

Kaggle không tự mount Google Drive; notebook kết nối Drive bằng OAuth. Làm các bước sau một lần:

1. Trong Google Cloud Console, tạo project, bật **Google Drive API**, cấu hình OAuth consent screen, rồi tạo OAuth Client ID loại **Desktop app** và tải file client JSON về máy tin cậy.
2. Trong thư mục dự án trên máy, chạy:

   ```powershell
   python -m pip install google-auth-oauthlib
   python scripts/google_drive_oauth.py --client-secrets "C:\duong-dan\client_secret.json" --output "gdrive_oauth_credentials.json"
   ```

3. Đồng ý quyền Google trong trình duyệt. Sao chép nội dung file `gdrive_oauth_credentials.json` vào **Kaggle → Add-ons → Secrets** với tên `GOOGLE_DRIVE_OAUTH_JSON`, rồi cho notebook sử dụng secret. Không chia sẻ hoặc tải JSON này lên GitHub/notebook.
4. Trong ô cấu hình Kaggle, điền ID thư mục `SignLanguage-Reproduction` (phần sau `/folders/` trong URL Drive).

Nếu OAuth app đang ở chế độ Testing, Google có thể yêu cầu cấp quyền lại khi token hết hạn; khi đó chạy lại helper và cập nhật Kaggle Secret. Chỉ dùng secret trong notebook riêng tư tin cậy.

## Bấm chạy

- **Colab:** mở notebook, bật GPU và Internet trong Runtime settings, rồi chọn **Runtime → Run all**. Lần đầu Google yêu cầu cho phép gắn Drive.
- **Kaggle:** mở notebook, bật GPU và Internet trong Notebook options, điền folder ID và bật quyền dùng Kaggle Secret, rồi chọn **Run All**.

Notebook lưu `last.pt`, `best.pt`, log và metrics vào `SignLanguage-Reproduction/runs/`. Khi runtime bị ngắt, chạy lại cùng notebook với cùng dataset/phase/seed để tiếp tục từ epoch gần nhất đã hoàn tất. Cần đủ ổ đĩa tạm để giải nén dữ liệu; dung lượng output Kaggle có giới hạn, nên output/checkpoint được lưu lên Drive.

PHOENIX14T là bộ dữ liệu lớn, tải lần đầu có thể mất nhiều giờ tùy đường truyền. Notebook hỗ trợ tiếp tục tải file chưa hoàn tất; không xóa file `.part` nếu muốn tiếp tục.
