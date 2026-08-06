# Danh Sách Công Việc (Tasks List): Smart Portrait-to-Sketch Project

Dưới đây là danh sách các nhiệm vụ cụ thể cần triển khai để hoàn thiện dự án.

---

## 🛠️ Bước 1: Khởi Tạo Dự Án & Cấu Hình Môi Trường
- [x] Thiết lập môi trường ảo Python `venv`
- [x] Tạo file `requirements.txt` với các thư viện cần thiết (`opencv-python-headless`, `pillow`, `rembg`, `requests`, `gradio`)
- [x] Tạo cấu trúc thư mục dự án (`src/`, `output/`, `test_images/`)

## Tasks

- [x] Khởi tạo dự án & Cấu hình môi trường ảo, file `requirements.txt`
- [x] Lập trình Module Loader (`src/loader.py`) để tải/đọc ảnh
- [x] Lập trình Module xử lý ảnh nâng cao (`src/processor.py`):
    - Tách nền bằng AI local (`rembg`)
    - Tách contours bằng OpenCV và đơn giản hóa nét vẽ bằng RDP
    - Tạo các bộ lọc nét vẽ: Rung nét tay (Jitter), độ dày nét bút, vẽ gạch bóng (Hatching)
    - Tạo mảng màu K-Means cho chế độ vẽ màu
- [x] Lập trình Module xuất file (`src/exporter.py`) để tối ưu hóa ảnh PNG và xóa EXIF
- [x] Lập trình API Trợ lý Gemini (`src/assistant.py`) gọi model Gemini 3.1 Flash Lite
- [x] Lập trình giao diện Web App bằng Gradio (`app.py`):
    - Kéo thả ảnh đầu vào
    - Chọn chế độ vibe (Nét trắng đen, Nét chì màu, Nét đen mảng màu)
    - Vẽ động từng nét vẽ lên canvas web sử dụng generator `yield`
    - Tải ảnh PNG kết quả
- [x] Cấu hình Dockerization (`Dockerfile` & `docker-compose.yml`) ánh xạ cổng ra `localhost`
- [x] Kiểm thử toàn bộ hệ thống bằng ảnh mẫu
- [x] Lập trình bộ dò nét vẽ hỗn hợp Canny + Adaptive threshold.
- [x] Tích hợp cơ chế tự phục hồi khi rembg xóa trắng nền vẽ.
- [x] Lập trình cơ chế phát lại quá trình vẽ (Replay) phía client mượt mà 30 FPS.
- [x] Lập trình tạo video quá trình vẽ MP4 và đóng gói ZIP.
- [x] Thêm nút Xem toàn màn hình (HTML5 Fullscreen API).
- [x] Viết script kiểm thử tự động toàn bộ API Endpoint và ZIP file.

## 🎨 Bước 2: Phát Triển Core Engine (Xử Lý Ảnh & Vector)
- [x] Viết Module Loader (`src/loader.py`) để tải ảnh từ URL hoặc đường dẫn local
- [x] Viết Module Tách Nền (`src/processor.py` - phần tách nền bằng `rembg`)
- [x] Viết Module Vẽ Nét (`src/processor.py` - phần thuật toán lấy biên bằng OpenCV)
- [x] Viết thuật toán đơn giản hóa điểm nét vẽ (`approxPolyDP`) và module điều khiển vẽ bằng `turtle`
- [x] Lập trình mô phỏng nét vẽ tay: thuật toán rung tay (Gaussian Noise) và nét vẽ đậm nhạt (Dynamic Pensize)
- [x] Lập trình thuật toán gạch bóng tạo khối (Hatching) cho các vùng đổ bóng tối
- [x] Lập trình phân lớp phong cảnh: dùng K-Means phân tông màu xám và sắp xếp thứ tự vẽ từ xa đến gần
- [x] Viết Module Exporter (`src/exporter.py`) để xuất ảnh PNG tối ưu (nén tốt, xóa EXIF) và chuyển đổi PostScript sang PNG

## 🌐 Bước 3: Phát Triển Giao Diện Web App Local
- [x] Thiết lập tính năng tải ảnh PNG kết quả về máy người dùng
- [x] Chuyển đổi từ giao diện Gradio mặc định sang giao diện FastAPI + Vanilla CSS/JS tối giản, hiện đại, bản địa hóa tiếng Việt.
- [x] Tích hợp nút phóng to toàn màn hình HTML5 và tính năng xem lại (Replay) động phía Client.

## 🧪 Bước 4: Kiểm Thử & Kiểm Định Chất Lượng
- [x] Thu thập 5 ảnh test khác nhau (chân dung, thú cưng, đồ vật, phong cảnh)
- [x] Chạy thử nghiệm và tinh chỉnh tham số mặc định của thuật toán OpenCV để có nét vẽ tự nhiên nhất
- [x] Kiểm tra tính năng tách nền trong suốt trên ảnh PNG kết quả

## 🐳 Bước 5: Đóng Gói Docker & Tối Ưu Môi Trường
- [x] Tạo file `Dockerfile` tối ưu dựa trên image python-slim
- [x] Viết cấu hình `docker-compose.yml` để mount volume chứa model AI local và ảnh output
- [x] Lập trình module xuất SVG vẽ động trên web browser để có thể chạy độc lập trong Docker container không cần Tkinter host GUI
- [x] Chạy thử nghiệm build Docker và kiểm tra lỗi cài đặt OpenCV/rembg trong container

## 🤖 Bước 6: Tích Hợp Bộ Tiền Xử Lý Gemini Vision Auto-Tune
- [x] Tạo file cấu hình môi trường `.env` chứa `GEMINI_API_KEY`
- [x] Thiết lập module kết nối và gọi Gemini API (`src/assistant.py`) sử dụng thư viện `google-genai` mới
- [x] Sử dụng Structured Outputs (Pydantic schema) để Gemini tự động trả về bộ siêu tham số OpenCV tối ưu
- [x] Tích hợp Auto-Tune vào nút vẽ của giao diện `app.py` để tự động cập nhật thanh trượt thông số
- [x] Chạy thử nghiệm thực tế với Gemini API Key hợp lệ
