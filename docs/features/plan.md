# Kế Hoạch Triển Khai Chi Tiết: TranhVe GPU CUDA Pipeline

Kế hoạch này mô tả các giai đoạn phát triển, thiết kế kiến trúc xử lý song song trên GPU và tích hợp mô hình ngôn ngữ lớn (LLM - Gemini 3.1 Flash Lite) của dự án **TranhVe** (Photo-to-Art Engine).

---

## 🚀 Giai Đoạn 1: Cấu Trúc Dự Án & Tăng Tốc GPU CUDA (Đã Hoàn Thành)
*   **Mục tiêu:** Thiết lập môi trường chạy CUDA chuyên dụng để giải quyết quá tải RAM/CPU khi xử lý ảnh độ phân giải cao.
*   **Các thành phần:**
    1.  Dựng Docker base image từ `nvidia/cuda:12.1.1-runtime-ubuntu22.04` để trực tiếp liên kết driver card đồ họa NVIDIA (RTX 3050 trở lên).
    2.  Tích hợp thư viện **CuPy** và `cupyx.scipy.ndimage` làm GPU layer cho xử lý NumPy mảng ảnh 3 chiều trực tiếp trên VRAM.
    3.  Tải mô hình tách nền `rembg` (U2NET) chạy trực tiếp trên GPU qua ONNX Runtime `CUDAExecutionProvider`.

---

## 🎨 Giai Đoạn 2: Tối Ưu Hóa 5 Phong Cách Vẽ Nghệ Thuật (Đã Hoàn Thành)
*   **Realistic Sketch (Phác thảo tả thực):** Lọc bỏ đốm trắng ngẫu nhiên, giữ lại 4 lớp nét chì chồng chéo (dodge blend, shadow darkening, Canny lines).
*   **Colored Pencil (Chì màu):** Tách 3 kênh màu RGB vẽ song song độc lập trên GPU, tạo cảm giác chì màu vẽ tay sống động.
*   **Anime Outline (Viền hoạt hình):** Dùng bộ lọc Bilateral làm phẳng màu sắc, cấu hình sigmas lớn trong XDoG để bỏ qua các nét xoáy nhỏ, chỉ viền các khối chính sắc nét.
*   **Oil Painting (Tranh sơn dầu):** Sửa lỗi Bilateral filter trên GPU, khôi phục nét đắp màu nổi (impasto) nguyên bản kết hợp phủ vân linen canvas thực tế.
*   **Paint-by-Numbers (Tranh số hóa):** Gom 12 mảng màu phẳng lớn bằng K-Means, lọc nhiễu các hạt màu nhỏ (<500px), đi viền ranh giới màu sắc và tự động định vị đánh số thứ tự chỉ dẫn dễ tô.

---

## 🤖 Giai Đoạn 3: Tích Hợp Gemini 3.1 Flash Lite API (Đã Hoàn Thành)
*   **Mô hình:** Gemini 3.1 Flash Lite (thông qua SDK `google-genai` mới nhất).
*   **Tính năng chính:**
    *   **Vision-to-Parameter Optimizer:** Nhận diện thuộc tính ảnh gốc (độ phức tạp, độ tương phản, chủ thể) thông qua phân tích ảnh đầu vào của Gemini.
    *   **Structured JSON Output:** Tự động trả về bộ tham số cấu hình thanh trượt hoàn hảo (`DrawingParams` chứa `blur_size`, `threshold_block`, `threshold_c`, `jitter`) dựa trên Structured Schema, giúp người dùng không cần chỉnh tay.
    *   **Fallback An Toàn:** Nếu API Gemini bị giới hạn tốc độ (Rate Limit) hoặc không có internet, hệ thống tự động fallback về bộ điều chỉnh cục bộ (Local Parameter Analyzer) lập tức để không bao giờ làm gián đoạn luồng vẽ.

---

## 🛠️ Giai Đoạn 4: Vá Lỗi Biên Tách Nền & Trải Nghiệm Web (Đã Hoàn Thành)
*   **Khắc phục lỗi tách nền bị thô nét:** Khi bật tách nền `rembg`, đối tượng được ghép lên một nền màu ấm trung tính `(240, 238, 233)` kết hợp làm mờ nhẹ alpha mask (Edge Feathering) thay vì nền trắng tinh gắt. Cách làm này triệt tiêu độ tương phản gắt ở viền cắt, giúp bộ dò nét tập trung bắt nét vẽ chi tiết vào bên trong chủ thể (mắt, mũi, lông thú cưng).
*   **Bảng điều khiển & Đo độ phủ nét vẽ:** Vá lỗi logic Quality Check (Kiểm Tra Chất Lượng), hỗ trợ tự động tìm file ảnh kết quả và so sánh phân tích cấu trúc, đề xuất gợi ý chỉnh thanh trượt thông minh.
