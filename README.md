# TranhVe — Smart Portrait-to-Sketch Engine (GPU CUDA Accelerated)

Dự án chuyển đổi ảnh chân dung thành các tác phẩm nghệ thuật dạng nét vẽ tay, tranh sơn dầu, tranh anime và tranh số hóa tự tô màu thông minh. Hệ thống hỗ trợ xử lý tăng tốc phần cứng thông qua **NVIDIA GPU CUDA (CuPy + ONNX Runtime)** và tương tác với Gemini 3.1 Flash Lite API để tối ưu thông số vẽ.

> [!IMPORTANT]
> Toàn bộ pipeline xử lý hình ảnh phức tạp đã được tăng tốc bằng **GPU CUDA** thông qua thư viện **CuPy** (tối ưu hóa trên card đồ họa RTX 3050 trở lên), giúp nâng tốc độ vẽ lên gấp 5-10 lần và tiết kiệm tài nguyên CPU/RAM.

---

## 📖 Hệ Thống Tài Liệu Dự Án

1.  **Ý tưởng gốc & Tiến hóa:**
    *   [ideas.md](file:///c:/Users/QuangNe/Downloads/Projects/web/tranhve/ideas.md): Ý tưởng ban đầu.
    *   [ideas_evolution.md](file:///c:/Users/QuangNe/Downloads/Projects/web/tranhve/docs/features/ideas_evolution.md): Đánh giá 6 chiều (Darwin Engine) chọn lọc phương án tối ưu.
2.  **Đặc tả kỹ thuật & Kiến trúc:**
    *   [architecture.md](file:///c:/Users/QuangNe/Downloads/Projects/web/tranhve/architecture.md): Sơ đồ hệ thống, phân lớp module GPU/CPU.
    *   [spec.md](file:///c:/Users/QuangNe/Downloads/Projects/web/tranhve/docs/features/spec.md): Đặc tả luồng xử lý và tham số vẽ.

---

## 🎨 5 Chế Độ Vẽ Nghệ Thuật Flat Premium

*   **🖋️ Tả thực (Realistic Sketch)**: Bản vẽ bút chì đen nhiều lớp, đổ bóng hạt carbon siêu mịn kết hợp hiệu ứng run tay (hand-jitter).
*   **🖍️ Tranh chì màu (Colored Pencil)**: Phác họa chì màu đa kênh màu (độc lập RGB), tô màu nước nhạt dưới nền giấy hạt sần mịn.
*   **✒️ Nét vẽ Anime (Anime Outline)**: Lược bỏ chi tiết nhiễu nền, giữ lại đường nét cartoon dày dặn sạch sẽ bằng thuật toán **XDoG** cải tiến.
*   **🎨 Tranh sơn dầu (Oil Painting)**: Mảng màu impasto dày dặn sử dụng Median Blur sâu kết hợp màng phủ vân vải canvas linen thực tế trên GPU.
*   **🔢 Tranh số hóa tự tô (Paint-by-Numbers)**: Thuật toán K-Means gom 12 vùng màu phẳng lớn, tự động đi viền ranh giới màu sắc và đánh số thứ tự chỉ dẫn cực kỳ trực quan.

---

## 🚀 Các Tính Năng Đột Phá Mới Cập Nhật

1.  **🤖 Phân Loại Chủ Thể AI Trực Quan (Subject-Aware Vision AI)**
    *   Tự động phát hiện 4 loại ảnh: `portrait_human` (người), `animal_pet` (thú cưng, chó, mèo), `landscape_nature` (phong cảnh, thiên nhiên), và `object_still_life` (tĩnh vật).
    *   Cấu hình thông số vẽ chuyên biệt cho từng chủ thể (ví dụ: giảm bóng tối ở phong cảnh để tránh bết màu, tăng nét vẽ tơ mịn cho lông thú cưng).
2.  **🔍 Kiểm Tra Chất Lượng & Vẽ Lại Thông Minh (Auto-Redraw)**
    *   Chấm điểm độ tương đồng (0-100) dựa trên độ bao phủ nét viền (Canny), độ bao phủ canvas, và độ chi tiết.
    *   Nếu điểm số thấp (<75/100), hệ thống tự động đưa ra các đề xuất điều chỉnh thông số và hiển thị nút `🔄 Tự Động Vẽ Lại Tối Ưu Hơn`. Khi click, các slider sẽ tự động dịch chuyển về vị trí lý tưởng và khởi tạo vẽ lại ngay lập tức.
3.  **💾 Quản Lý VRAM RTX 3050 Laptop An Toàn**
    *   **Giới hạn độ phân giải vàng 1200px**: Giúp tranh cực kỳ chi tiết nhưng tiết kiệm tới 45% bộ nhớ VRAM so với mức 1600px cũ.
    *   **Giải phóng VRAM tự động (`gpu_clear_cache`)**: Ép giải phóng bộ nhớ đệm ma trận CuPy VRAM trước và sau mỗi lượt vẽ để loại bỏ hoàn toàn lỗi tràn/sập nguồn VRAM.

---

## 🛠️ Công Nghệ Sử Dụng Chủ Đạo

*   **GPU Acceleration:** **CuPy** (CUDA-accelerated NumPy) & **cupyx.scipy.ndimage** (convolutions trên VRAM).
*   **AI Tách nền:** `rembg` (U2NET) chạy trực tiếp trên GPU thông qua **ONNX Runtime CUDA Provider**.
*   **Xử lý ảnh:** OpenCV (`opencv-python-headless`) & Pillow (`PIL`)
*   **Mô phỏng AI & Vision:** Gemini 3.1 Flash Lite (SDK `google-genai` mới nhất) để tự động hóa tinh chỉnh thanh trượt.
*   **Máy chủ Backend:** FastAPI & Uvicorn (Event Streaming chunks truyền base64 JPEG tối ưu băng thông).
*   **Giao diện:** HTML/CSS/JS thuần (Premium Dark Mode, phông chữ Outfit, responsive).

---

## 🚀 Hướng Dẫn Khởi Chạy Nhanh (Có GPU)

### Triển khai bằng Docker Compose (Khuyên dùng)
1.  Đảm bảo máy chủ đã cài đặt **NVIDIA Container Toolkit** để Docker có thể giao tiếp với GPU.
2.  Tạo tệp `.env` ở thư mục gốc và cấu hình API Key:
    ```env
    GEMINI_API_KEY=AIzaSy...
    PRIMARY_TEXT_MODEL=gemini-3.1-flash-lite
    PRIMARY_VISION_MODEL=gemini-3.1-flash-lite
    ```
3.  Khởi chạy hệ thống:
    ```bash
    docker-compose up --build -d
    ```
4.  Truy cập giao diện web tại địa chỉ: **[http://127.0.0.1:7860](http://127.0.0.1:7860)**

### Kiểm thử hiệu năng GPU
Để xác minh GPU đã nhận diện chính xác trong container:
```bash
docker exec tranhve-container python -c "from src.gpu_utils import gpu_status; print(gpu_status())"
```
Kết quả mong đợi: `GPU ON — NVIDIA GeForce RTX 3050 ... | VRAM ... MB free`
