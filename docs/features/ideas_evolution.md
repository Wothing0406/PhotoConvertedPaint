# Đảo Tiến Hóa Ý Tưởng - Vòng 1 (Idea Evolution - Round 1)

Chào mừng bạn đến với hệ thống tiến hóa ý tưởng (Darwin Engine) cho dự án **tranhve**. Dưới đây là bảng đánh giá 6 chiều và quá trình tiến hóa cho các loài ý tưởng ban đầu.

---

## 1. Đánh giá Chi tiết các Loài (Species Cards & Scoring)

### 🦎 IDEA-0001: Pipeline Chuyển Đổi Ảnh Thành Tranh Vẽ Nét (Line Art Generator)
* **Mô tả:** Tải ảnh từ internet/local -> Xử lý lọc nhiễu & tách nét (OpenCV Canny/Sobel/Contour) -> Xuất file PNG nền trong suốt để làm tranh tô màu/vẽ phác thảo.
* **Điểm số 6 chiều:**
  * **Novelty (10%):** 7/10 (Ý tưởng phổ biến nhưng cách đóng gói phục vụ việc vẽ tranh thì rất thiết thực).
  * **Feasibility (20%):** 9/10 (Sử dụng OpenCV và Pillow rất dễ cài đặt, chạy nhẹ nhàng trên local).
  * **Value (20%):** 9/10 (Phù hợp trực tiếp với tên thư mục dự án `tranhve`, tạo ra giá trị thực tế cho người thích vẽ).
  * **Logic (20%):** 8/10 (Luồng xử lý rõ ràng: Input -> Grayscale -> Blur -> Edge Detection -> Alpha Channel Insertion -> PNG Output).
  * **Cross Potential (10%):** 9/10 (Dễ dàng kết hợp với AI để tự động tách vật thể rồi mới lấy nét).
  * **Verifiability (20%):** 9/10 (Có thể kiểm thử nhanh chóng bằng các script Python ngắn với OpenCV).
  * **Điểm trung bình trọng số:** **8.55/10**

### 🐋 IDEA-0002: Trình Tải & Tối Ưu Hóa Ảnh Hàng Loạt (Bulk PNG Downloader & Optimizer)
* **Mô tả:** Tải ảnh đa luồng -> Convert mọi định dạng sang PNG -> Nén không mất chi tiết (lossless) -> Xóa EXIF data.
* **Điểm số 6 chiều:**
  * **Novelty (10%):** 4/10 (Các tool download và convert đã có quá nhiều).
  * **Feasibility (20%):** 9/10 (Rất dễ code bằng `requests` / `aiohttp` kết hợp với `Pillow`).
  * **Value (20%):** 6/10 (Tiện ích tốt nhưng không tạo ra sự đột phá hay hứng thú cao).
  * **Logic (20%):** 9/10 (Luồng kỹ thuật cực kỳ đơn giản và vững chắc).
  * **Cross Potential (10%):** 6/10 (Khó phát triển thêm các tính năng nghệ thuật).
  * **Verifiability (20%):** 9/10 (Viết unit test rất dễ dàng).
  * **Điểm trung bình trọng số:** **7.10/10**

### 🦅 IDEA-0003: Nâng Cấp Chất Lượng Ảnh Cũ Bằng AI Local (Local AI Upscaler)
* **Mô tả:** Sử dụng Real-ESRGAN/GFPGAN local để nâng phân giải ảnh mờ lên PNG sắc nét.
* **Điểm số 6 chiều:**
  * **Novelty (10%):** 8/10 (Nâng cấp chất lượng ảnh bằng AI offline luôn là nhu cầu hot).
  * **Feasibility (20%):** 6/10 (Yêu cầu cấu hình phần cứng local tốt, cài đặt PyTorch/ONNX Runtime có thể phức tạp với một số máy).
  * **Value (20%):** 8/10 (Ảnh chất lượng cao rất hữu ích cho các dự án thiết kế/in ấn tranh vẽ).
  * **Logic (20%):** 7/10 (Đôi khi AI sinh lỗi họa tiết hoặc sai lệch chi tiết gốc).
  * **Cross Potential (10%):** 8/10 (Kết hợp tốt với bộ lọc tranh vẽ nét để làm tranh nét siêu sắc nét).
  * **Verifiability (20%):** 7/10 (Đánh giá chất lượng ảnh AI cần kiểm tra trực quan bằng mắt nhiều hơn).
  * **Điểm trung bình trọng số:** **7.10/10**

### 🐆 IDEA-0004: Hệ Thống Tự Động Tách Nền Ảnh Số (Local Background Remover)
* **Mô tả:** Sử dụng `rembg` (U2NET) chạy local để tách chủ thể khỏi nền, lưu PNG trong suốt.
* **Điểm số 6 chiều:**
  * **Novelty (10%):** 7/10 (Thay thế các dịch vụ online trả phí như remove.bg bằng giải pháp offline miễn phí).
  * **Feasibility (20%):** 8/10 (Thư viện `rembg` rất dễ dùng trong Python, tự động tải model khi chạy lần đầu).
  * **Value (20%):** 8/10 (Tách nền là bước chuẩn bị cực kỳ quan trọng trước khi vẽ hoặc thiết kế tranh).
  * **Logic (20%):** 8/10 (Mô hình U2NET hoạt động ổn định cho chân dung và vật thể rõ ràng).
  * **Cross Potential (10%):** 9/10 (Có thể ghép nối với Line Art Generator để chỉ lấy nét vẽ của chủ thể chính).
  * **Verifiability (20%):** 8/10 (Kiểm thử dễ dàng trên các bộ ảnh test tiêu chuẩn).
  * **Điểm trung bình trọng số:** **7.90/10**

---

## 2. Lai Ghép & Đột Biến (Crossbreeding & Mutation)

Để tối ưu hóa, chúng ta sẽ thực hiện lai ghép (Crossbreed) các loài có điểm số cao nhất: **IDEA-0001 (Line Art)** và **IDEA-0004 (Background Remover)** để tạo ra một loài siêu việt mới:

### 🌟 HYBRID-01: Trình Tạo Phác Thảo Tranh Vẽ Từ Ảnh Chân Dung/Vật Thể (Smart Portrait-to-Sketch Pipeline)
* **Cha mẹ:** IDEA-0001 × IDEA-0004
* **Cơ chế hoạt động:**
  1. Người dùng cung cấp link ảnh hoặc đường dẫn thư mục chứa ảnh gốc.
  2. Hệ thống tự động tách nền (loại bỏ chi tiết hậu cảnh gây nhiễu) bằng `rembg`.
  3. Áp dụng thuật toán OpenCV lọc nét của chủ thể chính (người, thú cưng, đồ vật).
  4. Lưu trữ kết quả thành file PNG chất lượng cao với nền trong suốt (hoặc nền trắng tùy chọn) để người dùng có thể in ra làm tranh tô màu hoặc đưa vào phần mềm vẽ (Photoshop, Procreate) để vẽ đè lên.
* **Đánh giá vượt trội:** Giải quyết được bài toán hậu cảnh quá rối mắt khi chuyển đổi sang tranh nét nét vẽ thông thường.

---

## 3. Tiến Hóa Vòng 2: Đột Biến Nét Vẽ Dạng Turtle (Round 2: Turtle Vector Drawing Mutation)

Nhận được phản hồi quan trọng từ người dùng: **Muốn nét vẽ có cảm giác được vẽ bằng thư viện Turtle của Python (nét bút nối tiếp nhau chạy động từng nét).**

Đây là một đột biến cực kỳ giá trị giúp dự án thoát khỏi định dạng ảnh tĩnh truyền thống và nâng cao trải nghiệm thị giác.

### 🐢 Đánh giá IDEA-0005: Trình Vẽ Nét Ký Họa Bằng Đồ Họa Rùa (Turtle Vector Plotter)
* **Điểm số 6 chiều:**
  * **Novelty (10%):** 9/10 (Khác biệt hoàn toàn với việc xuất ảnh PNG thông thường, tạo hiệu ứng vẽ tranh động thú vị).
  * **Feasibility (20%):** 8/10 (Rất khả thi bằng cách lấy tọa độ Contour từ OpenCV và truyền vào lệnh di chuyển của Turtle. Python Turtle có sẵn không cần cài thêm).
  * **Value (20%):** 9/10 (Tạo ra trải nghiệm tương tác cực cao, người dùng có thể xem rùa vẽ từng nét chân dung/phong cảnh trước khi xuất file).
  * **Logic (20%):** 8/10 (Luồng hoạt động: OpenCV Contour Extraction -> Path Simplification (Ramer-Douglas-Peucker) -> Turtle Script Generation -> Playback).
  * **Cross Potential (10%):** 9/10 (Kết hợp hoàn hảo với thuật toán tách nền để vẽ riêng nét của chủ thể chính).
  * **Verifiability (20%):** 9/10 (Dễ dàng chạy thử script vẽ tự động trên cửa sổ Tkinter).
  * **Điểm trung bình trọng số:** **8.60/10** (Vượt qua cả IDEA-0001 ban đầu).

---

## 4. Lai Ghép Siêu Cấp: HYBRID-02 (Smart Portrait-to-Turtle Sketcher)
* **Cha mẹ:** HYBRID-01 × IDEA-0005 (Background Remover + OpenCV Edge + Turtle Vector Rendering)
* **Cơ chế hoạt động tiến hóa:**
  1. **Tách nền:** Xóa hậu cảnh bằng AI (`rembg`) giữ lại chủ thể chính.
  2. **Trích xuất nét vẽ:** Dùng OpenCV Canny & Tìm đường viền (`cv2.findContours`).
  3. **Đơn giản hóa đường đi (Vectorization):** Sử dụng thuật toán `approxPolyDP` để giảm số lượng điểm vẽ thừa, giúp bút vẽ Turtle chạy mượt và nhanh hơn (không bị quá chi tiết gây giật lag).
  4. **Vẽ động (Turtle Playback):** Turtle di chuyển vẽ nét lên màn hình Tkinter local. Người dùng có thể chỉnh tốc độ (`turtle.speed()`).
  5. **Xuất file:** Lưu lại canvas thành định dạng PostScript (`.ps`), sau đó convert qua PNG chất lượng cao có độ phân giải tùy chỉnh.

---

## 5. Tiến Hóa Vòng 3: Tối ưu hóa Docker & Trí Tuệ Nhân Tạo Gemini (Round 3: Docker & Gemini AI Integration)

Vòng này giải quyết các thách thức vận hành thực tế và bổ sung tính năng "Trợ lý nghệ thuật AI".

### A. Tối Ưu Hóa Môi Trường Bằng Docker
*   **Thách thức:** Các thư viện như `OpenCV` và `rembg` (cần tải AI model nặng ~170MB, ONNX runtime, C++ compilation) rất dễ xảy ra lỗi xung đột môi trường (Dependency Hell) trên các hệ điều hành khác nhau.
*   **Giải pháp Docker:** Đóng gói toàn bộ nhân xử lý ảnh, AI model, và Gradio Web UI vào Docker Container.
*   **Lưu ý kỹ thuật về đồ họa:**
    *   Vì Docker chạy ở chế độ Headless (không có giao diện hiển thị đồ họa Tkinter của host), chúng ta sẽ có **2 chế độ chạy**:
        1.  *Chế độ Hybrid (Khuyên dùng):* Chạy Core Engine (Loader, AI Background Remover, OpenCV) trong Docker như một API Service. Chạy mã điều khiển Turtle trực tiếp trên máy Host (Windows/macOS) để hiển thị Tkinter bình thường.
        2.  *Chế độ Web-Only:* Không dùng đồ họa Tkinter nữa mà xuất trực tiếp dữ liệu Contour dạng SVG, sử dụng JavaScript (Anime.js hoặc HTML5 Canvas) chạy ngay trên giao diện trình duyệt của Gradio. Cách này giúp Docker chạy độc lập 100%.

### B. Tích Hợp Mô Hình Đa Phương Tiện Gemini (Gemini AI Multi-modal)
Tích hợp Gemini API sẽ nâng tầm dự án từ một công cụ convert ảnh cơ khí thành một **Trợ Lý Vẽ Tranh Trực Quan**:
1.  **AI Art Critic (Đánh giá & Gợi ý nét vẽ):** Gemini sẽ nhìn ảnh gốc và nét vẽ phác thảo đầu ra. Nó có thể đưa ra nhận xét: *"Vùng mắt hơi bị mờ nét, bạn nên tăng thông số Adaptive Threshold thêm 10%"* hoặc *"Ảnh phong cảnh này hậu cảnh hơi rối, bạn nên bỏ bớt chi tiết mây."*
2.  **Smart Color-by-Numbers Palette Generator:** Đối với tranh phong cảnh/tĩnh vật số hóa, Gemini sẽ tự động gợi ý bảng phối màu (Color Palette) nghệ thuật tùy theo mong muốn của người dùng (ví dụ: "phối màu phong cách Vintage cổ điển", "phối màu Bắc Âu lạnh", "phối màu cyberpunk"). Nó sẽ chỉ định màu cụ thể cho từng số 1 đến 8 trên tranh.
3.  **Tạo ảnh từ Text (Imagen):** Người dùng nhập mô tả: *"Vẽ một con sư tử uy nghi đứng dưới mưa tuyết phong cách tối giản"*, Gemini/Imagen sinh ảnh, và luồng xử lý tự động chạy để cho ra tranh nét vẽ Rùa.
