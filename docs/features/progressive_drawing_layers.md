# Kiến Trúc Vẽ Đa Lớp & Tiến Trình Sinh Frame Hình (Progressive Drawing Layers)

Tài liệu này mô tả chi tiết kiến trúc vẽ đa lớp (multi-layer drawing) và cơ chế sinh frame động (progressive frame generation) của 5 model trong hệ thống **TranhVe**. Giải thích lý do tại sao các bức ảnh cực kỳ chi tiết như [chinhi.jpg](file:///c:/Users/QuangNe/Downloads/Projects/web/tranhve/test/chinhi.jpg) lại sinh ra số lượng frame tương ứng và cách các lớp được xây dựng.

---

## 🎨 1. Cấu Trúc Các Lớp Vẽ (Drawing Layers) Của 5 Model

Để tái tạo quá trình vẽ như một họa sĩ thực thụ chứ không phải dán bộ lọc ảnh (filter), mỗi phong cách vẽ được chia nhỏ thành các lớp vẽ riêng biệt và được vẽ đè lên nhau theo thứ tự logic:

| Phong Cách | Lớp 1 (Nền & Khung) | Lớp 2 (Phác Thảo) | Lớp 3 (Đánh Bóng/Tô Màu) | Lớp 4 (Chi Tiết & Phản Quang) | Lớp 5 (Hiệu Ứng Vật Liệu) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Realistic Sketch** | Canvas trắng tinh | Nét dựng hình ngoài (Outlines) | Nét đánh bóng chéo $45^\circ$ (Midtones) | Nét đánh bóng chéo $135^\circ$ (Shadows) | Smudge chì mờ mịn |
| **Colored Pencil** | Canvas trắng tinh | Nét phác chì màu nhạt | Nét đánh bóng chì màu chéo | Lớp màu nước wash lót dưới | Vân giấy nhám mịn (Linen Texture) |
| **Anime Outline** | Canvas trắng tinh | Nét mực đen (Ink Lines) | Phục hồi phản quang mắt | Tô màu phẳng (Cel-shading) lót dưới | Đậm nhạt viền ngoài |
| **Oil Painting** | Gesso canvas xám | Blocking-In (Cọ thô) | Medium Brushwork (Cọ vừa) | Detail Brushwork (Cọ mịn) | Đổ bóng nổi 3D Impasto |
| **Paint-by-Numbers** | Canvas giấy ngà | Nét lưới chia mảng màu | Nhãn số thứ tự màu | Tô màu phẳng mảng kín | Viền đen chốt ranh giới |

---

## 🎞️ 2. Giải Thích Số Lượng Frame Sinh Ra (150 - 500+ Frames)

Số lượng frame được sinh ra phụ thuộc vào hai yếu tố chính:
1. **Độ chi tiết của ảnh gốc**: Ảnh càng nhiều chi tiết (ví dụ: hoa, ren áo, tóc tơ trong [chinhi.jpg](file:///c:/Users/QuangNe/Downloads/Projects/web/tranhve/test/chinhi.jpg)) $\rightarrow$ Số lượng đường viền (contours) trích xuất được càng lớn $\rightarrow$ Số nét vẽ càng nhiều.
2. **Tham số `batch_size` (Tốc độ vẽ)**: Xác định số nét vẽ được gộp lại trước khi yield ra 1 frame.
   - Công thức tính số frame nét vẽ: $\text{Frames} = \frac{\text{Tổng số nét phác thảo} + \text{Tổng số nét đánh bóng}}{\text{batch\_size}}$

### Ví dụ thực tế kiểm thử với ảnh chi tiết [chinhi.jpg](file:///c:/Users/QuangNe/Downloads/Projects/web/tranhve/test/chinhi.jpg):

*   **Realistic Sketch (456 Frames)**:
    *   **Nét phác thảo**: Trích xuất được 350+ đường nét chi tiết trên khuôn mặt, bó hoa và trang phục tốt nghiệp.
    *   **Nét đánh bóng**: Tạo ra khoảng 8,000+ nét chì vector ngắn che phủ các vùng tối của nhân vật. Với `batch_size=8`, hệ thống liên tục yield ra tổng cộng **456 hình ảnh nối tiếp**, cho thấy rõ từng nét chì được đặt lên giấy.
*   **Colored Pencil (471 Frames)**:
    *   Tương tự như Sketch nhưng vẽ bằng các bút chì màu khác nhau, tạo ra **471 frames** vẽ màu lấp đầy từng lớp.
*   **Anime Outline (139 Frames)**:
    *   Do chỉ tập trung vào nét lineart lớn của anime và lược bỏ các nét chi tiết nhỏ/shading, số lượng nét vẽ ít hơn, tạo ra **139 frames** sạch sẽ.
*   **Oil Painting (751 Frames)**:
    *   Gồm 3 lượt đi cọ thô $\rightarrow$ mịn với hàng ngàn vệt sơn đắp nổi, tạo ra **751 frames** sơn chồng lên nhau cực kỳ sống động.

---

## ⚡ 3. Cơ Chế Xử Lý Tối Ưu Tốc Độ Render

Để đảm bảo sinh hàng trăm frame hình trong thời gian ngắn mà không gây nghẽn RAM/VRAM hoặc quá tải CPU:
1. **Streaming JPEGs qua SSE**: Thay vì chuyển đổi PNG nặng nề, mỗi frame yielded được nén thành JPEG chất lượng 75% trên RAM (nhỏ hơn 10 lần) rồi truyền trực tiếp dưới dạng chuỗi Base64 qua SSE (Server-Sent Events) tới UI của Client.
2. **Tính toán ma trận song song trên GPU**: Các tác vụ nặng như lọc nhám giấy (Linen texture), tính toán hướng đi cọ (Sobel gradients) và dò tìm vùng nổi bật (Saliency Map) được chạy trực tiếp bằng CUDA (thông qua thư viện CuPy trên card đồ họa RTX 3050).
3. **Cắt biên an toàn**: Kích thước ảnh được tối ưu hóa về mức tối đa 1200px giúp hạn chế việc tràn bộ nhớ VRAM nhưng vẫn giữ được độ sắc nét hiển thị.
