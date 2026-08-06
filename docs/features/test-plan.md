# Kế Hoạch Kiểm Thử (Test Plan): Smart Portrait-to-Sketch Pipeline

Tài liệu này hướng dẫn cách kiểm thử hệ thống để đảm bảo chất lượng xử lý ảnh ổn định, thuật toán chạy chính xác và đầu ra đạt chuẩn.

---

## 1. Phương Pháp Kiểm Thử (Testing Methodology)

### A. Kiểm thử Kỹ thuật (Technical Testing)
*   **Mục tiêu:** Đảm bảo code chạy không có lỗi ngoại lệ (Exceptions), ném đúng lỗi khi gặp input hỏng, và xuất đúng định dạng PNG.
*   **Cách thực hiện:** Tạo các script test tự động trong thư mục `tests/` chạy thử các hàm đơn lẻ (Unit tests).

### B. Kiểm thử Trực quan (Visual Quality Testing)
*   **Mục tiêu:** Đánh giá độ sắc nét của đường vẽ, khả năng tách nền chính xác (không bị lẹm vào chủ thể), và độ thẩm mỹ của tranh nét vẽ.
*   **Cách thực hiện:** Chạy thử nghiệm thủ công trên bộ ảnh kiểm thử tiêu chuẩn và quan sát kết quả bằng mắt thường qua giao diện Web UI.

---

## 2. Kịch Bản Kiểm Thử Chi Tiết (Test Cases)

### Kịch bản 1: Kiểm thử Loader (Đầu vào ảnh)
| Bước thực hiện | Kết quả mong đợi | Trạng thái |
| :--- | :--- | :--- |
| Nhập URL ảnh hợp lệ (ví dụ: `https://example.com/avatar.jpg`) | Ảnh được tải về, lưu tạm và chuyển đổi thành đối tượng Pillow thành công. | [ ] |
| Nhập đường dẫn file local hợp lệ (ví dụ: `C:/images/test.jpg`) | Hệ thống đọc file ngay lập tức, không gây lag. | [ ] |
| Nhập URL bị lỗi 404 hoặc file local không tồn tại | Hệ thống báo lỗi thân thiện (ví dụ: "Không tìm thấy ảnh"), không bị sập app. | [ ] |
| Nhập file không phải định dạng ảnh (ví dụ: file `.txt`) | Hệ thống từ chối xử lý và hiển thị thông báo định dạng không hợp lệ. | [ ] |

### Kịch bản 2: Kiểm thử Tách nền & Xử lý nét vẽ (Processing Engine)
| Bước thực hiện | Kết quả mong đợi | Trạng thái |
| :--- | :--- | :--- |
| Chọn tùy chọn `Tách nền` trên ảnh chân dung người | Nền xung quanh người bị xóa hoàn toàn, biên tóc và vai sắc nét. | [ ] |
| Điều chỉnh thanh trượt `Độ đậm nét (Threshold)` từ thấp đến cao | Đường nét trên bức tranh vẽ dày/mỏng tương ứng theo thời gian thực. | [ ] |
| Điều chỉnh thanh trượt `Độ mịn (Blur)` | Giảm bớt các nét vẽ nhỏ li ti (noise) khi tăng giá trị blur. | [ ] |

### Kịch bản 3: Kiểm thử Exporter (Đầu ra file PNG)
| Bước thực hiện | Kết quả mong đợi | Trạng thái |
| :--- | :--- | :--- |
| Tải ảnh sketch kết quả về máy | File tải về có đuôi `.png`, nền trong suốt (chỉ có nét đen). | [ ] |
| Kiểm tra dung lượng file PNG xuất ra | File đã được tối ưu hóa nén (thường dung lượng dưới 1MB đối với nét vẽ đen trắng). | [ ] |
| Kiểm tra Metadata của file PNG đầu ra | Các thông tin nhạy cảm (GPS, Model máy ảnh, Ngày chụp) của ảnh gốc đã bị loại bỏ hoàn toàn. | [ ] |

### Kịch bản 4: Kiểm thử Nét Vẽ Tay & Phân Lớp Phong Cảnh (Human Jitter & Landscape Rendering)
| Bước thực hiện | Kết quả mong đợi | Trạng thái |
| :--- | :--- | :--- |
| Bật tùy chọn `Hand Jitter` | Nét vẽ trên cửa sổ Turtle có độ run nhẹ ngẫu nhiên, tạo cảm giác vẽ bằng tay (không bị thẳng đúp cứng nhắc). | [ ] |
| Bật tùy chọn `Hatching` trên vùng tối của ảnh | Rùa thực hiện vẽ các đường gạch chéo đan xen tạo độ khối bóng tối thành công. | [ ] |
| Nạp ảnh phong cảnh và chọn chế độ phân lớp tông màu | Hệ thống phân rã ảnh thành các lớp từ 1 đến 8 phân tông xám và Rùa bắt đầu vẽ lớp nền xa (mây, trời) trước, lớp gần (đất đá) sau. | [ ] |
| Xuất bản vẽ dạng rỗng (Paint-by-Numbers Blueprint) | Ảnh PNG xuất ra có các vùng khép kín với các nhãn số in nhỏ chính xác ở giữa các vùng để tự tô màu. | [ ] |

### Kịch bản 5: Kiểm thử Khả năng Phục hồi Tách nền & Ảnh Ngang
| Bước thực hiện | Kết quả mong đợi | Trạng thái |
| :--- | :--- | :--- |
| Nạp ảnh vẽ chì/mịn (như ảnh vẽ chân dung Bác Hồ) có tùy chọn `Tách nền` | Mô hình `rembg` sẽ xóa trắng ảnh, nhưng bộ lọc kiểm tra mask sẽ phát hiện mask rỗng (< 2% diện tích) và tự động khôi phục ảnh gốc, tránh sập app hoặc hiển thị canvas trắng. | [ ] |
| Nạp một bức ảnh ngang (Landscape aspect ratio) | Hệ thống tự điều chỉnh kích thước video writer chuẩn chẵn (`w-1`, `h-1`), xuất file video `.mp4` và ảnh PNG đúng tỷ lệ, không bị méo. | [ ] |

### Kịch bản 6: Kiểm thử Trình diễn Toàn màn hình & Tải gói ZIP
| Bước thực hiện | Kết quả mong đợi | Trạng thái |
| :--- | :--- | :--- |
| Nhấp chọn nút floating "Xem toàn màn hình" trong `.canvas-container` | Bảng vẽ phóng to chiếm toàn bộ màn hình điều hành, ẩn các thanh sidebar, ảnh vẽ co dãn khớp màn hình. Nhấn ESC để thoát. | [ ] |
| Nhấp chọn nút "Xem Lại Quá Trình" sau khi vẽ xong | Giao diện vẽ lại từng bước từ mảng cached base64 ở tốc độ 30 FPS. | [ ] |
| Nhấp chọn tải tệp khi quá trình kết xuất kết thúc | Nhận được file `tranhve_package.zip` có dung lượng tối ưu, giải nén chứa đúng `final_artwork.png` và `drawing_process.mp4`. | [ ] |

---

## 3. Bộ Ảnh Kiểm Thử Tiêu Chuẩn (Standard Test Assets)

Để đánh giá thuật toán chính xác nhất, chúng ta cần chuẩn bị bộ ảnh test bao gồm:
1.  **Chân dung đơn giản:** Ảnh chân dung cận cảnh một người có nền trơn (Dễ tách nền, dễ lấy nét vẽ khuôn mặt).
2.  **Chân dung phức tạp:** Ảnh người có tóc bay, nền nhiều chi tiết rối mắt (Thách thức lớn đối với bộ tách nền `rembg`).
3.  **Thành phẩm nghệ thuật (Charcoal/Pencil drawings):** Bản vẽ tay sẵn có độ mịn (ví dụ: ảnh chân dung Bác Hồ vẽ chì mịn). Dùng để kiểm chứng thuật toán dò biên hỗn hợp (Hybrid Canny/Adaptive Threshold) và bộ thu hồi tách nền hỏng.
4.  **Thú cưng:** Ảnh chó hoặc mèo lông xù (Kiểm tra độ chi tiết của nét vẽ lông và bộ lọc nhẵn nét).
5.  **Ảnh Ngang (Landscape Aspect Ratio):** Kiểm tra tính tương thích co dãn khung và xuất video MP4 đúng tỷ lệ ngang.
6.  **Ảnh độ phân giải thấp/Mờ:** Ảnh chất lượng kém (Kiểm tra xem bộ lọc nhiễu hoạt động hiệu quả không).
