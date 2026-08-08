# Nâng Cấp Chất Lượng Mỹ Thuật & Nét Vẽ 5 Model (Aesthetic Upgrades Documentation)

Tài liệu này mô tả chi tiết các cải tiến thuật toán mỹ thuật và xử lý ảnh cho 5 phong cách vẽ trong công cụ **TranhVe** nhằm khắc phục lỗi kết cấu thô, mất chi tiết mắt, quầng sáng giả dập nổi (double-edge/metallic) và lỗi vẽ thiếu nét.

---

## 🖋️ 1. Phác thảo Tả thực (Realistic Sketch - `sketch.py`)
### Vấn đề gốc:
- Phép toán `Color Dodge` chia cho ảnh nghịch đảo bị mờ làm xuất hiện quầng sáng kép (double outline) xung quanh các đường biên tương phản mạnh, tạo cảm giác tranh kim loại dập nổi (emboss) thay vì phác thảo chì trên giấy.
- Tròng mắt bị mờ xám phẳng lì, làm nhân vật bị vô thần.
- Các nét vẽ ở vùng sáng (như da, quần áo sáng màu) có tông xám quá nhạt nên gần như vô hình, gây cảm giác bản vẽ tự dừng giữa chừng.

### Cải tiến:
1. **Lọc Shading Charcoal tuyến tính (`_charcoal_shading`)**: Hủy bỏ phép chia Dodge. Sử dụng phép nhân tuyến tính với ảnh xám lọc Gaussian mịn trong tầm [135, 255] để mô phỏng bóng than chì mịn, phẳng trên mặt giấy, triệt tiêu 100% quầng sáng hào quang.
2. **Giải thuật Tìm Nét Đơn (Centerline Skeletonization)**:
   - Thay thế việc cắt đôi mảng điểm contour thủ công dễ gây răng cưa.
   - Sử dụng phép toán **Dilation** (Giãn nở) để chập hai biên song song của Canny làm một mảng nét dày.
   - Sau đó chạy thuật toán **Skeletonization** (Trích xuất xương nét xám) bằng vòng lặp Morphological Erosion/Subtraction để co mảng nét dày về đúng **đường tâm duy nhất có độ rộng 1 pixel**.
   - Cách này loại bỏ hoàn toàn hiện tượng viền kép, tạo ra các đường vẽ chì trung tâm duy nhất, sạch sẽ và sắc nét 100%.
3. **Phục hồi phản quang tròng mắt (Layer 5)**: Dò tìm đốm sáng trắng Catchlight (độ sáng > 222) và vẽ bù màu giấy sáng lên tròng mắt để mắt sáng, có hồn.
4. **Tông nét vẽ đậm đà, rõ nét**: 
   - Đổi công thức tính độ xám của nét vẽ sang khoảng tối ổn định `[10, 45]`.
   - Các nét vẽ ở vùng sáng không bị hóa trắng hay nhạt màu nữa, mà vẫn giữ được độ xám chì đậm vừa phải, đảm bảo nhìn rõ toàn bộ khuôn mặt và nếp nhăn.

---

## 🖍️ 2. Phong cách Chì màu (Colored Pencil - `pencil.py`)
### Cải tiến:
1. **Hủy bỏ Color Dodge đa kênh**: Thay thế dodge kênh R, G, B bằng phép nhân màu ảnh gốc với lớp xám shading và vân giấy canvas nhám (`gpu_canvas_texture`).
2. **Nét chì màu sắc động & Trích xuất xương nét**:
   - Tương tự như Realistic Sketch, áp dụng giải thuật dilation + skeletonization cho toàn bộ nét vẽ và nét đánh bóng (hatching).
   - Màu nét chì lấy từ màu gốc nhưng được dìm tối động bằng công thức `stroke_factor = 0.16 + 0.12 * (lum / 255.0)`. Điều này giúp nét vẽ có màu sắc sống động, đậm nét và không bị chìm màu ở vùng sáng.
3. **Phục hồi phản quang mắt**: Bảo toàn và dán đè đốm sáng trắng catchlight.

---

## ✒️ 3. Phong cách Anime (Anime Outline - `anime.py`)
### Cải tiến:
1. **Lọc nét đơn (Skeletonization) trên XDoG**:
   - Sử dụng giải thuật co xương nét để loại bỏ toàn bộ hiện tượng viền đen kép của nét vẽ Anime.
   - Nét vẽ sau khi co xương được giãn rộng đồng đều bằng `line_art_width` chỉ định, mang lại đường viền mượt, đồng nhất đúng phong cách đồ họa vector.
2. **Hạ ngưỡng lọc nét vẽ**: Hạ ngưỡng lọc chiều dài nét tối thiểu từ `20px` xuống `8px` để vẽ đầy đủ các chi tiết mắt, lông mày, các lọn tóc nhỏ của nhân vật.
3. **Mắt lấp lánh**: Dán đè chấm phản quang mắt trắng cel-shaded lên nền màu phẳng.

---

## 🎨 4. Phong cách Tranh sơn dầu (Oil Painting - `oil.py`)
### Cải tiến:
1. **3D Impasto Texture**: Tích lũy độ dày cọ vẽ của từng pass vẽ qua một mảng độ cao (`heightmap_np`).
2. **Đổ bóng hướng sáng**: Tính toán normal map của mảng độ cao bằng Sobel và đổ bóng diffuse theo nguồn sáng hướng trên-trái (`[-1, -1, 1.2]`). Điều này tạo ra độ gồ ghề của lớp sơn khô, cho cảm giác sơn dầu đắp nổi chân thực.
3. **Tăng mật độ cọ vẽ**: Tối ưu hóa kích thước bước cọ vẽ (step size) nhỏ hơn để cọ đi dày đặc, tăng độ phân giải chi tiết bề mặt tranh.

---

## 🔢 5. Tranh số hóa tự tô (Paint-by-Numbers - `blueprint.py`)
### Cải tiến:
1. **Hạ ngưỡng lọc mảng màu**:
  - Hạ diện tích mảng màu tối thiểu từ `250` pixels.
  - Hạ ngưỡng lọc đảo nhỏ (contourArea) từ `200` pixels.
  - Hạ ngưỡng lọc nét vẽ ranh giới tối thiểu từ `12` pixels.
2. **Định vị nhãn số bằng Distance Transform**:
  - Thay vì dùng Moments (tâm hình học dễ bị lệch ra ngoài đối với mảng màu hình chữ C, chữ U), thuật toán sử dụng `cv2.distanceTransform` để tìm tâm của vòng tròn nội tiếp lớn nhất. Nhãn số mã màu chắc chắn nằm bên trong mảng và có khoảng cách xa đường biên nhất, không bao giờ bị đè lên nét vẽ.

---

## 💻 6. Cải tiến Trải nghiệm Web (Frontend Evolution)
- **Xóa bản vẽ cũ ngay khi bắt đầu**: 
  - Trong `app.js`, ngay khi người dùng nhấn nút **"Bắt Đầu Vẽ Động"**, ảnh vẽ của lượt chạy cũ sẽ lập tức bị xóa và thay thế bằng khung placeholder tối.
  - Điều này giải quyết triệt để hiểu lầm hệ thống "không xóa phác thảo từ ảnh cũ" trong lúc Gemini đang tính toán tham số.
