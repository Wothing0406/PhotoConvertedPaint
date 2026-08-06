# Species Bank (Ý tưởng ban đầu)

Chào mừng đến với Đảo Tiến Hóa Ý Tưởng. Dưới đây là các ý tưởng ban đầu (loài sinh vật) cho hệ thống tải và xử lý ảnh số bằng Python local, hướng tới việc xuất ra file PNG chất lượng cao.

## 1. Pipeline Chuyển Đổi Ảnh Thành Tranh Vẽ Nét (Line Art/Coloring Book Generator)
- **Mã định danh:** IDEA-0001
- **Mô tả:** Hệ thống tự động tải ảnh từ URL hoặc quét thư mục local. Sử dụng OpenCV/Pillow để xử lý ảnh số: tách biên (Canny edge detection), lọc nhiễu, tăng độ tương phản để chuyển đổi ảnh chụp thông thường thành tranh vẽ nét đen trắng (dùng để tô màu hoặc vẽ phác thảo). Xuất ra file PNG có nền trong suốt (transparent).
- **Phù hợp mục tiêu:** Tên thư mục dự án hiện tại là `tranhve`. Ý tưởng này cực kỳ khớp với định hướng làm công cụ hỗ trợ vẽ tranh.

## 2. Trình Tải & Tối Ưu Hóa Ảnh Hàng Loạt (Bulk Image Downloader & PNG Optimizer)
- **Mã định danh:** IDEA-0002
- **Mô tả:** Công cụ CLI hoặc GUI đơn giản chạy bằng Python. Cho phép người dùng nhập vào danh sách URL (hoặc file chứa URL), tải ảnh đa luồng (multi-threading), sau đó tự động convert mọi định dạng (WebP, JPEG, HEIC) sang PNG. Đồng thời áp dụng thuật toán nén không mất chi tiết (lossless PNG compression như `pyoptipng` hoặc `tinypng API`) và loại bỏ thông tin EXIF bảo mật.

## 3. Trình Quét & Nâng Cấp Chất Lượng Ảnh Cũ Bằng AI Local (Local AI-Powered Image Upscaler)
- **Mã định danh:** IDEA-0003
- **Mô tả:** Hệ thống sử dụng các mô hình học máy nhỏ chạy trực tiếp trên máy local (như Real-ESRGAN hoặc GFPGAN thông qua thư viện ONNX Runtime hoặc PyTorch). Nó sẽ quét thư mục ảnh, tải các ảnh mờ/ảnh cũ, chạy AI nâng cấp độ phân giải (upscale 2x/4x), khử nhiễu (denoise), và xuất ra file PNG sắc nét.

## 4. Hệ Thống Tự Động Tách Nền Ảnh Số (Local Background Remover & PNG Portrait Exporter)
- **Mã định danh:** IDEA-0004
- **Mô tả:** Sử dụng thư viện `rembg` (dựa trên mô hình U2NET chạy local) để tự động nhận diện chủ thể và loại bỏ nền của các bức ảnh tải về. Kết quả được lưu thành file PNG với kênh alpha (trong suốt). Cực kỳ hữu ích cho việc chuẩn bị tài nguyên thiết kế, làm sticker hoặc ghép ảnh.

## 5. Trình Vẽ Nét Ký Họa Bằng Đồ Họa Rùa (Turtle Vector Plotter & Drawing Player)
- **Mã định danh:** IDEA-0005
- **Mô tả:** Chuyển đổi ảnh số thành tập hợp các đường vector (contour paths), sau đó sử dụng thư viện đồ họa tích hợp sẵn của Python là `turtle` để điều khiển bút vẽ và tái hiện lại quá trình vẽ bức tranh từng nét một trên màn hình (giống như pen plotter hay người thật đang vẽ). Kết quả vẽ xong có thể xuất ra file định dạng PostScript (`.ps`) rồi chuyển đổi thành PNG chất lượng cao.
- **Phù hợp mục tiêu:** Đáp ứng mong muốn tạo hiệu ứng vẽ nét nghệ thuật tối giản, có tính tương tác cao (nhìn nét vẽ chạy trực tiếp).

