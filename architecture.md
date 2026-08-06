# Kiến Trúc Hệ Thống: Smart Portrait-to-Sketch Pipeline (GPU CUDA Accelerated)

Tài liệu này mô tả kiến trúc tổng thể, mô hình phân lớp xử lý song song GPU/CPU và cấu trúc thư mục của hệ thống chuyển đổi ảnh số sang tranh nét nghệ thuật.

---

## 1. Cấu Trúc Hệ Thống (System Architecture)

Hệ thống được thiết kế theo kiến trúc **Module tách biệt (Modular Design)**, phân chia rõ ràng giữa giao diện hiển thị Web Event-Stream (Presentation Layer), bộ điều phối FastAPI, và nhân xử lý tăng tốc phần cứng (GPU/CPU Layer).

```
    ┌──────────────────────────────────────────────────────────┐
    │                  Giao diện người dùng                    │
    │           (HTML5 / CSS3 / Vanilla JS Web App)            │
    └────────────────────────────┬─────────────────────────────┘
                                 │
                                 ▼ (FastAPI Event Streaming)
    ┌──────────────────────────────────────────────────────────┐
    │                Bộ Điều Phối (app.py)                     │
    └────────────────────────────┬─────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
 ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
 │ Image Loader │        │ Core Engine  │        │ PNG Exporter │
 │ (loader.py)  │        │(processor.py)│        │ (exporter.py)│
 └──────┬───────┘        └──────┬───────┘        └──────┬───────┘
        │                       │                       │
        ▼                       ▼                       ▼
 ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
 │  • Pillow    │        │  • rembg     │        │  • Pillow    │
 │  • Exif Strip│        │ (ONNX GPU)   │        │ • EXIF strip │
 │  • Resize    │        │  • OpenCV    │        │ • Zip Package│
 └──────────────┘        └──────┬───────┘        └──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │    GPU Acceleration Layer   │
                 │       (src/gpu_utils.py)    │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
         ┌──────────────┐              ┌──────────────┐
         │ NVIDIA VRAM  │              │ CPU Fallback │
         │(CuPy / CUDA) │              │   (NumPy)    │
         └──────────────┘              └──────────────┘
```

---

## 2. Cấu Trúc Thư Mục Dự Án (Directory Structure)

```
tranhve/
│
├── docs/                        # Tài liệu dự án
│   └── features/
│       ├── ideas_evolution.md  # Tiến hóa ý tưởng Darwin Engine
│       ├── spec.md             # Đặc tả luồng xử lý và tham số vẽ
│       ├── plan.md             # Kế hoạch phát triển dự án
│       ├── tasks.md            # Bảng theo dõi đầu việc
│       └── test-plan.md        # Kế hoạch kiểm thử tự động
│
├── src/                         # Mã nguồn cốt lõi (Core Engine)
│   ├── __init__.py
│   ├── gpu_utils.py            # Lớp tăng tốc GPU CUDA thông qua CuPy
│   ├── loader.py               # Tải và chuẩn hóa định dạng ảnh đầu vào
│   ├── processor.py            # Điều phối tiến trình vẽ động
│   ├── exporter.py             # Đóng gói tối ưu hóa ảnh đầu ra
│   ├── assistant.py            # Kết nối local analyzer và Gemini API
│   └── modes/                  # Thuật toán vẽ 5 phong cách nghệ thuật
│       ├── __init__.py
│       ├── sketch.py           # Realistic Sketch
│       ├── pencil.py           # Colored Pencil Sketch
│       ├── anime.py            # Anime Outline
│       ├── oil.py              # Oil Painting
│       └── blueprint.py        # Paint-by-Numbers Blueprint
│
├── static/                      # Giao diện Web Client
│   ├── index.html              # HTML5 Layout
│   ├── style.css               # Vanilla CSS Premium Flat UI
│   └── app.js                  # Logic kết nối API & Replay vẽ động
│
├── test_images/                 # Thư mục ảnh kiểm thử mẫu
├── output/                      # Thư mục lưu kết quả xuất file đầu ra
│
├── app.py                       # Khởi chạy FastAPI & Uvicorn máy chủ local
├── requirements.txt             # Thư viện Python yêu cầu cài đặt
└── README.md                    # Hướng dẫn cài đặt và sử dụng nhanh
```

---

## 3. Lựa Chọn Công Nghệ (Technology Decisions)

*   **Ngôn ngữ chính:** **Python 3.10+**. Tốt nhất cho xử lý ảnh số và tích hợp AI.
*   **Tách nền tự động:** **rembg (U2NET)** chạy trực tiếp trên GPU qua **ONNX Runtime CUDA Provider**, cho tốc độ phản hồi tức thì dưới 1 giây.
*   **Lớp tính toán tăng tốc:** **CuPy** (sử dụng CUDA của NVIDIA RTX) để song song hóa các phép tính chập ma trận (Gaussian Blur, Bilateral Filter, XDoG, blend màu) trực tiếp trên VRAM. CPU đóng vai trò dự phòng (fallback) tự động.
*   **Máy chủ REST API:** **FastAPI + Uvicorn**. Hỗ trợ cơ chế Streaming để đẩy ảnh vẽ động liên tục về UI với độ trễ cực thấp.
