# Äáº·c Táº£ TÃ­nh NÄƒng: Smart Portrait-to-Sketch Pipeline (Äáº·c táº£ Há»‡ thá»‘ng Xá»­ lÃ½ áº¢nh Tranh Váº½)

TÃ i liá»‡u nÃ y Ä‘áº·c táº£ chi tiáº¿t vá» há»‡ thá»‘ng Python local giÃºp táº£i áº£nh, tÃ¡ch ná»n, chuyá»ƒn Ä‘á»•i sang tranh nÃ©t (Line Art) vÃ  xuáº¥t ra file PNG cháº¥t lÆ°á»£ng cao.

---

## 1. Kiáº¿n TrÃºc Luá»“ng Dá»¯ Liá»‡u (Data Flow)

DÆ°á»›i Ä‘Ã¢y lÃ  sÆ¡ Ä‘á»“ luá»“ng hoáº¡t Ä‘á»™ng tá»« lÃºc nháº­p áº£nh Ä‘áº§u vÃ o cho Ä‘áº¿n khi xuáº¥t file PNG hoÃ n chá»‰nh:

```mermaid
graph TD
    A[Image Input: URL or Local Path] --> B[Image Loader module]
    B --> C{Is URL?}
    C -->|Yes| D[Download & cache image]
    C -->|No| E[Load local image file]
    D --> F[PIL Image Object]
    E --> F
    F --> G[Background Remover module: rembg/U2NET]
    G --> H[Foreground Image: Alpha Channel]
    H --> I[Line Art Generator module: OpenCV]
    I --> J[Pre-processing: Grayscale & Blur]
    J --> K[Edge Detection & Contour Extraction]
    K --> L{Select Output Mode}
    L -->|PNG Mode| M[Dilation & Raster Output]
    L -->|Turtle Mode| N[RDP Simplification & Turtle Draw]
    M --> O[PNG Exporter]
    N --> P[PostScript Exporter]
    O --> Q[Save PNG with transparent background]
    P --> R[Convert .ps to transparent PNG]
```

---

## 2. CÃ¡c ThÃ nh Pháº§n ChÃ­nh (Core Modules)

### A. Bá»™ Táº£i áº¢nh (Image Loader)
*   **ThÆ° viá»‡n:** `requests` (táº£i tá»« internet), `Pillow` (Ä‘á»c/ghi file áº£nh).
*   **Chá»©c nÄƒng:**
    *   Tá»± Ä‘á»™ng phÃ¡t hiá»‡n Ä‘áº§u vÃ o lÃ  URL hay Ä‘Æ°á»ng dáº«n thÆ° má»¥c local.
    *   Xá»­ lÃ½ lá»—i táº£i áº£nh (URL há»ng, khÃ´ng cÃ³ quyá»n truy cáº­p, Ä‘á»‹nh dáº¡ng áº£nh khÃ´ng Ä‘Æ°á»£c há»— trá»£).
    *   Tá»± Ä‘á»™ng chuáº©n hÃ³a áº£nh vá» há»‡ mÃ u RGB/RGBA.

### B. Bá»™ TÃ¡ch Ná»n (Background Remover)
*   **ThÆ° viá»‡n:** `rembg` (phiÃªn báº£n local dÃ¹ng ONNX Runtime, tá»± Ä‘á»™ng cháº¡y báº±ng CPU hoáº·c GPU náº¿u cÃ³ CUDA).
*   **Chá»©c nÄƒng:**
    *   Nháº­n diá»‡n chá»§ thá»ƒ chÃ­nh trong bá»©c áº£nh (con ngÆ°á»i, con váº­t, Ä‘á»“ váº­t).
    *   TÃ¡ch biá»‡t chá»§ thá»ƒ khá»i ná»n vÃ  tráº£ vá» áº£nh dáº¡ng RGBA vá»›i ná»n trong suá»‘t (Alpha = 0).

### C. Bá»™ Chuyá»ƒn Äá»•i NÃ©t Váº½ (Line Art/Sketch Generator)
*   **ThÆ° viá»‡n:** `opencv-python`, `numpy`.
*   **Chá»©c nÄƒng:**
    *   Chuyá»ƒn Ä‘á»•i chá»§ thá»ƒ Ä‘Ã£ tÃ¡ch ná»n thÃ nh áº£nh xÃ¡m (Grayscale).
    *   Ãp dá»¥ng bá»™ lá»c Gauss (Gaussian Blur) Ä‘á»ƒ giáº£m nhiá»…u háº¡t.
    *   Sá»­ dá»¥ng thuáº­t toÃ¡n **Adaptive Thresholding** hoáº·c lá»c Canny káº¿t há»£p Ä‘áº£o ngÆ°á»£c mÃ u (Invert) Ä‘á»ƒ táº¡o ra nÃ©t váº½ Ä‘en trÃªn ná»n tráº¯ng.
    *   Sá»­ dá»¥ng toÃ¡n tá»­ hÃ¬nh thÃ¡i há»c (Morphological Operations - Dilation/Erosion) Ä‘á»ƒ Ä‘iá»u chá»‰nh Ä‘á»™ dÃ y/má»ng cá»§a nÃ©t váº½.
    *   Gá»™p kÃªnh Alpha gá»‘c cá»§a bÆ°á»›c tÃ¡ch ná»n vÃ o áº£nh nÃ©t váº½ Ä‘á»ƒ nÃ©t váº½ cÃ³ ná»n trong suá»‘t (transparent).

### D. Bá»™ Xuáº¥t File (PNG Exporter)
*   **ThÆ° viá»‡n:** `Pillow`.
*   **Chá»©c nÄƒng:**
    *   LÆ°u file dÆ°á»›i Ä‘á»‹nh dáº¡ng `.png`.
    *   Ãp dá»¥ng cÃ¡c má»©c tá»‘i Æ°u hÃ³a nÃ©n (compression level 1-9) Ä‘á»ƒ dung lÆ°á»£ng nháº¹ nháº¥t nhÆ°ng giá»¯ nguyÃªn cháº¥t lÆ°á»£ng vector nÃ©t váº½.
    *   Tá»± Ä‘á»™ng xÃ³a sáº¡ch metadata (EXIF) cá»§a thiáº¿t bá»‹ chá»¥p áº£nh ban Ä‘áº§u Ä‘á»ƒ báº£o vá»‡ quyá»n riÃªng tÆ°.

### E. Bá»™ Váº½ Minh Há»a Turtle (Turtle Vector Plotter)
*   **ThÆ° viá»‡n:** `turtle` (tiÃªu chuáº©n cá»§a Python), `opencv-python` (Ä‘á»ƒ láº¥y contours).
*   **Chá»©c nÄƒng:**
    *   **TrÃ­ch xuáº¥t Contours:** DÃ¹ng `cv2.findContours` Ä‘á»ƒ phÃ¡t hiá»‡n cÃ¡c Ä‘Æ°á»ng bao khÃ©p kÃ­n hoáº·c cÃ¡c Ä‘Æ°á»ng nÃ©t chÃ­nh cá»§a hÃ¬nh áº£nh xÃ¡m Ä‘Ã£ tÃ¡ch ná»n.
    *   **ÄÆ¡n giáº£n hÃ³a Vector:** DÃ¹ng thuáº­t toÃ¡n `cv2.approxPolyDP` (Ramer-Douglas-Peucker) Ä‘á»ƒ xáº¥p xá»‰ hÃ³a Ä‘Æ°á»ng cong thÃ nh cÃ¡c Ä‘oáº¡n tháº³ng ná»‘i tiáº¿p, lá»c bá»›t cÃ¡c Ä‘iá»ƒm áº£nh quÃ¡ vá»¥n Ä‘á»ƒ trÃ¡nh lÃ m Turtle váº½ quÃ¡ cháº­m.
    *   **MÃ´ phá»ng Váº½:** Láº§n lÆ°á»£t duyá»‡t qua tá»«ng Contour, sá»­ dá»¥ng lá»‡nh `turtle.penup()`, `turtle.goto(x, y)`, `turtle.pendown()` Ä‘á»ƒ váº½ tá»«ng nÃ©t váº½ chÃ¢n thá»±c lÃªn mÃ n hÃ¬nh.
    *   **Xuáº¥t File Vector:** Xuáº¥t káº¿t quáº£ váº½ tá»« canvas ra file Ä‘á»‹nh dáº¡ng PostScript (`.ps`), sau Ä‘Ã³ dÃ¹ng Pillow Ä‘á»ƒ chuyá»ƒn Ä‘á»•i PostScript sang PNG trong suá»‘t Ä‘á»™ phÃ¢n giáº£i cao.


---

## 3. Giao Diá»‡n NgÆ°á»i DÃ¹ng Äá» Xuáº¥t (User Interfaces)

ChÃºng ta cÃ³ thá»ƒ há»— trá»£ 2 dáº¡ng giao diá»‡n Ä‘á»ƒ dá»… dÃ ng váº­n hÃ nh trÃªn mÃ¡y local:
1.  **Giao diá»‡n dÃ²ng lá»‡nh (CLI):**
    ```bash
    python main.py --input "https://example.com/art.jpg" --output "./output/sketch.png" --thickness 3 --no-bg
    ```
2.  **Giao diá»‡n Web Local (Gradio/Streamlit):**
    *   NgÆ°á»i dÃ¹ng kÃ©o tháº£ áº£nh vÃ o web app cháº¡y trÃªn trÃ¬nh duyá»‡t local.
    *   CÃ³ thanh trÆ°á»£t Ä‘iá»u chá»‰nh: Äá»™ Ä‘áº­m nÃ©t (Thickness), Äá»™ má»‹n nÃ©t (Blur), vÃ  TÃ¹y chá»n giá»¯/tÃ¡ch ná»n.
    *   Hiá»ƒn thá»‹ áº£nh so sÃ¡nh Before/After trá»±c quan.

---

## 4. Thuáº­t ToÃ¡n MÃ´ Phá»ng NÃ©t Váº½ Tay (Human-Like Stroke Simulation)

Äá»ƒ nÃ©t váº½ cá»§a RÃ¹a trÃ´ng tá»± nhiÃªn giá»‘ng nhÆ° ngÆ°á»i váº½ thá»§ cÃ´ng thay vÃ¬ cÃ¡c nÃ©t váº½ mÃ¡y mÃ³c hoÃ n háº£o, há»‡ thá»‘ng sáº½ tÃ­ch há»£p cÃ¡c ká»¹ thuáº­t mÃ´ phá»ng váº­t lÃ½ sau:

### A. Äá»™ Rung Tay Váº­t LÃ½ (Hand Jitter Simulation)
*   ThÃªm má»™t sai sá»‘ ngáº«u nhiÃªn cá»±c nhá» (Gaussian Noise) vÃ o cÃ¡c tá»a Ä‘á»™ di chuyá»ƒn cá»§a RÃ¹a:
    $$x_{má»›i} = x + \text{random.gauss}(0, \sigma)$$
    $$y_{má»›i} = y + \text{random.gauss}(0, \sigma)$$
    Trong Ä‘Ã³ $\sigma$ (Ä‘á»™ lá»‡ch chuáº©n) Ä‘Æ°á»£c cáº¥u hÃ¬nh tá»« `0.2` Ä‘áº¿n `0.8` pixel. Äiá»u nÃ y táº¡o ra Ä‘á»™ run tá»± nhiÃªn giá»‘ng nhÆ° tay ngÆ°á»i tháº­t Ä‘ang váº½ nÃ©t.

### B. Biáº¿n ThiÃªn Äá»™ DÃ y NÃ©t Váº½ (Dynamic Pensize)
*   NgÆ°á»i váº½ tháº­t thÆ°á»ng áº¥n máº¡nh bÃºt á»Ÿ Ä‘áº§u nÃ©t váº½ vÃ  nháº¥c nháº¹ á»Ÿ cuá»‘i nÃ©t váº½.
*   Há»‡ thá»‘ng sáº½ thay Ä‘á»•i liÃªn tá»¥c kÃ­ch thÆ°á»›c nÃ©t váº½ `turtle.pensize()` dá»±a trÃªn:
    *   **Giai Ä‘oáº¡n nÃ©t váº½:** Giáº£m dáº§n Ä‘á»™ dÃ y nÃ©t váº½ khi Ä‘i vá» phÃ­a cuá»‘i Ä‘Æ°á»ng viá»n (Contour).
    *   **Äá»™ cong cá»§a nÃ©t:** NÃ©t tháº³ng váº½ nhanh -> nÃ©t má»ng; nÃ©t cong váº½ cháº­m -> nÃ©t dÃ y hÆ¡n Ä‘á»ƒ mÃ´ táº£ chi tiáº¿t khá»‘i.

### C. Ká»¹ Thuáº­t ÄÃ¡nh BÃ³ng Táº¡o Khá»‘i (Hatching & Cross-hatching)
*   Äá»‘i vá»›i cÃ¡c máº£ng tá»‘i (shadows) trÃªn tranh tráº¯ng Ä‘en, thay vÃ¬ Ä‘á»ƒ trá»‘ng, há»‡ thá»‘ng sáº½ hÆ°á»›ng dáº«n RÃ¹a váº½ cÃ¡c nÃ©t gáº¡ch chÃ©o song song (Hatching) hoáº·c lÆ°á»›i gáº¡ch chÃ©o (Cross-hatching).
*   **Thuáº­t toÃ¡n:** QuÃ©t qua cÃ¡c vÃ¹ng cÃ³ sáº¯c Ä‘á»™ tá»‘i trong áº£nh gá»‘c, váº½ cÃ¡c Ä‘Æ°á»ng tháº³ng song song nghiÃªng 45 Ä‘á»™ vá»›i khoáº£ng cÃ¡ch tá»‰ lá»‡ nghá»‹ch vá»›i Ä‘á»™ tá»‘i (cÃ ng tá»‘i nÃ©t gáº¡ch cÃ ng khÃ­t).

---

## 5. Äáº·c Táº£ Xá»­ LÃ½ Phong Cáº£nh Sá»‘ HÃ³a (Landscape Digital Processing & Layering)

Tranh phong cáº£nh sá»‘ hÃ³a yÃªu cáº§u xá»­ lÃ½ khÃ´ng gian phá»©c táº¡p hÆ¡n tranh chÃ¢n dung do cÃ³ nhiá»u lá»›p chiá»u sÃ¢u. Há»‡ thá»‘ng Ã¡p dá»¥ng quy trÃ¬nh xá»­ lÃ½ phÃ¢n lá»›p:

```
[áº¢nh phong cáº£nh] â”€â”€> [K-Means PhÃ¢n máº£ng xÃ¡m] â”€â”€> [Sáº¯p xáº¿p chiá»u sÃ¢u] â”€â”€> [Turtle váº½ tá»« xa Ä‘áº¿n gáº§n]
```

### A. PhÃ¢n Máº£ng TÃ´ng MÃ u (Tonal Quantization)
*   Sá»­ dá»¥ng thuáº­t toÃ¡n **K-Means Clustering** trÃªn OpenCV Ä‘á»ƒ gom áº£nh phong cáº£nh vá» tá»« 4 Ä‘áº¿n 8 phÃ¢n Ä‘á»™ xÃ¡m cá»‘ Ä‘á»‹nh (vÃ­ dá»¥: Tráº¯ng, XÃ¡m nháº¡t, XÃ¡m trung bÃ¬nh, XÃ¡m Ä‘áº­m, Äen).
*   Má»—i phÃ¢n Ä‘á»™ xÃ¡m sáº½ Ä‘áº¡i diá»‡n cho má»™t lá»›p chiá»u sÃ¢u hoáº·c má»™t máº£ng mÃ u sá»‘ hÃ³a riÃªng biá»‡t.

### B. Váº½ PhÃ¢n Lá»›p Tá»« Xa Äáº¿n Gáº§n (Depth Sorting & Layering)
*   Há»‡ thá»‘ng tá»± Ä‘á»™ng phÃ¢n loáº¡i cÃ¡c Ä‘Æ°á»ng nÃ©t dá»±a trÃªn vá»‹ trÃ­ y (trá»¥c dá»c) vÃ  sáº¯c Ä‘á»™ Ä‘á»ƒ váº½ theo thá»© tá»±:
    1.  **Lá»›p ná»n (Background):** Báº§u trá»i, mÃ¢y, nÃºi á»Ÿ xa váº½ trÆ°á»›c báº±ng nÃ©t má»ng (`pensize=1`) vÃ  tá»‘c Ä‘á»™ nhanh.
    2.  **Lá»›p trung cáº£nh (Midground):** Äá»“i nÃºi gáº§n, hÃ ng cÃ¢y, nhÃ  cá»­a váº½ tiáº¿p theo báº±ng nÃ©t vá»«a (`pensize=2`).
    3.  **Lá»›p tiá»n cáº£nh (Foreground):** Chi tiáº¿t Ä‘Ã¡, sÃ´ng, ngÆ°á»i, cÃ¢y cá»‘i á»Ÿ gáº§n váº½ cuá»‘i cÃ¹ng báº±ng nÃ©t dÃ y (`pensize=3`) Ä‘á»ƒ táº¡o cáº£m giÃ¡c chiá»u sÃ¢u.

### C. Xuáº¥t Báº£n Äá»“ Sá»‘ HÃ³a Tá»± TÃ´ (Paint-by-Numbers Blueprint)
*   Há»‡ thá»‘ng cÃ³ cháº¿ Ä‘á»™ xuáº¥t ra báº£n váº½ nÃ©t rá»—ng khÃ©p kÃ­n kÃ¨m theo cÃ¡c con sá»‘ Ä‘á»‹nh danh tÃ´ng mÃ u (1 Ä‘áº¿n 8) ghi nhá» á»Ÿ giá»¯a má»—i vÃ¹ng, giÃºp ngÆ°á»i dÃ¹ng cÃ³ thá»ƒ tá»± in ra giáº¥y vÃ  tá»± tÃ´ mÃ u theo Ä‘Ãºng sá»‘ chá»‰ Ä‘á»‹nh (tranh sá»‘ hÃ³a monochrome Ä‘Ã­ch thá»±c).

---

## 6. Kiáº¿n TrÃºc Container (Docker Containerization)

Äá»ƒ tá»‘i Æ°u hÃ³a hiá»‡u nÄƒng, tÃ­nh nháº¥t quÃ¡n cá»§a mÃ´i trÆ°á»ng OpenCV/AI vÃ  dá»… dÃ ng triá»ƒn khai, chÃºng tÃ´i thiáº¿t káº¿ 2 mÃ´ hÃ¬nh cháº¡y Docker:

### A. MÃ´ HÃ¬nh Cháº¡y Hybrid (KhuyÃªn dÃ¹ng cho nhu cáº§u hiá»ƒn thá»‹ GUI local)
*   **Docker Container:** Chá»‰ cháº¡y backend xá»­ lÃ½ áº£nh. Chá»©a OpenCV, Pillow, vÃ  `rembg` (tÃ­ch há»£p sáºµn model U2NET). Cung cáº¥p má»™t FastAPI endpoint nháº­n áº£nh Ä‘áº§u vÃ o vÃ  tráº£ vá» danh sÃ¡ch tá»a Ä‘á»™ Contour dÆ°á»›i dáº¡ng JSON.
*   **Host OS (MÃ¡y local cá»§a báº¡n):** Cháº¡y má»™t script Python má»ng Ä‘iá»u khiá»ƒn thÆ° viá»‡n `turtle` Ä‘á»“ há»a. Script nÃ y gá»i API tá»›i Docker Container Ä‘á»ƒ láº¥y danh sÃ¡ch tá»a Ä‘á»™ Contour Ä‘Ã£ qua xá»­ lÃ½ rá»“i hiá»ƒn thá»‹ váº½ Ä‘á»™ng ngay trÃªn mÃ¡y cá»§a báº¡n.
*   **Æ¯u Ä‘iá»ƒm:** Kháº¯c phá»¥c triá»‡t Ä‘á»ƒ lá»—i káº¿t ná»‘i Ä‘á»“ há»a X11/Tkinter cá»§a Docker trÃªn Windows.

### B. MÃ´ HÃ¬nh Cháº¡y Web-Only (Docker Äá»™c Láº­p 100%)
*   **Docker Container:** Cháº¡y cáº£ backend xá»­ lÃ½ áº£nh láº«n giao diá»‡n Gradio.
*   **Giáº£i phÃ¡p Ä‘á»“ há»a thay tháº¿:** Thay vÃ¬ dÃ¹ng thÆ° viá»‡n `turtle` (vá»‘n phá»¥ thuá»™c vÃ o Tkinter GUI cá»§a mÃ¡y), há»‡ thá»‘ng sáº½ biÃªn dá»‹ch Ä‘Æ°á»ng nÃ©t OpenCV thÃ nh file **SVG (Scalable Vector Graphics)**.
*   **TrÃ¬nh diá»…n:** Sá»­ dá»¥ng thÆ° viá»‡n JavaScript `Anime.js` hoáº·c tháº» `<canvas>` tÃ­ch há»£p trá»±c tiáº¿p trÃªn web Gradio Ä‘á»ƒ váº½ Ä‘á»™ng nÃ©t bÃºt cháº¡y trÃªn trÃ¬nh duyá»‡t web.
*   **Æ¯u Ä‘iá»ƒm:** CÃ³ thá»ƒ triá»ƒn khai lÃªn báº¥t ká»³ server VPS, Cloud nÃ o mÃ  khÃ´ng cáº§n káº¿t ná»‘i pháº§n cá»©ng mÃ n hÃ¬nh.

---

## 7. TÃ­ch Há»£p Gemini API (AI Art Pre-processor & Vision Guide)

Há»‡ thá»‘ng khÃ´ng sá»­ dá»¥ng giao diá»‡n Chatbot, thay vÃ o Ä‘Ã³ tÃ­ch há»£p Gemini 3.1 Flash Lite lÃ m bá»™ tiá»n xá»­ lÃ½ hÃ¬nh áº£nh (Vision Pre-processor) trá»±c tiáº¿p trong luá»“ng váº½ tranh.

### A. Cáº¥u HÃ¬nh & Code Import Chuáº©n
*   **ThÆ° viá»‡n:** Sá»­ dá»¥ng SDK chÃ­nh thá»©c `google-genai` má»›i cá»§a Google AI Studio.
*   **Code máº«u káº¿t ná»‘i API:**
    ```python
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field

    # Äá»‹nh nghÄ©a cáº¥u trÃºc dá»¯ liá»‡u tráº£ vá» mong muá»‘n báº±ng Pydantic
    class DrawingParams(BaseModel):
        blur_size: int = Field(description="Äá»™ má»‹n Gaussian Blur, pháº£i lÃ  sá»‘ láº» tá»« 1 Ä‘áº¿n 21")
        threshold_block: int = Field(description="KÃ­ch thÆ°á»›c block tÃ­nh ngÆ°á»¡ng, sá»‘ láº» tá»« 3 Ä‘áº¿n 51")
        threshold_c: int = Field(description="Háº±ng sá»‘ C hiá»‡u chá»‰nh biÃªn nÃ©t, tá»« 1 Ä‘áº¿n 20")
        jitter: float = Field(description="Äá»™ rung nÃ©t váº½ tay, tá»« 0.0 Ä‘áº¿n 2.0")
        hatching: float = Field(description="Má»©c Ä‘á»™ gáº¡ch bÃ³ng tá»‘i táº¡o khá»‘i, tá»« 0.0 Ä‘áº¿n 1.0")
        explanation: str = Field(description="LÃ½ do AI lá»±a chá»n bá»™ tham sá»‘ nÃ y cho bá»©c áº£nh vÃ  phong cÃ¡ch tÆ°Æ¡ng á»©ng")

    # Khá»Ÿi táº¡o client
    client = genai.Client()
    ```

### B. Vai TrÃ² Bá»™ Tiá»n Xá»­ LÃ½ (Vision Parameter Optimizer)
*   **Äáº§u vÃ o:** Gá»­i bá»©c áº£nh gá»‘c ngÆ°á»i dÃ¹ng upload kÃ¨m theo phong cÃ¡ch váº½ (Vibe) Ä‘Ã£ chá»n.
*   **Xá»­ lÃ½:** Gemini phÃ¢n tÃ­ch bá»‘ cá»¥c, máº­t Ä‘á»™ chi tiáº¿t, Ä‘á»™ sÃ¡ng tá»‘i cá»§a áº£nh gá»‘c vÃ  tá»± Ä‘á»™ng tráº£ vá» bá»™ tham sá»‘ tá»‘i Æ°u hÃ³a (`DrawingParams`) dÆ°á»›i Ä‘á»‹nh dáº¡ng **Structured JSON** thÃ´ng qua cÆ¡ cháº¿ cáº¥u hÃ¬nh JSON Schema cá»§a SDK.
*   **Äáº§u ra:** CÃ¡c tham sá»‘ Ä‘Æ°á»£c Ä‘Æ°a trá»±c tiáº¿p vÃ o OpenCV vÃ  bá»™ váº½ nÃ©t Pillow Ä‘á»ƒ táº¡o káº¿t quáº£ nghá»‡ thuáº­t cao nháº¥t vÃ  cÃ³ nÃ©t váº½ "vibe" nháº¥t.

### C. CÃ¡c Phong CÃ¡ch Váº½ (Art Vibes) Há»— Trá»£
Há»‡ thá»‘ng cho phÃ©p ngÆ°á»i dÃ¹ng tÃ¹y chá»n cÃ¡c vibe váº½ tranh khÃ¡c nhau:
1.  **Váº½ Táº£ Thá»±c (Realistic Sketch):** NÃ©t váº½ chi tiáº¿t, Ä‘á»™ dÃ y biáº¿n thiÃªn máº¡nh, táº­n dá»¥ng tá»‘i Ä‘a gáº¡ch bÃ³ng gáº¡ch chÃ©o (`hatching` cao) Ä‘á»ƒ diá»…n táº£ khá»‘i tá»‘i sÃ¡ng cá»§a chÃ¢n dung hoáº·c tÄ©nh váº­t.
2.  **NÃ©t Váº½ Anime (Anime Outline):** ÄÆ°á»ng viá»n dÃ y dáº·n, cÃ¡c Ä‘Æ°á»ng nÃ©t Ä‘Æ°á»£c tá»‘i giáº£n vÃ  lÃ m ráº¥t mÆ°á»£t (RDP `epsilon` tÄƒng nháº¹), loáº¡i bá» hoÃ n toÃ n gáº¡ch bÃ³ng táº¡o khá»‘i (`hatching = 0`).
3.  **Tranh ChÃ¬ MÃ u (Colored Pencil Sketch):** NÃ©t váº½ Ä‘á»•i mÃ u bÃºt linh hoáº¡t theo tÃ´ng mÃ u thá»±c táº¿ cá»§a áº£nh gá»‘c hoáº·c máº£ng mÃ u K-Means.
4.  **Tranh Sá»‘ HÃ³a Tá»± TÃ´ (Paint-by-Numbers Blueprint):** Báº£n váº½ nÃ©t rá»—ng khÃ©p kÃ­n, phÃ¢n tÃ¡ch cÃ¡c máº£ng sáº¯c Ä‘á»™ xÃ¡m rÃµ rá»‡t vÃ  gÃ¡n sá»‘ thá»© tá»± tá»± tÃ´ mÃ u.


---

## 8. Äáº·c Táº£ Chi Tiáº¿t Cháº¿ Äá»™ Váº½ NÃ©t & Xuáº¥t Báº£n ÄÃ³ng GÃ³i (ZIP & Video)

### A. Bá»™ Nháº­n Diá»‡n BiÃªn Lai Há»—n Há»£p (Hybrid Edge Detection)
Äá»ƒ xá»­ lÃ½ cÃ¡c bá»©c tranh phá»©c táº¡p, Ä‘á»™ tÆ°Æ¡ng pháº£n tháº¥p, tranh váº½ chÃ¬ bÃ³ng má»‹n (nhÆ° tranh chÃ¢n dung nghá»‡ thuáº­t) hoáº·c tranh sÆ¡n dáº§u phong cáº£nh, há»‡ thá»‘ng Ã¡p dá»¥ng ká»¹ thuáº­t dÃ² biÃªn há»—n há»£p:
1.  **Canny Edge Detection:** Äá»‹nh vá»‹ cÃ¡c Ä‘Æ°á»ng viá»n cáº¥u trÃºc lá»›n rÃµ rá»‡t (silhouettes, Ä‘Æ°á»ng biÃªn cÆ¡ thá»ƒ, náº¿p gáº¥p quáº§n Ã¡o lá»›n).
2.  **Adaptive Thresholding:** Äá»‹nh vá»‹ cÃ¡c chi tiáº¿t nhá» tinh táº¿ (mÅ©i, náº¿p nhÄƒn nhá», máº¯t, miá»‡ng).
3.  **Bilateral Filter & CLAHE:** Tá»± Ä‘á»™ng tÄƒng cÆ°á»ng chi tiáº¿t cá»¥c bá»™ vÃ  lá»c nháºµn lÃ´ng thÃº cÆ°ng, vÃ¢n háº¡t trÆ°á»›c khi rÃºt biÃªn.
4.  **Káº¿t há»£p (OR gate):** Trá»™n cáº£ hai báº£n Ä‘á»“ biÃªn Ä‘á»ƒ sinh ra cÃ¡c Ä‘Æ°á»ng nÃ©t váº½ trá»n váº¹n nháº¥t, Ä‘áº£m báº£o cÃ¡c chi tiáº¿t nhÆ° máº¯t, mÅ©i vÃ  cÃ¡c vÃ¹ng tá»‘i má» xung quanh Ä‘á»u Ä‘Æ°á»£c váº½ sáº¯c nÃ©t.

### B. CÆ¡ Cháº¿ PhÃ¡t Láº¡i QuÃ¡ TrÃ¬nh Váº½ (Replay Feature)
*   **Client-side Caching:** ToÃ n bá»™ cÃ¡c khung hÃ¬nh áº£nh Base64 Ä‘Æ°á»£c tráº£ vá» tá»« luá»“ng stream cá»§a server sáº½ Ä‘Æ°á»£c lÆ°u trá»¯ táº¡m thá»i vÃ o máº£ng lÆ°u trá»¯ JavaScript á»Ÿ phÃ­a Client.
*   **Replay Button (Xem láº¡i quÃ¡ trÃ¬nh):** NgÆ°á»i dÃ¹ng cÃ³ thá»ƒ click nÃºt "Xem Láº¡i" báº¥t ká»³ lÃºc nÃ o Ä‘á»ƒ phÃ¡t láº¡i chuyá»ƒn Ä‘á»™ng váº½ nÃ©t tá»« Ä‘áº§u Ä‘áº¿n cuá»‘i trÃªn Canvas vá»›i tá»‘c Ä‘á»™ mÆ°á»£t mÃ  (30 FPS) mÃ  khÃ´ng cáº§n gá»i láº¡i API server, giÃºp giáº£m táº£i tá»‘i Ä‘a cho há»‡ thá»‘ng.

### C. Xuáº¥t Video QuÃ¡ TrÃ¬nh Váº½ & ÄÃ³ng GÃ³i Tá»‡p ZIP
Äá»ƒ ngÆ°á»i dÃ¹ng cÃ³ thá»ƒ chia sáº» quÃ¡ trÃ¬nh váº½ tranh lÃªn máº¡ng xÃ£ há»™i dÆ°á»›i dáº¡ng video ngáº¯n, há»‡ thá»‘ng tÃ­ch há»£p module sinh video vÃ  Ä‘Ã³ng gÃ³i:
1.  **OpenCV VideoWriter:** Trong quÃ¡ trÃ¬nh váº½ tranh Ä‘á»™ng á»Ÿ backend, má»—i khung hÃ¬nh Pillow sáº½ Ä‘Æ°á»£c ghi trá»±c tiáº¿p vÃ o tá»‡p video `.mp4` (sá»­ dá»¥ng codec `mp4v` chuáº©n).
2.  **ÄÃ³ng gÃ³i ZIP (zipfile):** Sau khi quÃ¡ trÃ¬nh váº½ káº¿t thÃºc, há»‡ thá»‘ng nÃ©n:
    *   Tá»‡p áº£nh káº¿t quáº£ sáº¯c nÃ©t cuá»‘i cÃ¹ng: `final_artwork.png`
    *   Tá»‡p video quÃ¡ trÃ¬nh váº½: `drawing_process.mp4`
    VÃ o má»™t tá»‡p lÆ°u trá»¯ duy nháº¥t: `tranhve_package.zip`.
3.  **Táº£i vá»:** NÃºt táº£i vá» trÃªn giao diá»‡n sáº½ táº£i tá»‡p ZIP Ä‘Ã³ng gÃ³i nÃ y.





