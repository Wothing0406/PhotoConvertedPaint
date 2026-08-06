document.addEventListener("DOMContentLoaded", () => {
    // ── DOM refs ──────────────────────────────────────────────────────────────
    const uploadZone        = document.getElementById("upload-zone");
    const fileInput         = document.getElementById("file-input");
    const uploadPlaceholder = document.getElementById("upload-placeholder");
    const imagePreview      = document.getElementById("image-preview");
    const vibeMode          = document.getElementById("vibe-mode");
    const removeBg          = document.getElementById("remove-bg");
    const geminiTune        = document.getElementById("gemini-tune");   // optional Gemini checkbox

    // Sliders
    const blurSlider     = document.getElementById("blur-slider");
    const blockSlider    = document.getElementById("block-slider");
    const cSlider        = document.getElementById("c-slider");
    const jitterSlider   = document.getElementById("jitter-slider");
    const hatchingSlider = document.getElementById("hatching-slider");
    const speedSlider    = document.getElementById("speed-slider");

    // Slider value labels
    const blurVal     = document.getElementById("blur-val");
    const blockVal    = document.getElementById("block-val");
    const cVal        = document.getElementById("c-val");
    const jitterVal   = document.getElementById("jitter-val");
    const hatchingVal = document.getElementById("hatching-val");
    const speedVal    = document.getElementById("speed-val");

    // Buttons & output
    const drawBtn        = document.getElementById("draw-btn");
    const downloadBtn    = document.getElementById("download-btn");
    const replayBtn      = document.getElementById("replay-btn");
    const fullscreenBtn  = document.getElementById("fullscreen-btn");
    const canvasContainer= document.getElementById("canvas-container");
    const statusBox      = document.getElementById("status-box");
    const canvasImg      = document.getElementById("drawing-canvas");
    const qualityBtn     = document.getElementById("quality-btn");

    // ── State ─────────────────────────────────────────────────────────────────
    let selectedFile      = null;
    let drawingFrames     = [];
    let replayInterval    = null;
    let currentResultPath = null;

    // ── Slider sync ───────────────────────────────────────────────────────────
    const syncVal = (slider, valEl) => {
        if (!slider || !valEl) return;
        slider.addEventListener("input", () => { valEl.textContent = slider.value; });
    };
    syncVal(blurSlider,     blurVal);
    syncVal(blockSlider,    blockVal);
    syncVal(cSlider,        cVal);
    syncVal(jitterSlider,   jitterVal);
    syncVal(hatchingSlider, hatchingVal);
    syncVal(speedSlider,    speedVal);

    // ── File upload ───────────────────────────────────────────────────────────
    if (uploadZone)  uploadZone.addEventListener("click", () => fileInput && fileInput.click());

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) handleFile(e.target.files[0]);
        });
    }

    if (uploadZone) {
        uploadZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            uploadZone.classList.add("dragover");
        });
        uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
        uploadZone.addEventListener("drop", (e) => {
            e.preventDefault();
            uploadZone.classList.remove("dragover");
            if (e.dataTransfer && e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
        });
    }

    function handleFile(file) {
        if (!file || !file.type.startsWith("image/")) {
            alert("Vui lòng chọn file ảnh hợp lệ (JPG, PNG, WEBP...).");
            return;
        }
        selectedFile = file;
        if (uploadPlaceholder) uploadPlaceholder.style.display = "none";
        if (imagePreview) {
            imagePreview.style.display = "block";
            imagePreview.src = URL.createObjectURL(file);
        }
    }

    // ── Replay ────────────────────────────────────────────────────────────────
    if (replayBtn) {
        replayBtn.addEventListener("click", () => {
            if (drawingFrames.length === 0 || replayBtn.classList.contains("disabled")) return;
            if (replayInterval) clearInterval(replayInterval);

            let frameIndex = 0;
            replayBtn.disabled = true;
            replayBtn.classList.add("disabled");
            replayBtn.textContent = "Đang phát lại...";

            replayInterval = setInterval(() => {
                if (frameIndex < drawingFrames.length) {
                    if (canvasImg) canvasImg.src = "data:image/jpeg;base64," + drawingFrames[frameIndex];
                    frameIndex++;
                } else {
                    clearInterval(replayInterval);
                    replayBtn.disabled = false;
                    replayBtn.classList.remove("disabled");
                    replayBtn.textContent = "Xem Lại Quá Trình";
                }
            }, 50);
        });
    }

    // ── Quality check ─────────────────────────────────────────────────────────
    if (qualityBtn) {
        qualityBtn.addEventListener("click", async () => {
            if (!currentResultPath || !selectedFile) {
                alert("Hãy vẽ xong một bức tranh trước khi kiểm tra chất lượng.");
                return;
            }
            qualityBtn.disabled = true;
            qualityBtn.textContent = "Đang kiểm tra...";

            try {
                const qForm = new FormData();
                qForm.append("original",    selectedFile);
                qForm.append("result_path", currentResultPath);

                const qRes = await fetch("/api/quality-check", { method: "POST", body: qForm });
                if (qRes.ok) {
                    const q = await qRes.json();
                    const suggestHtml = q.suggestions && q.suggestions.length > 0
                        ? `<br><br><strong>💡 Gợi ý cải thiện:</strong><br>• ${q.suggestions.join("<br>• ")}`
                        : "<br><br>✅ Bức tranh đạt chất lượng tốt!";

                    statusBox.innerHTML = `
                        🔍 <strong>Kết Quả Kiểm Tra Chất Lượng</strong><br><br>
                        📊 <strong>Điểm tổng: ${q.quality_score}/100 (${q.grade})</strong><br>
                        🎯 Độ phủ nét viền: ${q.edge_coverage_pct}%<br>
                        🖼️ Độ phủ canvas: ${q.canvas_coverage_pct}%<br>
                        🔬 Mật độ chi tiết: ${q.detail_density_pct}%
                        ${suggestHtml}
                    `;
                } else {
                    statusBox.innerText = "Kiểm tra chất lượng thất bại.";
                }
            } catch (e) {
                statusBox.innerText = `Lỗi kiểm tra: ${e.message}`;
            } finally {
                qualityBtn.disabled = false;
                qualityBtn.textContent = "🔍 Kiểm Tra Chất Lượng";
            }
        });
    }

    // ── Main drawing flow ─────────────────────────────────────────────────────
    if (drawBtn) {
        drawBtn.addEventListener("click", async () => {
            if (!selectedFile) {
                alert("Vui lòng tải ảnh lên trước.");
                return;
            }

            drawBtn.disabled = true;
            drawBtn.textContent = "Đang xử lý...";

            if (downloadBtn)  { downloadBtn.classList.add("disabled"); downloadBtn.href = "#"; }
            if (replayBtn)    { replayBtn.classList.add("disabled"); replayBtn.disabled = true; }
            if (qualityBtn)   { qualityBtn.disabled = true; }
            currentResultPath = null;

            drawingFrames = [];
            if (statusBox) statusBox.classList.remove("success");
            if (replayInterval) { clearInterval(replayInterval); replayInterval = null; }

            statusBox.innerText = "Bắt đầu khởi tạo luồng...";
            let explanationText = "";
            let aiParams = null;

            try {
                // ── Step 1: Optional Gemini param tuning ──────────────────────
                const useGemini = geminiTune && geminiTune.checked;

                if (useGemini) {
                    statusBox.innerText = "🤖 Gemini 3.1 Flash Lite đang phân tích ảnh...";

                    const optimizeForm = new FormData();
                    optimizeForm.append("image",        selectedFile);
                    optimizeForm.append("vibe",         vibeMode.value);
                    optimizeForm.append("force_gemini", "true");

                    try {
                        const optRes = await fetch("/api/optimize-params", {
                            method: "POST",
                            body:   optimizeForm
                        });
                        if (optRes.ok) {
                            aiParams = await optRes.json();

                            // Update sliders with Gemini's recommendations
                            if (blurSlider)     { blurSlider.value     = aiParams.blur_size;      blurVal.textContent     = aiParams.blur_size; }
                            if (blockSlider)    { blockSlider.value    = aiParams.threshold_block; blockVal.textContent    = aiParams.threshold_block; }
                            if (cSlider)        { cSlider.value        = aiParams.threshold_c;    cVal.textContent        = aiParams.threshold_c; }
                            if (jitterSlider)   { jitterSlider.value   = aiParams.jitter;         jitterVal.textContent   = aiParams.jitter; }
                            if (hatchingSlider) { hatchingSlider.value = aiParams.hatching;       hatchingVal.textContent = aiParams.hatching; }

                            explanationText = `📊 Gemini 3.1 Flash Lite: ${aiParams.explanation}\n\n`;
                            statusBox.innerText = `${explanationText}Thông số đã được Gemini tối ưu!`;
                        } else {
                            console.warn("Gemini optimize thất bại, dùng thông số slider hiện tại.");
                        }
                    } catch (geminiErr) {
                        console.warn("Gemini call lỗi:", geminiErr.message);
                    }
                }

                // ── Step 2: Stream drawing ────────────────────────────────────
                const vibeLabel = {
                    "Realistic Sketch":           "🖋️ Vẽ phác thảo tả thực (5 lớp)...",
                    "Colored Pencil Sketch":       "🖍️ Vẽ nét chì màu (4 lớp)...",
                    "Anime Outline":               "✒️ Vẽ viền anime (XDoG ink)...",
                    "Oil Painting":                "🎨 Vẽ sơn dầu impasto (3 lượt)...",
                    "Paint-by-Numbers Blueprint":  "🔢 Phân vùng màu Paint-by-Numbers..."
                }[vibeMode.value] || "Đang vẽ...";

                statusBox.innerText = `${explanationText}${vibeLabel}`;

                const drawForm = new FormData();
                drawForm.append("image",          selectedFile);
                drawForm.append("vibe",           vibeMode.value);
                drawForm.append("remove_bg",      removeBg ? removeBg.checked : false);
                drawForm.append("blur",           blurSlider   ? blurSlider.value   : "3");
                drawForm.append("thresh_block",   blockSlider  ? blockSlider.value  : "11");
                drawForm.append("thresh_c",       cSlider      ? cSlider.value      : "5");
                drawForm.append("jitter",         jitterSlider ? jitterSlider.value : "0.4");
                drawForm.append("hatching",       hatchingSlider ? hatchingSlider.value : "0");
                drawForm.append("speed",          speedSlider  ? speedSlider.value  : "10");
                drawForm.append("bg_color_wash",  aiParams ? String(aiParams.bg_color_wash ?? true)  : "true");
                drawForm.append("wash_opacity",   aiParams ? String(aiParams.wash_opacity   ?? 60)   : "60");
                drawForm.append("sketch_opacity", aiParams ? String(aiParams.sketch_opacity ?? 0.12) : "0.12");
                drawForm.append("line_art_width", aiParams ? String(aiParams.line_art_width ?? 1)    : "1");

                const response = await fetch("/api/draw-stream", { method: "POST", body: drawForm });
                if (!response.ok) throw new Error("Lỗi kết nối tới máy chủ vẽ.");

                const reader  = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let buffer = "";
                let frameCount = 0;

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split("\n");
                    buffer = lines.pop(); // keep incomplete line

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed.startsWith("data: ")) continue;

                        const payload = trimmed.substring(6).trim();

                        if (payload.startsWith("error:")) {
                            statusBox.innerText = `❌ Lỗi: ${payload.substring(6)}`;
                        } else if (payload.startsWith("done:")) {
                            const total = parseInt(payload.substring(5));
                            statusBox.innerText = `${explanationText}✅ Hoàn tất ${total} khung hình. Đang đóng gói ZIP...`;
                        } else if (payload.startsWith("filepath:")) {
                            const filepath = payload.substring(9).trim();
                            // filepath = "output/tranhve_{id}.zip"
                            // final PNG = "output/final_{id}.png"
                            const sessionId = filepath.replace(/.*tranhve_/, "").replace(".zip", "");
                            currentResultPath = `output/final_${sessionId}.png`;

                            if (downloadBtn) {
                                downloadBtn.href = `/api/download?path=${encodeURIComponent(filepath)}`;
                                downloadBtn.classList.remove("disabled");
                            }
                            if (replayBtn)  { replayBtn.classList.remove("disabled"); replayBtn.disabled = false; }
                            if (qualityBtn) { qualityBtn.disabled = false; }

                            statusBox.classList.add("success");
                            statusBox.innerHTML = `🎉 <strong>ĐÃ VẼ XONG TÁC PHẨM!</strong><br><br>
                                ${explanationText.replace(/\n/g, "<br>")}
                                Tác phẩm đã được đóng gói thành công. Bạn có thể tải xuống ZIP hoặc nhấn <strong>Kiểm Tra Chất Lượng</strong> để xem điểm số.`;
                        } else if (payload.length > 20) {
                            // Base64 JPEG frame
                            frameCount++;
                            drawingFrames.push(payload);
                            if (canvasImg) canvasImg.src = "data:image/jpeg;base64," + payload;
                            if (frameCount % 15 === 0) {
                                statusBox.innerText = `${explanationText}${vibeLabel} — ${frameCount} khung hình`;
                            }
                        }
                    }
                }

            } catch (error) {
                console.error(error);
                statusBox.innerText = `❌ Lỗi hệ thống: ${error.message}`;
            } finally {
                drawBtn.disabled = false;
                drawBtn.textContent = "Bắt Đầu Vẽ Động";
            }
        });
    }

    // ── Fullscreen ────────────────────────────────────────────────────────────
    if (fullscreenBtn && canvasContainer) {
        fullscreenBtn.addEventListener("click", () => {
            if (!document.fullscreenElement) {
                canvasContainer.requestFullscreen().catch(err => {
                    console.error(`Không thể vào chế độ toàn màn hình: ${err.message}`);
                });
            } else {
                document.exitFullscreen();
            }
        });
    }
});
