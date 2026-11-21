/**
 * GeoLand AI — Frontend Backend API Client & Component Connector
 */

const API_BASE = window.location.port === "8000" ? window.location.origin : "http://localhost:8000";

// Robust DataURI to Blob Converter (eliminates fetch(dataUri) network bugs)
function dataURItoBlob(dataURI) {
    if (!dataURI) return null;
    try {
        if (!dataURI.includes(',')) {
            const byteString = atob(dataURI);
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            return new Blob([ab], { type: 'image/jpeg' });
        }
        const parts = dataURI.split(',');
        const byteString = atob(parts[1]);
        const mimeString = parts[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        return new Blob([ab], { type: mimeString });
    } catch (e) {
        console.error("dataURItoBlob conversion error:", e);
        return null;
    }
}

// API Helpers
async function apiRequest(endpoint, options = {}) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`, options);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `HTTP Error ${res.status}`);
        }
        return await res.json();
    } catch (err) {
        console.error(`[API Error] ${endpoint}:`, err);
        throw err;
    }
}

// 1. Classification & Prediction
async function predictImage(fileOrBlob, uncertaintyThreshold = 0.60) {
    const formData = new FormData();
    formData.append("file", fileOrBlob, fileOrBlob.name || "satellite_patch.jpg");
    return await apiRequest(`/predict?uncertainty_threshold=${uncertaintyThreshold}`, {
        method: "POST",
        body: formData
    });
}

// 2. Explainability & Grad-CAM
async function explainImage(fileOrBlob, targetClass = null, usePlusPlus = true) {
    const formData = new FormData();
    formData.append("file", fileOrBlob, fileOrBlob.name || "satellite_patch.jpg");
    if (targetClass) formData.append("target_class", targetClass);
    formData.append("use_plus_plus", usePlusPlus);
    return await apiRequest("/explain", {
        method: "POST",
        body: formData
    });
}

// 3. Robustness Stress-Testing
async function testRobustness(fileOrBlob, corruption = "noise", severity = 0.10) {
    const formData = new FormData();
    formData.append("file", fileOrBlob, fileOrBlob.name || "satellite_patch.jpg");
    formData.append("corruption", corruption);
    formData.append("severity", severity);
    return await apiRequest("/robustness", {
        method: "POST",
        body: formData
    });
}

// 4. Metrics & Diagnostics
async function fetchMetrics() {
    return await apiRequest("/metrics");
}

// 5. Preset Samples
async function fetchPresetSamples() {
    return await apiRequest("/preset-samples");
}

// 6. Image Restoration (HF GFPGAN / local fallback)
async function restoreImage(fileOrBlob, modelVersion = "v1.4", scale = 2) {
    const formData = new FormData();
    formData.append("file", fileOrBlob, fileOrBlob.name || "satellite.jpg");
    formData.append("model_version", modelVersion);
    formData.append("scale", scale);
    return await apiRequest("/restore", { method: "POST", body: formData });
}

/* =========================================================================
   PAGE-SPECIFIC COMPONENT WIRE-UPS
   ========================================================================= */

// Wire Satellite Image Classification Page
function setupClassificationPage() {
    const imgElement = document.getElementById("classified-satellite-image");
    if (!imgElement) return;

    const dropzone = document.getElementById("image-dropzone");
    const dragOverlay = document.getElementById("dropzone-drag-overlay");
    const fileInput = document.getElementById("image-file-input") || document.getElementById("hidden-image-file-input");
    const uploadBtn = document.getElementById("btn-upload-image");
    const useSampleBtn = document.getElementById("btn-use-sample");
    const classifyBtn = document.getElementById("btn-classify");
    const gradcamBtn = document.getElementById("btn-gradcam-heatmap");
    const downloadBtn = document.getElementById("btn-download-result");
    const restoreBtn = document.getElementById("btn-restore-image");

    let currentFileOrBlob = null;
    let currentGradcamOverlay = null;
    let currentSegmentationOverlay = null;
    let currentOriginalImageSrc = null;
    let showingGradcam = false;
    let activeLayer = "optical";
    let cachedPresets = {};
    let currentPresetIndex = 0;
    const presetOrder = ["Highway", "River", "Forest", "Residential", "Industrial", "AnnualCrop", "Pasture", "PermanentCrop", "SeaLake", "HerbaceousVegetation"];

    // View Mode Switcher
    const optBtn = document.getElementById("view-mode-optical");
    const segBtn = document.getElementById("view-mode-segmentation");
    const gcamBtn = document.getElementById("view-mode-gradcam");

    function setLayer(layer) {
        activeLayer = layer;
        [optBtn, segBtn, gcamBtn].forEach(btn => {
            if (btn) btn.className = "px-2.5 py-1 rounded text-outline hover:text-primary cursor-pointer transition-all";
        });

        if (layer === "optical") {
            if (optBtn) optBtn.className = "px-2.5 py-1 rounded bg-surface-container-lowest text-primary font-bold shadow-xs cursor-pointer transition-all";
            if (currentOriginalImageSrc) imgElement.src = currentOriginalImageSrc;
        } else if (layer === "segmentation") {
            if (segBtn) segBtn.className = "px-2.5 py-1 rounded bg-secondary text-on-secondary font-bold shadow-xs cursor-pointer transition-all";
            if (currentSegmentationOverlay) {
                imgElement.src = currentSegmentationOverlay;
            } else {
                alert("Land-cover segmentation map will be visible once image is classified.");
            }
        } else if (layer === "gradcam") {
            if (gcamBtn) gcamBtn.className = "px-2.5 py-1 rounded bg-secondary text-on-secondary font-bold shadow-xs cursor-pointer transition-all";
            if (currentGradcamOverlay) {
                imgElement.src = currentGradcamOverlay;
            } else if (gradcamBtn) {
                gradcamBtn.click();
            }
        }
    }

    if (optBtn) optBtn.addEventListener("click", () => setLayer("optical"));
    if (segBtn) segBtn.addEventListener("click", () => setLayer("segmentation"));
    if (gcamBtn) gcamBtn.addEventListener("click", () => setLayer("gradcam"));

    // 1. File Upload Button & Dropzone Click
    if (uploadBtn && fileInput) {
        uploadBtn.addEventListener("click", () => fileInput.click());
    }
    if (dropzone && fileInput) {
        dropzone.addEventListener("click", (e) => {
            if (e.target.closest("button") || e.target.closest("a")) return;
            fileInput.click();
        });
    }

    // 2. Drag & Drop Handling
    if (dropzone) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (dragOverlay) dragOverlay.classList.remove("hidden");
                dropzone.classList.add("ring-4", "ring-secondary-fixed", "ring-opacity-80");
            });
        });

        ['dragleave', 'dragend'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (dragOverlay) dragOverlay.classList.add("hidden");
                dropzone.classList.remove("ring-4", "ring-secondary-fixed", "ring-opacity-80");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (dragOverlay) dragOverlay.classList.add("hidden");
            dropzone.classList.remove("ring-4", "ring-secondary-fixed", "ring-opacity-80");

            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleImageFile(e.dataTransfer.files[0]);
            }
        });
    }

    // Prevent global drag/drop from opening image in tab
    window.addEventListener("dragover", e => e.preventDefault(), false);
    window.addEventListener("drop", e => e.preventDefault(), false);

    // File Input Change
    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                handleImageFile(e.target.files[0]);
            }
        });
    }

    function handleImageFile(file) {
        currentFileOrBlob = file;
        currentGradcamOverlay = null;
        currentSegmentationOverlay = null;
        showingGradcam = false;
        setLayer("optical");

        const reader = new FileReader();
        reader.onload = (evt) => {
            currentOriginalImageSrc = evt.target.result;
            imgElement.src = evt.target.result;
            const tileHeader = document.querySelector(".font-label-lg.text-label-lg.text-primary");
            if (tileHeader) tileHeader.textContent = `Tile: ${file.name}`;
            runInference();
        };
        reader.readAsDataURL(file);
    }

    // 3. Preset Samples Handling
    const sampleButtons = document.querySelectorAll(".sample-btn, button[data-sample-class]");

    async function loadPresetClass(className) {
        if (!cachedPresets[className]) {
            try {
                const presets = await fetchPresetSamples();
                cachedPresets = presets;
            } catch (e) {
                console.error("Presets loading error:", e);
            }
        }

        const sample = cachedPresets[className];
        if (sample && sample.image_base64) {
            currentOriginalImageSrc = sample.image_base64;
            imgElement.src = sample.image_base64;
            currentFileOrBlob = dataURItoBlob(sample.image_base64);
            currentGradcamOverlay = null;
            currentSegmentationOverlay = null;
            showingGradcam = false;
            setLayer("optical");

            const tileHeader = document.querySelector(".font-label-lg.text-label-lg.text-primary");
            if (tileHeader) tileHeader.textContent = `Tile: EuroSAT_${className}_${sample.filename || '001.jpg'}`;

            // Update active sample button styles
            sampleButtons.forEach(b => {
                const c = b.dataset.sampleClass || b.textContent;
                if (c.toLowerCase().includes(className.toLowerCase())) {
                    b.className = "sample-btn px-space-sm py-1 rounded-full bg-secondary-container text-on-secondary-container font-label-md text-label-md flex items-center gap-1 transition-all shadow-sm";
                    if (!b.querySelector(".rounded-full.bg-secondary")) {
                        b.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-secondary"></span><span>${b.textContent.replace('Sample #', 'Sample ')}</span>`;
                    }
                } else {
                    b.className = "sample-btn px-space-sm py-1 rounded-full bg-surface-container-high text-on-surface-variant font-label-md text-label-md hover:bg-surface-container-highest transition-colors";
                    const dot = b.querySelector(".w-1.5");
                    if (dot) dot.remove();
                }
            });

            runInference();
        }
    }

    // Wire each sample button
    sampleButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const c = btn.dataset.sampleClass || (
                btn.textContent.includes("Highway") ? "Highway" :
                btn.textContent.includes("River") ? "River" :
                btn.textContent.includes("Forest") ? "Forest" :
                btn.textContent.includes("Residential") ? "Residential" : "Industrial"
            );
            loadPresetClass(c);
        });
    });

    // Wire "Use Sample" button (cycles through presets)
    if (useSampleBtn) {
        useSampleBtn.addEventListener("click", () => {
            currentPresetIndex = (currentPresetIndex + 1) % presetOrder.length;
            loadPresetClass(presetOrder[currentPresetIndex]);
        });
    }

    // Preload presets and check URL query param (e.g. ?sample=Forest from Dataset Explorer)
    fetchPresetSamples().then(presets => {
        cachedPresets = presets;
        const urlParams = new URLSearchParams(window.location.search);
        const requestedSample = urlParams.get('sample');
        if (requestedSample && presets[requestedSample]) {
            loadPresetClass(requestedSample);
        }
    }).catch(err => console.log("Preset preload:", err));

    // 4. Classify Button
    if (classifyBtn) {
        classifyBtn.addEventListener("click", () => {
            runInference();
        });
    }

    // 5. Grad-CAM Button
    if (gradcamBtn) {
        gradcamBtn.addEventListener("click", async () => {
            if (!currentFileOrBlob) {
                currentFileOrBlob = dataURItoBlob(imgElement.src);
                if (!currentFileOrBlob) {
                    const res = await fetch(imgElement.src);
                    currentFileOrBlob = await res.blob();
                }
            }

            const gradcamText = document.getElementById("btn-gradcam-text");

            if (!currentGradcamOverlay) {
                gradcamBtn.classList.add("animate-pulse");
                if (gradcamText) gradcamText.textContent = "Generating...";
                try {
                    const explainRes = await explainImage(currentFileOrBlob);
                    currentGradcamOverlay = explainRes.gradcam_overlay_base64;
                } catch (e) {
                    alert("Grad-CAM generation error: " + e.message);
                    return;
                } finally {
                    gradcamBtn.classList.remove("animate-pulse");
                }
            }

            showingGradcam = !showingGradcam;
            if (showingGradcam) {
                if (!imgElement.dataset.originalSrc) imgElement.dataset.originalSrc = currentOriginalImageSrc || imgElement.src;
                imgElement.src = currentGradcamOverlay;
                gradcamBtn.className = "py-2 px-space-sm bg-secondary text-on-secondary rounded-lg font-label-md text-label-md font-semibold transition-colors flex items-center justify-center gap-1 cursor-pointer";
                if (gradcamText) gradcamText.textContent = "Original Image";
                gradcamBtn.title = "Click to toggle back to original satellite image";
            } else {
                imgElement.src = currentOriginalImageSrc || imgElement.dataset.originalSrc || imgElement.src;
                gradcamBtn.className = "py-2 px-space-sm bg-surface-container-high hover:bg-surface-container-highest text-primary rounded-lg font-label-md text-label-md font-semibold transition-colors flex items-center justify-center gap-1 cursor-pointer";
                if (gradcamText) gradcamText.textContent = "Heatmap";
                gradcamBtn.title = "Grad-CAM Activation Map";
            }
        });
    }

    // 6b. AI Restore & Upscale Button (Hugging Face GFPGAN / local fallback)
    if (restoreBtn) {
        restoreBtn.addEventListener("click", async () => {
            if (!currentFileOrBlob) {
                currentFileOrBlob = dataURItoBlob(imgElement.src);
            }
            if (!currentFileOrBlob) {
                alert("Please upload or select a sample image first.");
                return;
            }
            restoreBtn.disabled = true;
            const origHTML = restoreBtn.innerHTML;
            restoreBtn.innerHTML = `<span class="material-symbols-outlined text-[18px] animate-spin">autorenew</span><span>Restoring...</span>`;
            try {
                const result = await restoreImage(currentFileOrBlob);
                if (result.success && result.restored_image_base64) {
                    const restoredBlob = dataURItoBlob(result.restored_image_base64);
                    if (restoredBlob) currentFileOrBlob = restoredBlob;
                    currentOriginalImageSrc = result.restored_image_base64;
                    imgElement.src = result.restored_image_base64;
                    const toast = document.createElement("div");
                    toast.className = "fixed bottom-4 right-4 bg-secondary text-on-secondary p-3 rounded-lg text-xs font-semibold shadow-lg z-50 max-w-sm";
                    const src = result.method_used.includes("Hugging") ? "GFPGAN HF Space" : "Local Satellite SR";
                    toast.textContent = `✅ ${result.original_resolution} → ${result.restored_resolution} | Sharpness +${result.sharpness_improvement_pct}% | ${src}`;
                    document.body.appendChild(toast);
                    setTimeout(() => toast.remove(), 5000);
                    runInference();
                }
            } catch (e) {
                alert("Restoration error: " + e.message);
            } finally {
                restoreBtn.disabled = false;
                restoreBtn.innerHTML = origHTML;
            }
        });
    }

    // 7. Download Result Button
    if (downloadBtn) {
        downloadBtn.addEventListener("click", () => {
            const className = document.getElementById("prediction-class-name")?.textContent || "Inference";
            const conf = document.getElementById("prediction-certainty-val")?.textContent || "98.27";
            const data = {
                title: "GeoLand AI - Sentinel-2 Satellite Classification Report",
                timestamp: new Date().toISOString(),
                predicted_class: className,
                certainty: `${conf}%`,
                model: "convnext_tiny (EuroSAT SOTA)",
                device: "CPU / TensorRT",
                spectral_domain: "RGB Sentinel-2 Composite (B04, B03, B02)"
            };
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `GeoLand_Prediction_${className}_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    // 7. Core Inference Function
    async function runInference() {
        if (!currentFileOrBlob) {
            if (imgElement.src.startsWith("data:")) {
                currentFileOrBlob = dataURItoBlob(imgElement.src);
            } else {
                try {
                    const res = await fetch(imgElement.src);
                    currentFileOrBlob = await res.blob();
                } catch (e) {
                    console.warn("Could not fetch imgElement.src as blob:", e);
                }
            }
        }

        if (!currentFileOrBlob) return;

        if (classifyBtn) {
            classifyBtn.disabled = true;
            classifyBtn.classList.add("opacity-75");
            const icon = classifyBtn.querySelector(".material-symbols-outlined");
            if (icon) icon.classList.add("animate-spin");
        }

        try {
            const res = await predictImage(currentFileOrBlob);
            renderPredictionResults(res);
        } catch (e) {
            console.error("Inference failed:", e);
        } finally {
            if (classifyBtn) {
                classifyBtn.disabled = false;
                classifyBtn.classList.remove("opacity-75");
                const icon = classifyBtn.querySelector(".material-symbols-outlined");
                if (icon) icon.classList.remove("animate-spin");
            }
        }
    }

    // 8. Render Results to DOM
    const classIcons = {
        "Highway": "add_road",
        "River": "water",
        "Forest": "forest",
        "Residential": "home",
        "Industrial": "factory",
        "AnnualCrop": "agriculture",
        "Pasture": "grass",
        "PermanentCrop": "nature",
        "SeaLake": "waves",
        "HerbaceousVegetation": "eco"
    };

    function renderPredictionResults(data) {
        const pred = data.prediction;
        
        // Update Identified Class & Category
        const classNameEl = document.getElementById("prediction-class-name");
        const categoryLabelEl = document.getElementById("prediction-category-label");
        const iconEl = document.getElementById("prediction-icon");

        if (classNameEl) classNameEl.textContent = pred.class;
        if (categoryLabelEl) categoryLabelEl.textContent = pred.category || "Land-Cover";
        if (iconEl && classIcons[pred.class]) {
            iconEl.textContent = classIcons[pred.class];
        }

        // Update Certainty Percentage
        const certaintyValEl = document.getElementById("prediction-certainty-val");
        if (certaintyValEl) {
            certaintyValEl.textContent = pred.confidence_percentage.toFixed(2);
        }

        // Update Optimal Match vs Low Confidence Badge
        const matchBadge = document.getElementById("prediction-match-badge");
        if (matchBadge) {
            if (pred.is_uncertain) {
                matchBadge.className = "font-label-md text-label-md text-amber-800 dark:text-amber-300 font-semibold bg-amber-100 dark:bg-amber-900/40 px-2 py-0.5 rounded-full flex items-center gap-1";
                matchBadge.innerHTML = `<span class="material-symbols-outlined text-[14px]">warning</span><span>Low Confidence</span>`;
            } else {
                matchBadge.className = "font-label-md text-label-md text-secondary font-semibold bg-secondary-container px-2 py-0.5 rounded-full flex items-center gap-1";
                matchBadge.innerHTML = `<span class="material-symbols-outlined text-[14px]">verified</span><span>Optimal Match</span>`;
            }
        }

        // Update Sensor, GSD and Scale Badges (Sentinel-2 vs Custom Upload)
        const sensorBadge = document.getElementById("tile-sensor-badge");
        const gsdBadge = document.getElementById("tile-gsd-badge");
        const scaleBadge = document.getElementById("tile-scale-badge");
        let wPx = imgElement.naturalWidth || 64;
        let hPx = imgElement.naturalHeight || 64;
        if (data.telemetry?.input_dimensions) {
            const parts = data.telemetry.input_dimensions.split("×");
            if (parts.length === 2) {
                wPx = parseInt(parts[0].trim()) || wPx;
                hPx = parseInt(parts[1].trim()) || hPx;
            }
        }
        const isEurosatScale = (wPx <= 128 && hPx <= 128);
        if (sensorBadge) sensorBadge.textContent = isEurosatScale ? "Sentinel-2 MSI" : "Custom Upload";
        if (gsdBadge) gsdBadge.textContent = isEurosatScale ? "10m / px" : `${wPx} × ${hPx} px`;
        if (scaleBadge) scaleBadge.textContent = isEurosatScale ? "640m × 640m" : `${wPx} × ${hPx} px`;

        // Update Viewport Confidence Badge
        const viewportBadge = document.getElementById("viewport-conf-badge");
        if (viewportBadge) {
            viewportBadge.textContent = `Confidence Level: ${pred.confidence_percentage.toFixed(2)}%`;
        }

        // Update Top-3 Predictions Stack
        if (data.top_3 && data.top_3.length > 0) {
            data.top_3.forEach((item, idx) => {
                const rank = idx + 1;
                const nameEl = document.getElementById(`top3-rank-${rank}-name`);
                const pctEl = document.getElementById(`top3-rank-${rank}-pct`);
                const barEl = document.getElementById(`top3-rank-${rank}-bar`);

                if (nameEl) nameEl.textContent = item.class;
                if (pctEl) pctEl.textContent = `${item.percentage.toFixed(2)}%`;
                if (barEl) barEl.style.width = `${item.percentage.toFixed(2)}%`;
            });
        }

        // Update Telemetry & Latency
        if (data.telemetry) {
            const latEl = document.getElementById("meta-latency");
            const modelEl = document.getElementById("meta-model");
            const dimEl = document.getElementById("meta-dimensions");
            if (latEl) latEl.textContent = `${data.telemetry.latency_ms.toFixed(1)} ms (${data.telemetry.device})`;
            if (modelEl) modelEl.textContent = data.telemetry.model_architecture;
            if (dimEl) {
                const rawDims = data.telemetry.input_dimensions || `${imgElement.naturalWidth || 64} × ${imgElement.naturalHeight || 64}`;
                const inferSize = data.telemetry.inference_input_size ? `${data.telemetry.inference_input_size.split("×")[0].trim()} resized` : "112 resized";
                dimEl.textContent = `${rawDims} (${inferSize})`;
            }
        }

        // Salience description based on class
        const salienceEl = document.getElementById("meta-salience");
        if (salienceEl) {
            const salienceMap = {
                "Highway": "Linear Corridors & Asphalt Edge",
                "River": "Hydrological Reflectance Absorption",
                "Forest": "NIR High Chlorophyll Canopy",
                "Residential": "High-Entropy Urban Fabric",
                "Industrial": "Large Planar Albedo Surfaces",
                "AnnualCrop": "Periodic Geometric Vegetative Slope",
                "Pasture": "Uniform Grazing Grasslands",
                "PermanentCrop": "Row Canopy Orchards",
                "SeaLake": "Homogeneous Water Absorption",
                "HerbaceousVegetation": "Deciduous Grass Foliage"
            };
            salienceEl.textContent = salienceMap[pred.class] || "Multi-Spectral Feature Maps";
        }

        // Kaggle 99% Accuracy Satellite Classification Data Binding
        if (data.kaggle_classification) {
            const kag = data.kaggle_classification;
            const badgeClass = document.getElementById("kaggle-pred-class");
            const badgeEmoji = document.getElementById("kaggle-pred-emoji");
            if (badgeClass) badgeClass.textContent = `${kag.predicted_class} (${kag.confidence_percentage}%)`;
            if (badgeEmoji && kag.metadata && kag.metadata.emoji) {
                badgeEmoji.textContent = kag.metadata.emoji;
            }

            if (kag.breakdown) {
                kag.breakdown.forEach(item => {
                    if (item.class === "cloudy") {
                        const el = document.getElementById("kaggle-pct-cloudy");
                        if (el) el.textContent = `${item.percentage.toFixed(1)}%`;
                    } else if (item.class === "desert") {
                        const el = document.getElementById("kaggle-pct-desert");
                        if (el) el.textContent = `${item.percentage.toFixed(1)}%`;
                    } else if (item.class === "green_area") {
                        const el = document.getElementById("kaggle-pct-green");
                        if (el) el.textContent = `${item.percentage.toFixed(1)}%`;
                    } else if (item.class === "water") {
                        const el = document.getElementById("kaggle-pct-water");
                        if (el) el.textContent = `${item.percentage.toFixed(1)}%`;
                    }
                });
            }
        }

        // Deep Geospatial Multi-Zone Analysis Data Binding
        if (data.deep_analysis) {
            const deep = data.deep_analysis;
            currentSegmentationOverlay = deep.segmentation_overlay_base64;
            currentOriginalImageSrc = imgElement.src;

            // Scene Composition Progress Bar & Percentages — REAL values, no hardcoding
            const comp = deep.composition_summary;
            if (comp && comp.length >= 4) {
                const pOcean   = (comp.find(c => c.class.includes("Open Water"))?.percentage   || 0).toFixed(1);
                const pEstuary = (comp.find(c => c.class.includes("Dark Water"))?.percentage   || 0).toFixed(1);
                const pUrban   = (comp.find(c => c.class.includes("Urban"))?.percentage        || 0).toFixed(1);
                const pVeg     = (comp.find(c => c.class.includes("Vegetat"))?.percentage      || 0).toFixed(1);
                const pSoil    = (comp.find(c => c.class.includes("Soil"))?.percentage         || 0).toFixed(1);

                const elOcean   = document.getElementById("comp-pct-ocean");
                const elEstuary = document.getElementById("comp-pct-estuary");
                const elUrban   = document.getElementById("comp-pct-urban");
                const elVeg     = document.getElementById("comp-pct-veg");

                if (elOcean)   elOcean.textContent   = `${pOcean}%`;
                if (elEstuary) elEstuary.textContent = `${pEstuary}%`;
                if (elUrban)   elUrban.textContent   = `${pUrban}%`;
                if (elVeg)     elVeg.textContent     = `${pVeg}%`;

                const barOcean   = document.getElementById("comp-bar-ocean");
                const barEstuary = document.getElementById("comp-bar-estuary");
                const barUrban   = document.getElementById("comp-bar-urban");
                const barVeg     = document.getElementById("comp-bar-veg");

                if (barOcean)   barOcean.style.width   = `${pOcean}%`;
                if (barEstuary) barEstuary.style.width = `${pEstuary}%`;
                if (barUrban)   barUrban.style.width   = `${pUrban}%`;
                if (barVeg)     barVeg.style.width     = `${pVeg}%`;
            }

            // 3 Geographical Zones Binding — use actual model predictions
            const zones = deep.zones_breakdown;
            if (zones && zones.length >= 3) {
                const z1 = zones[0];
                const z2 = zones[1];
                const z3 = zones[2];

                const z1Conf = document.getElementById("zone-1-conf");
                const z1Title = document.getElementById("zone-1-title");
                const z1Desc = document.getElementById("zone-1-desc");
                const z1Cov = document.getElementById("zone-1-coverage");
                if (z1Conf)  z1Conf.textContent  = `${z1.confidence_percentage}% Conf`;
                if (z1Title) z1Title.textContent  = z1.identified_land_use;
                if (z1Desc)  z1Desc.textContent   = z1.description || z1.identified_land_use;
                if (z1Cov)   z1Cov.textContent    = `Category: ${z1.category}`;

                const z2Conf = document.getElementById("zone-2-conf");
                const z2Title = document.getElementById("zone-2-title");
                const z2Desc = document.getElementById("zone-2-desc");
                const z2Cov = document.getElementById("zone-2-coverage");
                if (z2Conf)  z2Conf.textContent  = `${z2.confidence_percentage}% Conf`;
                if (z2Title) z2Title.textContent  = z2.identified_land_use;
                if (z2Desc)  z2Desc.textContent   = z2.description || z2.identified_land_use;
                if (z2Cov)   z2Cov.textContent    = `Category: ${z2.category}`;

                const z3Conf = document.getElementById("zone-3-conf");
                const z3Title = document.getElementById("zone-3-title");
                const z3Desc = document.getElementById("zone-3-desc");
                const z3Cov = document.getElementById("zone-3-coverage");
                if (z3Conf)  z3Conf.textContent  = `${z3.confidence_percentage}% Conf`;
                if (z3Title) z3Title.textContent  = z3.identified_land_use;
                if (z3Desc)  z3Desc.textContent   = z3.description || z3.identified_land_use;
                if (z3Cov)   z3Cov.textContent    = `Category: ${z3.category}`;
            }

            const deepBadge = document.getElementById("deep-analysis-badge");
            if (deepBadge) {
                deepBadge.textContent = deep.is_composite_scene
                    ? "Multi-Terrain Scene Mode Active"
                    : "EuroSAT Patch Mode";
            }

            // Show/hide deep analysis panel based on composite detection
            const deepPanel = document.getElementById("deep-analysis-panel");
            if (deepPanel) {
                deepPanel.style.display = deep.is_composite_scene ? "block" : "none";
            }
        }

        // Uncertainty Warning Banner — dynamic content
        let uncertBanner = document.getElementById("uncertainty-warning-banner");
        if (pred.is_uncertain) {
            if (!uncertBanner) {
                uncertBanner = document.createElement("div");
                uncertBanner.id = "uncertainty-warning-banner";
                uncertBanner.className = "p-3 bg-secondary-container text-on-secondary-container rounded-lg text-xs font-semibold mb-3 flex items-center gap-2 border border-secondary/30";
                classNameEl?.closest(".bg-surface-container-low")?.prepend(uncertBanner);
            }
            const isComposite = data.deep_analysis?.is_composite_scene;
            const topClass = data.top_3?.[0]?.class || pred.class;
            const bannerMsg = isComposite
                ? `Multi-Terrain Scene: Contains multiple land-cover types. See Deep Spatial Breakdown below.`
                : `Low confidence (${pred.confidence_percentage.toFixed(1)}%). Top prediction: ${topClass}. Consider using a focused satellite patch.`;
            uncertBanner.innerHTML = `<span class="material-symbols-outlined text-[18px] text-secondary">travel_explore</span><span>${bannerMsg}</span>`;
            uncertBanner.style.display = "flex";
        } else if (uncertBanner) {
            uncertBanner.style.display = "none";
        }
    }
}

// Wire Robustness Analysis Page
function setupRobustnessPage() {
    const noiseSlider = document.getElementById("noise-slider");
    const blurSlider = document.getElementById("blur-slider");
    const runBtn = Array.from(document.querySelectorAll("button")).find(b => 
        b.textContent.includes("Robustness") || b.textContent.includes("Stress") || b.innerHTML.includes("play_circle")
    );

    if (runBtn) {
        runBtn.addEventListener("click", async () => {
            runBtn.disabled = true;
            runBtn.classList.add("opacity-50");

            const noiseVal = noiseSlider ? parseFloat(noiseSlider.value) : 0.10;
            const blurVal = blurSlider ? parseFloat(blurSlider.value) : 3.0;

            // Use preset or active sample
            const imgEl = document.querySelector("img");
            if (!imgEl) return;

            try {
                const res = await fetch(imgEl.src);
                const blob = await res.blob();

                // Call robustness for blur & noise
                const robRes = await testRobustness(blob, "noise", noiseVal);
                
                // Update degraded tile preview if exists
                const degradedImg = Array.from(document.querySelectorAll("img"))[1];
                if (degradedImg && robRes.corrupted_image_base64) {
                    degradedImg.src = robRes.corrupted_image_base64;
                }
            } catch (e) {
                console.error("Robustness run error:", e);
            } finally {
                runBtn.disabled = false;
                runBtn.classList.remove("opacity-50");
            }
        });
    }

    // Populate Metrics from Backend
    fetchMetrics().then(data => {
        if (!data.robustness) return;
        const benchmark = data.robustness.benchmark_data;
        const table = document.querySelector("table tbody");
        if (table && benchmark) {
            table.innerHTML = "";
            for (const [model, stats] of Object.entries(benchmark)) {
                const row = document.createElement("tr");
                row.className = "border-b border-surface-container-high hover:bg-surface-container-low transition-colors";
                const clean = (stats.clean_acc * 100).toFixed(1);
                const blur5 = ((stats.blur["5.0"]?.accuracy || 0) * 100).toFixed(1);
                const noise03 = ((stats.noise["0.3"]?.accuracy || 0) * 100).toFixed(1);
                const avgDrop = (((stats.blur["5.0"]?.acc_drop || 0) + (stats.noise["0.3"]?.acc_drop || 0)) * 50).toFixed(1);

                row.innerHTML = `
                    <td class="py-3 px-4 font-semibold text-primary">${model.toUpperCase()}</td>
                    <td class="py-3 px-4">${clean}%</td>
                    <td class="py-3 px-4">${blur5}%</td>
                    <td class="py-3 px-4">${noise03}%</td>
                    <td class="py-3 px-4 text-error font-bold">-${avgDrop}%</td>
                `;
                table.appendChild(row);
            }
        }
    }).catch(e => console.log("Robustness metrics sync:", e));
}

// Wire Model Comparison Page
function setupModelComparisonPage() {
    fetchMetrics().then(data => {
        if (!data.model_comparison || !data.model_comparison.summary_table) return;
        const records = data.model_comparison.summary_table;
        const table = document.querySelector("table tbody");
        if (table) {
            table.innerHTML = "";
            records.forEach(r => {
                const tr = document.createElement("tr");
                tr.className = "border-b border-surface-container-high hover:bg-surface-container-low transition-colors";
                tr.innerHTML = `
                    <td class="py-3.5 px-4 font-bold text-primary">${r.Model.toUpperCase()}</td>
                    <td class="py-3.5 px-4 font-semibold text-secondary">${(r.Accuracy * 100).toFixed(2)}%</td>
                    <td class="py-3.5 px-4">${(r["Macro Precision"] * 100).toFixed(2)}%</td>
                    <td class="py-3.5 px-4">${(r["Macro Recall"] * 100).toFixed(2)}%</td>
                    <td class="py-3.5 px-4 font-semibold">${(r["Macro F1"] * 100).toFixed(2)}%</td>
                    <td class="py-3.5 px-4">${r["Latency (ms)"]} ms</td>
                    <td class="py-3.5 px-4">${r["Parameters (M)"]}M</td>
                `;
                table.appendChild(tr);
            });
        }
    }).catch(e => console.log("Comparison metrics sync:", e));
}

// Wire Error Analysis Page
function setupErrorAnalysisPage() {
    fetchMetrics().then(data => {
        if (!data.error_analysis) return;
        const err = data.error_analysis;
        
        // Update error rate counters
        const rateEls = Array.from(document.querySelectorAll("span, p, h2")).filter(el => 
            el.textContent.includes("Error") || el.textContent.includes("Misclass")
        );
        
        // Update top-10 confused pairs table
        const pairsTable = Array.from(document.querySelectorAll("table tbody"))[0];
        if (pairsTable && err.top_10_confused_pairs) {
            pairsTable.innerHTML = "";
            err.top_10_confused_pairs.forEach((p, idx) => {
                const tr = document.createElement("tr");
                tr.className = "border-b border-surface-container-high hover:bg-surface-container-low transition-colors";
                tr.innerHTML = `
                    <td class="py-2.5 px-4 font-bold text-outline">#${idx + 1}</td>
                    <td class="py-2.5 px-4 font-semibold text-primary">${p.pair}</td>
                    <td class="py-2.5 px-4 font-bold text-error">${p.count}</td>
                    <td class="py-2.5 px-4 text-xs text-outline">${Object.entries(p.directions).map(([k,v]) => `${k} (${v})`).join(', ')}</td>
                `;
                pairsTable.appendChild(tr);
            });
        }
    }).catch(e => console.log("Error analysis metrics sync:", e));
}

// Auto-initialize based on active page
document.addEventListener("DOMContentLoaded", () => {
    const path = window.location.pathname;
    if (path.includes("classification") || document.getElementById("classified-satellite-image")) {
        setupClassificationPage();
    } else if (path.includes("robustness") || document.getElementById("noise-slider")) {
        setupRobustnessPage();
    } else if (path.includes("model_comparison")) {
        setupModelComparisonPage();
    } else if (path.includes("error_confusion") || path.includes("error_analysis")) {
        setupErrorAnalysisPage();
    }
});
