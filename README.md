# GeoLand AI — Intelligent EuroSAT Land-Use Classification & Diagnostic Platform

GeoLand AI is an end-to-end, mission-critical satellite image classification system built on the European Space Agency **EuroSAT (Sentinel-2 RGB)** benchmark. The platform categorizes optical satellite patches into 10 land-use classes while placing primary engineering focus on **Robustness** under environmental degradations and **Explainability** via Grad-CAM / Grad-CAM++ spatial attention maps.

---

## 🛰️ Problem Definition & Cost of Error

* **Land-Use Categories (10 Classes):**
  `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`.
* **Cost-of-Error Matrix:**
  * **Urban/Industrial $\leftrightarrow$ Natural:** Flagged as **High-Impact Errors**. Misclassifying urban sprawl as natural reserves or wetlands leads to severe environmental planning failures; misclassifying natural habitats as industrial zones risks illegal zoning permits and ecological damage.
  * **Fine-grained Confusion:** Highway vs. River (curvilinear linear structures) and Pasture vs. Herbaceous Vegetation (chromatic green overlap).

---

## 🏗️ Architecture & Project Structure

```
ML_SAT/
├── backend/
│   ├── app.py                # FastAPI REST & inference API (CORS enabled)
│   └── device.py             # Hardware diagnostic & device selector (GPU/CPU)
├── src/
│   ├── dataset.py            # EuroSAT loader, 70/15/15 stratified split, data pipeline
│   ├── models.py             # Model factory (ResNet50, EfficientNet-B0, ConvNeXt-Tiny)
│   ├── train.py              # Fine-tuning engine with cosine LR, AdamW, label smoothing
│   ├── evaluate.py           # Test set evaluation, metrics, and confusion matrix generation
│   ├── robustness.py         # Environmental corruption testing (Gaussian blur & noise)
│   ├── error_analysis.py     # Confused pair ranking, cost-of-error flagging, error gallery
│   ├── explainability.py     # Grad-CAM / Grad-CAM++ attribution and feature analysis
│   ├── main_pipeline.py      # Master orchestrator for end-to-end execution
│   └── utils.py              # Helper functions, cost-of-error rules, reproducibility seeds
├── frontend/
│   ├── index.html            # Landing page & platform showcase
│   ├── satellite_image_classification.html  # Interactive inference & Grad-CAM studio
│   ├── robustness_analysis.html             # Blur & noise degradation stress-lab
│   ├── model_comparison.html                # Pretrained architecture benchmark studio
│   ├── error_confusion_analysis.html        # 10x10 confusion matrix & error gallery
│   ├── experiments_reports.html             # Training dynamics & learning curves
│   ├── model_management_settings.html       # Checkpoints and runtime telemetry
│   ├── multispectral_analysis.html          # Multispectral band inspection
│   ├── image_summary.html                   # EuroSAT dataset summary
│   └── api.js                # Frontend API client and component connector
├── models/
│   ├── best_model.pt         # Top performing model checkpoint
│   ├── class_names.json      # 10 EuroSAT class labels
│   └── best_model_metadata.json
├── outputs/
│   ├── samples/              # Sample image grids
│   ├── curves/               # Training loss & accuracy curves (PNG and JSON)
│   ├── robustness/           # Corruption accuracy curves, class fragility rankings
│   ├── errors/               # Error galleries and top-10 confused pairs
│   ├── explainability/       # Grad-CAM overlays and feature interpretation reports
│   └── model_comparison.csv # Test benchmark comparison across all 3 models
├── dataset/EuroSAT_RGB/      # EuroSAT dataset (26,990 patches)
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup

Ensure Python 3.10+ is installed:
```powershell
# Create and activate virtual environment
py -3 -m venv .venv
.\.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Single Command to Run Backend

Start the FastAPI inference and diagnostics engine:
```powershell
py -3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at: **http://localhost:8000/docs**

### 3. Single Command to Run Frontend

Serve the UI:
```powershell
py -3 -m http.server 3000 --directory frontend
```
Open **http://localhost:3000** in your browser.
*(Alternatively, navigate to **http://localhost:8000/static/index.html** directly from the backend server)*.

---

## 🚀 Model Training & Full Pipeline Run

To reproduce the complete pipeline from scratch (dataset preparation, fine-tuning ResNet50, EfficientNet-B0, and ConvNeXt-Tiny, standard test evaluation, corruption stress testing, error diagnostics, and Grad-CAM generation):

```powershell
py -3 -m src.main_pipeline --epochs 5 --batch-size 64
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict` | Classifies uploaded patch, returns class, confidence, top-3 probabilities, and uncertainty flag |
| `POST` | `/explain` | Returns Grad-CAM / Grad-CAM++ activation heatmap overlay as base64 |
| `POST` | `/robustness` | Applies blur or noise degradation with adjustable severity and runs inference |
| `GET` | `/metrics` | Returns model comparison table, confusion matrices, robustness data, and error analysis |
| `GET` | `/preset-samples` | Provides verified EuroSAT samples (Highway, River, Forest, Residential) for UI testing |
| `GET` | `/health` | Health check endpoint returning loaded model status and hardware device |

---

## 📊 Measured Benchmark Results

### 1. Model Comparison on EuroSAT Test Set

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Latency (ms) | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** (Baseline) | 85.47% | 86.63% | 85.47% | 85.52% | 13.57 ms | 23.53M |
| **EfficientNet-B0** (Lightweight) | 85.20% | 85.34% | 85.20% | 84.99% | **6.07 ms** | **4.02M** |
| **ConvNeXt-Tiny** (Selected Winner) | **93.33%** | **93.70%** | **93.33%** | **93.36%** | 11.41 ms | 27.83M |

### 2. Environmental Robustness & Degradation Resistance

* **Blur Resilience:** ConvNeXt-Tiny maintains **87.87% accuracy at $\sigma=1.0$** and **68.80% at $\sigma=2.0$**, demonstrating $3\times$ higher degradation tolerance than standard ResNet-50 (which collapses to 31.73% and 17.33%).
* **Noise Resilience:** ConvNeXt-Tiny retains **63.47% accuracy under Gaussian noise ($\text{std}=0.05$)**, proving modern inverted bottleneck architectures withstand sensor thermal noise significantly better.
* **Cost-of-Error Matrix:** Out of 50 total errors in the best model, **17 errors (34%)** crossed the critical Urban/Industrial $\leftrightarrow$ Natural boundary (notably `River` confused with `Highway` due to curvilinear geometry).
* **Explainability:** Grad-CAM overlays in `outputs/explainability/` demonstrate that ConvNeXt-Tiny learns true geometric structures (linear roads, building footprints, river channels) rather than spurious background correlations.

Detailed analytics and visual galleries are documented in [`outputs/REPORT.md`](file:///c:/Users/nikhi/OneDrive/Desktop/ML_SAT/outputs/REPORT.md).
