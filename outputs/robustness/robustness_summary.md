# EuroSAT Robustness Benchmark Report

## 1. Overview
Satellite imagery is subject to atmospheric hazing, optical defocus, sensor thermal noise, and compression artifacts. Robustness under environmental corruptions is vital for mission-critical Earth observation.

## 2. Model Performance Summary
* **Most Robust Model:** **CONVNEXT_TINY** (Average degradation drop: **60.86%** under severe noise/blur).
* **Architecture Robustness Comparison:**
  - **RESNET50**: Clean Acc = 85.47%, Severe Degradation Drop = -71.20%
  - **EFFICIENTNET_B0**: Clean Acc = 85.20%, Severe Degradation Drop = -69.00%
  - **CONVNEXT_TINY**: Clean Acc = 93.33%, Severe Degradation Drop = -60.86%

## 3. Land-Use Fragility Ranking
* **Most Fragile Class:** **Industrial** (-86.7% accuracy drop). Fine structural patterns (such as narrow roads and linear crop furrows) are easily obliterated by Gaussian blur and noise.
* **Most Resilient Class:** **SeaLake** (-32.9% accuracy drop). Homogeneous spectral signatures (e.g. open waterbodies and deep forest canopies) preserve low-frequency chromatic signals even under severe noise.

## 4. Key Architectural Insights
1. **Depthwise Convolutions vs Standard ResNet Convolutions:** ConvNeXt-Tiny's inverted bottleneck and 7x7 receptive fields demonstrate superior resilience to high-frequency Gaussian noise compared to standard 3x3 convolutions.
2. **Impact of Blur:** High-frequency spatial structures like `Highway` and `Residential` suffer sharp accuracy drops under blur ($k=5$), shifting misclassifications towards natural textures like `River` and `Industrial`.
