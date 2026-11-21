# EuroSAT Explainability & Feature Attribution Report

## 1. Executive Summary
Grad-CAM and Grad-CAM++ visualizations highlight the specific spatial and spectral features that drive land-use classification across the 10 EuroSAT categories.

## 2. Visual Feature Drivers by Land-Use Class

| Land-Use Class | Key Driving Visual Features | Vulnerability / Confusion Drivers |
| :--- | :--- | :--- |
| **Highway** | High-contrast linear asphalt/concrete corridors with parallel lane markings. | Curved highway segments bordered by vegetation can mimic rivers; blur destroys sharp edges. |
| **River** | Sinuous organic curvilinear geometry, low specular reflectance, bordering riparian zones. | Dry riverbeds or paved embankments can be misclassified as highways. |
| **Residential** | Regular geometric grid of small rooftops, driveways, high spatial frequency density. | Low-density suburbs with mature trees border with `Forest` and `Pasture`. |
| **Industrial** | Large rectilinear warehouse footprints, metallic/bright roofs, wide loading bays. | Commercial complexes adjacent to residential zones cause border confusion. |
| **Forest** | Irregular granular organic texture, deep green low-reflectance canopy clusters. | Deep shade in orchards or dense herbaceous patches mimic forest edges. |
| **AnnualCrop** | Rectangular agricultural plots with uniform seasonal furrows and tilled soil tones. | Bare soil between crop cycles can be confused with pasture. |
| **PermanentCrop** | Regular dot/stripe matrix grid patterns (orchards, vineyards, olive groves). | Under blur, repetitive row patterns soften into homogeneous herbaceous fields. |
| **Pasture & Herbaceous**| Homogeneous green/golden grasslands devoid of structures. | High spectral overlap; primarily separated by subtle textural variations. |
| **SeaLake** | Uniform, low-texture, monochromatic deep blue/cyan expanse. | Extremely robust; only compromised by severe cloud glint or high noise. |

## 3. Explainability on Perturbed Inputs
* **Under Gaussian Blur:** The model loses the sharp high-frequency edge activations along highways and industrial building boundaries, diluting attention across surrounding background terrain.
* **Under Gaussian Noise:** Random speckled noise introduces spurious local gradients, causing diffuse activation maps that diminish confidence in fine-grained agricultural classes.
