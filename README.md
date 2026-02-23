# 🌍 Automated Surface Parameter Extraction & Hybrid Point Simulation Pipeline
#### (*Landslide Susceptibility Modeling using Python and Machine Learning*)

---
## Overview

This repository contains the source code for a Python-based geospatial pipeline developed for automated surface parameter extraction and landslide susceptibility modeling under data-scarce conditions.

The system integrates:

- Digital Elevation Model (DEM) processing  
- Raster-based surface parameter derivation  
- Hybrid spatial point simulation  
- Machine learning model training and validation  
- Performance evaluation and visualization  

This pipeline was developed as part of a Senior High School STEM Capstone research project focused on landslide susceptibility mapping in Cagayan de Oro City, Philippines.

---

## Objectives

The pipeline was designed to:

- Automate extraction of topographic and hydrological parameters from DEM data  
- Address limited landslide inventory data through spatial simulation techniques  
- Compare multiple sampling strategies  
- Evaluate different machine learning classifiers  
- Provide a reproducible, open-source alternative to proprietary GIS workflows  

---

## Core Features

### 1.) Automated Raster-Based Surface Parameter Extraction

Derived from SRTM DEM:
- Elevation  
- Slope (Zevenbergen & Thorne method)  
- Aspect  
- Profile Curvature  
- Planform Curvature  
- Relative Relief  
- Flow Direction (D8 algorithm)  
- Flow Accumulation  
- Topographic Wetness Index (TWI)  
- Distance to River  

All computations are raster-based and optimized for large-scale grid processing.

---

### 2.) Hybrid Landslide Point Simulation

To address limited landslide inventories, the pipeline implements five sampling strategies:

- Buffer-Controlled Sampling (BCS)  
- Dynamic Buffer-Controlled Sampling (DBCS)  
- Kernel Density Sampling (KDS)  
- Hybrid Density Sampling (HDS)  
- Dynamic Hybrid Density Sampling (DHDS)  

These methods simulate synthetic landslide and non-landslide points based on spatial and environmental similarity principles.

---

### 3.) Machine Learning Integration

Models implemented via WEKA:

- Logistic Regression (LR)  
- Random Forest (RF)  
- Random Subspace (RS)  

Performance Metrics:

- Accuracy  
- F1-Score  
- Area Under the Curve (AUC)  
- Root Mean Square Error (RMSE)  

---

### 4.) Validation Strategy

- 10-Fold Cross Validation  
- Leave-One-Out Cross Validation (for small datasets)  
- Independent hold-out validation  

---

### 5.) Computational Efficiency

- Raster-level processing optimized using NumPy  
- Batch computation support  
- Caching system for parameter reuse  
- GUI-based user interface  
- Exportable `.exe` version for Windows  

---

## 🛠 Requirements

- Python 3.9+
	Required Libraries:
	- numpy
	- pandas
	- matplotlib
	- rasterio
	- pysheds
	- scikit-learn
	- weka-wrapper3

---

## 🚀 How to Run

### 1. Prepare Input Data

- Place DEM raster inside `/data/dem/`
- Add landslide inventory CSV file

### 2. Run the Pipeline

Command-line version: *"python main.py"*
GUI version: *"python ui_main.py"*

---

## ⚠ Limitations

- Limited landslide inventory size (during initial testing)
- Default 90m DEM resolution may not capture fine-scale topographic variations  
- Synthetic point generation introduces potential structural dependence  
- Validation limited to available inventory data  

This pipeline is currently released as an **Alpha version** and intended for research and educational purposes.

---

## 📚 Research Context

Developed for:

**“Automated Surface Parameter Extraction with Spatial and Environmental Hybrid Points Simulation Pipeline for Landslide Mapping and Ensemble Machine Learning Modelling”**

Capstone Research – STEM Strand  
Gusa Regional Science High School – X  
Cagayan de Oro City, Philippines  
January 2026  

---

## 📄 License

This project is released under the  
GNU General Public License (GPL) Version 2.

---