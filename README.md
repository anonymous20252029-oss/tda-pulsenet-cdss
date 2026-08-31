# 🫀 TDA-PulseNet: Topological Clinical Decision Support System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official web-based Clinical Decision Support System (CDSS) and pre-trained models for the manuscript: **"Topological Waveform Profiling of Ambulatory Cardiovascular Signals using Dual-Sensor Adaptive Filtering and Explainable Ensemble Learning: A Trust-Centric Clinical Decision Support System"** (Target Journal: *BMC Medical Informatics and Decision Making*).

## 📖 Project Overview

Traditional Chinese Medicine (TCM) radial pulse diagnosis is highly vulnerable to motion artifacts during ambulatory data acquisition. Standard linear time-frequency signal processing fails to capture the non-linear, multi-harmonic state-space manifold of anisotropic arterial pressure waves.

**TDA-PulseNet** addresses this by shifting from localized geometric landmarks to global, coordinate-free multi-scale topological invariants[cite: 5]. The framework integrates:
* **Motion-Adaptive Denoising:** Dual-channel Recursive Least Squares (RLS) adaptive noise cancellation using synchronized tri-axial accelerometry[cite: 6].
* **Topological Data Analysis (TDA):** Phase-space reconstruction via Takens' delay coordinate embedding and multi-scale Vietoris-Rips persistent homology to extract $b_0$ and $b_1$ Betti curves.
* **Explainable AI (XAI):** A regularized LightGBM ensemble interpreted via game-theoretic Shapley Additive Explanations (SHAP).

The models were trained and rigorously validated using Leave-One-Subject-Out Cross-Validation (GroupKFold) on the open-access **PhysioNet Pulse Transit Time PPG dataset**.

## 📁 Repository Structure

To ensure the web application runs successfully, the repository is structured as follows:

```text
tda-pulsenet-cdss/
├── app.py                      # Main Streamlit web application script
├── requirements.txt            # Python environment dependencies
├── models/
│   └── tda_pulsenet_lightgbm.pkl # Pre-trained LightGBM model weights[cite: 4]
├── web_demo_assets/
│   └── summary_metrics.json    # JSON file containing empirical CV evaluation metrics[cite: 3]
└── README.md                   # Project documentation
