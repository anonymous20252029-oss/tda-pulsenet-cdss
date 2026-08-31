import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
from ripser import ripser

# --- Page Configuration ---
st.set_page_config(page_title="TDA-PulseNet CDSS", page_icon="🫀", layout="wide")
st.title("🫀 TDA-PulseNet: Clinical Decision Support System")

# --- Load Colab Outputs (Model & JSON) ---
@st.cache_resource
def load_assets():
    # Load the LightGBM model trained in Colab
    model_path = os.path.join("models", "tda_pulsenet_lightgbm.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Load the metrics JSON generated in Colab
    json_path = os.path.join("web_demo_assets", "summary_metrics.json")
    with open(json_path, "r") as f:
        metrics = json.load(f)
        
    return model, metrics

try:
    model, metrics = load_assets()
    
    # --- Display Colab Cross-Validation Metrics ---
    st.sidebar.header("📊 Colab Training Metrics")
    st.sidebar.info(f"**ROC-AUC:** {metrics['mean_auc']:.4f}\n\n**F1-Score:** {metrics['mean_f1']:.4f}\n\n**MCC:** {metrics['mean_mcc']:.4f}")
except FileNotFoundError:
    st.error("Error: Could not find model or metrics files. Please ensure the 'models' and 'web_demo_assets' folders exist in your GitHub repository and contain the correct files.")
    st.stop()

# --- Helper Functions ---
def takens_embedding(time_series, delay=15, dimension=3):
    n_samples = len(time_series)
    max_index = n_samples - (dimension - 1) * delay
    embedded_matrix = np.zeros((max_index, dimension))
    for d in range(dimension):
        embedded_matrix[:, d] = time_series[d * delay : d * delay + max_index]
    return embedded_matrix

def compute_betti_features(embedded_cloud, n_bins=100):
    filtration_scales = np.linspace(0, 1.5, n_bins)
    res = ripser(embedded_cloud, maxdim=1)
    diagrams = res['dgms']
    
    betti_0, betti_1 = np.zeros(n_bins), np.zeros(n_bins)
    for birth, death in diagrams[0]:
        betti_0 += (filtration_scales >= birth) & (filtration_scales < death)
    for birth, death in diagrams[1]:
        betti_1 += (filtration_scales >= birth) & (filtration_scales < death)
        
    lifetimes_h1 = diagrams[1][:, 1] - diagrams[1][:, 0]
    lifetimes_h1 = lifetimes_h1[np.isfinite(lifetimes_h1)]
    entropy = -np.sum((lifetimes_h1/np.sum(lifetimes_h1)) * np.log((lifetimes_h1/np.sum(lifetimes_h1)) + 1e-10)) if len(lifetimes_h1) > 0 else 0.0
    
    return betti_0, betti_1, entropy

# --- Interface Input ---
st.sidebar.header("⚙️ Patient Signal Input")
sample_choice = st.sidebar.selectbox("Select Benchmark Clinical State:", ["Healthy Resting State", "Pathological Dynamic Tension"])

t = np.linspace(0, 10, 1000)
if sample_choice == "Healthy Resting State":
    noisy_wave = np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.08, 1000)
else:
    # Simulating the high-tension 'wiry' proxy with added accelerometer motion noise
    noisy_wave = np.sin(2 * np.pi * 1.2 * t) + 0.7 * np.sin(4 * np.pi * 1.2 * t + 0.3) + np.random.normal(0, 0.15, 1000)

# --- Process Pipeline & ACTUAL Inference ---
embedded_cloud = takens_embedding(noisy_wave, delay=15, dimension=3)
betti_0, betti_1, entropy = compute_betti_features(embedded_cloud)

# Assemble exact 201-dimensional feature vector expected by the Colab model
feature_vector = np.hstack([betti_0, betti_1, [entropy]]).reshape(1, -1)

# REAL Model Prediction
prob_tension = model.predict_proba(feature_vector)[0, 1]

# --- Dashboard Output ---
col1, col2 = st.columns(2)
col1.metric("CDSS Pathological Tension Probability", f"{prob_tension * 100:.1f}%")
col2.metric("Topological Entropy", f"{entropy:.4f}")

st.divider()

if prob_tension > 0.50:
    st.error(f"⚠️ **Pathological Dynamic Tension Detected**")
    st.write("**Clinical Recommendation:** High persistence in $b_1$ topological loops indicates elevated arterial wall tension (proxy for 'Wiry' pulse quality).")
else:
    st.success(f"✅ **Normal/Healthy Baseline**")
    st.write("**Clinical Recommendation:** Stable low-dimensional phase attractor. Normal physiological compliance.")

# --- Visualization ---
st.subheader("Signal & Phase Space Attractor")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(t[:500], noisy_wave[:500], color='#2c3e50')
ax1.set_title("Input 1D Pulse Waveform")
ax1.set_xlabel("Time")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(embedded_cloud[:, 0], embedded_cloud[:, 1], embedded_cloud[:, 2], color='#8e44ad', lw=0.8)
ax2.set_title("Reconstructed 3D Attractor")

st.pyplot(fig)

st.sidebar.info("💡 Powered by LightGBM model trained in Google Colab for BMC Medical Informatics and Decision Making submission.")
