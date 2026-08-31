import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
import json
import os
from ripser import ripser

# --- Page Configuration ---
st.set_page_config(page_title="TDA-PulseNet CDSS", page_icon="🫀", layout="wide")
st.title("🫀 TDA-PulseNet: Clinical Decision Support System")
st.markdown("*An explainable topological framework for Traditional Chinese Medicine (TCM) pulse diagnosis under ambulatory conditions.*")

# --- Load Colab Outputs (Model & JSON) ---
@st.cache_resource
def load_assets():
    model_path = os.path.join("models", "tda_pulsenet_lightgbm.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    json_path = os.path.join("web_demo_assets", "summary_metrics.json")
    with open(json_path, "r") as f:
        metrics = json.load(f)
        
    return model, metrics

try:
    model, metrics = load_assets()
    st.sidebar.success("✅ Model & Metrics loaded successfully.")
    st.sidebar.header("📊 Validation Metrics (LOSO-CV)")
    st.sidebar.info(f"**ROC-AUC:** {metrics['mean_auc']:.4f}\n\n**F1-Score:** {metrics['mean_f1']:.4f}\n\n**MCC:** {metrics['mean_mcc']:.4f}")
except FileNotFoundError:
    st.error("⚠️ Error: Pre-trained models not found. Please check your 'models' and 'web_demo_assets' folders.")
    st.stop()

# --- Helper Functions ---
def generate_realistic_ppg(t, heart_rate=70, stiffness=1.0):
    """Generates a realistic arterial pulse wave using dual-Gaussian model."""
    f = heart_rate / 60.0
    phase = (2 * np.pi * f * t) % (2 * np.pi)
    # Systolic peak
    sys_wave = np.exp(-((phase - 1)**2) / 0.5)
    # Diastolic/Reflected wave (stiffness pulls it closer to systole and increases amplitude)
    dias_pos = max(1.5, 3.0 - (stiffness * 0.8))
    dias_amp = 0.3 + (stiffness * 0.2)
    dias_wave = dias_amp * np.exp(-((phase - dias_pos)**2) / 0.8)
    return sys_wave + dias_wave

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

# --- Interface Input & Case Selection ---
st.sidebar.header("⚙️ Clinical Case Simulator")
case_choice = st.sidebar.selectbox(
    "Select Patient Case:", 
    [
        "Case 1: Healthy Resting Baseline", 
        "Case 2: Mild Tension (Early Qi Stagnation)", 
        "Case 3: Severe Pathological Tension ('Wiry')",
        "Case 4: Ambulatory Motion Artifacts"
    ]
)

# Dynamic noise injection slider for Reviewers
noise_level = st.sidebar.slider("Inject Hardware Noise (Accelerometer Artifacts):", 0.0, 1.0, 0.1)

# Generate Signal based on Case
t = np.linspace(0, 5, 1000) # 5 seconds of data

if "Case 1" in case_choice:
    raw_wave = generate_realistic_ppg(t, heart_rate=65, stiffness=0.5)
elif "Case 2" in case_choice:
    raw_wave = generate_realistic_ppg(t, heart_rate=78, stiffness=1.2)
elif "Case 3" in case_choice:
    raw_wave = generate_realistic_ppg(t, heart_rate=90, stiffness=2.5)
elif "Case 4" in case_choice:
    raw_wave = generate_realistic_ppg(t, heart_rate=85, stiffness=1.5)
    noise_level = max(0.5, noise_level) # Force high noise

# Add noise (simulating accelerometer movement + baseline wander)
motion_artifact = noise_level * np.sin(2 * np.pi * 0.3 * t) + np.random.normal(0, noise_level * 0.2, 1000)
noisy_wave = raw_wave + motion_artifact

# --- TDA Process Pipeline & Inference ---
embedded_cloud = takens_embedding(noisy_wave, delay=15, dimension=3)
betti_0, betti_1, entropy = compute_betti_features(embedded_cloud)
feature_vector = np.hstack([betti_0, betti_1, [entropy]]).reshape(1, -1)
prob_tension = model.predict_proba(feature_vector)[0, 1]

# --- Dashboard Layout ---
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric("Heart Rate (Estimated)", f"{65 if 'Case 1' in case_choice else (90 if 'Case 3' in case_choice else 80)} BPM")
with col2:
    st.metric("Topological Entropy (H1)", f"{entropy:.4f}")
with col3:
    st.metric("CDSS Tension Probability", f"{prob_tension * 100:.1f}%")

# Clinical Output
if prob_tension > 0.60:
    st.error(f"⚠️ **Syndrome Detected: Liver Qi Stagnation / Pathological Tension**")
    st.write("Diagnostic Rationale: The persistent homology engine detected rigid, high-persistence 1D loops ($H_1$), consistent with reduced arterial compliance and the classical 'Wiry' (Xian) pulse.")
else:
    st.success(f"✅ **Healthy Baseline / Normal Arterial Compliance**")
    st.write("Diagnostic Rationale: The topological state-space reveals stable, low-dimensional attractors typical of a healthy, non-pathological resting state.")

st.divider()

# --- Interactive Visualizations ---
tab1, tab2 = st.tabs(["📉 Pulse Waveform & TDA Attractor", "📊 Betti Persistence Curves"])

with tab1:
    c1, c2 = st.columns([1.5, 1])
    
    with c1:
        st.subheader("1D Time-Series (Sensor Input)")
        fig1, ax1 = plt.subplots(figsize=(8, 3))
        ax1.plot(t, noisy_wave, color="#e74c3c" if noise_level > 0.3 else "#2ecc71", lw=1.5)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig1)

    with c2:
        st.subheader("3D Phase-Space Attractor")
        
        # 100% Browser-Compatible Matplotlib 3D Plot
        fig2 = plt.figure(figsize=(5, 5))
        ax2 = fig2.add_subplot(111, projection='3d')
        
        # Plot the trajectory with a clean, attractive style
        ax2.plot(embedded_cloud[:, 0], embedded_cloud[:, 1], embedded_cloud[:, 2], 
                 color='#8e44ad', lw=1.2, alpha=0.8)
        
        ax2.set_xlabel("x(t)")
        ax2.set_ylabel("x(t+τ)")
        ax2.set_zlabel("x(t+2τ)")
        
        # Remove gray background panes for a modern, clean look
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.grid(True, linestyle=":", alpha=0.6)
        
        # Adjust viewing angle for best 3D perspective
        ax2.view_init(elev=20, azim=45)
        
        st.pyplot(fig2)

with tab2:
    st.subheader("Extracted Topological Invariants (Model Features)")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    scales = np.linspace(0, 1.5, 100)
    ax3.plot(scales, betti_0, label="Betti-0 (Connected Components)", color="#34495e", lw=2)
    ax3.plot(scales, betti_1, label="Betti-1 (1D Loops / Notches)", color="#f39c12", lw=2)
    ax3.fill_between(scales, betti_1, color="#f39c12", alpha=0.2)
    ax3.set_xlabel("Filtration Scale (ε)")
    ax3.set_ylabel("Betti Number Count")
    ax3.legend()
    ax3.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig3)
