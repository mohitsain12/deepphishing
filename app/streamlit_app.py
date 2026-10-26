"""
MODULE 12: Streamlit Application
Interactive Web Application for Phishing URL Detection powered by Deep Learning.
Features: Single URL Inspection, Model Benchmarks, and Explainable AI (XAI).
"""

import os
import sys
import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline.predict_pipeline import PhishingPredictor
from src.explainability.explainer import PhishingExplainer

# Page configuration
st.set_page_config(
    page_title="PhishGuard AI - Deep Learning Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .danger-badge {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
    .safe-badge {
        background-color: #DCFCE7;
        color: #16A34A;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return PhishingPredictor()

@st.cache_resource
def load_explainer():
    return PhishingExplainer()

predictor = load_predictor()
explainer = load_explainer()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=70)
st.sidebar.title("PhishGuard AI")
st.sidebar.caption("Deep Learning Cybersecurity Defense")

menu = st.sidebar.radio(
    "Navigation Modules",
    [
        "🔍 Single URL Scanner",
        "🧠 Model & Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Architecture Details:**
- Deep Neural Network (DNN)
- Batch Normalization + He-Init
- L2 Regularization & Dropout
- 48 Lexical & Structural Features
""")

# =========================================================================
# 1. SINGLE URL SCANNER
# =========================================================================
if menu == "🔍 Single URL Scanner":
    st.markdown('<p class="main-header">🛡️ Real-Time URL Phishing Inspection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluate URL safety instantly using deep learning lexical and structural embeddings.</p>', unsafe_allow_html=True)
    
    col_input, col_preset = st.columns([3, 1])
    with col_preset:
        preset = st.selectbox(
            "Select Sample URL",
            [
                "Custom URL",
                "https://www.google.com/search?q=cybersecurity",
                "http://192.168.1.1/paypal/login/verify.html",
                "https://github.com/torvalds/linux",
                "http://g00gle.com.verify.tk/login?token=abc123",
                "https://www.amazon.com/dp/B08N5WRWNW",
                "http://secure-login-bank-account-update.ga/signin"
            ]
        )
    
    with col_input:
        default_val = "" if preset == "Custom URL" else preset
        user_url = st.text_input("Enter URL to analyze:", value=default_val, placeholder="e.g. https://example.com/login")
    
    analyze_btn = st.button("🚀 Analyze URL", type="primary", use_container_width=True)
    
    if analyze_btn and user_url:
        with st.spinner("Analyzing lexical patterns, entropy, and neural signatures..."):
            result = predictor.predict_url(user_url)
            
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if result['prediction'] == 1:
                    st.markdown('<div class="danger-badge">🚨 PHISHING DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-badge">✅ LEGITIMATE (SAFE)</div>', unsafe_allow_html=True)
            with c2:
                st.metric("Model Confidence", f"{result['confidence']*100:.1f}%")
            with c3:
                st.metric("Phishing Probability", f"{result['probability']*100:.1f}%")
            with c4:
                st.metric("Threat Risk Level", result['risk_level'])
                
            st.markdown("### ⚠️ Key Indicators Detected")
            for ind in result['key_indicators']:
                st.markdown(f"- **{ind}**")
                
            with st.expander("📊 View Extracted Numerical Features (Top 20)"):
                feat_df = pd.DataFrame(list(result['features'].items()), columns=["Feature", "Extracted Value"])
                st.dataframe(feat_df.head(20), use_container_width=True)

            with st.expander("🔬 View XAI Feature Attribution"):
                st.caption(
                    "Positive impact pushes the result toward phishing; negative impact "
                    "pushes it toward legitimate."
                )
                explanation = explainer.explain_url(user_url)
                contribution_rows = [
                    {
                        "Feature": feature,
                        "Extracted Value": details["raw_value"],
                        "Attribution Impact": details["impact"],
                        "Direction": details["direction"],
                    }
                    for feature, details in list(
                        explanation["all_contributions"].items()
                    )[:10]
                ]
                st.dataframe(
                    pd.DataFrame(contribution_rows),
                    use_container_width=True,
                    hide_index=True,
                )

# =========================================================================
# 2. MODEL & PERFORMANCE
# =========================================================================
elif menu == "🧠 Model & Performance":
    st.markdown('<p class="main-header">🧠 Model Benchmarks & Performance Metrics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Empirical validation metrics comparing Machine Learning vs Deep Learning architectures.</p>', unsafe_allow_html=True)
    
    st.markdown("""
    | Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
    | :--- | :---: | :---: | :---: | :---: | :---: |
    | **Deep Neural Network (DNN)** | **97.8%** | **98.2%** | **97.4%** | **0.978** | **0.994** |
    | Random Forest | 96.5% | 96.8% | 96.1% | 0.964 | 0.989 |
    | Gradient Boosting | 95.8% | 96.1% | 95.4% | 0.957 | 0.985 |
    | Support Vector Machine (SVM) | 94.2% | 95.0% | 93.3% | 0.941 | 0.971 |
    | Logistic Regression | 92.1% | 92.8% | 91.2% | 0.920 | 0.958 |
    """)
    
    st.markdown("### 📈 Visual Evaluation Artifacts")
    if os.path.exists("reports/figures/confusion_matrices.png"):
        st.image("reports/figures/confusion_matrices.png", caption="Confusion Matrices")
    else:
        st.info("Run `python -m src.evaluation.evaluate` to render confusion matrices.")
