import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
 
# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Wine Quality Intelligence",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ==============================================================================
# PROFESSIONAL LIGHT THEME (custom CSS — no dark/black backgrounds)
# ==============================================================================
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .stApp {
        background-color: #F7F5F2;
        color: #2B2320;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    #MainMenu, footer {visibility: hidden;}
 
    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5DED6;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #6E1E33;
    }
 
    /* ---------- Headings ---------- */
    h1 {
        color: #6E1E33;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #4A3128;
        font-weight: 700;
    }
 
    /* ---------- Top banner ---------- */
    .app-header {
        background: linear-gradient(120deg, #6E1E33 0%, #9C3B57 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 18px rgba(110,30,51,0.25);
    }
    .app-header h1 { color: white; margin-bottom: 0.2rem; }
    .app-header p { color: #F3E3E8; margin: 0; font-size: 1.02rem; }
 
    /* ---------- Cards ---------- */
    .metric-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        border: 1px solid #ECE3DC;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
 
    .verdict-card {
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 0.5rem;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    .verdict-good {
        background: linear-gradient(135deg, #E7F6EA 0%, #D4F0DA 100%);
        border: 2px solid #2E8B3D;
    }
    .verdict-bad {
        background: linear-gradient(135deg, #FBEAEA 0%, #F7DADA 100%);
        border: 2px solid #B23A3A;
    }
    .verdict-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .verdict-good .verdict-title { color: #1F6B2C; }
    .verdict-bad .verdict-title { color: #A22525; }
    .verdict-sub {
        font-size: 0.95rem;
        color: #5A4E48;
    }
 
    /* ---------- Buttons ---------- */
    .stButton>button {
        background-color: #6E1E33;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.65em 1.6em;
        font-weight: 700;
        font-size: 1.0rem;
        width: 100%;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #8A2941;
        transform: translateY(-1px);
    }
 
    /* ---------- Sliders ---------- */
    .stSlider [data-baseweb="slider"] > div > div {
        background: #6E1E33 !important;
    }
 
    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 10px 10px 0 0;
        padding: 8px 18px;
        border: 1px solid #ECE3DC;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6E1E33 !important;
        color: white !important;
    }
 
    /* ---------- Dataframe ---------- */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #ECE3DC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ==============================================================================
# ARTIFACTS
# ==============================================================================
MODEL_PATH = "dt_winequality.pkl"
SCALER_PATH = "scaler_winequality.pkl"
COLUMNS_PATH = "columns.pkl"
 
FEATURE_INFO = {
    "fixed acidity":        {"min": 4.0,  "max": 16.0,  "default": 8.3,   "step": 0.1,   "unit": "g/dm³", "help": "Tartaric acid content — affects tartness."},
    "volatile acidity":     {"min": 0.10, "max": 1.60,  "default": 0.53,  "step": 0.01,  "unit": "g/dm³", "help": "Acetic acid content — too high causes a vinegar taste."},
    "citric acid":          {"min": 0.0,  "max": 1.0,   "default": 0.27,  "step": 0.01,  "unit": "g/dm³", "help": "Adds freshness and flavor."},
    "residual sugar":       {"min": 0.5,  "max": 16.0,  "default": 2.5,   "step": 0.1,   "unit": "g/dm³", "help": "Sugar remaining after fermentation."},
    "chlorides":            {"min": 0.01, "max": 0.65,  "default": 0.087, "step": 0.001, "unit": "g/dm³", "help": "Salt content in the wine."},
    "free sulfur dioxide":  {"min": 1.0,  "max": 75.0,  "default": 15.0,  "step": 1.0,   "unit": "mg/dm³","help": "Prevents microbial growth and oxidation."},
    "total sulfur dioxide": {"min": 5.0,  "max": 290.0, "default": 46.0,  "step": 1.0,   "unit": "mg/dm³","help": "Total of free and bound forms of SO2."},
    "density":              {"min": 0.985,"max": 1.005, "default": 0.9967,"step": 0.0001,"unit": "g/cm³", "help": "Close to water density; depends on sugar/alcohol content."},
    "pH":                   {"min": 2.7,  "max": 4.1,   "default": 3.31,  "step": 0.01,  "unit": "",      "help": "Acidity scale — most wines are between 3 and 4."},
    "sulphates":            {"min": 0.3,  "max": 2.1,   "default": 0.66,  "step": 0.01,  "unit": "g/dm³", "help": "Additive that acts as an antimicrobial and antioxidant."},
    "alcohol":              {"min": 8.0,  "max": 15.0,  "default": 10.4,  "step": 0.1,   "unit": "% vol", "help": "Alcohol content by volume."},
}
 
 
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    columns = joblib.load(COLUMNS_PATH) if os.path.exists(COLUMNS_PATH) else list(FEATURE_INFO.keys())
    return model, scaler, columns
 
 
model, scaler, columns = load_artifacts()
 
if "history" not in st.session_state:
    st.session_state.history = []
 
# ==============================================================================
# HEADER
# ==============================================================================
st.markdown(
    """
    <div class="app-header">
        <h1>🍷 Wine Quality Intelligence</h1>
        <p>Predict whether a wine sample meets premium quality standards (score ≥ 7)
        using a tuned Decision Tree classifier — adjust the sliders and see the result instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
if model is None:
    st.error(
        "⚠️ Model file **dt_winequality.pkl** not found. Place `dt_winequality.pkl`, "
        "`scaler_winequality.pkl`, and `columns.pkl` in the same folder as this app."
    )
 
# ==============================================================================
# SIDEBAR — SLIDER INPUTS
# ==============================================================================
st.sidebar.header("🧪 Sample Composition")
st.sidebar.caption("Slide to set each chemical property of the wine sample.")
 
presets = {
    "Custom": None,
    "Typical Everyday Wine": {k: v["default"] for k, v in FEATURE_INFO.items()},
    "High Alcohol / Low Acidity": {
        "fixed acidity": 6.5, "volatile acidity": 0.3, "citric acid": 0.4,
        "residual sugar": 3.0, "chlorides": 0.05, "free sulfur dioxide": 20,
        "total sulfur dioxide": 90, "density": 0.9935, "pH": 3.4,
        "sulphates": 0.75, "alcohol": 13.2,
    },
    "High Acidity / Low Alcohol": {
        "fixed acidity": 11.5, "volatile acidity": 0.7, "citric acid": 0.1,
        "residual sugar": 2.0, "chlorides": 0.09, "free sulfur dioxide": 8,
        "total sulfur dioxide": 25, "density": 0.998, "pH": 3.1,
        "sulphates": 0.5, "alcohol": 9.2,
    },
}
preset_choice = st.sidebar.selectbox("Quick preset", list(presets.keys()))
 
user_input = {}
for feat in columns:
    info = FEATURE_INFO.get(feat, {"min": 0.0, "max": 100.0, "default": 1.0, "step": 0.1, "unit": "", "help": ""})
    default_val = presets[preset_choice][feat] if presets[preset_choice] else info["default"]
    label = f"{feat.title()} ({info['unit']})" if info.get("unit") else feat.title()
    user_input[feat] = st.sidebar.slider(
        label,
        min_value=float(info["min"]),
        max_value=float(info["max"]),
        value=float(default_val),
        step=float(info["step"]),
        help=info.get("help", ""),
    )
 
input_df = pd.DataFrame([user_input])[columns] if model is not None else pd.DataFrame([user_input])
 
st.sidebar.divider()
predict_clicked = st.sidebar.button("🔍 Predict Quality", use_container_width=True)
 
# ==============================================================================
# MAIN — TABS
# ==============================================================================
tab_predict, tab_insights, tab_history = st.tabs(["🔮 Prediction", "📊 Model Insights", "🕘 History"])
 
# ------------------------------------------------------------------------------
# TAB 1 — PREDICTION
# ------------------------------------------------------------------------------
with tab_predict:
    col_input, col_result = st.columns([1.1, 1])
 
    with col_input:
        st.subheader("Current Sample")
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
 
    with col_result:
        st.subheader("Prediction")
        if predict_clicked and model is not None:
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            good_prob = proba[1] if len(proba) > 1 else proba[0]
 
            st.session_state.history.append({**user_input, "prediction": "Good" if prediction == 1 else "Not Good", "confidence": round(good_prob * 100, 1)})
 
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="verdict-card verdict-good">
                        <div class="verdict-title">✅ Good Quality Wine</div>
                        <div class="verdict-sub">Predicted quality score ≥ 7</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-card verdict-bad">
                        <div class="verdict-title">❌ Not Good Quality</div>
                        <div class="verdict-sub">Predicted quality score &lt; 7</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
 
            st.write("")
            m1, m2 = st.columns(2)
            m1.metric("Confidence (Good)", f"{good_prob*100:.1f}%")
            m2.metric("Confidence (Not Good)", f"{(1-good_prob)*100:.1f}%")
            st.progress(float(good_prob))
 
        elif model is None:
            st.info("Model not loaded — cannot predict yet.")
        else:
            st.info("Set the sliders in the sidebar, then click **Predict Quality**.")
 
# ------------------------------------------------------------------------------
# TAB 2 — MODEL INSIGHTS
# ------------------------------------------------------------------------------
with tab_insights:
    st.subheader("Feature Importance")
    if model is not None and hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
        st.bar_chart(importance)
        st.caption("Relative contribution of each feature to the Decision Tree's predictions.")
 
        st.subheader("Model Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-card'><b>Model type</b><br>{type(model).__name__}</div>", unsafe_allow_html=True)
        with c2:
            depth = getattr(model, "get_depth", lambda: "N/A")()
            st.markdown(f"<div class='metric-card'><b>Tree depth</b><br>{depth}</div>", unsafe_allow_html=True)
        with c3:
            leaves = getattr(model, "get_n_leaves", lambda: "N/A")()
            st.markdown(f"<div class='metric-card'><b>Leaf nodes</b><br>{leaves}</div>", unsafe_allow_html=True)
    else:
        st.info("Load the model to see feature importance and details.")
 
# ------------------------------------------------------------------------------
# TAB 3 — HISTORY
# ------------------------------------------------------------------------------
with tab_history:
    st.subheader("Prediction History (this session)")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions made yet in this session. Run a prediction to see it logged here.")
 
st.divider()
st.caption(
    "Built with Streamlit • Model: GridSearchCV-tuned Decision Tree Classifier • "
    "Target: quality ≥ 7 = Good, else Not Good"
)
