import streamlit as st
from modules.dashboard import show_dashboard
from modules.prediction import show_prediction
from modules.performance import show_performance


st.set_page_config(
    page_title="AI Breast Cancer Diagnostic System",
    page_icon="🧠",
    layout="wide"
)


st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)


st.sidebar.title("🧠 AI Diagnostic Panel")


page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "🔬 Prediction",
        "📈 Model Performance"
    ]
)


if page == "🏠 Dashboard":
    show_dashboard()

elif page == "🔬 Prediction":
    show_prediction()

elif page == "📈 Model Performance":
    show_performance()


st.markdown("---")
st.caption(
    "AI Powered Breast Cancer Diagnostic System | ANN + TensorFlow + Streamlit"
)