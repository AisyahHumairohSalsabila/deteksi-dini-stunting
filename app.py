"""Entry point aplikasi Streamlit deteksi dini stunting.

Navigasi didefinisikan eksplisit agar sidebar hanya memuat halaman layanan
yang relevan. Jalankan dengan: streamlit run app.py
"""
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Deteksi Dini Stunting",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    """Memuat satu sumber gaya bersama untuk seluruh halaman."""
    css_path = Path(__file__).parent / "assets" / "style.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


load_css()

dashboard = st.Page("pages/_1_Visualisasi.py", title="Dashboard Visualisasi", default=True)
deteksi = st.Page("pages/_2_Deteksi_Dini.py", title="Deteksi Dini")
navigation = st.navigation([dashboard, deteksi], position="sidebar")
navigation.run()
