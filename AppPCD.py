import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import joblib

from preprocessing import ekstrak_semua_fitur

# ------------------------------
# Konfigurasi Tampilan
# ------------------------------
st.set_page_config(
    page_title="Klasifikasi Kematangan Pisang 🍌",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown(
    "<h1 style='text-align: center; color: #f4b400;'>🍌 Klasifikasi Kematangan Buah Pisang</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ------------------------------
# Load Model & Fitur Order
# ------------------------------
model_rf = joblib.load("model/rf_model.pkl")
model_svm = joblib.load("model/svm_model.pkl")
fitur_order = joblib.load("model/fitur_order.pkl")

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.image("images/pisang_sample.jpg", width=500)
    st.markdown("### Pengaturan")
    model_option = st.selectbox("Pilih Model Klasifikasi", ["Random Forest", "SVM"])
    st.markdown("---")
    st.info("Unggah gambar pisang untuk diklasifikasikan tingkat kematangannya berdasarkan segmentasi warna.")

# ------------------------------
# Upload Gambar
# ------------------------------
uploaded_file = st.file_uploader("Unggah gambar pisang", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_column_width=True)

    if st.button("Prediksi Tingkat Kematangan"):
        with st.spinner("Mengekstraksi fitur dan melakukan prediksi..."):
            model = model_rf if model_option == "Random Forest" else model_svm

            fitur, hasil_prediksi, kelas_mask, proba = ekstrak_semua_fitur(img, model, fitur_order)

        st.success(f"Hasil Prediksi: **{hasil_prediksi.upper()}** (dari mask: {kelas_mask})")

        # Probabilitas
        st.markdown("### Probabilitas Prediksi")
        df_proba = pd.DataFrame({
            "Kelas": model.classes_,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False)
        st.bar_chart(df_proba.set_index("Kelas"))

        # Fitur
        st.markdown("### Fitur yang Diekstrak")
        df_fitur = pd.DataFrame([fitur])
        st.dataframe(df_fitur)

else:
    st.markdown("<i>Belum ada gambar yang diunggah.</i>", unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>© 2025 - Projek PCD Pisang - Streamlit Deployment</p>", unsafe_allow_html=True)
