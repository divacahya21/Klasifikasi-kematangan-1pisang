import streamlit as st
from PIL import Image
import numpy as np
import joblib
import pandas as pd
from preprocessing import ekstrak_semua_fitur

# ------------------------------
# Konfigurasi tampilan Streamlit
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
# Load model
# ------------------------------
model_rf = joblib.load("model/rf_model.pkl")
model_svm = joblib.load("model/svm_model.pkl")

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg", width=200)
    st.markdown("### Pengaturan")
    model_option = st.selectbox("Pilih Model Klasifikasi", ["Random Forest", "SVM"])
    st.markdown("---")
    st.info("Silakan unggah gambar pisang dari perangkat Anda untuk diklasifikasikan tingkat kematangannya.")

# ------------------------------
# Upload gambar
# ------------------------------
uploaded_file = st.file_uploader("Unggah gambar pisang di sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang Diupload", use_column_width=True)

    # Tombol prediksi
    if st.button("Prediksi Tingkat Kematangan"):
        with st.spinner("Memproses gambar dan mengekstraksi fitur..."):
            fitur, kelas_segmentasi = ekstrak_semua_fitur(img)
            fitur_df = pd.DataFrame([fitur])

            if model_option == "Random Forest":
                hasil_prediksi = model_rf.predict(fitur_df)[0]
                proba = model_rf.predict_proba(fitur_df)[0]
                label_list = model_rf.classes_
            else:
                hasil_prediksi = model_svm.predict(fitur_df)[0]
                proba = model_svm.predict_proba(fitur_df)[0]
                label_list = model_svm.classes_

        # Tampilkan hasil
        st.success(f"Hasil Klasifikasi: **{hasil_prediksi.upper()}**")
        st.markdown(f"<p style='color: gray;'>*Deteksi awal dari segmentasi: <b>{kelas_segmentasi.upper()}</b></p>", unsafe_allow_html=True)

        # Tampilkan probabilitas
        st.markdown("### Probabilitas Prediksi")
        prob_df = pd.DataFrame({
            "Kelas": label_list,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False)

        st.bar_chart(data=prob_df.set_index("Kelas"))

else:
    st.markdown("<i>Belum ada gambar yang diunggah.</i>", unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>© 2025 - Projek PCD Pisang - Streamlit Deployment</p>", unsafe_allow_html=True)
