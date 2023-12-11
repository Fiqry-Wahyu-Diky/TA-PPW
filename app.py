import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import altair as alt
from streamlit_option_menu import option_menu
from joblib import load
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

showWarningOnDirectExecution = False

with st.sidebar:
    selected = option_menu(
        menu_title="MENU",
        options=["HOME", "PROJECT"],
    )
# ====================== Home ====================
if selected == "HOME":
    st.markdown('<h1 style = "text-align: center;"> Aplikasi Klasifikasi Berita </h1>', unsafe_allow_html=True)
    # gambar = Image.open("nlp.jpg")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('')
    with col2:
        st.image("data/nlp.jpg", width=250)
    with col3:
        st.write(' ')

    st.write(' ')
    st.markdown(
        '<p style = "text-align: justify;"> <b> Text processing </b> adalah proses manipulasi atau analisis teks menggunakan komputer atau alat otomatis lainnya. Tujuannya dapat bervariasi, termasuk pengolahan teks untuk ekstraksi informasi, pemrosesan bahasa alami, pengenalan pola teks, pemfilteran teks, atau tugas lainnya yang melibatkan teks. Ada beberapa tahapan dalam text processing sebagai berikut:</p>',
        unsafe_allow_html=True)
    st.write('- Crawling Data <br> <p style = "text-align: justify;"><b>Crawling</b> merupakan alat otomatis yang mengumpulkan beragam informasi dengan menjelajahi berbagai halaman web. Proses ini mencakup identifikasi serta ekstraksi elemen-elemen seperti teks, gambar, dan unsur lainnya, sehingga membentuk pemahaman menyeluruh tentang konten yang tersebar di internet.</p>',unsafe_allow_html=True)
    st.write(
        '- Normalisasi Text <br> <p style = "text-align: justify;"> <b>Normalisasi Text</b> merupakan suatu proses mengubah data teks menjadi bentuk standar, sehingga dapat digunakan dalam pengolahan lebih lanjut.</p>',
        unsafe_allow_html=True)
    st.write(
        '- Reduksi Dimensi <br> <p style = "text-align: justify;"> <b>Reduksi dimensi Text</b> adalah proses mengurangi jumlah atribut (fitur) dalam suatu dataset dengan tetap mempertahankan informasi yang signifikan. Tujuan utama dari reduksi dimensi adalah untuk mengatasi masalah "kutukan dimensi" (curse of dimensionality), di mana dataset dengan banyak fitur dapat mengakibatkan masalah komputasi yang mahal dan pemodelan yang kurang akurat. Reduksi dimensi juga dapat membantu dalam memahami struktur data, menghilangkan atribut yang tidak relevan, dan memungkinkan visualisasi yang lebih baik dari data yang kompleks.</p>',
        unsafe_allow_html=True)


# ====================== Project ====================
else:
    st.markdown('<h1 style = "text-align: center;">Text Processing</h1>', unsafe_allow_html=True)
    st.write("Oleh | FIQRY WAHYU DIKY W | 200411100125")
    dataset, preprocessing, lda, modelling, implementasi = st.tabs(
["Dataset", "Preprocessing", "Reduksi LDA", "Modeling","Implementasi"])


# ====================== Crawling ====================
    with dataset:
        dataset = pd.read_csv("data/dataset crawling antaranews.csv")
        st.dataframe(dataset)
        st.info(f"Banyak Dataset : {len(dataset)}")
        st.warning(f'Informasi Dataset')
        st.write(dataset.describe())


# ======================= Preprocessing =================================
    with preprocessing:
        st.write("# Normalisasi")


# ======== cleanned ===================

    # Tombol untuk menghapus data NaN dari kolom "Abstrak"
        datasetClean = pd.read_csv("data/dataset cleaning.csv")
        st.info("#### Data sudah dibersihkan")
        st.warning("Proses pembersihan data 4 tahapan:")
        col1, col2, col3, col4= st.columns(4)
        with col1:
            st.write("Missing Values")
        with col2:
            st.write("Duplikasi Data")
        with col3:
            st.write("Punctuation")
        with col4:
            st.write("Stopwords")

        st.dataframe(datasetClean)

        st.success("Panjang Dataset Setelah Preprocessing")
        st.info(f'{len(datasetClean)} data')



# =========================== LDA ===============================
    with lda:
        st.info("### Hasil dari reduksi LDA sebagai berikut:")

        st.warning("Proporsi Topik Pada Dokumen")
        ptd = pd.read_csv("data/proporsi topik dokumen.csv")
        st.dataframe(ptd)

        st.warning("Proporsi Topik Pada Kata")
        ptk = pd.read_csv("data/proporsi topik kata.csv")
        st.dataframe(ptk)

# # =========================== Modelling ==================================
    with modelling:
        st.info("Hasil Evaluasi 4 Model")
        st.image("data/akurasi.png")

        st.success("Hasil pengujian")
        history = pd.read_csv("data/history.csv")
        st.dataframe(history)

        st.success("Dari evaluasi diatas didapatkan model terbaik Random Forest")
        best = pd.read_csv("data/best param.csv")
        st.dataframe(best)

#
#     # =========================== Implementasi ===============================
    with implementasi:
        st.write("# Implementasi")
        st.info(f"Dalam implementasi akan digunakan metode yang paling tinggi akurasinya (dalam evaluasi) yaitu: metode Random Forest dan menggunakan 7 Topik")
        inputan = st.text_area("Masukkan Teks Berita")
        inp = [inputan]
        st.warning(f"VSM yang digunakan yaitu TFIDF")

        vectorizer = load("data/tfidf vectorizer.pkl")
        lda = load("data/model lda.pkl")
        model = load("data/model.pkl")


        if st.button("predict"):
            vecinp = vectorizer.transform(inp)
            ldainp = lda.transform(vecinp)
            modelinp = model.predict(ldainp)
            st.success("Hasil dari prediksi berita anda adalah:")
            if modelinp == 'politik':
                st.info("Politik")
            elif modelinp == 'ekonomi':
                st.info("Ekonomi")
            else:
                st.info("Olahraga")
