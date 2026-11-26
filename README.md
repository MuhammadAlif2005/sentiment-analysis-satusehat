# UAS Pemrograman Big Data: Analisis Sentimen Ulasan Google Play Store

Aplikasi yang dianalisis: `com.telkom.tracencare`

## Alur
Scraping ulasan -> pelabelan dari rating -> preprocessing -> TF-IDF + Logistic Regression -> dashboard Streamlit.

## Struktur
- `model/` : model terlatih dan artefak evaluasi
- `data/` : data ulasan mentah dan bersih
- `preprocessing.py` : fungsi pembersihan teks (dipakai Colab dan dashboard)
- `requirements.txt` : daftar library
