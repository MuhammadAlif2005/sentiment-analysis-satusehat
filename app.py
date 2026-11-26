import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from preprocessing import clean_text

# ------------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Sentimen Ulasan Aplikasi",
    page_icon="💬",
    layout="wide",
)

BASE = Path(__file__).parent
URUTAN = ["positif", "netral", "negatif"]
WARNA = {"positif": "#16a34a", "netral": "#f59e0b", "negatif": "#e11d48"}


def muat_css():
    css = (BASE / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Pemuatan data (model dimuat malas / lazy, hanya saat dibutuhkan)
# ------------------------------------------------------------------
@st.cache_data
def muat_data():
    df = pd.read_csv(BASE / "data" / "reviews_clean.csv")
    df["clean"] = df["clean"].fillna("")
    return df


@st.cache_data
def muat_json(nama):
    with open(BASE / "model" / nama, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def muat_perbandingan():
    return pd.read_csv(BASE / "model" / "model_comparison.csv")


@st.cache_resource
def muat_model():
    return joblib.load(BASE / "model" / "sentimen_model.joblib")


@st.cache_data
def buat_wordcloud(teks):
    wc = WordCloud(
        width=900, height=420, background_color="white",
        colormap="viridis", max_words=120,
    ).generate(teks)
    return wc.to_array()


def figur(lebar=5, tinggi=3.2):
    fig, ax = plt.subplots(figsize=(lebar, tinggi))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def kpi(label, nilai, sub="", warna="#4f46e5"):
    st.markdown(
        f"""
        <div class="kpi" style="--accent:{warna}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{nilai}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


muat_css()
df = muat_data()
metrics = muat_json("metrics.json")
top_words = muat_json("top_words.json")

total = len(df)
hitung = df["sentiment"].value_counts()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">💬 Dashboard Analisis Sentimen Ulasan</div>
        <div class="hero-sub">
            Aplikasi: <b>{metrics['app_id']}</b> &nbsp;•&nbsp; Sumber: Google Play Store
            &nbsp;•&nbsp; Model: TF-IDF + Logistic Regression
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard", "🧪 Evaluasi Model", "🔍 Coba Prediksi", "🗂️ Data Ulasan"]
)

# ==================================================================
# TAB 1: DASHBOARD
# ==================================================================
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Total ulasan", f"{total:,}", "setelah pembersihan")
    for kolom, label in zip([k2, k3, k4], URUTAN):
        n = int(hitung.get(label, 0))
        with kolom:
            kpi(label.capitalize(), f"{n / total:.1%}", f"{n:,} ulasan", WARNA[label])

    st.write("")
    kiri, kanan = st.columns(2)
    with kiri:
        with st.container(border=True):
            st.markdown("##### Distribusi sentimen")
            fig, ax = figur()
            ada = [s for s in URUTAN if s in hitung.index]
            ax.bar(ada, [hitung[s] for s in ada], color=[WARNA[s] for s in ada])
            ax.set_ylabel("Jumlah ulasan")
            st.pyplot(fig, transparent=True)
    with kanan:
        with st.container(border=True):
            st.markdown("##### Distribusi rating")
            fig, ax = figur()
            rating = df["score"].value_counts().sort_index()
            ax.bar(rating.index.astype(str), rating.values, color="#6366f1")
            ax.set_xlabel("Rating")
            ax.set_ylabel("Jumlah ulasan")
            st.pyplot(fig, transparent=True)

    st.write("")
    wc_kol, kata_kol = st.columns([2, 1])
    with wc_kol:
        with st.container(border=True):
            st.markdown("##### Word cloud")
            pilih = st.radio("Sentimen", URUTAN, horizontal=True, label_visibility="collapsed")
            teks = " ".join(df.loc[df["sentiment"] == pilih, "clean"])
            if teks.strip():
                st.image(buat_wordcloud(teks))
            else:
                st.info("Tidak ada data untuk sentimen ini.")
    with kata_kol:
        with st.container(border=True):
            st.markdown(f"##### Kata paling berpengaruh: {pilih}")
            for i, kata in enumerate(top_words.get(pilih, []), start=1):
                st.markdown(f"**{i}.** {kata}")

    st.write("")
    st.markdown("##### Contoh ulasan per sentimen")
    cols = st.columns(3)
    for kolom, label in zip(cols, URUTAN):
        with kolom:
            with st.container(border=True):
                st.markdown(
                    f"<span class='badge' style='background:{WARNA[label]}'>{label.upper()}</span>",
                    unsafe_allow_html=True,
                )
                subset = df[df["sentiment"] == label]
                contoh = subset.sample(min(3, len(subset)), random_state=7)
                for teks_ulasan in contoh["content"]:
                    ringkas = str(teks_ulasan)
                    ringkas = ringkas if len(ringkas) <= 160 else ringkas[:157] + "..."
                    st.markdown(f"<div class='quote'>{ringkas}</div>", unsafe_allow_html=True)

# ==================================================================
# TAB 2: EVALUASI MODEL
# ==================================================================
with tab2:
    e1, e2, e3 = st.columns(3)
    with e1:
        kpi("Akurasi", f"{metrics['akurasi']:.2%}", "data uji 20%")
    with e2:
        kpi("F1 macro", f"{metrics['f1_macro']:.2%}", "rata-rata antar kelas", "#0ea5e9")
    with e3:
        n_uji = int(metrics["laporan"]["macro avg"]["support"])
        kpi("Data latih / uji", f"{total - n_uji:,} / {n_uji:,}", "pembagian 80:20", "#8b5cf6")

    st.write("")
    kiri, kanan = st.columns(2)
    with kiri:
        with st.container(border=True):
            st.markdown("##### Confusion matrix")
            st.image(str(BASE / "model" / "confusion_matrix.png"))
    with kanan:
        with st.container(border=True):
            st.markdown("##### Metrik per kelas")
            laporan = metrics["laporan"]
            baris = []
            for k in URUTAN:
                if k in laporan:
                    baris.append({
                        "Kelas": k,
                        "Precision": laporan[k]["precision"],
                        "Recall": laporan[k]["recall"],
                        "F1-score": laporan[k]["f1-score"],
                        "Support": int(laporan[k]["support"]),
                    })
            st.dataframe(
                pd.DataFrame(baris).style.format(
                    {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}
                ),
                hide_index=True,
            )
        st.write("")
        with st.container(border=True):
            st.markdown("##### Perbandingan algoritma")
            st.dataframe(muat_perbandingan(), hide_index=True)

    st.caption(
        f"Dilatih pada {metrics['tanggal']} dengan scikit-learn {metrics['versi_sklearn']}. "
        "Label sentimen diturunkan dari rating (1-2 negatif, 3 netral, 4-5 positif)."
    )

# ==================================================================
# TAB 3: COBA PREDIKSI
# ==================================================================
with tab3:
    with st.container(border=True):
        st.markdown("##### Prediksi sentimen ulasan baru")
        ulasan = st.text_area(
            "Tulis ulasan aplikasi",
            value="aplikasinya bagus, driver ramah dan cepat",
            height=120,
        )
        jalankan = st.button("Prediksi sentimen", type="primary")

    if jalankan:
        bersih = clean_text(ulasan)
        if not bersih:
            st.warning("Teks kosong atau tidak mengandung kata yang bisa dianalisis.")
        else:
            with st.spinner("Memuat model..."):
                model = muat_model()
            label = model.predict([bersih])[0]
            proba = model.predict_proba([bersih])[0]

            h1, h2 = st.columns([1, 2])
            with h1:
                kpi("Hasil prediksi", label.upper(), f"keyakinan {proba.max():.1%}", WARNA.get(label, "#4f46e5"))
            with h2:
                with st.container(border=True):
                    st.markdown("##### Probabilitas tiap kelas")
                    fig, ax = figur(5, 2.2)
                    kelas = list(model.classes_)
                    ax.barh(kelas, proba, color=[WARNA.get(k, "#6366f1") for k in kelas])
                    ax.set_xlim(0, 1)
                    st.pyplot(fig, transparent=True)

            # Kata yang paling mendorong hasil prediksi
            vec = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]
            fitur = vec.get_feature_names_out()
            X = vec.transform([bersih])
            idx = X.nonzero()[1]
            if len(idx) > 0:
                koef = clf.coef_[list(clf.classes_).index(label)]
                kontribusi = sorted(
                    ((fitur[i], float(X[0, i] * koef[i])) for i in idx),
                    key=lambda t: t[1], reverse=True,
                )[:5]
                with st.container(border=True):
                    st.markdown("##### Kata yang paling mendorong hasil ini")
                    st.write(", ".join(f"**{k}**" for k, _ in kontribusi))
                    st.caption(f"Teks setelah preprocessing: {bersih}")

# ==================================================================
# TAB 4: DATA ULASAN
# ==================================================================
with tab4:
    with st.container(border=True):
        f1, f2 = st.columns([2, 1])
        with f1:
            cari = st.text_input("Cari kata dalam ulasan", placeholder="contoh: error")
        with f2:
            saring = st.multiselect("Sentimen", URUTAN, default=URUTAN)

        tampil = df[df["sentiment"].isin(saring)]
        if cari.strip():
            tampil = tampil[tampil["content"].str.contains(cari, case=False, na=False)]

        st.caption(f"Menampilkan {len(tampil):,} dari {total:,} ulasan")
        st.dataframe(
            tampil[["content", "score", "sentiment"]],
            height=420,
            hide_index=True,
        )
        st.download_button(
            "Unduh data terfilter (CSV)",
            tampil.to_csv(index=False).encode("utf-8"),
            file_name="ulasan_terfilter.csv",
            mime="text/csv",
        )

st.markdown(
    "<div class='footer'>UAS Pemrograman Big Data • Analisis Sentimen Ulasan Google Play Store</div>",
    unsafe_allow_html=True,
)
