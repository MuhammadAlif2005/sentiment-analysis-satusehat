import html
import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from wordcloud import WordCloud

from preprocessing import clean_text

# ------------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Analisis Sentimen SATUSEHAT Mobile",
    page_icon=":material/monitoring:",
    layout="wide",
)

BASE = Path(__file__).parent

# ------------------------------------------------------------------
# Token desain (satu sumber untuk warna yang dipakai grafik Python;
# padanannya untuk CSS ada di style.css)
# ------------------------------------------------------------------
URUTAN = ["positif", "netral", "negatif"]
LABEL = {"positif": "Positif", "netral": "Netral", "negatif": "Negatif"}
WARNA = {"positif": "#16A34A", "netral": "#D97706", "negatif": "#DC2626"}
WARNA_TEKS = {"positif": "#15803D", "netral": "#B45309", "negatif": "#B91C1C"}
WARNA_SOFT = {"positif": "#DCFCE7", "netral": "#FEF3C7", "negatif": "#FEE2E2"}
PRIMARY = "#0E7490"
PRIMARY_SOFT = "#E0F2F7"
INK = "#0F172A"
MUTED = "#64748B"
GRID = "#E2E8F0"

PALET_WORDCLOUD = {
    "positif": ["#14532D", "#166534", "#15803D", "#16A34A"],
    "netral": ["#78350F", "#92400E", "#B45309", "#D97706"],
    "negatif": ["#7F1D1D", "#991B1B", "#B91C1C", "#DC2626"],
}

# Ikon SVG bergaya garis (stroke 1.75) agar konsisten
IKON = {
    "pesan": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "positif": '<circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" x2="9.01" y1="9" y2="9"/><line x1="15" x2="15.01" y1="9" y2="9"/>',
    "netral": '<circle cx="12" cy="12" r="10"/><line x1="8" x2="16" y1="15" y2="15"/><line x1="9" x2="9.01" y1="9" y2="9"/><line x1="15" x2="15.01" y1="9" y2="9"/>',
    "negatif": '<circle cx="12" cy="12" r="10"/><path d="M16 16s-1.5-2-4-2-4 2-4 2"/><line x1="9" x2="9.01" y1="9" y2="9"/><line x1="15" x2="15.01" y1="9" y2="9"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "aktivitas": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "data": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
}


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
def buat_wordcloud(teks, sentimen):
    wc = WordCloud(
        width=900, height=420, background_color="white",
        colormap=ListedColormap(PALET_WORDCLOUD[sentimen]),
        max_words=120, prefer_horizontal=0.95,
    ).generate(teks)
    return wc.to_array()


# ------------------------------------------------------------------
# Format angka gaya Indonesia (1.234 dan 12,3%)
# ------------------------------------------------------------------
def fmt_int(n):
    return f"{int(n):,}".replace(",", ".")


def fmt_pct(x, desimal=1):
    return f"{x * 100:.{desimal}f}%".replace(".", ",")


# ------------------------------------------------------------------
# Komponen presentasi
# ------------------------------------------------------------------
def ikon(nama, ukuran=20):
    return (
        f'<svg width="{ukuran}" height="{ukuran}" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{IKON[nama]}</svg>'
    )


def kpi(label, nilai, sub="", nama_ikon="pesan", fg=PRIMARY, bg=PRIMARY_SOFT):
    st.markdown(
        '<div class="kpi">'
        f'<div class="kpi-icon" style="color:{fg};background:{bg}">{ikon(nama_ikon)}</div>'
        '<div class="kpi-body">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(nilai)}</div>'
        f'<div class="kpi-sub">{html.escape(sub)}</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def bagian(judul):
    st.markdown(f'<div class="section-title">{html.escape(judul)}</div>', unsafe_allow_html=True)


def kartu(kunci):
    return st.container(border=True, key=f"card-{kunci}")


def judul_kartu(judul, sub=""):
    st.markdown(
        f'<div class="card-title">{html.escape(judul)}</div>'
        f'<div class="card-sub">{html.escape(sub)}</div>',
        unsafe_allow_html=True,
    )


def figur(lebar=5.2, tinggi=3.0):
    fig, ax = plt.subplots(figsize=(lebar, tinggi))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for sisi in ("top", "right", "left"):
        ax.spines[sisi].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def tampil_figur(fig):
    """Simpan grafik ke PNG di memori lalu tampilkan lewat st.image
    (menggantikan st.pyplot dengan argumen savefig yang sudah deprecated)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    st.image(buf)


def ringkas(teks, batas=160):
    teks = str(teks)
    return teks if len(teks) <= batas else teks[: batas - 3] + "..."


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
    '<div class="page-header">'
    '<h1 class="page-title">Dashboard Analisis Sentimen Ulasan Aplikasi SATUSEHAT Mobile</h1>'
    '<div class="page-meta">'
    "<span>Google Play Store</span>"
    f"<span>{html.escape(str(metrics['app_id']))}</span>"
    "<span>TF-IDF + Logistic Regression</span>"
    f"<span>Model dilatih {html.escape(str(metrics['tanggal']))}</span>"
    "</div></div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    ":material/dashboard: Dashboard",
    ":material/analytics: Evaluasi Model",
    ":material/search: Coba Prediksi",
    ":material/table_rows: Data Ulasan",
])

# ==================================================================
# TAB 1: DASHBOARD
# ==================================================================
with tab1:
    bagian("Ringkasan")
    k = st.columns(4)
    with k[0]:
        kpi("Total ulasan", fmt_int(total), "setelah pembersihan", "pesan")
    for kolom, label in zip(k[1:], URUTAN):
        n = int(hitung.get(label, 0))
        with kolom:
            kpi(LABEL[label], fmt_pct(n / total), f"{fmt_int(n)} ulasan",
                label, WARNA_TEKS[label], WARNA_SOFT[label])

    bagian("Sebaran data")
    kiri, kanan = st.columns(2)
    with kiri:
        with kartu("sentimen"):
            judul_kartu("Distribusi sentimen", "Jumlah ulasan pada tiap kategori")
            fig, ax = figur()
            ada = [s for s in URUTAN if s in hitung.index]
            nilai = [int(hitung[s]) for s in ada]
            batang = ax.bar([LABEL[s] for s in ada], nilai,
                            color=[WARNA[s] for s in ada], width=0.55)
            ax.bar_label(batang, labels=[fmt_int(v) for v in nilai], padding=4,
                         color=INK, fontsize=9)
            ax.set_ylim(0, max(nilai) * 1.15)
            ax.set_ylabel("Jumlah ulasan", color=MUTED, fontsize=9)
            tampil_figur(fig)
    with kanan:
        with kartu("rating"):
            judul_kartu("Distribusi rating", "Jumlah ulasan pada tiap bintang")
            fig, ax = figur()
            rating = df["score"].value_counts().sort_index()
            batang = ax.bar(rating.index.astype(str), rating.values,
                            color=PRIMARY, width=0.55)
            ax.bar_label(batang, labels=[fmt_int(v) for v in rating.values], padding=4,
                         color=INK, fontsize=9)
            ax.set_ylim(0, rating.max() * 1.15)
            ax.set_xlabel("Rating", color=MUTED, fontsize=9)
            ax.set_ylabel("Jumlah ulasan", color=MUTED, fontsize=9)
            tampil_figur(fig)

    bagian("Kata kunci")
    wc_kol, kata_kol = st.columns([2, 1])
    with wc_kol:
        with kartu("wordcloud"):
            judul_kartu("Word cloud", "Kata yang paling sering muncul pada sentimen terpilih")
            pilih = st.radio("Sentimen", URUTAN, horizontal=True,
                             label_visibility="collapsed", format_func=lambda s: LABEL[s])
            teks = " ".join(df.loc[df["sentiment"] == pilih, "clean"])
            if teks.strip():
                st.image(buat_wordcloud(teks, pilih))
            else:
                st.info("Tidak ada data untuk sentimen ini.")
    with kata_kol:
        with kartu("katakunci"):
            judul_kartu("Kata paling berpengaruh", f"Sentimen {LABEL[pilih].lower()}")
            daftar = "".join(
                f'<div class="rank"><span class="rank-no">{i}</span>'
                f"<span>{html.escape(str(kata))}</span></div>"
                for i, kata in enumerate(top_words.get(pilih, []), start=1)
            )
            st.markdown(daftar, unsafe_allow_html=True)

    bagian("Contoh ulasan")
    for kolom, label in zip(st.columns(3), URUTAN):
        with kolom:
            with kartu(f"contoh-{label}"):
                st.markdown(
                    f'<div class="tag" style="color:{WARNA_TEKS[label]}">'
                    f'<span class="dot" style="background:{WARNA[label]}"></span>{LABEL[label]}</div>',
                    unsafe_allow_html=True,
                )
                subset = df[df["sentiment"] == label]
                contoh = subset.sample(min(3, len(subset)), random_state=7)
                isi = "".join(
                    f'<div class="review">{html.escape(ringkas(t))}</div>'
                    for t in contoh["content"]
                )
                st.markdown(isi, unsafe_allow_html=True)

# ==================================================================
# TAB 2: EVALUASI MODEL
# ==================================================================
with tab2:
    bagian("Kinerja model")
    e1, e2, e3 = st.columns(3)
    with e1:
        kpi("Akurasi", fmt_pct(metrics["akurasi"], 2), "pada data uji 20%", "target")
    with e2:
        kpi("F1 macro", fmt_pct(metrics["f1_macro"], 2), "rata-rata antar kelas", "aktivitas")
    with e3:
        n_uji = int(metrics["laporan"]["macro avg"]["support"])
        kpi("Data latih / uji", f"{fmt_int(total - n_uji)} / {fmt_int(n_uji)}",
            "pembagian 80:20", "data")

    bagian("Detail evaluasi")
    kiri, kanan = st.columns(2)
    with kiri:
        with kartu("confusion"):
            judul_kartu("Confusion matrix", "Prediksi model dibandingkan label sebenarnya")
            st.image(str(BASE / "model" / "confusion_matrix.png"))
    with kanan:
        with kartu("metrik-kelas"):
            judul_kartu("Metrik per kelas", "Precision, recall, dan F1-score")
            laporan = metrics["laporan"]
            baris = []
            for kelas in URUTAN:
                if kelas in laporan:
                    baris.append({
                        "Kelas": LABEL[kelas],
                        "Precision": laporan[kelas]["precision"],
                        "Recall": laporan[kelas]["recall"],
                        "F1-score": laporan[kelas]["f1-score"],
                        "Support": int(laporan[kelas]["support"]),
                    })
            st.dataframe(
                pd.DataFrame(baris).style.format(
                    {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}
                ),
                hide_index=True,
            )
        with kartu("perbandingan"):
            judul_kartu("Perbandingan algoritma", "Akurasi dan F1 macro pada data uji")
            st.dataframe(muat_perbandingan(), hide_index=True)

    st.markdown(
        '<div class="note">'
        f"Dilatih pada {html.escape(str(metrics['tanggal']))} dengan scikit-learn "
        f"{html.escape(str(metrics['versi_sklearn']))}. "
        "Label sentimen diturunkan dari rating (1-2 negatif, 3 netral, 4-5 positif)."
        "</div>",
        unsafe_allow_html=True,
    )

# ==================================================================
# TAB 3: COBA PREDIKSI
# ==================================================================
with tab3:
    bagian("Prediksi sentimen")
    with kartu("input-prediksi"):
        judul_kartu("Ulasan baru", "Tulis ulasan, lalu model akan memperkirakan sentimennya")
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

            bagian("Hasil")
            h1, h2 = st.columns([1, 2])
            with h1:
                kpi("Hasil prediksi", str(label).upper(), f"keyakinan {fmt_pct(proba.max())}",
                    label if label in IKON else "pesan",
                    WARNA_TEKS.get(label, PRIMARY), WARNA_SOFT.get(label, PRIMARY_SOFT))
            with h2:
                with kartu("probabilitas"):
                    judul_kartu("Probabilitas tiap kelas", "Peluang ulasan masuk ke masing-masing sentimen")
                    fig, ax = figur(5, 2.0)
                    kelas = list(model.classes_)
                    batang = ax.barh([LABEL.get(c, c) for c in kelas], proba,
                                     color=[WARNA.get(c, PRIMARY) for c in kelas], height=0.5)
                    ax.bar_label(batang, labels=[fmt_pct(p) for p in proba], padding=4,
                                 color=INK, fontsize=9)
                    ax.set_xlim(0, 1.15)
                    ax.xaxis.grid(False)
                    ax.yaxis.grid(False)
                    ax.set_xticks([])
                    ax.spines["bottom"].set_visible(False)
                    tampil_figur(fig)

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
                with kartu("pendorong"):
                    judul_kartu("Kata yang paling mendorong hasil ini",
                                f"Teks setelah preprocessing: {bersih}")
                    chips = "".join(f'<span class="chip">{html.escape(str(k))}</span>'
                                    for k, _ in kontribusi)
                    st.markdown(chips, unsafe_allow_html=True)

# ==================================================================
# TAB 4: DATA ULASAN
# ==================================================================
with tab4:
    bagian("Data ulasan")
    with kartu("data"):
        f1, f2 = st.columns([2, 1])
        with f1:
            cari = st.text_input("Cari kata dalam ulasan", placeholder="contoh: error")
        with f2:
            saring = st.multiselect("Sentimen", URUTAN, default=URUTAN,
                                    format_func=lambda s: LABEL[s])

        tampil = df[df["sentiment"].isin(saring)]
        if cari.strip():
            tampil = tampil[tampil["content"].str.contains(cari, case=False, na=False)]

        st.markdown(
            f'<div class="note">Menampilkan {fmt_int(len(tampil))} dari {fmt_int(total)} ulasan</div>',
            unsafe_allow_html=True,
        )
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
    '<div class="footer">UAS Pemrograman Big Data &middot; Analisis Sentimen Ulasan Google Play Store</div>',
    unsafe_allow_html=True,
)