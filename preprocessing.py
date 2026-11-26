import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Kata negasi dipertahankan karena penting untuk sentimen ("tidak bagus")
NEGASI = {"tidak", "tak", "bukan", "belum", "jangan", "kurang"}
STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) - NEGASI

SLANG = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak",
    "tdk": "tidak", "yg": "yang", "bgt": "banget", "udah": "sudah", "udh": "sudah",
    "sdh": "sudah", "aja": "saja", "krn": "karena", "dgn": "dengan", "sy": "saya",
    "jg": "juga", "lg": "lagi", "trs": "terus", "blm": "belum", "tp": "tapi",
    "dr": "dari", "utk": "untuk", "bs": "bisa", "kyk": "seperti", "emang": "memang",
    "bagu": "bagus", "mantul": "mantap", "oke": "ok",
}


def clean_text(teks):
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)
    teks = re.sub(r"(.)\1{2,}", r"\1", teks)      # bagusssss -> bagus
    teks = re.sub(r"[^a-z\s]", " ", teks)         # hapus angka, emoji, simbol
    kata = [SLANG.get(k, k) for k in teks.split()]
    kata = [k for k in kata if k not in STOPWORDS and len(k) > 1]
    return " ".join(kata)
