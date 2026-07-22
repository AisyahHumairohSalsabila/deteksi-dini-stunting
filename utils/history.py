"""Penyimpanan ringan riwayat hasil deteksi tanpa database.

Cara kerja:
- Setiap hasil deteksi disimpan ke file CSV lokal (data/riwayat_prediksi.csv)
  dan ke st.session_state supaya langsung terlihat di halaman yang sama.
- Karena disimpan ke file di disk, riwayat akan tetap ada selama proses
  aplikasi TIDAK dijalankan ulang dari awal (redeploy) di layanan hosting
  yang disknya sementara (ephemeral), misalnya Streamlit Community Cloud.
  Jika aplikasi dijalankan di server/VM sendiri (disk permanen), riwayat
  akan tetap tersimpan meskipun aplikasi di-restart.
- Sebagai jaring pengaman tambahan, pengguna bisa mengunduh riwayat sebagai
  CSV lalu mengunggahnya kembali (menu "Muat riwayat sebelumnya") jika suatu
  saat file lokal hilang karena redeploy.

Jika ke depan dibutuhkan riwayat yang benar-benar permanen di banyak
perangkat/pengguna sekaligus, opsi paling sederhana adalah menghubungkan ke
Google Sheets (via gspread) atau database gratis seperti Supabase/SQLite
dengan disk permanen.
"""
from pathlib import Path

import pandas as pd

HISTORY_PATH = Path(__file__).parents[1] / "data" / "riwayat_prediksi.csv"
COLUMNS = [
    "waktu",
    "umur_bulan",
    "jenis_kelamin",
    "berat_kg",
    "tinggi_cm",
    "status",
    "tingkat_keyakinan",
]


def load_history() -> pd.DataFrame:
    """Muat riwayat dari file lokal. Mengembalikan DataFrame kosong jika belum ada."""
    if HISTORY_PATH.exists():
        try:
            return pd.read_csv(HISTORY_PATH)
        except Exception:
            return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(columns=COLUMNS)


def save_record(record: dict) -> None:
    """Tambahkan satu baris hasil deteksi ke file riwayat lokal (best effort)."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([record], columns=COLUMNS)
    try:
        if HISTORY_PATH.exists():
            row.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
        else:
            row.to_csv(HISTORY_PATH, mode="w", header=True, index=False)
    except Exception:
        # Jika disk bersifat read-only (beberapa layanan hosting gratis),
        # riwayat tetap tersimpan di session_state selama sesi berjalan.
        pass


def merge_uploaded_history(uploaded_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan riwayat yang diunggah pengguna dengan riwayat yang sudah ada, hapus duplikat."""
    combined = pd.concat([current_df, uploaded_df], ignore_index=True)
    return combined.drop_duplicates()
