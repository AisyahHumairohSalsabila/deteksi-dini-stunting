"""Fungsi praproses inferensi yang mengikuti notebook penelitian."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


def load_preprocessing_artifacts() -> dict[str, Any]:
    """Memuat scaler, encoder, dan metadata yang diekspor oleh notebook."""
    required = {
        "scaler": MODELS_DIR / "scaler.pkl",
        "encoder_jk": MODELS_DIR / "label_encoder_jk.pkl",
        "encoder_target": MODELS_DIR / "label_encoder_target.pkl",
        "metadata": MODELS_DIR / "metadata.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Artefak praproses belum tersedia: {', '.join(missing)}")
    return {
        "scaler": joblib.load(required["scaler"]),
        "encoder_jk": joblib.load(required["encoder_jk"]),
        "encoder_target": joblib.load(required["encoder_target"]),
        "metadata": pd.read_json(required["metadata"], typ="series").to_dict(),
    }


def align_to_model(df_scaled: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Menyamakan nama kolom hasil scaler dengan nama fitur yang diharapkan model.

    Catatan penting: berkas ``scaler.pkl`` dilatih dengan nama kolom
    ``jk_encoded``, tetapi ``model_terbaik.pkl`` dan ``model_xgboost.pkl``
    ternyata dilatih dengan DataFrame yang kolomnya masih bernama ``jk``
    (isinya tetap nilai hasil label encoding, hanya nama kolomnya berbeda).
    scikit-learn versi baru memvalidasi nama fitur secara ketat sehingga
    ketidaksesuaian ini menyebabkan galat saat predict(). Fungsi ini
    mengganti nama kolom secara berurutan (posisi fitur tetap sama) agar
    cocok dengan ``model.feature_names_in_`` tanpa perlu melatih ulang model.
    """
    expected = getattr(model, "feature_names_in_", None)
    if expected is None or len(expected) != df_scaled.shape[1]:
        return df_scaled
    return df_scaled.set_axis(list(expected), axis=1)


def prepare_input(umur: float, jenis_kelamin: str, berat: float, tinggi: float) -> pd.DataFrame:
    """Membentuk fitur terstandardisasi sesuai urutan yang dipakai saat scaler dilatih.

    Catatan: metadata.json menyimpan nama fitur mentah (['umur','jk','berat','tinggi']),
    tetapi scaler.pkl sebenarnya dilatih dengan nama kolom hasil encoding
    (['umur','jk_encoded','berat','tinggi']). Nama kolom di sini HARUS mengikuti
    scaler, bukan metadata, kalau tidak scaler.transform() akan gagal karena
    scikit-learn memvalidasi nama fitur secara ketat.
    """
    artifacts = load_preprocessing_artifacts()
    encoder_jk = artifacts["encoder_jk"]
    scaler = artifacts["scaler"]
    fitur_scaler = list(getattr(scaler, "feature_names_in_", ["umur", "jk_encoded", "berat", "tinggi"]))
    if jenis_kelamin not in encoder_jk.classes_:
        raise ValueError("Jenis kelamin harus menggunakan nilai yang tersedia pada model.")
    raw_input = pd.DataFrame(
        [[umur, encoder_jk.transform([jenis_kelamin])[0], berat, tinggi]],
        columns=fitur_scaler,
    )
    return pd.DataFrame(scaler.transform(raw_input), columns=fitur_scaler)
