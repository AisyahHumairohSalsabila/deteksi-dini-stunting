"""Fungsi inferensi status gizi: gabungan Z-Score TB/U baku WHO + model ML.

Kenapa hibrida, bukan model ML saja:
Model machine learning (model_terbaik.pkl) hanya pernah dilatih dengan label
"Stunted" dan "Severely Stunted", karena data primer 13 Posyandu tidak
memiliki satu pun contoh balita berkategori "Normal" atau "Tinggi" (lihat
Tabel 4.3 skripsi). Akibatnya model_terbaik.pkl TIDAK PERNAH bisa memprediksi
dua kategori tersebut, walau data masukan jelas menunjukkan pertumbuhan
normal/tinggi (mis. balita tinggi 110 cm tetap dipaksa "Stunted").

Solusinya bukan menambal tampilan, tetapi memakai jalur klasifikasi yang
sesuai domain masing-masing:
1. Hitung Z-Score TB/U dari rumus baku WHO (utils/zscore.py) -- ini persis
   metode yang dipakai notebook (label_stunting()) untuk membentuk label
   ground-truth, sehingga selalu benar untuk keempat kategori.
2. Jika Z-Score menunjukkan kategori Normal atau Tinggi, kategori itulah
   yang ditampilkan sebagai status akhir (di luar cakupan yang pernah
   dipelajari model).
3. Jika Z-Score menunjukkan kategori Stunted atau Severely Stunted, model
   Random Forest terbaik dipakai untuk menajamkan keputusan di antara kedua
   kategori tersebut -- ini persis cakupan yang dievaluasi pada Bab IV
   skripsi (Accuracy 92,8%, F1-Score makro 0,89 pada dua kategori ini).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import numpy as np

from utils.preprocessing import MODELS_DIR, align_to_model, load_preprocessing_artifacts, prepare_input
from utils.zscore import hitung_zscore_tbu, klasifikasi_zscore


@lru_cache(maxsize=1)
def load_model_artifacts() -> dict[str, Any]:
    """Memuat model terbaik dan seluruh artefak yang diperlukan untuk inferensi.

    Catatan: notebook (Listing "Penyimpanan Model Terbaik") HANYA pernah
    menyimpan berkas ``model_terbaik.pkl``. Berkas lama bernama
    ``random_forest.pkl`` bukan hasil ekspor notebook ini sehingga tidak
    dipakai lagi di sini, agar prediksi selalu konsisten dengan model yang
    dilaporkan pada Bab IV skripsi.
    """
    model_path = MODELS_DIR / "model_terbaik.pkl"
    if not model_path.exists():
        raise FileNotFoundError("File models/model_terbaik.pkl tidak ditemukan.")
    artifacts = load_preprocessing_artifacts()
    artifacts["model"] = joblib.load(model_path)
    return artifacts


def _prediksi_model_ml(umur: float, jenis_kelamin: str, berat: float, tinggi: float) -> dict[str, Any]:
    """Prediksi murni dari model ML (hanya valid untuk 2 kategori stunting)."""
    artifacts = load_model_artifacts()
    model = artifacts["model"]
    encoder_target = artifacts["encoder_target"]
    prepared = prepare_input(umur, jenis_kelamin, berat, tinggi)
    # Kolom hasil scaler bernama 'jk_encoded', tetapi model_terbaik.pkl dilatih
    # dengan kolom bernama 'jk'. Diselaraskan dulu agar validasi nama fitur
    # scikit-learn tidak menggagalkan prediksi.
    prepared = align_to_model(prepared, model)
    predicted_code = int(model.predict(prepared)[0])
    probabilities = model.predict_proba(prepared)[0]
    labels = encoder_target.inverse_transform(model.classes_.astype(int))
    probability_map = {label: float(value) for label, value in zip(labels, probabilities)}
    return {
        "status": str(encoder_target.inverse_transform([predicted_code])[0]),
        "probabilitas": probability_map,
        "confidence": float(np.max(probabilities)),
    }


def predict_stunting(umur: float, jenis_kelamin: str, berat: float, tinggi: float) -> dict[str, Any]:
    """Menghasilkan status gizi akhir, Z-Score TB/U, probabilitas, dan confidence.

    ``sumber_status`` menandai apakah status akhir ditentukan langsung dari
    Z-Score WHO ("zscore_who") atau dipertajam oleh model ML ("model_ml").
    """
    zscore = hitung_zscore_tbu(umur, jenis_kelamin, tinggi)
    kategori_zscore = klasifikasi_zscore(zscore)

    if kategori_zscore in ("Normal", "Tinggi"):
        # Di luar cakupan yang pernah dipelajari model -> Z-Score WHO menjadi acuan.
        return {
            "status": kategori_zscore,
            "zscore": zscore,
            "sumber_status": "zscore_who",
            "probabilitas": {kategori_zscore: 1.0},
            "confidence": 1.0,
        }

    # Berada pada rentang Stunted/Severely Stunted -> pertajam dengan model ML
    # yang memang dievaluasi khusus untuk membedakan dua kategori ini.
    hasil_model = _prediksi_model_ml(umur, jenis_kelamin, berat, tinggi)
    hasil_model["zscore"] = zscore
    hasil_model["sumber_status"] = "model_ml"
    return hasil_model
