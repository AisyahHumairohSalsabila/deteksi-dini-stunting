"""Perhitungan Z-Score Tinggi Badan menurut Umur (TB/U) berbasis standar resmi WHO.

Sumber data M (median) dan S (koefisien variasi) diunduh langsung dari tabel
resmi WHO Child Growth Standards (cdn.who.int/.../length-height-for-age),
indikator Length-for-age (0-2 tahun, panjang badan diukur berbaring) dan
Height-for-age (2-5 tahun, tinggi badan diukur berdiri), untuk laki-laki dan
perempuan usia 0-60 bulan. Parameter L (Box-Cox power) bernilai 1 untuk
seluruh usia pada indikator TB/U sehingga rumus LMS
    Z = [(X/M)^L - 1] / (L * S)
dapat disederhanakan menjadi
    Z = (X - M) / (M * S)

Alasan modul ini dibutuhkan: model machine learning (model_terbaik.pkl) hanya
pernah dilatih dengan label "Stunted" dan "Severely Stunted" karena data
primer 13 Posyandu tidak memiliki contoh balita berkategori "Normal" maupun
"Tinggi". Akibatnya model TIDAK PERNAH bisa memprediksi dua kategori tersebut
walau data masukan jelas menunjukkan pertumbuhan normal/tinggi. Perhitungan
Z-Score langsung berbasis rumus baku WHO ini dipakai sebagai penentu status
gizi yang sebenarnya (persis logika label_stunting() pada notebook, lihat
Tabel 3.6 skripsi), sehingga tidak bergantung pada apa yang "pernah dilihat"
model saat pelatihan.
"""
from __future__ import annotations

# month -> (M, S). Bulan 0-23 memakai tabel Length-for-age (panjang, berbaring),
# bulan 24-60 memakai tabel Height-for-age (tinggi, berdiri), sesuai konvensi
# pengukuran resmi WHO/Kemenkes RI pada usia tersebut.
_LMS_BOYS: dict[int, tuple[float, float]] = {
    0: (49.8842, 0.03795), 1: (54.7244, 0.03557), 2: (58.4249, 0.03424),
    3: (61.4292, 0.03328), 4: (63.8860, 0.03257), 5: (65.9026, 0.03204),
    6: (67.6236, 0.03165), 7: (69.1645, 0.03139), 8: (70.5994, 0.03124),
    9: (71.9687, 0.03117), 10: (73.2812, 0.03118), 11: (74.5388, 0.03125),
    12: (75.7488, 0.03137), 13: (76.9186, 0.03154), 14: (78.0497, 0.03174),
    15: (79.1458, 0.03197), 16: (80.2113, 0.03222), 17: (81.2487, 0.03250),
    18: (82.2587, 0.03279), 19: (83.2418, 0.03310), 20: (84.1996, 0.03342),
    21: (85.1348, 0.03376), 22: (86.0477, 0.03410), 23: (86.9410, 0.03445),
    24: (87.1161, 0.03507), 25: (87.9720, 0.03542), 26: (88.8065, 0.03576),
    27: (89.6197, 0.03610), 28: (90.4120, 0.03642), 29: (91.1828, 0.03674),
    30: (91.9327, 0.03704), 31: (92.6631, 0.03733), 32: (93.3753, 0.03761),
    33: (94.0711, 0.03787), 34: (94.7532, 0.03812), 35: (95.4236, 0.03836),
    36: (96.0835, 0.03858), 37: (96.7337, 0.03879), 38: (97.3749, 0.03900),
    39: (98.0073, 0.03919), 40: (98.6310, 0.03937), 41: (99.2459, 0.03954),
    42: (99.8515, 0.03971), 43: (100.4485, 0.03986), 44: (101.0374, 0.04002),
    45: (101.6186, 0.04016), 46: (102.1933, 0.04031), 47: (102.7625, 0.04045),
    48: (103.3273, 0.04059), 49: (103.8886, 0.04073), 50: (104.4473, 0.04086),
    51: (105.0041, 0.04100), 52: (105.5596, 0.04113), 53: (106.1138, 0.04126),
    54: (106.6668, 0.04139), 55: (107.2188, 0.04152), 56: (107.7697, 0.04165),
    57: (108.3198, 0.04177), 58: (108.8689, 0.04190), 59: (109.4170, 0.04202),
    60: (109.9638, 0.04214),
}

_LMS_GIRLS: dict[int, tuple[float, float]] = {
    0: (49.1477, 0.03790), 1: (53.6872, 0.03640), 2: (57.0673, 0.03568),
    3: (59.8029, 0.03520), 4: (62.0899, 0.03486), 5: (64.0301, 0.03463),
    6: (65.7311, 0.03448), 7: (67.2873, 0.03441), 8: (68.7498, 0.03440),
    9: (70.1435, 0.03444), 10: (71.4818, 0.03452), 11: (72.7710, 0.03464),
    12: (74.0150, 0.03479), 13: (75.2176, 0.03496), 14: (76.3817, 0.03514),
    15: (77.5099, 0.03534), 16: (78.6055, 0.03555), 17: (79.6710, 0.03576),
    18: (80.7079, 0.03598), 19: (81.7182, 0.03620), 20: (82.7036, 0.03643),
    21: (83.6654, 0.03666), 22: (84.6040, 0.03688), 23: (85.5202, 0.03711),
    24: (85.7153, 0.03764), 25: (86.5904, 0.03786), 26: (87.4462, 0.03808),
    27: (88.2830, 0.03830), 28: (89.1004, 0.03851), 29: (89.8991, 0.03872),
    30: (90.6797, 0.03893), 31: (91.4430, 0.03913), 32: (92.1906, 0.03933),
    33: (92.9239, 0.03952), 34: (93.6444, 0.03971), 35: (94.3533, 0.03989),
    36: (95.0515, 0.04006), 37: (95.7399, 0.04024), 38: (96.4187, 0.04041),
    39: (97.0885, 0.04057), 40: (97.7493, 0.04073), 41: (98.4015, 0.04089),
    42: (99.0448, 0.04105), 43: (99.6795, 0.04120), 44: (100.3058, 0.04135),
    45: (100.9238, 0.04150), 46: (101.5337, 0.04164), 47: (102.1360, 0.04179),
    48: (102.7312, 0.04193), 49: (103.3197, 0.04206), 50: (103.9021, 0.04220),
    51: (104.4786, 0.04233), 52: (105.0494, 0.04246), 53: (105.6148, 0.04259),
    54: (106.1748, 0.04272), 55: (106.7295, 0.04285), 56: (107.2788, 0.04298),
    57: (107.8227, 0.04310), 58: (108.3613, 0.04322), 59: (108.8948, 0.04334),
    60: (109.4233, 0.04347),
}


def hitung_zscore_tbu(umur_bulan: float, jenis_kelamin: str, tinggi_cm: float) -> float:
    """Menghitung Z-Score TB/U sesuai rumus resmi WHO (L=1 untuk indikator ini)."""
    bulan = min(60, max(0, round(umur_bulan)))
    tabel = _LMS_BOYS if jenis_kelamin.upper().startswith("L") else _LMS_GIRLS
    m, s = tabel[bulan]
    return (tinggi_cm - m) / (m * s)


def klasifikasi_zscore(z: float) -> str:
    """Aturan pelabelan identik dengan label_stunting() pada notebook (Tabel 3.6)."""
    if z < -3:
        return "Severely Stunted"
    if z < -2:
        return "Stunted"
    if z <= 3:
        return "Normal"
    return "Tinggi"
