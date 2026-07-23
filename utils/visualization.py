"""Pemuatan data primer dan visualisasi Plotly untuk tahap deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.preprocessing import ROOT_DIR

DATASET_DIR = ROOT_DIR / "DATASET-PRIMER"

WARNA = {"Normal": "#1c9b72", "Stunted": "#e08b2d", "Severely Stunted": "#c74b50", "Tinggi": "#2477b9"}
STATUS_ORDER = ["Severely Stunted", "Stunted", "Normal", "Tinggi"]

# Batas tahun yang masuk akal untuk data pemeriksaan (mencegah entri tanggal
# yang salah ketik/parsing menghasilkan tahun ekstrem seperti 1900 atau 2099).
TAHUN_MIN = 2000
TAHUN_MAX_OFFSET = 1  # tahun berjalan + 1, untuk mengakomodasi input mendekati akhir tahun


def _baca_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".xls":
        try:
            return pd.read_html(path)[0]
        except ValueError:
            return pd.read_excel(path, engine="xlrd")
    return pd.read_excel(path, engine="openpyxl")


def load_primary_data() -> pd.DataFrame:
    """Membaca seluruh Excel primer dan hanya menyisakan kolom non-identitas."""
    frames: list[pd.DataFrame] = []
    pii = ("NIK", "NAMA", "AYAH", "IBU", "ORTU", "ALAMAT", "RT", "RW", "HP", "TELP")
    for path in sorted(DATASET_DIR.rglob("*")):
        if path.suffix.lower() not in {".xls", ".xlsx", ".csv"} or path.name.startswith("~$"):
            continue
        try:
            frame = _baca_file(path)
            frame.columns = frame.columns.astype(str).str.strip()
            remove = [c for c in frame.columns if any(word in c.upper() for word in pii)]
            frames.append(frame.drop(columns=remove, errors="ignore"))
        except Exception:
            continue

    if not frames:
        raise ValueError("Tidak ada file dataset primer yang dapat dibaca.")

    data = pd.concat(frames, ignore_index=True)
    data.columns = (
        data.columns.astype(str).str.strip().str.lower()
        .str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False).str.replace(".", "", regex=False)
    )

    # dayfirst=True: berkas sumber memakai format tanggal Indonesia
    # (dd/mm/yyyy). Tanpa ini, pandas bisa salah menafsirkan tanggal
    # sehingga banyak baris gagal parse (NaT) dan kolom "tahun" berakhir
    # kosong semua -> grafik tren hilang meski kolomnya ada.
    data["tgl_lahir"] = pd.to_datetime(data["tgl_lahir"], errors="coerce", dayfirst=True)
    data["tanggal_pengukuran"] = pd.to_datetime(data["tanggal_pengukuran"], errors="coerce", dayfirst=True)

    data["umur"] = np.floor((data["tanggal_pengukuran"] - data["tgl_lahir"]).dt.days / 30.44)
    data.loc[data["umur"] < 0, "umur"] = np.nan

    # PENTING: kolom tahun yang dipakai di seluruh dashboard SELALU dibangun
    # dari tanggal pemeriksaan (tanggal_pengukuran), bukan dari kolom mentah
    # semacam "tahun_data" pada berkas sumber — kolom itu ternyata berisi
    # kode/ID (mis. "Daftar-Status-Gizi-PMT-20042604-0006" atau nama
    # berkas/folder lain), bukan tahun murni, sehingga tidak aman dipakai
    # langsung untuk grafik tren.
    tahun_batas_atas = pd.Timestamp.now().year + TAHUN_MAX_OFFSET
    tahun = data["tanggal_pengukuran"].dt.year
    data["tahun"] = tahun.where(tahun.between(TAHUN_MIN, tahun_batas_atas)).astype("Int64")

    # Diagnostik sementara: hapus/nonaktifkan setelah penyebab kolom tahun
    # kosong (bila terjadi) sudah dipastikan dan diperbaiki.
    total_baris = len(data)
    nat_tanggal = int(data["tanggal_pengukuran"].isna().sum())
    nan_tahun = int(data["tahun"].isna().sum())
    print(f"[load_primary_data] total baris: {total_baris}")
    print(f"[load_primary_data] tanggal_pengukuran gagal parse (NaT): {nat_tanggal}")
    print(f"[load_primary_data] tahun kosong (NaN): {nan_tahun}")

    data["status_stunting"] = pd.cut(
        data["zs_tb_u"], [-np.inf, -3, -2, 3, np.inf],
        labels=STATUS_ORDER, right=False,
    ).astype("string")
    data.loc[data["zs_tb_u"].eq(3), "status_stunting"] = "Normal"

    return data


def build_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Menyusun ringkasan yang tidak memuat identitas anak."""
    counts = data["status_stunting"].value_counts()
    stunting = int(counts.get("Stunted", 0) + counts.get("Severely Stunted", 0))
    return {
        "total_data": len(data),
        "total_anak": len(data),
        "normal": int(counts.get("Normal", 0)),
        "stunting": stunting,
        "persentase": (stunting / len(data) * 100) if len(data) else 0,
    }


def base_layout(fig: go.Figure, title: str) -> go.Figure:
    # Font disamakan dengan font UI utama ("Plus Jakarta Sans" di
    # assets/style.css). st.plotly_chart merender grafik di dalam iframe
    # terpisah yang terisolasi dari CSS halaman utama, jadi font grafik
    # WAJIB diatur di sini (bukan lewat CSS) agar tampilannya konsisten
    # dengan teks lain di dashboard.
    fig.update_layout(
        title=title, template="plotly_white", margin=dict(l=20, r=20, t=55, b=20),
        font=dict(family="'Plus Jakarta Sans', Arial, sans-serif", color="#244058"),
        paper_bgcolor="white", plot_bgcolor="white", legend_title_text="",
    )
    return fig


def _status_order(data: pd.DataFrame) -> list[str]:
    return [s for s in STATUS_ORDER if s in data["status_stunting"].unique()]


def build_data_figures(data: pd.DataFrame) -> dict[str, go.Figure]:
    """Grafik eksplorasi inti: komposisi status + distribusi antropometri."""
    order = _status_order(data)
    figures: dict[str, go.Figure] = {}

    figures["pie"] = base_layout(
        px.pie(data, names="status_stunting", color="status_stunting",
               category_orders={"status_stunting": order}, color_discrete_map=WARNA, hole=.55),
        "Komposisi Status Stunting",
    )
    figures["bar"] = base_layout(
        px.histogram(data, x="status_stunting", color="status_stunting",
                     category_orders={"status_stunting": order}, color_discrete_map=WARNA),
        "Jumlah Data per Status",
    )
    figures["umur"] = base_layout(px.histogram(data, x="umur", nbins=24, color_discrete_sequence=["#2477b9"]), "Distribusi Umur Balita")
    figures["berat"] = base_layout(px.histogram(data, x="berat", nbins=24, color_discrete_sequence=["#1c9b72"]), "Distribusi Berat Badan")
    figures["tinggi"] = base_layout(px.histogram(data, x="tinggi", nbins=24, color_discrete_sequence=["#2477b9"]), "Distribusi Tinggi Badan")
    figures["scatter"] = base_layout(
        px.scatter(data, x="umur", y="tinggi", color="status_stunting", color_discrete_map=WARNA, opacity=.65),
        "Hubungan Umur dan Tinggi Badan",
    )
    figures["box"] = base_layout(
        px.box(data, x="jk", y="berat", color="jk", color_discrete_sequence=["#2477b9", "#1c9b72"]),
        "Sebaran Berat Badan menurut Jenis Kelamin",
    )

    numeric = data[["umur", "berat", "tinggi", "zs_tb_u"]].corr().round(2)
    figures["heatmap"] = base_layout(
        px.imshow(numeric, text_auto=True, color_continuous_scale="Blues", zmin=-1, zmax=1),
        "Korelasi Variabel Antropometri",
    )
    return figures


def build_wilayah_figures(data: pd.DataFrame, wilayah_column: str | None) -> dict[str, go.Figure]:
    """Grafik jumlah kasus per wilayah (kecamatan/puskesmas/desa) — kosong jika kolom tidak ada."""
    if not wilayah_column or wilayah_column not in data.columns:
        return {}

    order = _status_order(data)
    counted = (
        data.groupby([wilayah_column, "status_stunting"], observed=True).size()
        .reset_index(name="jumlah")
    )
    fig = px.bar(
        counted, x=wilayah_column, y="jumlah", color="status_stunting",
        category_orders={"status_stunting": order}, color_discrete_map=WARNA, barmode="stack",
    )
    label = wilayah_column.replace("_", " ").title()
    return {"wilayah": base_layout(fig, f"Jumlah Kasus per {label}")}


def build_trend_figures(data: pd.DataFrame, tahun_column: str | None) -> dict[str, go.Figure]:
    """Grafik tren jumlah kasus per tahun — kosong jika kolom tahun tidak ada.

    Sengaja memakai pd.to_numeric(..., errors="coerce") alih-alih
    .astype(int) langsung: apa pun isi kolom tahun (idealnya kolom "tahun"
    bersih dari load_primary_data), nilai yang bukan angka akan otomatis
    menjadi kosong (bukan membuat aplikasi error/crash).
    """
    if not tahun_column or tahun_column not in data.columns:
        return {}

    working = data.copy()
    working[tahun_column] = pd.to_numeric(working[tahun_column], errors="coerce")
    working = working.dropna(subset=[tahun_column])
    if working.empty:
        return {}

    working[tahun_column] = working[tahun_column].astype(int)
    order = _status_order(working)
    counted = (
        working.groupby([tahun_column, "status_stunting"], observed=True).size()
        .reset_index(name="jumlah").sort_values(tahun_column)
    )
    fig = px.line(
        counted, x=tahun_column, y="jumlah", color="status_stunting", markers=True,
        category_orders={"status_stunting": order}, color_discrete_map=WARNA,
    )
    fig.update_xaxes(title="Tahun", dtick=1, tickformat="d")
    fig.update_yaxes(title="Jumlah")
    return {"tren": base_layout(fig, "Tren Jumlah Kasus per Tahun")}
