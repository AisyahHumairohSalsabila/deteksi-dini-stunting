"""Dashboard visualisasi data primer untuk pemantauan Posyandu."""
import re

import pandas as pd
import streamlit as st

from utils.visualization import (
    build_data_figures,
    build_model_figures,
    build_summary,
    build_trend_figures,
    build_wilayah_figures,
    load_primary_data,
)


@st.cache_data(show_spinner="Memuat data dashboard...")
def get_data():
    return load_primary_data()


def find_column(df: pd.DataFrame, *candidates: str):
    """Mengambil kolom yang namanya sesuai kandidat, tanpa peka kapitalisasi."""
    lookup = {column.strip().lower(): column for column in df.columns}
    return next((lookup[name.strip().lower()] for name in candidates if name.strip().lower() in lookup), None)


def extract_valid_years(series: pd.Series) -> list:
    """Ambil hanya nilai tahun yang valid (format 19xx/20xx) dari kolom tahun.

    Ini mencegah nilai non-tahun (misalnya nama file seperti "PMT" yang
    tercampur di kolom tahun) ikut muncul sebagai opsi filter.
    """
    years = set()
    for value in series.dropna():
        match = re.search(r"(19|20)\d{2}", str(value).strip())
        if match:
            years.add(int(match.group()))
    return sorted(years)


def section_heading(number: str, title: str, subtitle: str, description: str) -> None:
    st.markdown(
        f"""<section class="section-heading"><span class="section-number">{number}</span>
        <div><h2>{title}</h2><p class="section-subtitle">{subtitle}</p>
        <p class="section-description">{description}</p></div></section>""",
        unsafe_allow_html=True,
    )


def render_chart(figure, description: str, insight: str) -> None:
    """Menampilkan grafik bersama konteks pembacaan untuk dashboard."""
    st.plotly_chart(figure, use_container_width=True)
    st.markdown(f"<p class='chart-description'>{description}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='insight'>Insight: {insight}</p>", unsafe_allow_html=True)


try:
    data = get_data()
    col_tahun = find_column(data, "tahun", "year", "tahun_data")
    # Dataset ePPGBM menggunakan singkatan/variasi nama kolom berikut.
    col_kecamatan = find_column(data, "kecamatan", "kec")
    col_puskesmas = find_column(data, "puskesmas", "pukesmas")
    col_desa = find_column(data, "desa", "kelurahan", "desa_kel")
    col_posyandu = find_column(data, "posyandu")
    filtered = data.copy()

    st.markdown(
        """<section class="hero dashboard-hero"><div class="eyebrow">MONITORING PERTUMBUHAN BALITA</div>
        <h1>Dashboard Visualisasi</h1>
        <p>Ringkasan data pemeriksaan balita untuk membantu pemantauan pertumbuhan
        dan penentuan tindak lanjut di wilayah layanan.</p></section>""",
        unsafe_allow_html=True,
    )

    available_filters = [column for column in (col_tahun, col_kecamatan, col_puskesmas) if column]
    if available_filters:
        #st.markdown("<div class='filter-panel'><p>Filter data</p>", unsafe_allow_html=True)
        filter_cols = st.columns(len(available_filters), gap="large")
        for container, column in zip(filter_cols, available_filters):
            with container:
                label = column.replace("_", " ").title()
                is_tahun = column == col_tahun

                if is_tahun:
                    # Khusus kolom tahun: hanya tampilkan nilai tahun yang valid
                    # (angka 19xx/20xx), bukan teks lain yang mungkin tercampur.
                    options = extract_valid_years(data[column])
                else:
                    options = sorted(data[column].dropna().unique().tolist())

                selected = st.multiselect(label, options, placeholder=f"Semua {label.lower()}")

                if selected:
                    if is_tahun:
                        tahun_series = (
                            filtered[column]
                            .astype(str)
                            .str.extract(r"((?:19|20)\d{2})")[0]
                        )
                        filtered = filtered[tahun_series.astype(float).isin(selected)]
                    else:
                        filtered = filtered[filtered[column].isin(selected)]
        st.markdown("</div>", unsafe_allow_html=True)

    if filtered.empty:
        st.warning("Tidak ada data pada kombinasi filter ini. Silakan ubah pilihan filter.")
        st.stop()

    summary = build_summary(filtered)
    kpis = [
        ("Total Balita", f"{summary['total_data']:,}", "Data pemeriksaan aktif"),
        ("Persentase Stunting", f"{summary['persentase']:.1f}%", "Perlu pemantauan"),
        ("Total Stunting", f"{summary['stunting']:,}", "Kasus prioritas"),
        ("Total Normal", f"{summary['normal']:,}", "Pertumbuhan sesuai"),
        ("Jumlah Posyandu", f"{filtered[col_posyandu].nunique():,}" if col_posyandu else "–", "Unit pemantauan"),
        ("Jumlah Desa", f"{filtered[col_desa].nunique():,}" if col_desa else "–", "Wilayah desa"),
        ("Jumlah Puskesmas", f"{filtered[col_puskesmas].nunique():,}" if col_puskesmas else "–", "Fasilitas layanan"),
    ]
    st.markdown("<div class='kpi-label'>RINGKASAN DATA</div>", unsafe_allow_html=True)
    for start in range(0, len(kpis), 4):
        row_cols = st.columns(4, gap="medium")
        for column, (label, value, caption) in zip(row_cols, kpis[start:start + 4]):
            column.markdown(
                f"<article class='metric-card'><span>{label}</span><strong>{value}</strong><p>{caption}</p></article>",
                unsafe_allow_html=True,
            )

    figures = build_data_figures(filtered)
    wilayah_column = col_kecamatan or col_puskesmas or col_desa
    wilayah_figures = build_wilayah_figures(filtered, wilayah_column)
    trend_figures = build_trend_figures(filtered, col_tahun)

    tab_names = ["Ringkasan"]
    if wilayah_figures:
        tab_names.append("Wilayah")
    tab_names += ["Tren & Demografi", "Antropometri", "Model"]
    tabs = st.tabs(tab_names)
    tab_map = dict(zip(tab_names, tabs))

    with tab_map["Ringkasan"]:
        section_heading("01", "Karakteristik Data", "Gambaran umum pemeriksaan balita",
                         "Komposisi status pertumbuhan membantu menentukan kelompok yang perlu mendapat perhatian lebih awal.")
        left, right = st.columns(2, gap="large")
        with left:
            render_chart(figures["pie"], "Proporsi setiap status pertumbuhan pada data yang dipilih.",
                         "kelompok dengan jumlah terbesar perlu dipantau melalui layanan rutin.")
        with right:
            render_chart(figures["bar"], "Jumlah pemeriksaan untuk setiap status pertumbuhan.",
                         "gunakan jumlah kasus untuk membantu penentuan prioritas tindak lanjut.")

    if "Wilayah" in tab_map:
        with tab_map["Wilayah"]:
            section_heading("02", "Sebaran Wilayah", "Jumlah kasus per wilayah layanan",
                             "Bandingkan wilayah untuk membantu penentuan prioritas kunjungan dan penyuluhan.")
            render_chart(wilayah_figures["wilayah"], "Jumlah kasus per wilayah menurut status pertumbuhan.",
                         "wilayah dengan proporsi stunting tinggi perlu perhatian lebih dahulu.")

    with tab_map["Tren & Demografi"]:
        section_heading("03", "Tren & Demografi", "Pola dari waktu ke waktu dan menurut kelompok",
                         "Grafik berikut memperlihatkan tren tahunan (bila tersedia) serta sebaran umur balita.")
        if trend_figures:
            render_chart(trend_figures["tren"], "Jumlah kasus per tahun menurut status pertumbuhan.",
                         "tren naik pada satu status menandakan perlu evaluasi program di periode tersebut.")
        else:
            st.info("Kolom tahun tidak ditemukan pada data, grafik tren dilewati.")
        render_chart(figures["umur"], "Sebaran umur balita pada saat pemeriksaan.",
                     "data yang terkonsentrasi pada umur tertentu dapat membantu pengaturan kegiatan pemantauan.")

    with tab_map["Antropometri"]:
        section_heading("04", "Distribusi Antropometri", "Sebaran berat, tinggi, dan hubungan antar variabel",
                         "Grafik berikut memperlihatkan pola pengukuran yang tercatat pada data terpilih.")
        first, second = st.columns(2, gap="large")
        with first:
            render_chart(figures["berat"], "Sebaran berat badan balita dalam kilogram.",
                         "amati nilai yang jauh dari sebaran utama untuk memastikan pencatatan dan pemeriksaan lanjutan.")
        with second:
            render_chart(figures["tinggi"], "Sebaran tinggi badan balita dalam sentimeter.",
                         "pola tinggi badan perlu dibaca bersama umur dan hasil pemeriksaan langsung.")
        left, right = st.columns(2, gap="large")
        with left:
            render_chart(figures["scatter"], "Hubungan umur dan tinggi badan menurut status pertumbuhan.",
                         "setiap titik perlu dibaca bersama riwayat pertumbuhan balita.")
        with right:
            render_chart(figures["box"], "Sebaran berat badan menurut jenis kelamin.",
                         "perbedaan sebaran adalah konteks pemantauan, bukan diagnosis individual.")
        render_chart(figures["heatmap"], "Kekuatan hubungan antarvariabel antropometri.",
                     "hubungan statistik membantu memahami pola data, tetapi keputusan layanan tetap melalui pemeriksaan tenaga kesehatan.")

    with tab_map["Model"]:
        section_heading("05", "Evaluasi Model", "Kelayakan penggunaan sebagai alat bantu",
                         "Sistem digunakan sebagai deteksi dini untuk mendukung, bukan menggantikan, pemeriksaan Bidan atau dokter.")
        st.markdown(
            """<article class='evaluation-card'><h3>Visual evaluasi untuk dokumentasi implementasi</h3>
            <p>Bagian ini menampilkan keluaran evaluasi dari artefak model penelitian untuk kebutuhan dokumentasi
            dan pengujian sistem. Halaman Deteksi Dini tidak menampilkan istilah teknis ini kepada Bidan.</p></article>""",
            unsafe_allow_html=True,
        )
        try:
            with st.spinner("Menyiapkan visual evaluasi model..."):
                model_figures = build_model_figures(filtered)
            left, right = st.columns(2, gap="large")
            with left:
                render_chart(model_figures["rf_importance"], "Kontribusi relatif setiap atribut pada model Random Forest.",
                             "atribut dengan kontribusi lebih besar lebih sering membantu pemisahan kelas pada data penelitian.")
            with right:
                render_chart(model_figures["xgb_importance"], "Kontribusi relatif setiap atribut pada model XGBoost.",
                             "perbandingan membantu dokumentasi perilaku kedua algoritma yang diuji pada notebook.")
            left, right = st.columns(2, gap="large")
            with left:
                render_chart(model_figures["cm_rf"], "Perbandingan kelas aktual dan hasil prediksi Random Forest.",
                             "nilai pada diagonal menunjukkan prediksi yang selaras dengan label data uji.")
            with right:
                render_chart(model_figures["cm_xgb"], "Perbandingan kelas aktual dan hasil prediksi XGBoost.",
                             "sel yang tidak berada pada diagonal menunjukkan prediksi berbeda dari label data uji.")
            render_chart(model_figures["roc"], "Kurva kemampuan pemisahan kelas untuk setiap algoritma.",
                         "kurva yang berada lebih jauh dari garis acuan menunjukkan kemampuan pemisahan yang lebih baik pada data uji.")
        except Exception as error:
            st.info(f"Visual evaluasi belum dapat ditampilkan pada filter ini: {error}")

except Exception as error:
    st.error(f"Dashboard visualisasi belum dapat dimuat: {error}")
