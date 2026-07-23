"""Form pemeriksaan Posyandu untuk deteksi dini status pertumbuhan, sekaligus riwayatnya."""
from datetime import datetime
import pandas as pd
import streamlit as st
from utils.history import COLUMNS, load_history, merge_uploaded_history, save_record
from utils.prediction import predict_stunting

KATEGORI_URUT = ["Severely Stunted", "Stunted", "Normal", "Tinggi"]


def recommendation_for(status: str) -> tuple[str, str, list[str]]:
    """Mengubah label menjadi bahasa tindak lanjut yang mudah dipahami."""
    if status == "Normal":
        return "normal", "Pertumbuhan anak dalam batas normal", [
            "Terus berikan makanan bergizi seimbang setiap hari.",
            "Rutin timbang dan ukur anak di Posyandu setiap bulan.",
            "Jaga kebersihan diri dan lingkungan anak.",
        ]
    if status == "Tinggi":
        return "tinggi", "Tinggi badan anak di atas rata-rata untuk usianya", [
            "Ukur ulang tinggi badan untuk memastikan hasilnya sudah tepat.",
            "Tetap rutin timbang dan ukur anak di Posyandu setiap bulan.",
            "Tanyakan ke Bidan jika ada hal yang dikhawatirkan pada tumbuh kembang anak.",
        ]
    if status == "Severely Stunted":
        return "severe", "Perlu segera diperiksa lebih lanjut", [
            "Ukur ulang berat dan tinggi badan untuk memastikan hasilnya sudah tepat.",
            "Segera bawa anak ke Bidan, Puskesmas, atau dokter.",
            "Ikuti anjuran pemberian gizi dan pemantauan dari tenaga kesehatan.",
        ]
    return "stunting", "Perlu dipantau dan diperiksa lebih lanjut", [
        "Ukur ulang berat dan tinggi badan untuk memastikan hasilnya sudah tepat.",
        "Konsultasikan ke Bidan atau tenaga kesehatan terdekat.",
        "Pantau pertumbuhan dan asupan makanan anak secara rutin.",
    ]


st.markdown(
    """<section class="hero form-hero"><div class="eyebrow">LAYANAN POSYANDU</div>
    <h1>Cek Pertumbuhan Balita</h1>
    <p>Masukkan hasil timbang dan ukur anak yang paling baru. Pastikan alat ukur digunakan dengan benar dan hasilnya dicatat dengan teliti sebelum lanjut.</p></section>""",
    unsafe_allow_html=True,
)

st.markdown(
    """<article class="instruction-card"><h2>Cara mengisi</h2>
    <p>Isi umur anak dalam bulan, pilih jenis kelamin, lalu masukkan berat badan (kg) dan tinggi badan (cm) hasil pengukuran terakhir.</p></article>""",
    unsafe_allow_html=True,
)

st.markdown("<h2 class='section-title'>Form Pemeriksaan</h2>", unsafe_allow_html=True)

with st.form("form_deteksi", border=False):
    st.markdown("<div class='form-panel'>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        umur = st.number_input("Umur balita (bulan)", min_value=0.0, max_value=60.0, value=24.0, step=1.0)
        jenis_kelamin = st.selectbox(
            "Jenis kelamin", ["L", "P"],
            format_func=lambda value: "Laki-laki" if value == "L" else "Perempuan",
        )
    with right:
        berat = st.number_input("Berat badan (kg)", min_value=0.1, max_value=40.0, value=11.0, step=0.1)
        tinggi = st.number_input("Tinggi badan (cm)", min_value=30.0, max_value=140.0, value=85.0, step=0.1)
    submitted = st.form_submit_button("Cek Pertumbuhan", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    try:
        result = predict_stunting(umur, jenis_kelamin, berat, tinggi)
        tone, interpretation, recommendations = recommendation_for(result["status"])
        recommendation_html = "".join(f"<li>{item}</li>" for item in recommendations)
        probabilitas = result.get("probabilitas", {})
        probability_html = "".join(
            f"<li><span>{label}</span><b>{probabilitas.get(label, 0.0):.0%}</b></li>"
            for label in KATEGORI_URUT
        )
        if result.get("sumber_status") == "zscore_who":
            catatan_sumber = (
                "Status ini dihitung langsung dari standar tinggi badan menurut umur (Z-Score TB/U) "
                "yang ditetapkan WHO, karena hasilnya berada di luar kategori Stunted/Severely Stunted."
            )
        else:
            catatan_sumber = (
                "Status ini dihitung menggunakan model prediksi (Random Forest) yang dilatih khusus "
                "untuk membedakan tingkat keparahan antara Stunted dan Severely Stunted."
            )
        zscore = result.get("zscore")
        zscore_html = f"<div><span>Z-Score TB/U</span><b>{zscore:.2f} SD</b></div>" if zscore is not None else ""
        st.markdown(
            f"""<section class="prediction-card {tone}"><div class="tag">HASIL CEK</div>
            <h2>{result['status']}</h2><div class="result-grid"><div><span>Status</span><b>{interpretation}</b></div>
            <div><span>Tingkat Keyakinan</span><b>{result['confidence']:.0%}</b></div>{zscore_html}</div>
            <h3>Penjelasan</h3><p>{interpretation}. {catatan_sumber} Hasil ini hanya alat bantu deteksi dini, bukan diagnosis. Tetap perlu dikonfirmasi oleh tenaga kesehatan.</p>
            <h3>Rincian Kemungkinan Hasil</h3><ul class="probability-list">{probability_html}</ul>
            <h3>Saran Tindak Lanjut</h3><ul>{recommendation_html}</ul></section>""",
            unsafe_allow_html=True,
        )
        # Catat ke riwayat harian (CSV lokal + session_state).
        # Catatan: karena tidak memakai database, riwayat akan hilang jika
        # aplikasi di-redeploy pada hosting dengan disk sementara (ephemeral).
        record = {
            "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "umur_bulan": umur,
            "jenis_kelamin": "Laki-laki" if jenis_kelamin == "L" else "Perempuan",
            "berat_kg": berat,
            "tinggi_cm": tinggi,
            "status": result["status"],
            "tingkat_keyakinan": round(result["confidence"], 4),
        }
        save_record(record)
        st.session_state.setdefault("riwayat_sesi", []).append(record)
    except Exception as error:
        st.error(f"Pemeriksaan belum bisa dilakukan: {error}")

st.markdown(
    "<div class='disclaimer'><b>Catatan penting.</b> Hasil cek ini tidak menggantikan pemeriksaan langsung oleh Bidan atau dokter. "
    "Kategori Normal/Tinggi dihitung dari standar tinggi badan menurut umur (Z-Score TB/U) dari WHO, sedangkan tingkat keparahan "
    "Stunted/Severely Stunted ditentukan oleh model prediksi yang telah diuji dan dijelaskan pada skripsi ini.</div>",
    unsafe_allow_html=True,
)

st.divider()
st.markdown("<h2 class='section-title'>Riwayat Pemeriksaan</h2>", unsafe_allow_html=True)

history = load_history()
with st.expander("Muat riwayat sebelumnya (jika file di server hilang setelah redeploy)"):
    uploaded = st.file_uploader("Unggah file CSV riwayat", type="csv")
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            history = merge_uploaded_history(uploaded_df, history)
            st.success("Riwayat yang diunggah berhasil digabungkan untuk ditampilkan.")
        except Exception as error:
            st.error(f"Berkas tidak dapat dibaca: {error}")

if history.empty:
    st.markdown(
        "<div class='history-empty'>Belum ada riwayat pemeriksaan. Riwayat akan muncul di sini "
        "setelah ada pemeriksaan dari form di atas.</div>",
        unsafe_allow_html=True,
    )
else:
    history["waktu"] = pd.to_datetime(history["waktu"], errors="coerce")
    today = datetime.now().date()
    today_count = int((history["waktu"].dt.date == today).sum())
    stunting_today = int(
        history[(history["waktu"].dt.date == today) & (history["status"].isin(["Stunted", "Severely Stunted"]))].shape[0]
    )
    kpi_cols = st.columns(3, gap="medium")
    kpi_data = [
        ("Pemeriksaan Hari Ini", f"{today_count:,}", "Jumlah anak yang diperiksa hari ini"),
        ("Terindikasi Stunting Hari Ini", f"{stunting_today:,}", "Status Stunted / Severely Stunted"),
        ("Total Riwayat Tersimpan", f"{len(history):,}", "Sejak riwayat pertama kali dicatat"),
    ]
    for column, (label, value, caption) in zip(kpi_cols, kpi_data):
        column.markdown(
            f"<article class='metric-card'><span>{label}</span><strong>{value}</strong><p>{caption}</p></article>",
            unsafe_allow_html=True,
        )
    st.markdown("<div class='kpi-label'>DAFTAR RIWAYAT</div>", unsafe_allow_html=True)
    st.dataframe(
        history.sort_values("waktu", ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Unduh riwayat sebagai CSV",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="riwayat_prediksi.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown(
    "<div class='disclaimer'><b>Catatan.</b> Riwayat disimpan sebagai file CSV di server, bukan database — "
    "unduh secara berkala agar riwayat tidak hilang jika ingin disimpan dalam jangka panjang.</div>",
    unsafe_allow_html=True,
)
