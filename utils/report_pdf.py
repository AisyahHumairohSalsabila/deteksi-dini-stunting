"""Pembuatan PDF sementara untuk hasil pemeriksaan tanpa penyimpanan data."""
from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def buat_pdf_hasil_pemeriksaan(hasil: dict[str, Any], status: str, interpretasi: str, rekomendasi: list[str]) -> bytes:
    """Membuat berkas PDF di memori; tidak ada data pemeriksaan yang disimpan."""
    buffer = BytesIO()
    dokumen = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    judul = ParagraphStyle("JudulPemeriksaan", parent=styles["Heading1"], textColor=HexColor("#173b57"), spaceAfter=14)
    isi = ParagraphStyle("IsiPemeriksaan", parent=styles["BodyText"], leading=17, spaceAfter=7)

    jenis_kelamin = "Laki-laki" if hasil["jenis_kelamin"] == "L" else "Perempuan"
    tanggal = hasil["tanggal"].strftime("%d-%m-%Y %H:%M")
    baris = [
        ("Tanggal Pemeriksaan", tanggal), ("Umur Balita", f"{hasil['umur']:.0f} bulan"),
        ("Jenis Kelamin", jenis_kelamin), ("Berat Badan", f"{hasil['berat']:.1f} kg"),
        ("Tinggi Badan", f"{hasil['tinggi']:.1f} cm"), ("Status Pemeriksaan", status),
        ("Tingkat Keyakinan Sistem", f"{hasil['confidence']:.1%}"), ("Interpretasi", interpretasi),
    ]
    elemen = [Paragraph("Hasil Pemeriksaan Dini Risiko Stunting", judul)]
    elemen.extend(Paragraph(f"<b>{label}:</b> {nilai}", isi) for label, nilai in baris)
    elemen.append(Spacer(1, 8))
    elemen.append(Paragraph("<b>Rekomendasi</b>", isi))
    elemen.extend(Paragraph(f"• {item}", isi) for item in rekomendasi)
    elemen.append(Spacer(1, 8))
    elemen.append(Paragraph(
        "<b>Disclaimer:</b> Hasil ini merupakan alat bantu deteksi dini dan tidak menggantikan diagnosis tenaga kesehatan.",
        isi,
    ))
    dokumen.build(elemen)
    return buffer.getvalue()
