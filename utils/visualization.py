"""Pemuatan data primer dan visualisasi Plotly untuk tahap deployment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

from utils.preprocessing import MODELS_DIR, ROOT_DIR, align_to_model, load_preprocessing_artifacts

DATASET_DIR = ROOT_DIR / "DATASET-PRIMER"
WARNA = {"Normal": "#1c9b72", "Stunted": "#e08b2d", "Severely Stunted": "#c74b50", "Tinggi": "#2477b9"}
STATUS_ORDER = ["Severely Stunted", "Stunted", "Normal", "Tinggi"]


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
    data["tgl_lahir"] = pd.to_datetime(data["tgl_lahir"], errors="coerce")
    data["tanggal_pengukuran"] = pd.to_datetime(data["tanggal_pengukuran"], errors="coerce")
    data["umur"] = np.floor((data["tanggal_pengukuran"] - data["tgl_lahir"]).dt.days / 30.44)
    data.loc[data["umur"] < 0, "umur"] = np.nan

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
    fig.update_layout(
        title=title, template="plotly_white", margin=dict(l=20, r=20, t=55, b=20),
        font=dict(family="Inter, Arial, sans-serif", color="#244058"),
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
    """Grafik tren jumlah kasus per tahun — kosong jika kolom tahun tidak ada."""
    if not tahun_column or tahun_column not in data.columns:
        return {}
    order = _status_order(data)
    counted = (
        data.groupby([tahun_column, "status_stunting"], observed=True).size()
        .reset_index(name="jumlah").sort_values(tahun_column)
    )
    fig = px.line(
        counted, x=tahun_column, y="jumlah", color="status_stunting", markers=True,
        category_orders={"status_stunting": order}, color_discrete_map=WARNA,
    )
    return {"tren": base_layout(fig, "Tren Jumlah Kasus per Tahun")}


def _evaluation_data(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    artifacts = load_preprocessing_artifacts()
    clean = data[["umur", "jk", "berat", "tinggi", "status_stunting"]].copy()
    for col in ["umur", "berat", "tinggi"]:
        clean[col] = clean[col].fillna(clean[col].median())
    clean["jk"] = clean["jk"].fillna(clean["jk"].mode().iloc[0])
    clean = clean.drop_duplicates()
    clean["jk_encoded"] = artifacts["encoder_jk"].transform(clean["jk"].astype(str).str.upper())
    clean["target"] = artifacts["encoder_target"].transform(clean["status_stunting"])
    features = list(getattr(artifacts["scaler"], "feature_names_in_", ["umur", "jk_encoded", "berat", "tinggi"]))
    scaled = pd.DataFrame(artifacts["scaler"].transform(clean[features]), columns=features)
    _, x_test, _, y_test = train_test_split(
        scaled, clean["target"].to_numpy(), test_size=.25, random_state=42, stratify=clean["target"].to_numpy(),
    )
    return x_test, y_test, artifacts


def build_model_figures(data: pd.DataFrame) -> dict[str, go.Figure]:
    """Membuat evaluasi dengan inferensi artefak, tanpa melatih model kembali."""
    x_test, y_test, artifacts = _evaluation_data(data)
    # Memuat berkas kanonis hasil ekspor notebook (bukan berkas lama
    # random_forest.pkl / xgboost.pkl yang tidak pernah dihasilkan notebook
    # dan isinya bisa berbeda dari model yang dievaluasi di metadata.json).
    rf = joblib.load(MODELS_DIR / "model_random_forest.pkl")
    xgb = joblib.load(MODELS_DIR / "model_xgboost.pkl")
    internal = joblib.load(MODELS_DIR / "label_encoder_train_internal.pkl")
    target = artifacts["encoder_target"]
    features = list(getattr(artifacts["scaler"], "feature_names_in_", ["umur", "jk_encoded", "berat", "tinggi"]))

    # Kolom hasil scaler bernama 'jk_encoded', sedangkan model_random_forest.pkl
    # dan model_xgboost.pkl dilatih dengan kolom bernama 'jk'. Diselaraskan
    # per model agar validasi nama fitur scikit-learn/XGBoost tidak gagal.
    x_test_rf = align_to_model(x_test, rf)
    x_test_xgb = align_to_model(x_test, xgb)

    pred_rf = rf.predict(x_test_rf)
    pred_xgb_raw = np.asarray(xgb.predict(x_test_xgb))
    pred_xgb_code = np.argmax(pred_xgb_raw, axis=1) if pred_xgb_raw.ndim > 1 else pred_xgb_raw.astype(int)
    pred_xgb = internal.inverse_transform(pred_xgb_code.astype(int))
    labels = np.unique(np.concatenate([y_test, pred_rf, pred_xgb]))
    names = target.inverse_transform(labels)

    figures: dict[str, go.Figure] = {}
    for name, model, color in [("rf_importance", rf, "#2477b9"), ("xgb_importance", xgb, "#1c9b72")]:
        title = "Feature Importance " + ("Random Forest" if name.startswith("rf") else "XGBoost")
        figures[name] = base_layout(
            px.bar(x=list(model.feature_importances_), y=features, orientation="h", color_discrete_sequence=[color]),
            title,
        )

    for key, prediction, colorscale, title in [
        ("cm_rf", pred_rf, "Blues", "Confusion Matrix Random Forest"),
        ("cm_xgb", pred_xgb, "Greens", "Confusion Matrix XGBoost"),
    ]:
        matrix = confusion_matrix(y_test, prediction, labels=labels)
        figures[key] = base_layout(px.imshow(matrix, x=names, y=names, text_auto=True, color_continuous_scale=colorscale), title)
        figures[key].update_xaxes(title="Prediksi").update_yaxes(title="Aktual")

    roc = go.Figure()
    y_bin = np.column_stack([(y_test == label).astype(int) for label in labels])
    for model_name, model, line_color in [("Random Forest", rf, "#2477b9"), ("XGBoost", xgb, "#1c9b72")]:
        probability = model.predict_proba(x_test_rf if model_name == "Random Forest" else x_test_xgb)
        model_classes = list(model.classes_)
        for index, label in enumerate(labels):
            if model_name == "Random Forest":
                model_label = int(label)
            else:
                if label not in internal.classes_:
                    continue
                model_label = int(internal.transform([label])[0])
            if model_label not in model_classes:
                # Model tidak pernah dilatih dengan kategori ini pada dataset yang diekspor,
                # jadi tidak ada probabilitas untuk dihitung kurva ROC-nya.
                continue
            column = model_classes.index(model_label)
            fpr, tpr, _ = roc_curve(y_bin[:, index], probability[:, column])
            roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color=line_color, dash="solid" if index == 0 else "dot"),
                name=f"{model_name} — {target.inverse_transform([label])[0]} (AUC {auc(fpr, tpr):.2f})",
            ))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#93a6b5", dash="dash"), name="Acuan"))
    figures["roc"] = base_layout(roc, "ROC Curve One-vs-Rest")
    figures["roc"].update_xaxes(title="False Positive Rate").update_yaxes(title="True Positive Rate")
    return figures
