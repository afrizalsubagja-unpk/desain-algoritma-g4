import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ============================================================
# SIM-RS DASHBOARD - FEATURE SELECTION KANKER PAYUDARA
# Membaca hasil dari validasi_penelitian.py
# Tidak menjalankan ulang eksperimen.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "hasil_pengujian"
METHOD_ORDER = ["Baseline", "SFS", "RFE", "GA"]

# Palet warna khas SIM-RS (Hijau, Biru Langit, Oranye, Merah)
METHOD_COLORS = {
    "Baseline": "#3366cc",  # SIM-RS Blue
    "SFS": "#5cb85c",       # SIM-RS Green
    "RFE": "#f0ad4e",       # SIM-RS Orange
    "GA": "#d9534f",        # SIM-RS Red
}

st.set_page_config(
    page_title="SIM-RS · Analisis Feature Selection Kanker Payudara",
    page_icon=":material/domain:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- SIM-RS Classic Hospital Styling ---------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

      html, body, .stApp {
        font-family: 'Open Sans', Arial, Helvetica, sans-serif;
        color: #333333;
        background-color: #e5e8ec;
      }

      /* Pertahankan font Material Symbols bawaan Streamlit */
      span[data-testid="stIconMaterial"], .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
      }

      [data-testid="stHeader"] {
        background: #3d3d3d !important;
        border-bottom: 2px solid #2b2b2b;
      }
      [data-testid="stHeader"] * {
        color: #ffffff !important;
      }

      .block-container {
        max-width: 100%;
        padding-top: 0.6rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        padding-bottom: 2.5rem;
      }

      /* ---------------- SIM-RS Olive Green Sidebar ---------------- */
      [data-testid="stSidebar"] {
        background-color: #4c7c1d !important;
        border-right: 2px solid #385e13;
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] .stCaptionContainer {
        color: #ffffff !important;
      }
      [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-radius: 2px !important;
        border: 1px solid #2e4d0f !important;
      }
      [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #222222 !important;
      }
      [data-testid="stSidebarCollapseButton"] button {
        color: #ffffff !important;
      }

      /* Tag multiselect bergaya kotak tegas */
      span[data-baseweb="tag"] {
        background-color: #3a6114 !important;
        border: 1px solid #2b490e !important;
        color: #ffffff !important;
        border-radius: 2px !important;
        font-weight: 600 !important;
      }
      span[data-baseweb="tag"] span {
        color: #ffffff !important;
      }
      span[data-baseweb="tag"] svg {
        fill: #ffffff !important;
      }

      /* Sidebar Search & Menu Styling (Persis Gambar SIM-RS) */
      .simrs-side-search {
        background: #2f2f2f;
        color: #cccccc;
        padding: 9px 12px;
        font-size: 0.8rem;
        border: 1px solid #222222;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      .simrs-side-section {
        background: #558b22;
        color: #ffffff;
        font-size: 0.92rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 10px 12px;
        border-top: 1px solid #68a230;
        border-bottom: 1px solid #3b6313;
        margin: 12px -1rem 8px -1rem;
        letter-spacing: 0.03em;
      }
      .simrs-menu-list {
        margin: 0 -1rem 12px -1rem;
      }
      .simrs-menu-item {
        padding: 7px 14px;
        font-size: 0.8rem;
        color: #f1f8e9;
        border-bottom: 1px solid #416b18;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #4c7c1d;
      }
      .simrs-menu-item.active {
        background: #3d6615;
        font-weight: 700;
        border-left: 4px solid #ffffff;
      }
      .simrs-menu-val {
        font-weight: 700;
        color: #ffffff;
        background: rgba(0, 0, 0, 0.22);
        padding: 1px 6px;
        font-size: 0.73rem;
        border-radius: 2px;
      }
      .simrs-user-footer {
        background: #2f2f2f;
        color: #e0e0e0;
        padding: 12px;
        margin: 18px -1rem -1rem -1rem;
        border-top: 2px solid #222222;
        font-size: 0.76rem;
        line-height: 1.45;
      }
      .simrs-user-footer b {
        color: #8bc34a;
      }

      /* ---------------- SIM-RS Top Bar & Window Header ---------------- */
      .simrs-topbar {
        background: linear-gradient(180deg, #4b4b4b 0%, #383838 100%);
        color: #ffffff;
        padding: 10px 18px;
        border: 1px solid #292929;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
      }
      .simrs-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        color: #f5f5f5;
      }
      .simrs-subbar {
        background: #222222;
        color: #ffffff;
        padding: 6px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
        border: 1px solid #111111;
      }
      .win-controls {
        display: inline-flex;
        gap: 3px;
      }
      .win-btn {
        display: inline-block;
        width: 16px;
        height: 16px;
        line-height: 15px;
        text-align: center;
        font-size: 10px;
        font-weight: 700;
        background: #f0f0f0;
        color: #333333;
        border: 1px solid #999999;
        border-radius: 1px;
      }
      .win-btn.close {
        background: #d9534f;
        color: #ffffff;
        border-color: #b52b27;
      }

      /* ---------------- SIM-RS Box Windows (Green, Cyan, Orange, Red) ---------------- */
      .simrs-panel {
        background: #ffffff;
        border: 1px solid #cccccc;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
      }
      .simrs-panel-head {
        padding: 7px 12px;
        color: #ffffff;
        font-size: 0.82rem;
        font-weight: 700;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      .head-green {
        background: linear-gradient(180deg, #63b963 0%, #4ca64c 100%);
        border-bottom: 1px solid #3d8b3d;
      }
      .head-cyan {
        background: linear-gradient(180deg, #5bc0de 0%, #31b0d5 100%);
        border-bottom: 1px solid #269abc;
      }
      .head-orange {
        background: linear-gradient(180deg, #f5b041 0%, #ec971f 100%);
        border-bottom: 1px solid #d58512;
      }
      .head-red {
        background: linear-gradient(180deg, #de6764 0%, #c9302c 100%);
        border-bottom: 1px solid #ac2925;
      }
      .head-dark {
        background: linear-gradient(180deg, #4f4f4f 0%, #333333 100%);
        border-bottom: 1px solid #222222;
      }
      .simrs-panel-body {
        padding: 12px 14px;
        background: #ffffff;
      }

      /* Toggle Grafik | Tabel khas SIM-RS di sudut kanan panel */
      .simrs-toggle-group {
        display: inline-flex;
        border: 1px solid rgba(0,0,0,0.2);
      }
      .simrs-toggle-btn {
        background: #ffffff;
        color: #333333;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 7px;
      }
      .simrs-toggle-btn.active {
        background: #2b90b0;
        color: #ffffff;
      }

      /* ---------------- Classic HTML Table (Persis Tabel Tarif Layanan SIM-RS) ---------------- */
      .simrs-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.78rem;
        background: #ffffff;
      }
      .simrs-table th {
        background: #f2f4f7;
        color: #23527c;
        font-weight: 700;
        text-align: left;
        padding: 6px 8px;
        border: 1px solid #dddddd;
      }
      .simrs-table td {
        padding: 5px 8px;
        border: 1px solid #dddddd;
        color: #333333;
      }
      .simrs-table tr:nth-child(even) {
        background: #f9f9f9;
      }
      .simrs-table tr:hover {
        background: #eef7ea;
      }

      /* KPI Widget didalam kotak SIM-RS */
      .simrs-kpi-num {
        font-size: 1.55rem;
        font-weight: 700;
        color: #222222;
        line-height: 1.2;
      }
      .simrs-kpi-sub {
        font-size: 0.75rem;
        color: #666666;
        margin-top: 4px;
      }

      /* Tabs SIM-RS */
      div[data-testid="stTabs"] {
        background: #ffffff;
        border: 1px solid #cccccc;
        padding: 8px 14px 16px 14px;
      }
      div[data-testid="stTabs"] button {
        font-weight: 700;
        font-size: 0.82rem;
        color: #444444;
        border-radius: 0 !important;
      }
      div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #4c7c1d !important;
      }

      /* Sembunyikan dekorasi bawaan Streamlit */
      [data-testid="stToolbar"],
      [data-testid="stDecoration"],
      #MainMenu,
      footer {
        visibility: hidden !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def standardize_method(value):
    text = str(value).strip().lower()
    if text in {"baseline", "base line"}:
        return "Baseline"
    if text == "sfs" or "sequentialforward" in text:
        return "SFS"
    if text == "rfe" or "recursivefeature" in text:
        return "RFE"
    if text in {"ga", "genetic algorithm", "geneticalgorithm"} or "genetic" in text:
        return "GA"
    return str(value).strip()


def find_col(df, candidates):
    mapping = {norm(col): col for col in df.columns}
    for candidate in candidates:
        if norm(candidate) in mapping:
            return mapping[norm(candidate)]
    return None


def read_csv(path):
    """Membaca file CSV dengan beberapa pilihan encoding."""
    if not path or not Path(path).exists():
        logger.warning("File CSV tidak ditemukan: %s", path)
        return None

    last_error = None
    for kwargs in (
        {"encoding": "utf-8-sig"},
        {"encoding": "latin1"},
    ):
        try:
            df = pd.read_csv(path, **kwargs)
            if not df.empty:
                return df
            logger.warning("File CSV kosong: %s", path)
            return None
        except UnicodeDecodeError as err:
            last_error = err
            continue
        except pd.errors.EmptyDataError:
            logger.warning("File CSV tidak memiliki data atau header: %s", path)
            return None
        except (pd.errors.ParserError, PermissionError, OSError) as err:
            last_error = err
            logger.warning("Gagal membaca CSV '%s' dengan opsi %s: %s", path, kwargs, err)
            break

    if last_error is not None:
        logger.warning("Gagal memuat file CSV '%s'. Penyebab terakhir: %s", path, last_error)
    return None


def first_existing(folder, names):
    for name in names:
        path = folder / name
        if path.exists():
            return path
    return None


def _load_csv_with_fallback(primary_names, seed_filename, fallback_names):
    """Cari dan muat CSV dengan tiga lapis fallback."""
    path = first_existing(RESULT_DIR, primary_names)
    if path:
        return read_csv(path), path
    parts = [read_csv(RESULT_DIR / f"seed_{s}" / seed_filename) for s in [21, 42, 2026]]
    parts = [p for p in parts if p is not None]
    if parts:
        return pd.concat(parts, ignore_index=True), RESULT_DIR
    path = first_existing(BASE_DIR, fallback_names)
    return (read_csv(path), path) if path else (None, None)


def load_per_fold():
    return _load_csv_with_fallback(
        primary_names=["seluruh_hasil_per_fold.csv", "hasil_validasi_per_fold.csv"],
        seed_filename="hasil_per_fold.csv",
        fallback_names=[
            "seluruh_hasil_per_fold.csv",
            "hasil_validasi_per_fold.csv",
            "hasil_perbandingan.csv",
        ],
    )


def load_time_data():
    return _load_csv_with_fallback(
        primary_names=["seluruh_waktu_per_fold.csv", "waktu_validasi_per_fold.csv"],
        seed_filename="waktu_per_fold.csv",
        fallback_names=[
            "seluruh_waktu_per_fold.csv",
            "waktu_validasi_per_fold.csv",
            "waktu_seleksi.csv",
        ],
    )


def load_feature_data(fold_df=None):
    result, path = _load_csv_with_fallback(
        primary_names=["seluruh_fitur_per_fold.csv", "fitur_validasi_per_fold.csv"],
        seed_filename="fitur_per_fold.csv",
        fallback_names=[
            "seluruh_fitur_per_fold.csv",
            "fitur_validasi_per_fold.csv",
            "fitur_terpilih.csv",
        ],
    )
    if result is not None:
        return result, path
    if fold_df is not None and "Fitur_Terpilih" in fold_df.columns:
        cols = [
            c
            for c in ["Seed", "Fold", "Metode", "Fitur_Terpilih"]
            if c in fold_df.columns
        ]
        return fold_df[cols].copy(), None
    return None, None


def nice_method_order(values):
    values = list(values)
    return [m for m in METHOD_ORDER if m in values] + [
        m for m in values if m not in METHOD_ORDER
    ]


def simrs_window_open(title, color_class="head-green", right_mode="controls"):
    if right_mode == "toggle":
        right_html = """
        <div class="simrs-toggle-group">
          <span class="simrs-toggle-btn">Grafik</span>
          <span class="simrs-toggle-btn active">Tabel</span>
        </div>
        """
    else:
        right_html = """
        <div class="win-controls">
          <span class="win-btn">&minus;</span>
          <span class="win-btn">&#9723;</span>
          <span class="win-btn close">&times;</span>
        </div>
        """
    st.markdown(
        f"""
        <div class="simrs-panel">
          <div class="simrs-panel-head {color_class}">
            <span>&#9638; &nbsp;{title}</span>
            {right_html}
          </div>
          <div class="simrs-panel-body">
        """,
        unsafe_allow_html=True,
    )


def simrs_window_close():
    st.markdown("</div></div>", unsafe_allow_html=True)


def simrs_vertical_bar(metric_df, metrics_list, title):
    """Vertical grouped bar chart bergaya SIM-RS Kunjungan Pasien."""
    avail = [m for m in metrics_list if m in metric_df.columns]
    if not avail:
        st.info("Data metrik belum tersedia.")
        return
    frame = metric_df[avail].dropna(how="all").copy()
    if frame.empty:
        st.info("Data kosong.")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=135)
    x = np.arange(len(frame.index))
    width = 0.34 if len(avail) == 2 else 0.24
    palette = ["#3366cc", "#dc3912", "#ff9900", "#109618"]

    for idx, col in enumerate(avail):
        vals = frame[col].astype(float).to_numpy() * 100
        offset = (idx - (len(avail) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width=width * 0.9,
            label=col.replace("_Malignant", ""),
            color=palette[idx % len(palette)],
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.8,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(frame.index, fontsize=8.5, fontweight="bold", color="#333333")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Persentase (%)", fontsize=8, color="#555555")
    ax.tick_params(axis="y", labelsize=8, colors="#555555")
    ax.grid(axis="y", linestyle="-", linewidth=0.5, color="#dddddd")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.legend(
        loc="upper left",
        frameon=False,
         ncol=len(avail),
        fontsize=8,
    )
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    fig.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def simrs_pie_chart(series_data, labels, colors_list, legend_title=""):
    """Pie chart bulat dengan legenda di kanan persis gaya SIM-RS."""
    fig, ax = plt.subplots(figsize=(6.2, 2.7), dpi=135)
    wedges, texts, autotexts = ax.pie(
        series_data,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors_list,
        pctdistance=0.65,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 8, "color": "white", "fontweight": "bold"},
    )
    ax.legend(
        wedges,
        labels,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )
    ax.axis("equal")
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout(pad=0.6)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# --------------------------- Load data ---------------------------
fold_df, fold_path = load_per_fold()
time_df, time_path = load_time_data()
feature_df, feature_path = load_feature_data(fold_df)

if fold_df is None or fold_df.empty:
    st.error("File hasil eksperimen belum ditemukan di folder hasil_pengujian.")
    st.stop()

# Normalize names and numeric columns.
fold_df.columns = [str(c).strip() for c in fold_df.columns]
method_col = find_col(fold_df, ["Metode", "Method", "Model"])
if method_col is None:
    st.error("Kolom 'Metode' tidak ditemukan pada file hasil.")
    st.stop()
fold_df[method_col] = fold_df[method_col].map(standardize_method)

metric_map = {
    "Accuracy": find_col(fold_df, ["Accuracy", "Accuracy_mean", "Akurasi"]),
    "Precision_Malignant": find_col(
        fold_df, ["Precision_Malignant", "Precision_Malignant_mean"]
    ),
    "Recall_Malignant": find_col(
        fold_df, ["Recall_Malignant", "Recall_Malignant_mean"]
    ),
    "F1_Malignant": find_col(
        fold_df, ["F1_Malignant", "F1_Malignant_mean", "F1-score"]
    ),
    "Jumlah_Fitur": find_col(fold_df, ["Jumlah_Fitur", "Jumlah_Fitur_mean", "Fitur"]),
}
if metric_map["Accuracy"] is None:
    st.error("Kolom Accuracy tidak ditemukan pada file hasil.")
    st.stop()

# --------------------------- SIM-RS Olive Green Sidebar ---------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="simrs-side-search">
          <span>Live Search ...</span>
          <span>&#128269;</span>
        </div>
        <div class="simrs-side-section">FRONT OFFICE</div>
        """,
        unsafe_allow_html=True,
    )

    seed_col = find_col(fold_df, ["Seed", "Random_Seed"])
    all_seeds = (
        sorted(
            pd.to_numeric(fold_df[seed_col], errors="coerce").dropna().unique().tolist()
        )
        if seed_col
        else []
    )
    selected_seeds = (
        st.multiselect("Filter Random Seed", all_seeds, default=all_seeds)
        if all_seeds
        else []
    )
    all_methods = nice_method_order(fold_df[method_col].dropna().unique().tolist())
    selected_methods = st.multiselect(
        "Filter Metode Seleksi", all_methods, default=all_methods
    )

    st.markdown(
        """
        <div class="simrs-menu-list">
          <div class="simrs-menu-item active">
            <span>&#8962; &nbsp;Beranda Dashboard</span>
            <span>&lsaquo;</span>
          </div>
          <div class="simrs-menu-item">
            <span>&#9998; &nbsp;Dataset WDBC</span>
            <span class="simrs-menu-val">569 Sampel</span>
          </div>
          <div class="simrs-menu-item">
            <span>&#10010; &nbsp;Pasien Malignant</span>
            <span class="simrs-menu-val">212 Kasus</span>
          </div>
          <div class="simrs-menu-item">
            <span>&#9877; &nbsp;Pasien Benign</span>
            <span class="simrs-menu-val">357 Kasus</span>
          </div>
          <div class="simrs-menu-item">
            <span>&#9883; &nbsp;Laboratorium Sitologi</span>
            <span class="simrs-menu-val">30 Fitur</span>
          </div>
        </div>

        <div class="simrs-side-section">BACK OFFICE</div>
        <div class="simrs-menu-list">
          <div class="simrs-menu-item">
            <span>&#9776; &nbsp;Setup Estimator</span>
            <span class="simrs-menu-val">LogReg</span>
          </div>
          <div class="simrs-menu-item">
            <span>&#9881; &nbsp;Skema Validasi</span>
            <span class="simrs-menu-val">5-Fold CV</span>
          </div>
          <div class="simrs-menu-item">
            <span>&#128202; &nbsp;Total Eksperimen</span>
            <span class="simrs-menu-val">3 Seed</span>
          </div>
          <div class="simrs-menu-item">
            <span>&#128196; &nbsp;SIRS / Pelaporan</span>
            <span>&lsaquo;</span>
          </div>
        </div>

        <div class="simrs-user-footer">
          <div><b>User :</b> Group 4 (Studi Kasus #8)</div>
          <div><b>Dosen :</b> Dr. Fajar Delli Wihartiko, S.Si., MM., M.Kom.</div>
          <div style="margin-top:4px; color:#aaaaaa;">Analisis &amp; Desain Algoritma</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Filter folds for current view.
view_df = fold_df.copy()
if seed_col and selected_seeds:
    view_df = view_df[
        pd.to_numeric(view_df[seed_col], errors="coerce").isin(selected_seeds)
    ]
if not selected_methods:
    st.warning("Pilih setidaknya satu metode pada panel FRONT OFFICE di sebelah kiri.")
    st.stop()
view_df = view_df[view_df[method_col].isin(selected_methods)]

# Mean metrics from actual outer-fold rows.
metric_agg_cols = {k: v for k, v in metric_map.items() if v is not None}
for c in metric_agg_cols.values():
    view_df[c] = pd.to_numeric(view_df[c], errors="coerce")
agg = view_df.groupby(method_col).agg(
    {c: ["mean", "std"] for c in metric_agg_cols.values()}
)
agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
metrics = pd.DataFrame(index=agg.index)
for canonical, raw_col in metric_agg_cols.items():
    metrics[canonical] = agg.get(f"{raw_col}_mean")
    metrics[f"{canonical}_std"] = agg.get(f"{raw_col}_std")
metrics = metrics.reindex(nice_method_order(metrics.index))
metrics.index.name = "Metode"

# Average selection time using fold-level time CSV.
if time_df is not None and not time_df.empty:
    time_method_col = find_col(time_df, ["Metode", "Method", "Model"])
    time_value_col = find_col(
        time_df, ["Waktu_Detik", "Waktu_Rata_Rata_Detik", "Waktu_Seleksi_Detik"]
    )
    if time_method_col and time_value_col:
        time_df[time_method_col] = time_df[time_method_col].map(standardize_method)
        time_df[time_value_col] = pd.to_numeric(
            time_df[time_value_col], errors="coerce"
        )
        if seed_col:
            time_seed_col = find_col(time_df, ["Seed", "Random_Seed"])
            if time_seed_col and selected_seeds:
                time_df[time_seed_col] = pd.to_numeric(
                    time_df[time_seed_col], errors="coerce"
                )
                time_view = time_df[time_df[time_seed_col].isin(selected_seeds)]
            else:
                time_view = time_df
        else:
            time_view = time_df
        time_view = time_view[time_view[time_method_col].isin(selected_methods)]
        time_avg = time_view.groupby(time_method_col)[time_value_col].mean()
        metrics["Waktu_Seleksi_Detik"] = time_avg.reindex(metrics.index)

# --------------------------- SIM-RS Top Header & Sub-bar ---------------------------
st.markdown(
    """
    <div class="simrs-topbar">
      <div class="simrs-brand">
        <span>&#127973;</span>
        <span>SIM-RS &middot; Sistem Evaluasi Feature Selection Kanker Payudara</span>
      </div>
      <div style="font-size:0.8rem; color:#dddddd;">
        Group 4 &middot; Studi Kasus Nomor 8 &middot; Logistic Regression
      </div>
    </div>

    <div class="simrs-subbar">
      <span>&#9638; &nbsp;Dashboard Evaluasi Diagnostik - Breast Cancer Wisconsin (Diagnostic)</span>
      <div class="win-controls">
        <span class="win-btn">&minus;</span>
        <span class="win-btn">&#9723;</span>
        <span class="win-btn close">&times;</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------- 4 Colored SIM-RS KPI Boxes ---------------------------
k1, k2, k3, k4 = st.columns(4)
if "Accuracy" in metrics and metrics["Accuracy"].notna().any():
    best_acc = metrics["Accuracy"].idxmax()
    with k1:
        st.markdown(
            f"""
            <div class="simrs-panel">
              <div class="simrs-panel-head head-green">
                <span>&#9638; Akurasi Tertinggi</span>
                <span>{best_acc}</span>
              </div>
              <div class="simrs-panel-body">
                <div class="simrs-kpi-num">{metrics.loc[best_acc, 'Accuracy'] * 100:.2f}%</div>
                <div class="simrs-kpi-sub">Rata-rata Outer Fold ({best_acc})</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
if "Recall_Malignant" in metrics and metrics["Recall_Malignant"].notna().any():
    best_recall = metrics["Recall_Malignant"].idxmax()
    with k2:
        st.markdown(
            f"""
            <div class="simrs-panel">
              <div class="simrs-panel-head head-cyan">
                <span>&#9638; Recall Malignant</span>
                <span>{best_recall}</span>
              </div>
              <div class="simrs-panel-body">
                <div class="simrs-kpi-num">{metrics.loc[best_recall, 'Recall_Malignant'] * 100:.2f}%</div>
                <div class="simrs-kpi-sub">Sensitivitas Kelas Positif ({best_recall})</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
if "Jumlah_Fitur" in metrics and metrics["Jumlah_Fitur"].notna().any():
    fewest_method = metrics["Jumlah_Fitur"].idxmin()
    with k3:
        st.markdown(
            f"""
            <div class="simrs-panel">
              <div class="simrs-panel-head head-orange">
                <span>&#9638; Fitur Paling Sedikit</span>
                <span>{fewest_method}</span>
              </div>
              <div class="simrs-panel-body">
                <div class="simrs-kpi-num">{metrics.loc[fewest_method, 'Jumlah_Fitur']:.2f} Fitur</div>
                <div class="simrs-kpi-sub">Dari 30 Fitur Sitologi Awal</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
if "Waktu_Seleksi_Detik" in metrics and metrics["Waktu_Seleksi_Detik"].notna().any():
    times = metrics["Waktu_Seleksi_Detik"].dropna()
    if not times.empty:
        fastest = times.idxmin()
        with k4:
            st.markdown(
                f"""
                <div class="simrs-panel">
                  <div class="simrs-panel-head head-red">
                    <span>&#9638; Seleksi Tercepat</span>
                    <span>{fastest}</span>
                  </div>
                  <div class="simrs-panel-body">
                    <div class="simrs-kpi-num">{times.loc[fastest]:.3f} dtk</div>
                    <div class="simrs-kpi-sub">Durasi Rata-rata per Fold</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# --------------------------- Main SIM-RS 2x2 Grid (Persis Gambar Contoh) ---------------------------
row1_left, row1_right = st.columns([1.15, 1.0])

with row1_left:
    # Panel Hijau (Persis "Tarif layanan" di kiri atas pada gambar referensi)
    rows_html = ""
    for idx, (m_name, row) in enumerate(metrics.iterrows(), start=1):
        acc_txt = f"{row['Accuracy'] * 100:.2f}%" if pd.notna(row.get("Accuracy")) else "-"
        prec_txt = (
            f"{row['Precision_Malignant'] * 100:.2f}%"
            if pd.notna(row.get("Precision_Malignant"))
            else "-"
        )
        rec_txt = (
            f"{row['Recall_Malignant'] * 100:.2f}%"
            if pd.notna(row.get("Recall_Malignant"))
            else "-"
        )
        f1_txt = (
            f"{row['F1_Malignant'] * 100:.2f}%"
            if pd.notna(row.get("F1_Malignant"))
            else "-"
        )
        fitur_txt = (
            f"{row['Jumlah_Fitur']:.1f}" if pd.notna(row.get("Jumlah_Fitur")) else "-"
        )
        waktu_txt = (
            f"{row['Waktu_Seleksi_Detik']:.3f} dtk"
            if pd.notna(row.get("Waktu_Seleksi_Detik"))
            else "N/A"
        )
        rows_html += f"""
        <tr>
          <td>{idx}</td>
          <td><b>{m_name.upper()}</b></td>
          <td>{acc_txt}</td>
          <td>{prec_txt}</td>
          <td>{rec_txt}</td>
          <td>{f1_txt}</td>
          <td>{fitur_txt}</td>
          <td>{waktu_txt}</td>
        </tr>
        """

    st.markdown(
        f"""
        <div class="simrs-panel">
          <div class="simrs-panel-head head-green">
            <span>&#9638; &nbsp;Ringkasan Layanan Metode Feature Selection &amp; Tim Peneliti</span>
            <div class="win-controls">
              <span class="win-btn">&minus;</span>
              <span class="win-btn">&#9723;</span>
              <span class="win-btn close">&times;</span>
            </div>
          </div>
          <div class="simrs-panel-body">
            <table class="simrs-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Metode Seleksi</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>Fitur</th>
                  <th>Waktu</th>
                </tr>
              </thead>
              <tbody>
                {rows_html}
              </tbody>
            </table>
            <div style="margin-top: 12px;"></div>
            <table class="simrs-table">
              <thead>
                <tr>
                  <th>No</th>
                  <th>Nama Anggota Tim Peneliti (Group 4)</th>
                  <th>NIM</th>
                  <th>Dosen Pengampu</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>1</td>
                  <td>AFIFAH ZUNDATUL ISHLAH</td>
                  <td>072926011</td>
                  <td rowspan="4" style="vertical-align: middle; font-weight: 600;">
                    Dr. Fajar Delli Wihartiko,<br>S.Si., MM., M.Kom.
                  </td>
                </tr>
                <tr>
                  <td>2</td>
                  <td>HAYKAL SAPUTRA JOBAN</td>
                  <td>072926031</td>
                </tr>
                <tr>
                  <td>3</td>
                  <td>MUHAMMAD VICKY WAHYUDA</td>
                  <td>0729260007</td>
                </tr>
                <tr>
                  <td>4</td>
                  <td>AFRIZAL SUBAGJA</td>
                  <td>072926002</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with row1_right:
    # Panel Biru Muda 1 (Persis "Kualifikasi karyawan" di kanan atas)
    simrs_window_open(
        "Distribusi Diagnosis Pasien (WDBC Dataset)",
        color_class="head-cyan",
        right_mode="toggle",
    )
    simrs_pie_chart(
        series_data=[212, 357],
        labels=["Malignant / Ganas (212)", "Benign / Jinak (357)"],
        colors_list=["#dc3912", "#3366cc"],
        legend_title="Kelas Target",
    )
    simrs_window_close()

    # Panel Biru Muda 2 (Persis "Karyawan per bagian" di kanan tengah)
    simrs_window_open(
        "Proporsi Rata-rata Fitur Terpilih per Metode",
        color_class="head-cyan",
        right_mode="toggle",
    )
    if "Jumlah_Fitur" in metrics and metrics["Jumlah_Fitur"].notna().any():
        fitur_series = metrics["Jumlah_Fitur"].dropna()
        simrs_pie_chart(
            series_data=fitur_series.values,
            labels=[f"{m} ({v:.1f} fitur)" for m, v in fitur_series.items()],
            colors_list=["#3366cc", "#109618", "#ff9900", "#990099"][: len(fitur_series)],
            legend_title="Metode Seleksi",
        )
    simrs_window_close()

# --------------------------- Baris Bawah: Panel Oranye & Panel Merah ---------------------------
row2_left, row2_right = st.columns([1.15, 1.0])

with row2_left:
    # Panel Oranye (Persis "Kunjungan pasien 2017" dengan grafik batang vertikal Biru-Merah)
    simrs_window_open(
        "Perbandingan Kinerja Metrik Klasifikasi (%)",
        color_class="head-orange",
        right_mode="controls",
    )
    simrs_vertical_bar(
        metrics,
        ["Accuracy", "Recall_Malignant"],
        "Perbandingan Accuracy dan Recall Malignant",
    )
    simrs_window_close()

with row2_right:
    # Panel Merah (Persis "Rawat Inap per kelas 2017" di kanan bawah)
    simrs_window_open(
        "Distribusi Waktu Komputasi Seleksi Fitur",
        color_class="head-red",
        right_mode="controls",
    )
    if (
        "Waktu_Seleksi_Detik" in metrics
        and metrics["Waktu_Seleksi_Detik"].notna().any()
    ):
        waktu_valid = metrics["Waktu_Seleksi_Detik"].dropna()
        if not waktu_valid.empty:
            simrs_pie_chart(
                series_data=waktu_valid.values,
                labels=[f"{m} ({v:.3f} dtk)" for m, v in waktu_valid.items()],
                colors_list=["#109618", "#ff9900", "#dc3912"][: len(waktu_valid)],
                legend_title="Waktu / Fold",
            )
        else:
            st.info("Pilih metode SFS, RFE, atau GA untuk melihat proporsi waktu seleksi.")
    simrs_window_close()

# --------------------------- Detail Tabs di Bagian Bawah ---------------------------
tab_features, tab_folds, tab_about = st.tabs(
    [
        ":material/biotech: Stabilitas Fitur Terpilih",
        ":material/table_rows: Data Evaluasi Per Fold",
        ":material/description: Dokumentasi & Metodologi",
    ]
)

with tab_features:
    if feature_df is None or feature_df.empty:
        st.info("Data fitur terpilih belum ditemukan di folder hasil_pengujian.")
    else:
        f_method_col = find_col(feature_df, ["Metode", "Method", "Model"])
        f_feature_col = find_col(
            feature_df, ["Fitur_Terpilih", "Selected_Features", "Fitur"]
        )
        f_seed_col = find_col(feature_df, ["Seed", "Random_Seed"])
        if f_method_col is None or f_feature_col is None:
            st.info("Format kolom fitur terpilih belum dikenali.")
        else:
            feature_view = feature_df.copy()
            feature_view[f_method_col] = feature_view[f_method_col].map(
                standardize_method
            )
            feature_methods = [
                m
                for m in selected_methods
                if m != "Baseline" and m in feature_view[f_method_col].unique()
            ]
            chosen_feature_method = st.selectbox(
                "Pilih Metode Seleksi Fitur",
                feature_methods
                or [
                    m
                    for m in ["SFS", "RFE", "GA"]
                    if m in feature_view[f_method_col].unique()
                ],
            )
            if f_seed_col and selected_seeds:
                feature_view[f_seed_col] = pd.to_numeric(
                    feature_view[f_seed_col], errors="coerce"
                )
                feature_view = feature_view[
                    feature_view[f_seed_col].isin(selected_seeds)
                ]
            feature_view = feature_view[
                feature_view[f_method_col] == chosen_feature_method
            ]
            counts = {}
            for value in feature_view[f_feature_col].dropna().astype(str):
                for feature in value.split(","):
                    feature = feature.strip()
                    if feature:
                        counts[feature] = counts.get(feature, 0) + 1
            if counts:
                total = len(feature_view)
                freq = pd.DataFrame(
                    {"Nama Fitur": list(counts.keys()), "Frekuensi Fold": list(counts.values())}
                )
                freq["Persentase (%)"] = (freq["Frekuensi Fold"] / max(total, 1) * 100).round(2)
                freq = freq.sort_values(
                    ["Frekuensi Fold", "Nama Fitur"], ascending=[False, True]
                ).head(20)

                col_chart, col_tbl = st.columns([1.2, 1.0])
                with col_chart:
                    fig, ax = plt.subplots(
                        figsize=(7.5, max(3.6, min(6.8, len(freq) * 0.3))), dpi=135
                    )
                    freq_plot = freq.sort_values("Frekuensi Fold", ascending=True)
                    ax.barh(
                        freq_plot["Nama Fitur"],
                        freq_plot["Frekuensi Fold"],
                        color="#5cb85c",
                        height=0.62,
                    )
                    ax.set_title(
                        f"Top 20 Fitur Terpilih — {chosen_feature_method} ({total} Fold)",
                        loc="left",
                        fontsize=10,
                        fontweight="bold",
                        color="#333333",
                        pad=10,
                    )
                    ax.set_xlabel("Frekuensi Terpilih", color="#555555", fontsize=8.5)
                    ax.tick_params(axis="both", labelsize=8, colors="#333333", length=0)
                    ax.grid(axis="x", linestyle="-", linewidth=0.5, color="#dddddd")
                    ax.set_axisbelow(True)
                    for spine in ["top", "right"]:
                        ax.spines[spine].set_visible(False)
                    ax.set_xlim(0, max(freq_plot["Frekuensi Fold"].max() * 1.16, 1))
                    for i, value in enumerate(freq_plot["Frekuensi Fold"]):
                        ax.text(
                            value + 0.08,
                            i,
                            f"{int(value)}/{total}",
                            va="center",
                            fontsize=8,
                            color="#333333",
                            fontweight="bold",
                        )
                    fig.patch.set_facecolor("#ffffff")
                    ax.set_facecolor("#ffffff")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                with col_tbl:
                    st.dataframe(
                        freq.sort_values("Frekuensi Fold", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                        height=380,
                    )
            else:
                st.info("Tidak ada daftar fitur yang dapat dihitung pada pilihan ini.")

with tab_folds:
    fold_view = view_df.copy()
    display_cols = []
    for candidates in [
        ["Seed", "Random_Seed"],
        ["Fold", "Fold_No"],
        ["Metode", "Method"],
        ["Accuracy"],
        ["Precision_Malignant"],
        ["Recall_Malignant"],
        ["F1_Malignant"],
        ["Jumlah_Fitur"],
        ["Fitur_Terpilih"],
    ]:
        col = find_col(fold_view, candidates)
        if col and col not in display_cols:
            display_cols.append(col)
    if display_cols:
        show_df = fold_view[display_cols].copy()
        for column in show_df.columns:
            canonical = norm(column)
            if canonical in {
                "accuracy",
                "precisionmalignant",
                "recallmalignant",
                "f1malignant",
            }:
                show_df[column] = pd.to_numeric(show_df[column], errors="coerce") * 100
        col_config = {}
        for c in show_df.columns:
            if norm(c) in {
                "accuracy",
                "precisionmalignant",
                "recallmalignant",
                "f1malignant",
            }:
                col_config[c] = st.column_config.NumberColumn(
                    c + " (%)", format="%.2f%%"
                )
        st.dataframe(
            show_df,
            use_container_width=True,
            hide_index=True,
            column_config=col_config,
            height=420,
        )
        st.download_button(
            "Unduh Data Per Fold (CSV)",
            icon=":material/download:",
            data=fold_view[display_cols].to_csv(index=False).encode("utf-8-sig"),
            file_name="simrs_evaluasi_per_fold.csv",
            mime="text/csv",
        )

with tab_about:
    st.markdown(
        """
        <table class="simrs-table">
          <thead>
            <tr>
              <th style="width: 25%;">Komponen Penelitian</th>
              <th>Spesifikasi &amp; Keterangan</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><b>Dataset</b></td>
              <td>Breast Cancer Wisconsin (Diagnostic) — 569 sampel biopsi (212 Malignant, 357 Benign), 30 fitur numerik sitologi.</td>
            </tr>
            <tr>
              <td><b>Model Klasifikasi</b></td>
              <td>Pipeline <code>StandardScaler</code> + <code>LogisticRegression(max_iter=5000, random_state=42)</code>.</td>
            </tr>
            <tr>
              <td><b>Metode Seleksi Fitur</b></td>
              <td>Baseline (30 fitur), Sequential Forward Selection / SFS (10 fitur), Recursive Feature Elimination / RFE (10 fitur), Genetic Algorithm / GA.</td>
            </tr>
            <tr>
              <td><b>Skema Validasi</b></td>
              <td>Stratified 5-Fold Cross-Validation yang diulang pada 3 Random Seed (21, 42, 2026) = 15 eksperimen per metode.</td>
            </tr>
            <tr>
              <td><b>Catatan Interpretasi</b></td>
              <td>Evaluasi akademik pada dataset publik. Dalam konteks deteksi kanker, metrik Recall Malignant sangat penting untuk mencegah <i>False Negative</i>.</td>
            </tr>
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Diagnostik Sumber File CSV"):
        st.write("Folder hasil:", str(RESULT_DIR))
        st.write("Data per fold:", str(fold_path) if fold_path else "Tidak ditemukan")
        st.write("Data waktu:", str(time_path) if time_path else "Tidak ditemukan")
        st.write("Data fitur:", str(feature_path) if feature_path else "Tidak ditemukan")
