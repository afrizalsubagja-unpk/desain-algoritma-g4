import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ============================================================
# DASHBOARD INTERAKTIF - FEATURE SELECTION KANKER PAYUDARA
# Membaca hasil dari validasi_penelitian.py
# Tidak menjalankan ulang eksperimen.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "hasil_pengujian"
METHOD_ORDER = ["Baseline", "SFS", "RFE", "GA"]
METHOD_COLORS = {
    "Baseline": "#0284C7",  # Medical Sky Blue
    "SFS": "#0D9488",       # Surgical Teal
    "RFE": "#4F46E5",       # Clinical Indigo
    "GA": "#D97706",        # Diagnostic Amber
}

# --------------------------- SVG Icons (Clinical Grade) ---------------------------
ICONS = {
    "hospital": """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 6v4"/><path d="M14 14h-4"/><path d="M14 18h-4"/><path d="M14 8h-4"/><path d="M18 12h2a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2h2"/><path d="M18 22V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v18"/></svg>""",
    "pulse": """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/></svg>""",
    "shield_check": """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>""",
    "layers": """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>""",
    "clock": """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>""",
    "microscope": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/></svg>""",
    "users": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>""",
    "award": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m15.477 12.89 1.515 8.526a.5.5 0 0 1-.81.47l-3.58-2.687a1 1 0 0 0-1.197 0l-3.586 2.686a.5.5 0 0 1-.81-.469l1.514-8.526"/><circle cx="12" cy="8" r="6"/></svg>""",
    "info": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>""",
    "check_circle": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>""",
    "clipboard": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/></svg>""",
    "database": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>""",
}

st.set_page_config(
    page_title="Clinical Analytics · Breast Cancer Feature Selection",
    page_icon=":material/clinical_notes:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Styling (Hospital / Clinical UI) ---------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

      html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
      }

      /* Pastikan Material Icons bawaan Streamlit tidak tertimpa font custom */
      span[data-testid="stIconMaterial"], .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
      }

      .stApp {
        background:
          radial-gradient(circle at 92% 4%, rgba(13, 148, 136, 0.07), transparent 28%),
          radial-gradient(circle at 8% 18%, rgba(2, 132, 199, 0.06), transparent 24%),
          linear-gradient(180deg, #f4f8fb 0%, #edf3f8 100%);
      }

      [data-testid="stHeader"] {
        background: rgba(244, 248, 251, 0.85);
        backdrop-filter: blur(8px);
      }

      .block-container {
        max-width: 1420px;
        padding-top: 1.1rem;
        padding-bottom: 2.5rem;
      }

      /* ---------------- Sidebar Hospital Navigation ---------------- */
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a192f 0%, #0f2942 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.16);
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] .stCaptionContainer {
        color: #e2e8f0 !important;
      }
      [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #0f172a !important;
      }
      [data-testid="stSidebarCollapseButton"] button {
        color: #e2e8f0 !important;
      }

      .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 14px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        margin-bottom: 14px;
      }
      .sidebar-brand-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, #0d9488 0%, #0284c7 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.35);
      }
      .sidebar-brand-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.25;
      }
      .sidebar-brand-sub {
        font-size: 0.74rem;
        color: #94a3b8;
        margin-top: 2px;
      }
      .sidebar-meta-box {
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 12px;
        padding: 12px 14px;
        margin-top: 10px;
      }
      .sidebar-meta-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        padding: 5px 0;
        border-bottom: 1px dashed rgba(148, 163, 184, 0.16);
      }
      .sidebar-meta-row:last-child {
        border-bottom: none;
      }
      .sidebar-meta-key {
        color: #94a3b8;
      }
      .sidebar-meta-val {
        color: #f8fafc;
        font-weight: 600;
      }

      /* ---------------- Clinical Header Banner ---------------- */
      .clinical-hero {
        background: linear-gradient(125deg, #0a2540 0%, #0f3d5e 52%, #0d747c 100%);
        border-radius: 20px;
        padding: 28px 32px;
        color: #ffffff;
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 16px 36px rgba(10, 37, 64, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.12);
      }
      .clinical-hero::after {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 240px;
        height: 240px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(45, 212, 191, 0.18) 0%, transparent 70%);
        pointer-events: none;
      }
      .hero-top-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 14px;
      }
      .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(45, 212, 191, 0.16);
        border: 1px solid rgba(94, 234, 212, 0.35);
        color: #99f6e4;
        padding: 5px 13px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .hero-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #e2e8f0;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
      }
      .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;
        box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.25);
      }
      .clinical-hero h1 {
        color: #ffffff;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.22;
        margin: 0 0 8px 0;
        letter-spacing: -0.01em;
      }
      .clinical-hero .hero-subtitle {
        color: #cbd5e1;
        font-size: 0.96rem;
        max-width: 900px;
        line-height: 1.55;
        margin: 0 0 18px 0;
      }

      /* Team & Supervisor Grid inside Hero */
      .team-panel {
        background: rgba(15, 23, 42, 0.34);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 14px;
        padding: 14px 18px;
        margin-top: 14px;
        backdrop-filter: blur(6px);
      }
      .team-header {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
      }
      .team-title {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #99f6e4;
      }
      .lecturer-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.8rem;
        color: #f8fafc;
        background: rgba(2, 132, 199, 0.35);
        border: 1px solid rgba(125, 211, 252, 0.3);
        padding: 4px 11px;
        border-radius: 8px;
        font-weight: 600;
      }
      .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 8px;
      }
      .member-chip {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 9px;
        padding: 7px 11px;
        display: flex;
        flex-direction: column;
      }
      .member-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #ffffff;
      }
      .member-nim {
        font-size: 0.72rem;
        color: #94a3b8;
        font-family: monospace;
      }

      .hero-chips-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
      }
      .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.22);
        background: rgba(255, 255, 255, 0.09);
        color: #f8fafc;
        padding: 5px 12px;
        font-size: 0.77rem;
        font-weight: 500;
      }

      /* ---------------- Section Headers ---------------- */
      .section-head {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 10px 0 3px 0;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .section-sub {
        color: #475569;
        margin: 0 0 14px 0;
        font-size: 0.88rem;
      }

      /* ---------------- Clinical KPI Metric Cards ---------------- */
      .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid var(--accent-color, #0d9488);
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 132px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
      }
      .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
      }
      .metric-card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
      }
      .metric-label {
        color: #475569;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
      }
      .metric-icon-box {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: var(--accent-bg, #f0fdfa);
        color: var(--accent-color, #0d9488);
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .metric-value {
        color: #0f172a;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.15;
      }
      .metric-help {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 8px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 5px;
      }

      /* ---------------- Panels & Notices ---------------- */
      .panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.035);
        height: 100%;
      }
      .panel h4 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #0f172a;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1.02rem;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .panel ul {
        margin: 0;
        padding-left: 18px;
        color: #334155;
        font-size: 0.9rem;
        line-height: 1.65;
      }
      .note {
        border-left: 4px solid #0d9488;
        background: #f0fdfa;
        border-top: 1px solid #ccfbf1;
        border-right: 1px solid #ccfbf1;
        border-bottom: 1px solid #ccfbf1;
        padding: 13px 16px;
        border-radius: 0 12px 12px 0;
        color: #115e59;
        font-size: 0.88rem;
        display: flex;
        align-items: flex-start;
        gap: 10px;
        line-height: 1.55;
      }
      .note-icon {
        color: #0d9488;
        flex-shrink: 0;
        margin-top: 2px;
      }

      .result-reading {
        color: #0f172a !important;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        border-radius: 12px;
        padding: 13px 16px;
        margin: 8px 0;
        line-height: 1.6;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.025);
      }
      .result-reading-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: #e0f2fe;
        color: #0284c7;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }
      .result-reading, .result-reading p, .result-reading span, .result-reading strong {
        color: #0f172a !important;
        margin: 0;
        font-size: 0.9rem;
      }

      .footer {
        color: #64748b;
        font-size: 0.78rem;
        text-align: center;
        padding-top: 28px;
        border-top: 1px solid #e2e8f0;
        margin-top: 32px;
      }

      /* ---------------- Tabs Styling ---------------- */
      div[data-testid="stTabs"] button {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        color: #475569;
        padding: 10px 16px;
      }
      div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #0d9488 !important;
      }
      .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
      }

      /* Sembunyikan elemen dekoratif bawaan Streamlit */
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


def styled_bar(metric_df, metric, title, xlabel, percent=False, descending=False):
    """Create one clean horizontal bar chart with a separate figure."""
    if metric not in metric_df.columns:
        st.info(f"Metrik {metric} tidak tersedia pada CSV yang ditemukan.")
        return
    frame = metric_df[[metric]].dropna().copy()
    if frame.empty:
        st.info(f"Belum ada data valid untuk {title.lower()}.")
        return
    if descending:
        frame = frame.sort_values(metric, ascending=False)
    else:
        frame = frame.reindex(
            [m for m in METHOD_ORDER if m in frame.index]
            + [m for m in frame.index if m not in METHOD_ORDER]
        )
    fig, ax = plt.subplots(figsize=(7.3, 3.25), dpi=140)
    labels = list(frame.index)
    values = frame[metric].astype(float).to_numpy()
    if percent:
        values = values * 100
    colors = [METHOD_COLORS.get(label, "#64748b") for label in labels]
    bars = ax.barh(labels, values, color=colors, height=0.56)
    ax.invert_yaxis()
    ax.set_title(
        title, loc="left", fontsize=11.5, fontweight="bold", color="#0f172a", pad=14
    )
    ax.set_xlabel(xlabel, color="#64748b", fontsize=9)
    ax.tick_params(axis="both", labelsize=9, colors="#334155", length=0)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.3, color="#cbd5e1")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    max_val = max(values) if len(values) else 1
    ax.set_xlim(0, max_val * 1.18 if max_val > 0 else 1)
    for bar, value in zip(bars, values):
        if percent:
            label = f"{value:.2f}%"
        elif metric == "Jumlah_Fitur":
            label = f"{value:.2f}"
        elif metric == "Waktu_Seleksi_Detik":
            label = f"{value:.3f} dtk"
        else:
            label = f"{value:.4f}"
        ax.text(
            value + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=9,
            color="#0f172a",
            fontweight="bold",
        )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout(pad=1.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def clinical_card(label, value, help_text="", icon_svg="", accent="#0d9488", accent_bg="#f0fdfa"):
    st.markdown(
        f"""
        <div class="metric-card" style="--accent-color: {accent}; --accent-bg: {accent_bg};">
          <div>
            <div class="metric-card-top">
              <div class="metric-label">{label}</div>
              <div class="metric-icon-box">{icon_svg}</div>
            </div>
            <div class="metric-value">{value}</div>
          </div>
          <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------- Load data ---------------------------
fold_df, fold_path = load_per_fold()
time_df, time_path = load_time_data()
feature_df, feature_path = load_feature_data(fold_df)

if fold_df is None or fold_df.empty:
    st.markdown(
        """
        <div class="clinical-hero">
          <div class="hero-badge">Clinical Decision Support Research</div>
          <h1>Sistem Evaluasi Feature Selection Kanker Payudara</h1>
          <p class="hero-subtitle">Visualisasi evaluasi klasifikasi diagnostik menggunakan Logistic Regression.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error("File hasil eksperimen belum ditemukan atau tidak dapat dibaca.")
    st.markdown("Pastikan file hasil terbaru berada di folder `hasil_pengujian`.")
    st.code(str(RESULT_DIR))
    st.markdown(
        "Jalankan `python validasi_penelitian.py` terlebih dahulu, kemudian muat ulang dashboard."
    )
    st.stop()

# Normalize names and numeric columns.
fold_df.columns = [str(c).strip() for c in fold_df.columns]
method_col = find_col(fold_df, ["Metode", "Method", "Model"])
if method_col is None:
    st.error("Kolom 'Metode' tidak ditemukan pada file hasil.")
    st.write("Kolom yang tersedia:", list(fold_df.columns))
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
    st.write("Kolom yang tersedia:", list(fold_df.columns))
    st.stop()

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
          <div class="sidebar-brand-icon">{ICONS['hospital']}</div>
          <div>
            <div class="sidebar-brand-title">ONCO-ANALYTICS</div>
            <div class="sidebar-brand-sub">Clinical Feature Selection</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("PARAMETER FILTER DIAGNOSTIK")
    seed_col = find_col(fold_df, ["Seed", "Random_Seed"])
    all_seeds = (
        sorted(
            pd.to_numeric(fold_df[seed_col], errors="coerce").dropna().unique().tolist()
        )
        if seed_col
        else []
    )
    selected_seeds = (
        st.multiselect("Random Seed Eksperimen", all_seeds, default=all_seeds)
        if all_seeds
        else []
    )
    all_methods = nice_method_order(fold_df[method_col].dropna().unique().tolist())
    selected_methods = st.multiselect(
        "Metode Seleksi Fitur", all_methods, default=all_methods
    )

    st.markdown(
        """
        <div class="sidebar-meta-box">
          <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Dataset</span>
            <span class="sidebar-meta-val">WDBC (Diagnostic)</span>
          </div>
          <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Populasi Sampel</span>
            <span class="sidebar-meta-val">569 Pasien</span>
          </div>
          <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Dimensi Awal</span>
            <span class="sidebar-meta-val">30 Fitur Sitologi</span>
          </div>
          <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Model Klasifikasi</span>
            <span class="sidebar-meta-val">Logistic Regression</span>
          </div>
          <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Skema Validasi</span>
            <span class="sidebar-meta-val">5-Fold CV × 3 Seed</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "Mode Baca: Sistem menampilkan data tervalidasi dari repositori CSV tanpa komputasi ulang."
    )

# Filter folds for current view.
view_df = fold_df.copy()
if seed_col and selected_seeds:
    view_df = view_df[
        pd.to_numeric(view_df[seed_col], errors="coerce").isin(selected_seeds)
    ]
if not selected_methods:
    st.warning("Pilih setidaknya satu metode pada panel kontrol kiri.")
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

# --------------------------- Header Banner ---------------------------
st.markdown(
    f"""
    <div class="clinical-hero">
      <div class="hero-top-bar">
        <span class="hero-badge">{ICONS['microscope']} Analisis &amp; Desain Algoritma · Group 4 (Studi Kasus #8)</span>
        <span class="hero-status"><span class="status-dot"></span> Clinical Dataset Validated</span>
      </div>
      <h1>Analisis Feature Selection pada Klasifikasi Diagnostik Kanker Payudara</h1>
      <p class="hero-subtitle">
        Evaluasi komparatif algoritma reduksi dimensi <b>Baseline</b>, <b>Sequential Forward Selection (SFS)</b>,
        <b>Recursive Feature Elimination (RFE)</b>, dan <b>Genetic Algorithm (GA)</b> menggunakan pengklasifikasi
        <i>Logistic Regression</i> untuk deteksi keganasan (<i>Malignant</i>).
      </p>

      <div class="team-panel">
        <div class="team-header">
          <span class="team-title">{ICONS['users']} Tim Peneliti Klinis &amp; Algoritma</span>
          <span class="lecturer-badge">{ICONS['award']} Dosen Pengampu: Dr. Fajar Delli Wihartiko, S.Si., MM., M.Kom.</span>
        </div>
        <div class="team-grid">
          <div class="member-chip">
            <span class="member-name">Afifah Zundatul Ishlah</span>
            <span class="member-nim">NIM: 072926011</span>
          </div>
          <div class="member-chip">
            <span class="member-name">Haykal Saputra Joban</span>
            <span class="member-nim">NIM: 072926031</span>
          </div>
          <div class="member-chip">
            <span class="member-name">Muhammad Vicky Wahyuda</span>
            <span class="member-nim">NIM: 0729260007</span>
          </div>
          <div class="member-chip">
            <span class="member-name">Afrizal Subagja</span>
            <span class="member-nim">NIM: 072926002</span>
          </div>
        </div>
      </div>

      <div class="hero-chips-bar">
        <span class="hero-chip">{ICONS['database']} Breast Cancer Wisconsin (Diagnostic)</span>
        <span class="hero-chip">{ICONS['clipboard']} 569 Sampel Biopsi · 30 Fitur Morfologi Inti Sel</span>
        <span class="hero-chip">{ICONS['check_circle']} Validasi Silang: 3 Random Seed × 5 Outer Fold</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-head">Indikator Kinerja Utama (Clinical KPI)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-sub">Ringkasan metrik diagnostik terbaik berdasarkan parameter seed dan metode yang diaktifkan pada panel kiri.</div>',
    unsafe_allow_html=True,
)

# Four KPI cards with professional SVG icons & clinical color coding
k1, k2, k3, k4 = st.columns(4)
if "Accuracy" in metrics and metrics["Accuracy"].notna().any():
    best_acc = metrics["Accuracy"].idxmax()
    with k1:
        clinical_card(
            "AKURASI DIAGNOSTIK TERTINGGI",
            f"{metrics.loc[best_acc, 'Accuracy'] * 100:.2f}%",
            f"Metode: {best_acc} (Rata-rata Outer Fold)",
            icon_svg=ICONS["shield_check"],
            accent="#0284c7",
            accent_bg="#e0f2fe",
        )
if "Recall_Malignant" in metrics and metrics["Recall_Malignant"].notna().any():
    best_recall = metrics["Recall_Malignant"].idxmax()
    with k2:
        clinical_card(
            "SENSITIVITAS (RECALL MALIGNANT)",
            f"{metrics.loc[best_recall, 'Recall_Malignant'] * 100:.2f}%",
            f"Metode: {best_recall} (Deteksi Kasus Positif)",
            icon_svg=ICONS["pulse"],
            accent="#0d9488",
            accent_bg="#ccfbf1",
        )
if "Jumlah_Fitur" in metrics and metrics["Jumlah_Fitur"].notna().any():
    fewest_method = metrics["Jumlah_Fitur"].idxmin()
    with k3:
        clinical_card(
            "REDUKSI BIOMARKER TERBAIK",
            f"{metrics.loc[fewest_method, 'Jumlah_Fitur']:.2f} Fitur",
            f"Metode: {fewest_method} (Dari 30 Fitur Awal)",
            icon_svg=ICONS["layers"],
            accent="#4f46e5",
            accent_bg="#e0e7ff",
        )
if "Waktu_Seleksi_Detik" in metrics and metrics["Waktu_Seleksi_Detik"].notna().any():
    times = metrics["Waktu_Seleksi_Detik"].dropna()
    if not times.empty:
        fastest = times.idxmin()
        with k4:
            clinical_card(
                "EFISIENSI KOMPUTASI SELEKSI",
                f"{times.loc[fastest]:.3f} dtk",
                f"Metode Tercepat: {fastest}",
                icon_svg=ICONS["clock"],
                accent="#d97706",
                accent_bg="#fef3c7",
            )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="note">
      <div class="note-icon">{ICONS['info']}</div>
      <div>
        <b>Catatan Protokol Klinis &amp; Akademik:</b> Dalam skrining onkologi, <b>Recall (Sensitivitas) kelas Malignant</b>
        sangat krusial untuk meminimalkan <i>False Negative</i> (pasien kanker yang terlewat). Evaluasi metode terbaik
        perlu menyeimbangkan akurasi, sensitivitas deteksi, efisiensi jumlah fitur sitologi, serta waktu komputasi.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Tabs (Using Professional Material Icons) ---------------------------
tab_summary, tab_features, tab_folds, tab_about = st.tabs(
    [
        ":material/monitoring: Ringkasan Performa Klinis",
        ":material/biotech: Stabilitas Fitur Sitologi",
        ":material/table_chart: Log Evaluasi Per Fold",
        ":material/clinical_notes: Protokol & Metodologi",
    ]
)

with tab_summary:
    st.markdown(
        '<div class="section-head">Matriks Perbandingan Performa Diagnostik</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Nilai rata-rata beserta simpangan baku (Mean ± SD) dari seluruh outer fold yang aktif.</div>',
        unsafe_allow_html=True,
    )

    display = pd.DataFrame(index=metrics.index)
    for canonical, label in [
        ("Accuracy", "Accuracy"),
        ("Precision_Malignant", "Precision (Malignant)"),
        ("Recall_Malignant", "Recall / Sensitivitas (Malignant)"),
        ("F1_Malignant", "F1-Score (Malignant)"),
    ]:
        if canonical in metrics:
            std_col = f"{canonical}_std"
            if std_col in metrics:
                display[label] = [
                    f"{row[canonical] * 100:.2f}% ± {row[std_col] * 100:.2f}%"
                    if pd.notna(row[canonical]) and pd.notna(row[std_col])
                    else (
                        f"{row[canonical] * 100:.2f}%"
                        if pd.notna(row[canonical])
                        else "—"
                    )
                    for _, row in metrics.iterrows()
                ]
            else:
                display[label] = metrics[canonical].map(
                    lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "—"
                )
    if "Jumlah_Fitur" in metrics:
        display["Rata-rata Fitur Terpilih"] = metrics["Jumlah_Fitur"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "—"
        )
    if "Waktu_Seleksi_Detik" in metrics:
        display["Waktu Seleksi (Detik)"] = metrics["Waktu_Seleksi_Detik"].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "N/A (Tanpa Seleksi)"
        )
    st.dataframe(display, use_container_width=True)

    left, right = st.columns(2)
    with left:
        styled_bar(
            metrics, "Accuracy", "Akurasi Diagnostik Rata-rata", "Accuracy (%)", percent=True
        )
    with right:
        styled_bar(
            metrics,
            "Recall_Malignant",
            "Sensitivitas (Recall) Kelas Malignant",
            "Recall (%)",
            percent=True,
        )
    left2, right2 = st.columns(2)
    with left2:
        styled_bar(metrics, "Jumlah_Fitur", "Rata-rata Jumlah Fitur Terpilih", "Jumlah Fitur")
    with right2:
        styled_bar(
            metrics, "Waktu_Seleksi_Detik", "Rata-rata Waktu Komputasi Seleksi", "Waktu (Detik)"
        )

    st.markdown(
        '<div class="section-head">Temuan Utama &amp; Interpretasi Klinis</div>',
        unsafe_allow_html=True,
    )
    reading_items = []
    if "Accuracy" in metrics and metrics["Accuracy"].notna().any():
        best = metrics["Accuracy"].idxmax()
        reading_items.append(
            f"""
            <div class='result-reading'>
              <div class='result-reading-icon'>{ICONS['shield_check']}</div>
              <p><strong>Akurasi Diagnostik Tertinggi:</strong> Metode <b>{best}</b> mencapai rata-rata akurasi tertinggi sebesar <b>{metrics.loc[best, 'Accuracy'] * 100:.2f}%</b> pada pengujian validasi silang.</p>
            </div>
            """
        )
    if "Jumlah_Fitur" in metrics and metrics["Jumlah_Fitur"].notna().any():
        minimum_features = metrics["Jumlah_Fitur"].min()
        fewest_methods = [
            str(m)
            for m in metrics.index
            if pd.notna(metrics.loc[m, "Jumlah_Fitur"])
            and np.isclose(metrics.loc[m, "Jumlah_Fitur"], minimum_features)
        ]
        fewest_text = " dan ".join(fewest_methods)
        reading_items.append(
            f"""
            <div class='result-reading'>
              <div class='result-reading-icon'>{ICONS['layers']}</div>
              <p><strong>Efisiensi Dimensi Fitur:</strong> Metode <b>{fewest_text}</b> mereduksi atribut secara optimal menjadi rata-rata <b>{minimum_features:.2f} fitur</b> dari total 30 fitur sitologi.</p>
            </div>
            """
        )
    if (
        "Waktu_Seleksi_Detik" in metrics
        and metrics["Waktu_Seleksi_Detik"].notna().any()
    ):
        valid_time = metrics["Waktu_Seleksi_Detik"].dropna()
        if len(valid_time):
            fastest = valid_time.idxmin()
            reading_items.append(
                f"""
                <div class='result-reading'>
                  <div class='result-reading-icon'>{ICONS['clock']}</div>
                  <p><strong>Kecepatan Komputasi Seleksi:</strong> Metode <b>{fastest}</b> merupakan algoritma seleksi tercepat dengan rata-rata waktu <b>{valid_time.loc[fastest]:.3f} detik/fold</b> (Baseline tidak dihitung karena menggunakan seluruh fitur tanpa proses seleksi).</p>
                </div>
                """
            )
    for item in reading_items:
        st.markdown(item, unsafe_allow_html=True)

with tab_features:
    st.markdown(
        '<div class="section-head">Analisis Stabilitas Pemilihan Fitur Sitologi</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Frekuensi kemunculan karakteristik inti sel pada berbagai fold eksperimen. Menunjukkan konsistensi algoritma dalam memilih biomarker numerik.</div>',
        unsafe_allow_html=True,
    )
    if feature_df is None or feature_df.empty:
        st.info(
            "Data fitur terpilih belum ditemukan. Pastikan seluruh_fitur_per_fold.csv tersedia di folder hasil_pengujian."
        )
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
                    {"Fitur Sitologi": list(counts.keys()), "Frekuensi Muncul": list(counts.values())}
                )
                freq["Stabilitas (%)"] = (freq["Frekuensi Muncul"] / max(total, 1) * 100).round(2)
                freq = freq.sort_values(
                    ["Frekuensi Muncul", "Fitur Sitologi"], ascending=[False, True]
                ).head(20)
                st.caption(
                    f"Metode {chosen_feature_method}: dievaluasi dari {total} partisi fold pengujian."
                )
                fig, ax = plt.subplots(
                    figsize=(9, max(4, min(8, len(freq) * 0.34))), dpi=140
                )
                freq_plot = freq.sort_values("Frekuensi Muncul", ascending=True)
                ax.barh(
                    freq_plot["Fitur Sitologi"],
                    freq_plot["Frekuensi Muncul"],
                    color=METHOD_COLORS.get(chosen_feature_method, "#0D9488"),
                    height=0.65,
                )
                ax.set_title(
                    f"Top 20 Fitur Sitologi Paling Stabil — Metode {chosen_feature_method}",
                    loc="left",
                    fontsize=11.5,
                    fontweight="bold",
                    color="#0f172a",
                    pad=12,
                )
                ax.set_xlabel("Frekuensi Terpilih (Jumlah Fold)", color="#64748b", fontsize=9)
                ax.tick_params(axis="both", labelsize=8.5, colors="#334155", length=0)
                ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.3, color="#cbd5e1")
                ax.set_axisbelow(True)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xlim(0, max(freq_plot["Frekuensi Muncul"].max() * 1.18, 1))
                for i, value in enumerate(freq_plot["Frekuensi Muncul"]):
                    ax.text(
                        value + 0.08,
                        i,
                        f"{int(value)}/{total}",
                        va="center",
                        fontsize=8,
                        color="#0f172a",
                        fontweight="bold",
                    )
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.dataframe(
                    freq.sort_values("Frekuensi Muncul", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Tidak ada daftar fitur yang dapat dihitung pada pilihan ini.")

with tab_folds:
    st.markdown(
        '<div class="section-head">Log Evaluasi Diagnostik Per Fold</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Rekaman lengkap metrik performa pada setiap kombinasi random seed dan outer fold.</div>',
        unsafe_allow_html=True,
    )
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
            height=460,
        )
        st.download_button(
            "Unduh Laporan CSV (Data Terfilter)",
            icon=":material/download:",
            data=fold_view[display_cols].to_csv(index=False).encode("utf-8-sig"),
            file_name="laporan_evaluasi_fold_klinis.csv",
            mime="text/csv",
        )

with tab_about:
    st.markdown(
        '<div class="section-head">Protokol Dataset &amp; Metodologi Eksperimen</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Spesifikasi dataset medis dan arsitektur validasi algoritma yang digunakan dalam penelitian.</div>',
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            <div class="panel">
              <h4>{ICONS['database']} Spesifikasi Dataset Klinis</h4>
              <p style="color:#475569;font-size:0.88rem;margin-bottom:10px;">
                <b>Breast Cancer Wisconsin (Diagnostic) Dataset (WDBC)</b> — hasil ekstraksi citra digital aspirasi jarum halus (FNA) massa payudara:
              </p>
              <ul>
                <li><b>Total Populasi:</b> 569 sampel biopsi pasien</li>
                <li><b>Dimensi Fitur:</b> 30 atribut numerik morfologi inti sel (radius, tekstur, perimeter, area, smoothness, dll.)</li>
                <li><b>Distribusi Kelas:</b> 212 <i>Malignant</i> (Ganas) &amp; 357 <i>Benign</i> (Jinak)</li>
                <li><b>Konvensi Target:</b> Kelas <i>Malignant</i> ditetapkan sebagai kelas positif (1) untuk fokus deteksi onkologi</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""
            <div class="panel">
              <h4>{ICONS['microscope']} Arsitektur Validasi &amp; Algoritma</h4>
              <p style="color:#475569;font-size:0.88rem;margin-bottom:10px;">
                Seluruh metode diuji pada kondisi yang terkontrol untuk menjamin keadilan perbandingan (<i>apple-to-apple comparison</i>):
              </p>
              <ul>
                <li><b>Pengklasifikasi Utama:</b> Pipeline <code>StandardScaler</code> + <code>LogisticRegression</code></li>
                <li><b>Skema Evaluasi:</b> Stratified 5-Fold Cross-Validation yang diulang pada 3 Random Seed (21, 42, 2026)</li>
                <li><b>Metode Wrapper &amp; Metaheuristik:</b> Baseline (30 fitur), SFS (10 fitur), RFE (10 fitur), dan Genetic Algorithm</li>
                <li><b>Pencegahan Data Leakage:</b> Standardisasi dan seleksi fitur dilakukan murni di dalam partisi data latih setiap fold</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="note">
          <div class="note-icon">{ICONS['info']}</div>
          <div>
            <b>Pernyataan Batasan (Disclaimer):</b> Dashboard ini dirancang untuk tujuan penelitian akademik
            pada mata kuliah Analisis dan Desain Algoritma. Hasil metrik tidak ditujukan sebagai pengganti
            diagnosis medis profesional atau keputusan klinis langsung pada pasien.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Diagnostik Sistem & Sumber Data CSV", icon=":material/folder_open:"):
        st.write("Direktori Hasil Pengujian:", str(RESULT_DIR))
        st.write("File Hasil Per Fold:", str(fold_path) if fold_path else "Tidak ditemukan")
        st.write("File Waktu Komputasi:", str(time_path) if time_path else "Tidak ditemukan")
        st.write("File Fitur Terpilih:", str(feature_path) if feature_path else "Tidak ditemukan")
        st.write("Atribut Kolom Tersedia:", list(fold_df.columns))
        st.write("Total Baris Evaluasi Aktif:", len(view_df))

st.markdown(
    '<div class="footer">ONCO-ANALYTICS · Sistem Informasi Evaluasi Feature Selection Kanker Payudara · Group 4 Analisis dan Desain Algoritma</div>',
    unsafe_allow_html=True,
)
