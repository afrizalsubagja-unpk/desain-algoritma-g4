import base64
import io
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ============================================================
# GREEN HOSPITAL DASHBOARD - FEATURE SELECTION KANKER PAYUDARA
# Membaca hasil dari validasi_penelitian.py
# Tidak menjalankan ulang eksperimen.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "hasil_pengujian"
METHOD_ORDER = ["Baseline", "SFS", "RFE", "GA"]
METHOD_COLORS = {
    "Baseline": "#2563eb",  # Blue
    "SFS": "#10b981",       # Emerald Green
    "RFE": "#9333ea",       # Purple
    "GA": "#f97316",        # Orange
}

st.set_page_config(
    page_title="Green Hospital · Breast Cancer Analytics",
    page_icon=":material/local_hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- SVG Icons (Exact Green Hospital Style) ---------------------------
SVGS = {
    "leaf": """<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="#10b981" stroke="#10b981" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" stroke="#064e3b"/></svg>""",
    "search": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>""",
    "bell": """<svg xmlns="http://www.w3.org/2000/svg" width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="#1e293b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/><circle cx="18" cy="5" r="3.5" fill="#ef4444" stroke="#ffffff" stroke-width="1.5"/></svg>""",
    "calendar": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1e293b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/></svg>""",
    "kpi_green": """<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="#10b981"><path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/></svg>""",
    "kpi_blue": """<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="#2563eb"><path d="M19 4h-1V2h-2v2H8V2H6v2H5c-1.11 0-1.99.9-1.99 2L3 20a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 16H5V10h14v10zM9 14H7v-2h2v2zm4 0h-2v-2h2v2zm4 0h-2v-2h2v2zm-8 4H7v-2h2v2zm4 0h-2v-2h2v2zm4 0h-2v-2h2v2z"/></svg>""",
    "kpi_purple": """<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="#9333ea"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>""",
    "kpi_orange": """<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="#f97316"><path d="M7 13c1.66 0 3-1.34 3-3S8.66 7 7 7s-3 1.34-3 3 1.34 3 3 3zm12-6h-8v7H3V5H1v15h2v-3h18v3h2v-9c0-2.21-1.79-4-4-4z"/></svg>""",
    "nav_home": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>""",
    "nav_users": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>""",
    "nav_lab": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/></svg>""",
    "nav_records": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="20" x="4" y="2" rx="2"/><path d="M8 6h8"/><path d="M8 10h8"/><path d="M8 14h4"/></svg>""",
    "nav_reports": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/></svg>""",
    "nav_settings": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>""",
}


def fig_to_base64(fig):
    """Render matplotlib figure ke Base64 PNG agar bisa menyatu di dalam kartu putih."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=145, facecolor="#ffffff")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --------------------------- Green Hospital Modern SaaS CSS ---------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

      html, body, .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #111c44;
        background-color: #f4f7fb;
      }

      /* Pertahankan font Material Symbols bawaan Streamlit */
      span[data-testid="stIconMaterial"], .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
      }

      [data-testid="stHeader"] {
        background: transparent !important;
        height: 2.2rem;
      }

      .block-container {
        max-width: 1420px;
        padding-top: 0.4rem;
        padding-left: 1.8rem;
        padding-right: 1.8rem;
        padding-bottom: 2.5rem;
      }

      /* ---------------- Dark Forest-Teal Sidebar (Persis Gambar Green Hospital) ---------------- */
      [data-testid="stSidebar"] {
        background:
          radial-gradient(circle at 30% 98%, rgba(37, 99, 235, 0.45), transparent 36%),
          linear-gradient(180deg, #0b2e33 0%, #071f24 72%, #082b49 100%) !important;
        border-right: none;
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] .stCaptionContainer {
        color: #cbd5e1 !important;
      }
      [data-testid="stSidebarCollapseButton"] button {
        color: #cbd5e1 !important;
      }
      [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 10px !important;
        border: none !important;
      }
      [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #111c44 !important;
      }
      span[data-baseweb="tag"] {
        background-color: #d1fae5 !important;
        border: 1px solid #6ee7b7 !important;
        color: #065f46 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
      }
      span[data-baseweb="tag"] span {
        color: #065f46 !important;
      }
      span[data-baseweb="tag"] svg {
        fill: #065f46 !important;
      }

      .gh-logo {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 4px 6px 20px 6px;
      }
      .gh-logo-text {
        font-size: 1.2rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.15;
      }
      .gh-nav-active {
        display: flex;
        align-items: center;
        gap: 12px;
        background: linear-gradient(90deg, #00a86b 0%, #10b981 100%);
        color: #ffffff !important;
        padding: 11px 16px;
        border-radius: 11px;
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 8px;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.28);
      }
      .gh-nav-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #cbd5e1;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 3px;
      }
      .gh-nav-left {
        display: flex;
        align-items: center;
        gap: 12px;
      }
      .gh-nav-badge {
        font-size: 0.72rem;
        background: rgba(255, 255, 255, 0.1);
        color: #e2e8f0;
        padding: 2px 8px;
        border-radius: 999px;
        font-weight: 600;
      }

      /* ---------------- Top Search Bar & Profile Header ---------------- */
      .gh-topbar {
        background: #ffffff;
        border-radius: 16px;
        padding: 11px 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 18px;
        box-shadow: 0 4px 18px rgba(17, 28, 68, 0.035);
      }
      .gh-search-pill {
        background: #f1f5f9;
        border-radius: 999px;
        padding: 9px 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        width: 440px;
        max-width: 100%;
        color: #64748b;
        font-size: 0.85rem;
      }
      .gh-profile-area {
        display: flex;
        align-items: center;
        gap: 18px;
      }
      .gh-user-chip {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.88rem;
        font-weight: 700;
        color: #111c44;
      }
      .gh-user-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.82rem;
        border: 2px solid #ffffff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
      }

      /* ---------------- Welcome Back Row ---------------- */
      .gh-welcome-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 18px;
        gap: 12px;
      }
      .gh-welcome-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #111c44;
        margin: 0 0 3px 0;
        letter-spacing: -0.02em;
      }
      .gh-welcome-sub {
        font-size: 0.9rem;
        color: #64748b;
        margin: 0;
        font-weight: 500;
      }
      .gh-date-pill {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 9px 16px;
        display: inline-flex;
        align-items: center;
        gap: 9px;
        font-size: 0.83rem;
        font-weight: 600;
        color: #1e293b;
        box-shadow: 0 2px 8px rgba(17, 28, 68, 0.03);
      }

      /* ---------------- 4 Horizontal KPI Cards with Mini Bars ---------------- */
      .gh-kpi-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 18px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 18px rgba(17, 28, 68, 0.04);
        border: 1px solid rgba(226, 232, 240, 0.7);
        margin-bottom: 18px;
      }
      .gh-kpi-left {
        display: flex;
        align-items: center;
        gap: 15px;
      }
      .gh-kpi-icon {
        width: 58px;
        height: 58px;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }
      .gh-kpi-num {
        font-size: 1.55rem;
        font-weight: 800;
        color: #111c44;
        line-height: 1.15;
      }
      .gh-kpi-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
        margin: 2px 0 5px 0;
      }
      .gh-kpi-trend {
        font-size: 0.76rem;
        font-weight: 700;
        color: #10b981;
        display: inline-flex;
        align-items: center;
        gap: 3px;
      }
      .gh-sparkbars {
        display: flex;
        align-items: flex-end;
        gap: 3.5px;
        height: 32px;
        padding-bottom: 2px;
      }
      .gh-sparkbars span {
        width: 4.5px;
        border-radius: 2px;
        display: inline-block;
      }

      /* ---------------- White Rounded Content Cards ---------------- */
      .gh-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 4px 18px rgba(17, 28, 68, 0.04);
        border: 1px solid rgba(226, 232, 240, 0.7);
        margin-bottom: 18px;
      }
      .gh-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
      }
      .gh-card-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #111c44;
        margin: 0;
      }
      .gh-card-dropdown {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        padding: 5px 13px;
        font-size: 0.76rem;
        font-weight: 600;
        color: #334155;
      }
      .gh-card-link {
        font-size: 0.8rem;
        font-weight: 700;
        color: #2563eb;
      }

      /* ---------------- Bottom Left Table (Recent Patients Style) ---------------- */
      .gh-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.83rem;
      }
      .gh-table th {
        text-align: left;
        color: #64748b;
        font-weight: 600;
        padding: 10px 8px;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.78rem;
      }
      .gh-table td {
        padding: 11px 8px;
        border-bottom: 1px solid #f8fafc;
        color: #1e293b;
        font-weight: 500;
        vertical-align: middle;
      }
      .gh-avatar-cell {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
        color: #111c44;
      }
      .gh-mini-avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.72rem;
        font-weight: 800;
        flex-shrink: 0;
      }
      .gh-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
      }
      .pill-green { background: #d1fae5; color: #059669; }
      .pill-orange { background: #ffedd5; color: #d97706; }
      .pill-purple { background: #f3e8ff; color: #7e22ce; }
      .pill-blue { background: #dbeafe; color: #2563eb; }

      /* ---------------- Bottom Right Schedule List (Upcoming Appointments Style) ---------------- */
      .gh-sched-row {
        display: grid;
        grid-template-columns: 82px 36px 1fr auto;
        align-items: center;
        gap: 10px;
        padding: 10px 4px;
        border-bottom: 1px solid #f8fafc;
        font-size: 0.83rem;
      }
      .gh-sched-row:last-child {
        border-bottom: none;
      }
      .gh-sched-time {
        color: #475569;
        font-weight: 600;
        font-size: 0.78rem;
      }
      .gh-sched-icon {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 800;
      }
      .gh-sched-name {
        font-weight: 700;
        color: #111c44;
      }
      .gh-sched-doc {
        color: #475569;
        font-weight: 600;
        font-size: 0.8rem;
        text-align: right;
      }

      /* Tabs Styling */
      div[data-testid="stTabs"] {
        background: #ffffff;
        border-radius: 18px;
        padding: 10px 20px 20px 20px;
        box-shadow: 0 4px 18px rgba(17, 28, 68, 0.04);
        border: 1px solid rgba(226, 232, 240, 0.7);
      }
      div[data-testid="stTabs"] button {
        font-weight: 700;
        font-size: 0.86rem;
        color: #64748b;
      }
      div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #00a86b !important;
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


def render_kpi_card(icon_svg, icon_bg, number, label, trend_text, bar_color):
    """Kartu KPI horizontal persis gaya Green Hospital (Ikon Kiri + Angka + Mini Sparkbars Kanan)."""
    heights = [10, 14, 18, 23, 29]
    opacities = [0.35, 0.5, 0.65, 0.82, 1.0]
    bars_html = "".join(
        f'<span style="height:{h}px; background:{bar_color}; opacity:{op};"></span>'
        for h, op in zip(heights, opacities)
    )
    st.markdown(
        f"""
        <div class="gh-kpi-card">
          <div class="gh-kpi-left">
            <div class="gh-kpi-icon" style="background: {icon_bg};">
              {icon_svg}
            </div>
            <div>
              <div class="gh-kpi-num">{number}</div>
              <div class="gh-kpi-label">{label}</div>
              <div class="gh-kpi-trend">&uarr; {trend_text}</div>
            </div>
          </div>
          <div class="gh-sparkbars">
            {bars_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_smooth_wave_b64(view_df, method_col, acc_col, rec_col):
    """Membuat grafik kurva mulus berarsir (Smooth Area Line Chart) persis 'Patient Admissions'."""
    fig, ax = plt.subplots(figsize=(7.5, 2.75), dpi=145)

    fold_col = find_col(view_df, ["Fold", "Fold_No"])
    if fold_col and fold_col in view_df.columns:
        grouped = (
            view_df.groupby(fold_col)[[acc_col, rec_col]]
            .mean()
            .sort_index()
        )
        x_labels = [f"Fold {int(f)}" for f in grouped.index]
        y_acc = grouped[acc_col].to_numpy() * 100
        y_rec = grouped[rec_col].to_numpy() * 100
    else:
        grouped = view_df.groupby(method_col)[[acc_col, rec_col]].mean()
        x_labels = list(grouped.index)
        y_acc = grouped[acc_col].to_numpy() * 100
        y_rec = grouped[rec_col].to_numpy() * 100

    x = np.arange(len(x_labels))
    if len(x) >= 3:
        # Interpolasi kurva halus menggunakan polinomial / cosine
        x_smooth = np.linspace(x.min(), x.max(), 160)
        y_acc_smooth = np.interp(x_smooth, x, y_acc)
        y_rec_smooth = np.interp(x_smooth, x, y_rec)
        # Haluskan dengan moving window kecil
        kernel = np.ones(9) / 9
        y_acc_smooth = np.convolve(
            np.pad(y_acc_smooth, (4, 4), mode="edge"), kernel, mode="valid"
        )
        y_rec_smooth = np.convolve(
            np.pad(y_rec_smooth, (4, 4), mode="edge"), kernel, mode="valid"
        )
    else:
        x_smooth, y_acc_smooth, y_rec_smooth = x, y_acc, y_rec

    # Garis & Area Hijau (Accuracy)
    ax.plot(x_smooth, y_acc_smooth, color="#10b981", linewidth=2.3, label="Accuracy (%)")
    ax.fill_between(x_smooth, y_acc_smooth, 85, color="#10b981", alpha=0.14)
    ax.scatter(x, y_acc, color="#10b981", s=28, zorder=4)

    # Garis & Area Biru (Recall Malignant)
    ax.plot(x_smooth, y_rec_smooth, color="#2563eb", linewidth=2.3, label="Recall Malignant (%)")
    ax.fill_between(x_smooth, y_rec_smooth, 85, color="#2563eb", alpha=0.10)
    ax.scatter(x, y_rec, color="#2563eb", s=28, zorder=4)

    min_y = max(80, min(np.min(y_acc), np.min(y_rec)) - 3.5)
    ax.set_ylim(min_y, 100.5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8.5, color="#475569", fontweight="600")
    ax.tick_params(axis="y", labelsize=8, colors="#64748b", length=0)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="both", linestyle="-", linewidth=0.6, color="#f1f5f9")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.15),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.5,
    )
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    fig.tight_layout(pad=0.3)
    return fig_to_base64(fig)


def make_donut_b64(metrics):
    """Membuat Donut Chart dengan angka total di tengah persis 'Department-wise Patients'."""
    fig, ax = plt.subplots(figsize=(3.0, 2.75), dpi=145)

    if "Jumlah_Fitur" in metrics and metrics["Jumlah_Fitur"].notna().any():
        s = metrics["Jumlah_Fitur"].dropna()
        vals = s.to_numpy()
        labels = list(s.index)
    else:
        vals = np.array([30, 10, 10, 12])
        labels = ["Baseline", "SFS", "RFE", "GA"]

    colors = ["#10b981", "#9333ea", "#2563eb", "#38bdf8", "#f97316"][: len(vals)]
    ax.pie(
        vals,
        colors=colors,
        startangle=140,
        counterclock=False,
        wedgeprops={"width": 0.36, "edgecolor": "white", "linewidth": 2},
    )
    ax.text(
        0,
        0.08,
        "569",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#111c44",
    )
    ax.text(
        0,
        -0.22,
        "Samples",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#64748b",
    )
    ax.axis("equal")
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout(pad=0.1)
    return fig_to_base64(fig), labels, vals, colors


# --------------------------- Load data ---------------------------
fold_df, fold_path = load_per_fold()
time_df, time_path = load_time_data()
feature_df, feature_path = load_feature_data(fold_df)

if fold_df is None or fold_df.empty:
    st.error("File hasil eksperimen belum ditemukan di folder hasil_pengujian.")
    st.stop()

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

# --------------------------- Sidebar (Green Hospital Navigation) ---------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div class="gh-logo">
          {SVGS['leaf']}
          <div class="gh-logo-text">Green<br>Hospital</div>
        </div>
        <div class="gh-nav-active">
          {SVGS['nav_home']}
          <span>Dashboard</span>
        </div>
        <div class="gh-nav-item">
          <span class="gh-nav-left">{SVGS['nav_users']} Patients (WDBC)</span>
          <span class="gh-nav-badge">569</span>
        </div>
        <div class="gh-nav-item">
          <span class="gh-nav-left">{SVGS['nav_lab']} Cytology Features</span>
          <span class="gh-nav-badge">30</span>
        </div>
        <div class="gh-nav-item">
          <span class="gh-nav-left">{SVGS['nav_records']} Validation Folds</span>
          <span class="gh-nav-badge">5-Fold</span>
        </div>
        <div class="gh-nav-item">
          <span class="gh-nav-left">{SVGS['nav_reports']} Logistic Reg.</span>
          <span class="gh-nav-badge">L2</span>
        </div>
        <div style="margin: 14px 0 8px 4px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #6ee7b7;">
          Filter Parameters
        </div>
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
        st.multiselect("Random Seeds", all_seeds, default=all_seeds)
        if all_seeds
        else []
    )
    all_methods = nice_method_order(fold_df[method_col].dropna().unique().tolist())
    selected_methods = st.multiselect(
        "Feature Selection Methods", all_methods, default=all_methods
    )

    st.markdown(
        f"""
        <div style="margin-top: 16px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 12px;">
          <div class="gh-nav-item">
            <span class="gh-nav-left">{SVGS['nav_settings']} Group 4 · Case #8</span>
          </div>
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
    st.warning("Pilih setidaknya satu metode pada panel kiri.")
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

# --------------------------- Top Bar & Welcome Row ---------------------------
st.markdown(
    f"""
    <div class="gh-topbar">
      <div class="gh-search-pill">
        {SVGS['search']}
        <span>Search methods, biomarkers, fold evaluations...</span>
      </div>
      <div class="gh-profile-area">
        {SVGS['bell']}
        <div class="gh-user-chip">
          <div class="gh-user-avatar">G4</div>
          <span>Group 4 Admin &nbsp;&or;</span>
        </div>
      </div>
    </div>

    <div class="gh-welcome-row">
      <div>
        <div class="gh-welcome-title">Welcome Back!</div>
        <div class="gh-welcome-sub">Analisis Feature Selection pada Klasifikasi Kanker Payudara (Breast Cancer Wisconsin Diagnostic).</div>
      </div>
      <div class="gh-date-pill">
        {SVGS['calendar']}
        <span>3 Seeds &times; 5-Fold CV &nbsp;&or;</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------- 4 Horizontal KPI Cards (Green, Blue, Purple, Orange) ---------------------------
k1, k2, k3, k4 = st.columns(4)
if "Accuracy" in metrics and metrics["Accuracy"].notna().any():
    best_acc = metrics["Accuracy"].idxmax()
    with k1:
        render_kpi_card(
            icon_svg=SVGS["kpi_green"],
            icon_bg="#e6f8f1",
            number=f"{metrics.loc[best_acc, 'Accuracy'] * 100:.2f}%",
            label="Highest Accuracy",
            trend_text=f"{best_acc} Method",
            bar_color="#10b981",
        )
if "Recall_Malignant" in metrics and metrics["Recall_Malignant"].notna().any():
    best_recall = metrics["Recall_Malignant"].idxmax()
    with k2:
        render_kpi_card(
            icon_svg=SVGS["kpi_blue"],
            icon_bg="#e8f1ff",
            number=f"{metrics.loc[best_recall, 'Recall_Malignant'] * 100:.2f}%",
            label="Recall Malignant",
            trend_text=f"{best_recall} Sensitivity",
            bar_color="#2563eb",
        )
if "Jumlah_Fitur" in metrics and metrics["Jumlah_Fitur"].notna().any():
    fewest_method = metrics["Jumlah_Fitur"].idxmin()
    with k3:
        render_kpi_card(
            icon_svg=SVGS["kpi_purple"],
            icon_bg="#f5ebff",
            number=f"{metrics.loc[fewest_method, 'Jumlah_Fitur']:.0f} / 30",
            label="Selected Features",
            trend_text=f"-67% ({fewest_method})",
            bar_color="#9333ea",
        )
if "Waktu_Seleksi_Detik" in metrics and metrics["Waktu_Seleksi_Detik"].notna().any():
    times = metrics["Waktu_Seleksi_Detik"].dropna()
    if not times.empty:
        fastest = times.idxmin()
        with k4:
            render_kpi_card(
                icon_svg=SVGS["kpi_orange"],
                icon_bg="#fff1e6",
                number=f"{times.loc[fastest]:.3f}s",
                label="Selection Time",
                trend_text=f"Fastest ({fastest})",
                bar_color="#f97316",
            )

# --------------------------- Middle Row: Smooth Wave Chart + Donut Chart ---------------------------
mid_left, mid_right = st.columns([1.45, 1.0])

with mid_left:
    wave_b64 = make_smooth_wave_b64(
        view_df,
        method_col,
        metric_map["Accuracy"],
        metric_map["Recall_Malignant"] or metric_map["Accuracy"],
    )
    st.markdown(
        f"""
        <div class="gh-card">
          <div class="gh-card-header">
            <span class="gh-card-title">Model Performance Across Folds</span>
            <span class="gh-card-dropdown">5-Fold CV &nbsp;&or;</span>
          </div>
          <img src="data:image/png;base64,{wave_b64}" style="width:100%; height:auto; display:block;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

with mid_right:
    donut_b64, d_labels, d_vals, d_colors = make_donut_b64(metrics)
    legend_rows_html = "".join(
        f'<div style="display:flex;align-items:center;justify-content:space-between;padding:6px 0;font-size:0.84rem;">'
        f'<span style="display:flex;align-items:center;gap:9px;color:#334155;font-weight:500;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{clr};display:inline-block;"></span>'
        f"{lbl}</span>"
        f'<span style="font-weight:800;color:#111c44;">{val:.1f} Fitur</span>'
        f"</div>"
        for lbl, val, clr in zip(d_labels, d_vals, d_colors)
    )
    st.markdown(
        f'<div class="gh-card">'
        f'<div class="gh-card-header"><span class="gh-card-title">Method-wise Feature Subset</span></div>'
        f'<div style="display:grid;grid-template-columns:1.05fr 1fr;align-items:center;gap:10px;">'
        f'<div><img src="data:image/png;base64,{donut_b64}" style="width:100%;max-width:210px;margin:0 auto;display:block;" /></div>'
        f"<div>{legend_rows_html}"
        f'<div style="display:flex;align-items:center;justify-content:space-between;padding:6px 0;font-size:0.84rem;border-top:1px dashed #e2e8f0;margin-top:4px;">'
        f'<span style="display:flex;align-items:center;gap:9px;color:#334155;font-weight:500;">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:#f59e0b;display:inline-block;"></span>'
        f"Malignant / Benign</span>"
        f'<span style="font-weight:800;color:#111c44;">212 / 357</span>'
        f"</div></div></div></div>",
        unsafe_allow_html=True,
    )

# --------------------------- Bottom Row: Method Evaluation Table + Research Team Roster ---------------------------
bot_left, bot_right = st.columns([1.45, 1.0])

with bot_left:
    avatar_styles = [
        ("B", "#dbeafe", "#2563eb", "Baseline (30 Fitur)", "pill-blue"),
        ("S", "#d1fae5", "#059669", "Optimal Accuracy", "pill-green"),
        ("R", "#f3e8ff", "#7e22ce", "Fast Wrapper", "pill-purple"),
        ("G", "#ffedd5", "#d97706", "Metaheuristic", "pill-orange"),
    ]
    table_rows = ""
    for idx, (m_name, row) in enumerate(metrics.iterrows()):
        av_char, av_bg, av_clr, status_lbl, pill_cls = avatar_styles[idx % len(avatar_styles)]
        acc_v = f"{row['Accuracy'] * 100:.2f}%" if pd.notna(row.get("Accuracy")) else "—"
        rec_v = f"{row['Recall_Malignant'] * 100:.2f}%" if pd.notna(row.get("Recall_Malignant")) else "—"
        f1_v = f"{row['F1_Malignant'] * 100:.2f}%" if pd.notna(row.get("F1_Malignant")) else "—"
        fit_v = f"{row['Jumlah_Fitur']:.1f} Fitur" if pd.notna(row.get("Jumlah_Fitur")) else "—"
        time_v = f"{row['Waktu_Seleksi_Detik']:.3f}s" if pd.notna(row.get("Waktu_Seleksi_Detik")) else "0.000s"
        table_rows += (
            f"<tr>"
            f'<td><div class="gh-avatar-cell">'
            f'<span class="gh-mini-avatar" style="background:{av_bg};color:{av_clr};">{av_char}</span>'
            f"<span>{m_name}</span></div></td>"
            f"<td>{acc_v}</td>"
            f"<td>{rec_v}</td>"
            f"<td>{f1_v}</td>"
            f'<td><span class="gh-pill {pill_cls}">{status_lbl}</span></td>'
            f'<td style="color:#475569;font-weight:600;">{fit_v} ({time_v})</td>'
            f"</tr>"
        )

    st.markdown(
        f'<div class="gh-card">'
        f'<div class="gh-card-header">'
        f'<span class="gh-card-title">Feature Selection Evaluation Summary</span>'
        f'<span class="gh-card-link">Logistic Regression</span>'
        f"</div>"
        f'<table class="gh-table">'
        f"<thead><tr>"
        f"<th>Method</th><th>Accuracy</th><th>Recall (Malignant)</th>"
        f"<th>F1-Score</th><th>Status</th><th>Features &amp; Time</th>"
        f"</tr></thead>"
        f"<tbody>{table_rows}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )

with bot_right:
    st.markdown(
        """
        <div class="gh-card">
          <div class="gh-card-header">
            <span class="gh-card-title">Research Team · Group 4</span>
            <span class="gh-card-link">Case Study #8</span>
          </div>
          <div class="gh-sched-row">
            <span class="gh-sched-time">072926011</span>
            <span class="gh-sched-icon" style="background:#d1fae5; color:#059669;">AZ</span>
            <span class="gh-sched-name">Afifah Zundatul Ishlah</span>
            <span class="gh-sched-doc">Peneliti 1</span>
          </div>
          <div class="gh-sched-row">
            <span class="gh-sched-time">072926031</span>
            <span class="gh-sched-icon" style="background:#dbeafe; color:#2563eb;">HS</span>
            <span class="gh-sched-name">Haykal Saputra Joban</span>
            <span class="gh-sched-doc">Peneliti 2</span>
          </div>
          <div class="gh-sched-row">
            <span class="gh-sched-time">0729260007</span>
            <span class="gh-sched-icon" style="background:#f3e8ff; color:#7e22ce;">MV</span>
            <span class="gh-sched-name">Muhammad Vicky Wahyuda</span>
            <span class="gh-sched-doc">Peneliti 3</span>
          </div>
          <div class="gh-sched-row">
            <span class="gh-sched-time">072926002</span>
            <span class="gh-sched-icon" style="background:#fee2e2; color:#dc2626;">AS</span>
            <span class="gh-sched-name">Afrizal Subagja</span>
            <span class="gh-sched-doc">Peneliti 4</span>
          </div>
          <div class="gh-sched-row" style="background:#f8fafc; border-radius:10px; padding:10px 8px; margin-top:4px;">
            <span class="gh-sched-time" style="color:#059669; font-weight:700;">DOSEN</span>
            <span class="gh-sched-icon" style="background:#e0f2fe; color:#0284c7;">FD</span>
            <span class="gh-sched-name">Dr. Fajar Delli Wihartiko, S.Si., MM., M.Kom.</span>
            <span class="gh-sched-doc">Pengampu</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------- Detailed Tabs Below Main Dashboard ---------------------------
tab_features, tab_folds, tab_about = st.tabs(
    [
        ":material/biotech: Feature Stability Analysis",
        ":material/table_rows: Per-Fold Evaluation Records",
        ":material/clinical_notes: Dataset & Methodology",
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
                    {"Biomarker Feature": list(counts.keys()), "Selected Count": list(counts.values())}
                )
                freq["Stability (%)"] = (freq["Selected Count"] / max(total, 1) * 100).round(2)
                freq = freq.sort_values(
                    ["Selected Count", "Biomarker Feature"], ascending=[False, True]
                ).head(20)

                col_chart, col_tbl = st.columns([1.25, 1.0])
                with col_chart:
                    fig, ax = plt.subplots(
                        figsize=(7.2, max(3.6, min(6.6, len(freq) * 0.3))), dpi=140
                    )
                    freq_plot = freq.sort_values("Selected Count", ascending=True)
                    ax.barh(
                        freq_plot["Biomarker Feature"],
                        freq_plot["Selected Count"],
                        color=METHOD_COLORS.get(chosen_feature_method, "#10b981"),
                        height=0.6,
                    )
                    ax.set_title(
                        f"Top 20 Most Stable Features — {chosen_feature_method} ({total} Folds)",
                        loc="left",
                        fontsize=10.5,
                        fontweight="bold",
                        color="#111c44",
                        pad=10,
                    )
                    ax.set_xlabel("Selection Frequency", color="#64748b", fontsize=8.5)
                    ax.tick_params(axis="both", labelsize=8.5, colors="#334155", length=0)
                    ax.grid(axis="x", linestyle="-", linewidth=0.5, color="#f1f5f9")
                    ax.set_axisbelow(True)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.set_xlim(0, max(freq_plot["Selected Count"].max() * 1.15, 1))
                    for i, value in enumerate(freq_plot["Selected Count"]):
                        ax.text(
                            value + 0.08,
                            i,
                            f"{int(value)}/{total}",
                            va="center",
                            fontsize=8,
                            color="#111c44",
                            fontweight="bold",
                        )
                    fig.patch.set_facecolor("#ffffff")
                    ax.set_facecolor("#ffffff")
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                with col_tbl:
                    st.dataframe(
                        freq.sort_values("Selected Count", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                        height=380,
                    )

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
            "Download Filtered Fold Data (CSV)",
            icon=":material/download:",
            data=fold_view[display_cols].to_csv(index=False).encode("utf-8-sig"),
            file_name="green_hospital_fold_evaluation.csv",
            mime="text/csv",
        )

with tab_about:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div style="padding: 8px 4px;">
              <h4 style="color:#111c44; margin-top:0;">Breast Cancer Wisconsin (Diagnostic)</h4>
              <p style="color:#475569; font-size:0.88rem; line-height:1.6;">
                Dataset klinis hasil digitasi aspirasi jarum halus (FNA) massa payudara yang mencakup
                <b>569 pasien</b> (212 <i>Malignant</i> / Ganas dan 357 <i>Benign</i> / Jinak) dengan
                <b>30 fitur morfologi inti sel</b>.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div style="padding: 8px 4px;">
              <h4 style="color:#111c44; margin-top:0;">Validation Protocol &amp; Algorithms</h4>
              <p style="color:#475569; font-size:0.88rem; line-height:1.6;">
                Menggunakan pipeline <code>StandardScaler</code> + <code>LogisticRegression</code> yang dievaluasi
                dengan <b>Stratified 5-Fold Cross-Validation</b> pada 3 Random Seed (21, 42, 2026) untuk membandingkan
                <b>Baseline</b>, <b>SFS</b>, <b>RFE</b>, dan <b>Genetic Algorithm</b>.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
