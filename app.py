import streamlit as st
import gdown
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import pickle
import os
from datetime import datetime
import base64


def get_logo_base64():
    svg_logo = '''
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <rect width="100" height="100" rx="20" fill="url(#gradient)"/>
        <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#1a5f7a;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#0d3b4f;stop-opacity:1" />
            </linearGradient>
        </defs>
        <rect x="25" y="35" width="50" height="30" rx="15" fill="white" opacity="0.9"/>
        <text x="50" y="55" font-size="20" text-anchor="middle" fill="#1a5f7a" font-weight="bold">+</text>
    </svg>
    '''
    return base64.b64encode(svg_logo.encode()).decode()


st.set_page_config(
    page_title="Medical AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── APP BACKGROUND ── */
.stApp {
    background: #0b1e2d !important;
}

/* ── MAIN CONTENT TEXT BASE ── */
.stApp p, .stApp span, .stApp div, .stApp label {
    color: #cbd5e1;
}

/* ── Remove default streamlit padding issues ── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
}

/* ── KEYFRAMES ── */
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.96); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.3); }
    50%       { box-shadow: 0 0 0 8px rgba(220, 38, 38, 0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #0f2a3f !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    line-height: 1.7;
}
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
}
section[data-testid="stSidebar"] [data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.1rem !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricDelta"] {
    color: #34d399 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}
section[data-testid="stSidebar"] .stAlert {
    background: rgba(14,77,110,0.35) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    border-radius: 10px;
}
section[data-testid="stSidebar"] .stAlert p {
    color: #bae6fd !important;
    font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] .stSuccess {
    background: rgba(16,185,129,0.15) !important;
    border: 1px solid rgba(52,211,153,0.3) !important;
}
section[data-testid="stSidebar"] .stSuccess p {
    color: #6ee7b7 !important;
}
section[data-testid="stSidebar"] .stInfo p {
    color: #bae6fd !important;
}
section[data-testid="stSidebar"] small {
    color: #64748b !important;
}

/* ── HERO HEADER ── */
.medisafe-header {
    background: linear-gradient(135deg, #0f2a3f 0%, #0e4d6e 55%, #0f6e9e 100%);
    border-radius: 20px;
    padding: 3rem 3.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    animation: fadeDown 0.6s cubic-bezier(0.22,1,0.36,1) both;
    box-shadow: 0 20px 60px rgba(14,77,110,0.35), 0 4px 12px rgba(0,0,0,0.15);
}
.medisafe-header::after {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.medisafe-header::before {
    content: '';
    position: absolute;
    bottom: -100px; left: 30%;
    width: 380px; height: 380px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.brand-row {
    display: flex;
    align-items: center;
    gap: 1.1rem;
    margin-bottom: 0.6rem;
}
.brand-icon {
    width: 58px; height: 58px;
    background: rgba(255,255,255,0.15);
    border: 1.5px solid rgba(255,255,255,0.3);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.85rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
.brand-name {
    font-family: 'DM Serif Display', serif !important;
    font-size: 3rem !important;
    font-weight: 400 !important;
    color: #ffffff !important;
    letter-spacing: -0.01em;
    line-height: 1;
    text-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.header-tagline {
    font-size: 1.05rem !important;
    color: #93c5fd !important;
    font-weight: 400;
    margin: 0 0 1.5rem 0 !important;
    letter-spacing: 0.01em;
}
.header-badges {
    display: flex; gap: 0.55rem; flex-wrap: wrap;
}
.hbadge {
    padding: 0.32rem 1rem;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #e0f2fe !important;
    background: rgba(255,255,255,0.12);
    border: 1.5px solid rgba(255,255,255,0.22);
    backdrop-filter: blur(4px);
}

/* ── SECTION LABEL ── */
.section-label {
    font-size: 0.67rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #7dd3fc !important;
    margin-bottom: 0.85rem;
    display: block;
}

/* ── PANELS ── */
.panel {
    background: #112233;
    border: 1px solid #1e3a52;
    border-radius: 18px;
    padding: 1.8rem;
    animation: scaleIn 0.45s cubic-bezier(0.22,1,0.36,1) both;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.panel-title {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    margin: 0 0 0.25rem 0 !important;
}
.panel-sub {
    font-size: 0.82rem !important;
    color: #7fa8c9 !important;
    margin: 0 0 1.3rem 0 !important;
    line-height: 1.5;
}

/* ── THRESHOLD CHIP ── */
.threshold-chip {
    background: linear-gradient(135deg, #0d2137, #0e3350);
    border: 1.5px solid #1e4976;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.3rem;
}
.threshold-chip .tc-label {
    font-size: 0.82rem !important;
    color: #93c5fd !important;
    font-weight: 700 !important;
}
.threshold-chip .tc-sub {
    font-size: 0.72rem !important;
    color: #5b9bd5 !important;
    margin-top: 3px;
}
.threshold-chip .tc-value {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    color: #7dd3fc !important;
    font-variant-numeric: tabular-nums;
}

/* ── IMAGE INFO ── */
.img-info {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    margin-top: 0.9rem;
    font-size: 0.83rem !important;
    color: #14532d !important;
    line-height: 1.6;
}
.img-info strong { color: #052e16 !important; font-weight: 700 !important; }

/* ── ANALYZE BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #0f2a3f 0%, #0e4d6e 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.85rem 1.5rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.22s ease !important;
    box-shadow: 0 4px 14px rgba(14,77,110,0.4) !important;
    width: 100% !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0e3d5c 0%, #0f6e9e 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(14,77,110,0.45) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── RESULTS DIVIDER ── */
.results-divider {
    border: none;
    border-top: 2px solid #1e3a52;
    margin: 2rem 0 1.8rem;
}

/* ── KPI CARDS ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1.1rem;
    margin-bottom: 2rem;
    animation: fadeUp 0.45s 0.05s cubic-bezier(0.22,1,0.36,1) both;
}
.kpi-card {
    background: #112233;
    border: 1px solid #1e3a52;
    border-radius: 16px;
    padding: 1.5rem 1.7rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #0e4d6e, #38bdf8);
    border-radius: 16px 16px 0 0;
}
.kpi-card.kpi-danger::before {
    background: linear-gradient(90deg, #dc2626, #f87171);
}
.kpi-card .kpi-icon {
    font-size: 1.4rem;
    margin-bottom: 0.7rem;
    display: block;
}
.kpi-card .kpi-label {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    color: #7fa8c9 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
    display: block;
}
.kpi-card .kpi-value {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #e2f0fb !important;
    line-height: 1;
    display: block;
    font-variant-numeric: tabular-nums;
}
.kpi-card.kpi-danger .kpi-value { color: #f87171 !important; }
.kpi-card .kpi-sub {
    font-size: 0.77rem !important;
    color: #5b8aad !important;
    margin-top: 0.4rem;
    display: block;
}

/* ── MEDICINE PILLS ── */
.med-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}
.med-count-badge {
    background: #0d2e4a;
    color: #7dd3fc !important;
    font-size: 0.73rem !important;
    font-weight: 700 !important;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    border: 1px solid #1a4060;
}
.medicine-pill {
    background: #0d2035;
    border: 1px solid #1a3d5c;
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    transition: all 0.2s;
}
.medicine-pill:hover {
    border-color: #38bdf8;
    background: #0f2c48;
    box-shadow: 0 3px 12px rgba(56,189,248,0.15);
    transform: translateX(4px);
}
.pill-icon { font-size: 1rem; flex-shrink: 0; }
.pill-name {
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    color: #e2f0fb !important;
    flex: 1;
}
.pill-conf { display: none; }

/* ── INTERACTION SECTION HEADER ── */
.int-section-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1.2rem;
    padding: 1.1rem 1.4rem;
    background: #1a1200;
    border: 1px solid #3d2600;
    border-radius: 12px;
}
.int-section-icon { font-size: 1.3rem; }
.int-section-title {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #fbbf24 !important;
    margin: 0 !important;
}
.int-section-sub {
    font-size: 0.78rem !important;
    color: #d97706 !important;
    margin: 0 !important;
}

/* ── SEVERITY ROW ── */
.severity-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.85rem;
    margin-bottom: 1.4rem;
    animation: fadeUp 0.4s 0.05s cubic-bezier(0.22,1,0.36,1) both;
}
.sev-card {
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.sev-card.sev-critical {
    background: #fef2f2;
    border: 1.5px solid #fca5a5;
}
.sev-card.sev-moderate {
    background: #fff7ed;
    border: 1.5px solid #fdba74;
}
.sev-card.sev-minor {
    background: #fefce8;
    border: 1.5px solid #fde047;
}
.sev-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.8);
}
.sev-critical .sev-dot { background: #dc2626; }
.sev-moderate .sev-dot { background: #ea580c; }
.sev-minor    .sev-dot { background: #ca8a04; }
.sev-num {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    line-height: 1;
    display: block;
}
.sev-critical .sev-num { color: #b91c1c !important; }
.sev-moderate .sev-num { color: #c2410c !important; }
.sev-minor    .sev-num { color: #a16207 !important; }
.sev-lbl {
    font-size: 0.69rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    display: block;
    margin-top: 2px;
}
.sev-critical .sev-lbl { color: #991b1b !important; }
.sev-moderate .sev-lbl { color: #9a3412 !important; }
.sev-minor    .sev-lbl { color: #854d0e !important; }

/* ── INTERACTION CARDS ── */
.int-card {
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.85rem;
    animation: fadeUp 0.35s cubic-bezier(0.22,1,0.36,1) both;
    transition: transform 0.18s;
}
.int-card:hover { transform: translateX(5px); }
.int-card.int-critical {
    background: #fef2f2;
    border: 1.5px solid #fca5a5;
    border-left: 5px solid #dc2626;
    animation: pulseRed 2.8s 0.6s infinite;
}
.int-card.int-moderate {
    background: #fff7ed;
    border: 1.5px solid #fdba74;
    border-left: 5px solid #ea580c;
}
.int-card.int-minor {
    background: #fefce8;
    border: 1.5px solid #fde047;
    border-left: 5px solid #ca8a04;
}
.int-badge {
    display: inline-block;
    font-size: 0.66rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    margin-bottom: 0.5rem;
}
.int-critical .int-badge {
    background: #fee2e2;
    color: #991b1b !important;
    border: 1px solid #fca5a5;
}
.int-moderate .int-badge {
    background: #ffedd5;
    color: #9a3412 !important;
    border: 1px solid #fdba74;
}
.int-minor .int-badge {
    background: #fef9c3;
    color: #854d0e !important;
    border: 1px solid #fde047;
}
.int-drugs {
    font-size: 0.97rem !important;
    font-weight: 700 !important;
    color: #0f2a3f !important;
    margin-bottom: 0.35rem;
    line-height: 1.35;
}
.int-note {
    font-size: 0.81rem !important;
    color: #374151 !important;
    font-weight: 400 !important;
    line-height: 1.55;
}

/* ── SAFE CARD ── */
.safe-card {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1.5px solid #86efac;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: scaleIn 0.4s cubic-bezier(0.22,1,0.36,1) both;
    box-shadow: 0 4px 20px rgba(22,163,74,0.1);
}
.safe-icon { font-size: 2.5rem; margin-bottom: 0.7rem; display: block; }
.safe-title {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #14532d !important;
    margin-bottom: 0.45rem !important;
}
.safe-sub {
    font-size: 0.83rem !important;
    color: #166534 !important;
    line-height: 1.6;
}

/* ── FOOTER ── */
.medisafe-footer {
    margin-top: 3rem;
    padding: 1.8rem 2.5rem;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(15,42,63,0.06);
    animation: fadeIn 0.5s 0.2s both;
}
.footer-disclaimer {
    font-size: 0.8rem !important;
    color: #374151 !important;
    line-height: 1.7;
    max-width: 680px;
    margin: 0 auto 0.6rem;
}
.footer-disclaimer strong { color: #0f2a3f !important; font-weight: 700 !important; }
.footer-copy {
    font-size: 0.71rem !important;
    color: #94a3b8 !important;
    margin-top: 0.5rem;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #bfdbfe !important;
    border-radius: 12px !important;
    background: #f0f9ff !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #0e4d6e !important;
    background: #e0f2fe !important;
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span {
    color: #334155 !important;
    font-weight: 500 !important;
}

/* ── CHECKBOX ── */
.stCheckbox label {
    font-size: 0.9rem !important;
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* ── PROGRESS BAR ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #0e4d6e, #38bdf8) !important;
    border-radius: 999px !important;
}

/* ── ALERTS ── */
.stAlert {
    border-radius: 10px !important;
    font-size: 0.86rem !important;
}
.stAlert p {
    color: #1e293b !important;
    font-weight: 500 !important;
}
.stWarning { background: #fff7ed !important; border: 1px solid #fed7aa !important; }
.stWarning p { color: #7c2d12 !important; }
.stInfo p { color: #1e3a8a !important; }
.stSuccess p { color: #14532d !important; }

/* ── SPINNER ── */
.stSpinner p { color: #334155 !important; font-weight: 500 !important; }

/* ── IMAGE CAPTION ── */
.stImage figcaption {
    font-size: 0.74rem !important;
    color: #64748b !important;
    text-align: center;
}

/* ── DETAILED REPORT LABEL ── */
.report-label-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1.5px solid #f1f5f9;
}
</style>
""", unsafe_allow_html=True)


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="medisafe-header">
    <div class="brand-row">
        <div class="brand-icon">🧬</div>
        <span class="brand-name">Medical AI</span>
    </div>
    <p class="header-tagline">AI-Powered Prescription Drug Interaction Checker</p>
    <div class="header-badges">
        <span class="hbadge">Deep Learning Vision</span>
        <span class="hbadge">Real-time Analysis</span>
        <span class="hbadge">Drug Interaction Database</span>
        <span class="hbadge">Clinical Risk Assessment</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 About MediSafe")
    st.markdown("""
    MediSafe uses advanced deep learning to:
    - 🔍 Detect medicines from prescription images
    - ⚠️ Identify potential drug interactions
    - 📊 Provide risk assessment
    - 💡 Offer safety recommendations
    """)
    st.markdown("---")
    st.markdown("### 📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", "✅ Active", delta="Ready")
    with col2:
        st.metric("Mode", "Live", delta="Online")
    st.markdown("---")
    st.markdown("### 🎯 How to Use")
    st.markdown("""
    1. Upload a clear prescription image
    2. Click **Analyze Prescription**
    3. Review detected medicines
    4. Check for interactions
    5. Get safety recommendations
    """)
    st.markdown("---")
    st.markdown("### ⚡ Pro Tips")
    st.info("""
    - Use well-lit, clear images
    - Ensure text is readable
    - Multiple angles increase accuracy
    - Check interactions for all detected drugs
    """)


# ── MODEL & DATA LOADERS ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import gdown
    url = "https://drive.google.com/file/d/152zjFxo_ExjDMoR91SKyuOGMtUR0Ejt-/view?usp=sharing"
    model_path = "model.pth"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model (107MB)... This may take a minute"):
            gdown.download(url, model_path, quiet=False)

    with open('medicines.pkl', 'rb') as f:
        medicines = pickle.load(f)

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, len(medicines))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, medicines


@st.cache_data
def load_interactions():
    possible_files = ['ddinter_merged.csv', 'interactions.csv', 'ddinter.csv']
    for file in possible_files:
        if os.path.exists(file):
            ddinter = pd.read_csv(file)
            inter_dict = {}
            for _, row in ddinter.iterrows():
                drug_a = str(row['Drug_A']).strip().lower()
                drug_b = str(row['Drug_B']).strip().lower()
                level = str(row['Level']).strip()
                inter_dict[(drug_a, drug_b)] = level
                inter_dict[(drug_b, drug_a)] = level
            return inter_dict, file
    return {}, None


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

THRESHOLD = 0.01

brand_generic = {
    'neuro-b': 'vitamin b complex',
    'sergel': 'esomeprazole',
    'diasulin': 'insulin',
    'maxpro': 'esomeprazole',
    'clopid': 'clopidogrel',
    'aspirin': 'aspirin',
    'paracetamol': 'paracetamol',
    'pregaba': 'pregabalin',
    'tramadol': 'tramadol',
    'zolpidem': 'zolpidem',
}


def get_generic(med):
    med_clean = med.lower().split('.')[-1].strip().split()[0]
    return brand_generic.get(med_clean, med_clean)


# ── DB STATUS ──────────────────────────────────────────────────────────────────
inter_dict, inter_file = load_interactions()
if inter_file:
    st.sidebar.success(f"✅ Interaction DB: `{inter_file}`")
    st.sidebar.info(f"📊 {len(inter_dict):,} interaction pairs loaded")


# ── MAIN PANELS ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">📸 Upload Prescription</p>', unsafe_allow_html=True)
    st.markdown('<p class="panel-sub">Supported formats: JPG, JPEG, PNG &nbsp;·&nbsp; Use a clear, well-lit image for best results</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear, well-lit image of the prescription"
    )
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption="Uploaded Prescription", use_column_width=True)
        st.markdown(f"""
        <div class="img-info">
            <strong>✅ Image ready for analysis</strong><br>
            Resolution: {image.size[0]} × {image.size[1]} px &nbsp;·&nbsp; Mode: RGB
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">⚙️ Analysis Settings</p>', unsafe_allow_html=True)
    st.markdown('<p class="panel-sub">Configure detection parameters before running the analysis</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="threshold-chip">
        <div>
            <div class="tc-label">Detection Threshold</div>
            <div class="tc-sub">Minimum confidence score required</div>
        </div>
        <div class="tc-value">{THRESHOLD}</div>
    </div>
    """, unsafe_allow_html=True)
    check_interactions = st.checkbox(
        "⚠️ Enable Drug Interaction Checker",
        value=True,
        help="Cross-reference detected medicines against the interaction database"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("🔬 Analyze Prescription", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── RESULTS ────────────────────────────────────────────────────────────────────
if analyze and uploaded:
    with st.spinner("Analyzing prescription — please wait…"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Loading model…")
        progress_bar.progress(25)
        model, medicines = load_model()

        status_text.text("Processing image…")
        progress_bar.progress(50)
        img_tensor = transform(image).unsqueeze(0)

        status_text.text("Running inference…")
        progress_bar.progress(75)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.sigmoid(outputs)
            pred_values = probs[0].numpy()
            detected = [medicines[i] for i, val in enumerate(pred_values) if val > THRESHOLD]

        status_text.text("Analysis complete.")
        progress_bar.progress(100)

        import time
        time.sleep(0.4)
        progress_bar.empty()
        status_text.empty()

    st.markdown('<hr class="results-divider">', unsafe_allow_html=True)

    if detected:
        # Pre-compute interaction count
        interactions_found = 0
        if check_interactions and len(detected) >= 2 and inter_dict:
            for i in range(len(detected)):
                for j in range(i + 1, len(detected)):
                    gi = get_generic(detected[i])
                    gj = get_generic(detected[j])
                    if (gi, gj) in inter_dict or (gj, gi) in inter_dict:
                        interactions_found += 1

        kpi_class = "kpi-danger" if interactions_found > 0 else ""

        # ── 2-column KPI (Max Confidence removed) ──
        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <span class="kpi-icon">🧬</span>
                <span class="kpi-label">Medicines Detected</span>
                <span class="kpi-value">{len(detected)}</span>
                <span class="kpi-sub">unique medications found</span>
            </div>
            <div class="kpi-card {kpi_class}">
                <span class="kpi-icon">{"⚠️" if interactions_found > 0 else "✅"}</span>
                <span class="kpi-label">Interactions Found</span>
                <span class="kpi-value">{interactions_found}</span>
                <span class="kpi-sub">{"potential drug conflicts" if interactions_found > 0 else "no conflicts detected"}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Medicine List ──
        st.markdown(f"""
        <div class="med-header-row">
            <span class="section-label" style="margin:0;">Detected Medicines</span>
            <span class="med-count-badge">{len(detected)} found</span>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(3)
        for i, med in enumerate(detected[:30]):
            confidence = pred_values[medicines.index(med)]
            with cols[i % 3]:
                st.markdown(f"""
                <div class="medicine-pill">
                    <span class="pill-icon">💊</span>
                    <span class="pill-name">{med}</span>
                    <span class="pill-conf">{confidence:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

        # ── Drug Interactions ──
        if check_interactions and len(detected) >= 2:
            st.markdown("""
            <div class="int-section-header">
                <span class="int-section-icon">⚠️</span>
                <div>
                    <p class="int-section-title">Drug Interaction Analysis</p>
                    <p class="int-section-sub">Cross-referenced against the interaction database</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if inter_dict:
                interactions = []
                for i in range(len(detected)):
                    for j in range(i + 1, len(detected)):
                        gi = get_generic(detected[i])
                        gj = get_generic(detected[j])
                        pair = (gi, gj)
                        rev  = (gj, gi)
                        if pair in inter_dict:
                            interactions.append((detected[i], detected[j], inter_dict[pair]))
                        elif rev in inter_dict:
                            interactions.append((detected[i], detected[j], inter_dict[rev]))

                if interactions:
                    counts = {'major': 0, 'moderate': 0, 'minor': 0}
                    for _, _, lvl in interactions:
                        if lvl.lower() in counts:
                            counts[lvl.lower()] += 1

                    st.markdown(f"""
                    <div class="severity-row">
                        <div class="sev-card sev-critical">
                            <div class="sev-dot"></div>
                            <div>
                                <span class="sev-num">{counts['major']}</span>
                                <span class="sev-lbl">Critical</span>
                            </div>
                        </div>
                        <div class="sev-card sev-moderate">
                            <div class="sev-dot"></div>
                            <div>
                                <span class="sev-num">{counts['moderate']}</span>
                                <span class="sev-lbl">Moderate</span>
                            </div>
                        </div>
                        <div class="sev-card sev-minor">
                            <div class="sev-dot"></div>
                            <div>
                                <span class="sev-num">{counts['minor']}</span>
                                <span class="sev-lbl">Minor</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="report-label-row">
                        <span class="section-label" style="margin:0;">Detailed Interaction Report</span>
                    </div>
                    """, unsafe_allow_html=True)

                    for m1, m2, level in interactions:
                        ll = level.lower()
                        if ll == 'major':
                            st.markdown(f"""
                            <div class="int-card int-critical">
                                <div class="int-badge">🔴 Critical Interaction</div>
                                <div class="int-drugs">{m1} &nbsp;＋&nbsp; {m2}</div>
                                <div class="int-note">Severity: <strong>MAJOR</strong> — Immediate medical consultation required. Do not continue without professional guidance.</div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif ll == 'moderate':
                            st.markdown(f"""
                            <div class="int-card int-moderate">
                                <div class="int-badge">🟡 Moderate Interaction</div>
                                <div class="int-drugs">{m1} &nbsp;＋&nbsp; {m2}</div>
                                <div class="int-note">Severity: <strong>MODERATE</strong> — Monitor for side effects carefully. Consult your pharmacist before proceeding.</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="int-card int-minor">
                                <div class="int-badge">🟢 Minor Interaction</div>
                                <div class="int-drugs">{m1} &nbsp;＋&nbsp; {m2}</div>
                                <div class="int-note">Severity: <strong>MINOR</strong> — Generally considered safe. Monitor as a precaution and inform your doctor.</div>
                            </div>
                            """, unsafe_allow_html=True)

                else:
                    st.markdown("""
                    <div class="safe-card">
                        <span class="safe-icon">✅</span>
                        <div class="safe-title">No Dangerous Interactions Detected</div>
                        <div class="safe-sub">
                            The prescription appears safe based on available database records.<br>
                            Always consult your healthcare provider for final clinical approval.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Interaction database not found. Please place `ddinter_merged.csv` in the application directory.")
                st.info("📥 Download `ddinter_merged.csv` from your data source and place it in this folder.")
    else:
        st.warning("⚠️ No medicines detected. Please try again with a clearer, well-lit prescription image.")


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="medisafe-footer">
    <p class="footer-disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong>
        MediSafe is an AI-powered informational tool only and is not a substitute for professional
        medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider
        before making any medical decisions based on this output.
    </p>
    <p class="footer-copy">© 2025 MediSafe &nbsp;·&nbsp; AI-Powered Healthcare Solution &nbsp;·&nbsp; Version 3.0</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<small>Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>",
    unsafe_allow_html=True
)