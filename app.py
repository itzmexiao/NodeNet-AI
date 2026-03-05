import streamlit as st
import numpy as np
import onnxruntime as rt
from PIL import Image
import time
import plotly.graph_objects as go
import pandas as pd

# ═══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Lung  · Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
#  MASTER CSS  — Obsidian Biopunk theme
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
  --ink:       #080c10;
  --surface:   #0e1318;
  --card:      #131920;
  --card2:     #192028;
  --rim:       #1e2830;
  --border:    #243040;
  --cyan:      #00e5c8;
  --cyan-dim:  #00e5c820;
  --cyan-mid:  #00e5c860;
  --blue:      #38bdf8;
  --rose:      #fb4f6b;
  --amber:     #fbbf24;
  --emerald:   #10d98a;
  --text:      #d8e8f0;
  --muted:     #5a7080;
  --mono:      'JetBrains Mono', monospace;
  --sans:      'Syne', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: var(--sans); }
.stApp { background: var(--ink); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ══ SIDEBAR ════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1419 0%, #0e1822 100%) !important;
  border-right: 1px solid var(--border) !important;
  padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--muted) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  transition: all 0.2s !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  border-color: var(--cyan) !important;
  color: var(--cyan) !important;
  background: var(--cyan-dim) !important;
  transform: none !important;
  box-shadow: none !important;
}

.sb-brand {
  background: linear-gradient(135deg, #0a1c24, #0e2230);
  border-bottom: 1px solid var(--border);
  padding: 1.4rem 1.2rem 1.2rem;
  margin-bottom: 0.5rem;
}
.sb-brand .logo { font-size: 1.4rem; font-weight: 800; letter-spacing: -1px; color: #fff; }
.sb-brand .logo span { color: var(--cyan); }
.sb-brand .tagline {
  font-size: 0.7rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 2px;
  margin-top: 2px; font-family: var(--mono);
}

.sb-type {
  display: flex; align-items: center; gap: 0.6rem;
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;
  transition: border-color 0.2s;
}
.sb-type:hover { border-color: var(--cyan); }
.sb-type .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.sb-type .name { font-size: 0.82rem; font-weight: 600; color: var(--text); }
.sb-type .risk { font-size: 0.68rem; color: var(--muted); font-family: var(--mono); }

.sb-heading {
  font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 2.5px; color: var(--cyan); padding: 0.8rem 0 0.4rem;
  font-family: var(--mono);
}

.sb-step { display: flex; align-items: flex-start; gap: 0.7rem; margin-bottom: 0.6rem; }
.sb-step .num {
  width: 20px; height: 20px; border-radius: 50%;
  background: var(--cyan-dim); border: 1px solid var(--cyan-mid);
  color: var(--cyan); font-size: 0.65rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; font-family: var(--mono);
}
.sb-step .txt { font-size: 0.8rem; color: var(--muted); line-height: 1.4; padding-top: 2px; }

.hist-row {
  display: flex; align-items: center; gap: 0.8rem;
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.6rem 0.9rem; margin-bottom: 0.4rem;
  font-size: 0.8rem;
}
.hist-row .hnum { font-family: var(--mono); color: var(--muted); font-size: 0.68rem; width: 20px; }
.hist-row .hname { color: var(--text); font-weight: 600; flex: 1; }
.hist-row .hconf { font-family: var(--mono); font-size: 0.75rem; }

/* ══ HERO ════════════════════════════════════════════════ */
.hero {
  position: relative;
  background: linear-gradient(135deg, #090f15 0%, #0a1a22 40%, #091420 100%);
  border: 1px solid var(--border); border-radius: 20px;
  padding: 2.2rem 2.5rem; margin-bottom: 1.5rem; overflow: hidden;
}
.hero::before {
  content: ''; position: absolute; top: -60px; right: -60px;
  width: 320px; height: 320px;
  background: radial-gradient(circle, #00e5c812 0%, transparent 65%);
  border-radius: 50%; pointer-events: none;
}
.hero-grid {
  position: absolute; inset: 0;
  background-image: linear-gradient(var(--border) 1px, transparent 1px),
    linear-gradient(90deg, var(--border) 1px, transparent 1px);
  background-size: 40px 40px; opacity: 0.22;
  mask-image: radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%);
}
.hero-content { position: relative; z-index: 1; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: var(--cyan-dim); border: 1px solid var(--cyan-mid);
  border-radius: 20px; padding: 0.25rem 0.8rem;
  font-size: 0.68rem; font-weight: 600; color: var(--cyan);
  letter-spacing: 1.5px; text-transform: uppercase;
  font-family: var(--mono); margin-bottom: 0.9rem;
}
.hero-badge .pulse {
  width: 6px; height: 6px; background: var(--cyan);
  border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.7); }
}
.hero h1 {
  font-size: 2.6rem; font-weight: 800; color: #fff;
  margin: 0 0 0.4rem; letter-spacing: -1.5px; line-height: 1.1;
}
.hero h1 .hl { color: var(--cyan); }
.hero p { color: var(--muted); font-size: 0.95rem; margin: 0; font-weight: 400; max-width: 500px; }

/* ══ STAT PILLS ══════════════════════════════════════════ */
.stat-strip {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 0.8rem; margin-bottom: 1.6rem;
}
.stat-pill {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem 1.2rem;
  display: flex; align-items: center; gap: 0.8rem;
  transition: border-color 0.2s, transform 0.2s;
}
.stat-pill:hover { border-color: var(--cyan); transform: translateY(-2px); }
.stat-pill .icon { font-size: 1.5rem; line-height: 1; }
.stat-pill .val { font-size: 1.3rem; font-weight: 800; color: var(--cyan); font-family: var(--mono); letter-spacing: -0.5px; }
.stat-pill .lbl { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-top: 1px; }

/* ══ PANEL TITLE ═════════════════════════════════════════ */
.panel-title {
  font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 2.5px; color: var(--cyan); font-family: var(--mono);
  margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;
}
.panel-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border), transparent); }

/* ══ RESULT HERO CARD ════════════════════════════════════ */
.result-hero {
  border-radius: 14px; padding: 1.6rem; margin-bottom: 1.2rem;
  position: relative; overflow: hidden; text-align: center;
}
.result-hero.state-normal  { background: linear-gradient(135deg, #091f14, #0b2618); border: 1px solid #10d98a44; }
.result-hero.state-cancer  { background: linear-gradient(135deg, #1a0810, #200c14); border: 1px solid #fb4f6b44; }
.result-hero.state-warn    { background: linear-gradient(135deg, #1a1000, #1f1500); border: 1px solid #fbbf2444; }
.rh-glow {
  position: absolute; top: -40px; left: 50%; transform: translateX(-50%);
  width: 200px; height: 200px; border-radius: 50%; opacity: 0.15;
}
.rh-eye { font-size: 2.4rem; margin-bottom: 0.5rem; position: relative; }
.rh-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 3px; color: var(--muted); font-family: var(--mono); margin-bottom: 0.4rem; }
.rh-class { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 0.2rem; }
.rh-conf { font-size: 3.2rem; font-weight: 800; font-family: var(--mono); line-height: 1; letter-spacing: -2px; }
.rh-unit { font-size: 1rem; color: var(--muted); font-weight: 400; letter-spacing: 0; }
.rh-sub { font-size: 0.75rem; color: var(--muted); margin-top: 0.4rem; }

.risk-row { display: flex; gap: 0.6rem; margin-bottom: 1rem; flex-wrap: wrap; }
.risk-chip {
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.3rem 0.7rem;
  font-size: 0.72rem; font-family: var(--mono); color: var(--muted);
}
.risk-chip .val { font-weight: 600; }

/* ══ META STRIP ══════════════════════════════════════════ */
.meta-strip { display: flex; gap: 0.5rem; margin: 0.7rem 0; flex-wrap: wrap; }
.meta-chip {
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.25rem 0.6rem;
  font-size: 0.7rem; font-family: var(--mono); color: var(--muted);
}
.meta-chip .v { color: var(--cyan); }

/* ══ DROPZONE ════════════════════════════════════════════ */
.dropzone {
  background: var(--card); border: 1.5px dashed var(--border);
  border-radius: 14px; padding: 2.5rem 1.5rem; text-align: center;
}
.dz-icon { font-size: 2.5rem; margin-bottom: 0.6rem; opacity: 0.5; }
.dz-title { font-weight: 600; color: var(--text); margin-bottom: 0.3rem; }
.dz-sub { font-size: 0.82rem; color: var(--muted); line-height: 1.6; }

/* ══ BUTTONS ═════════════════════════════════════════════ */
.stButton > button {
  background: linear-gradient(135deg, #00c4ae 0%, #29b8f0 100%) !important;
  color: #05080b !important; border: none !important;
  border-radius: 10px !important; font-weight: 800 !important;
  font-family: var(--sans) !important; letter-spacing: 0.5px !important;
  font-size: 0.9rem !important; padding: 0.65rem 1.5rem !important;
  transition: all 0.18s !important;
  box-shadow: 0 0 20px #00e5c820 !important;
}
.stButton > button:hover {
  opacity: 0.9 !important; transform: translateY(-2px) !important;
  box-shadow: 0 6px 24px #00e5c830 !important;
}

/* ══ TABS ════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: var(--card2) !important; border-radius: 10px !important;
  padding: 3px !important; gap: 2px !important; border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important; color: var(--muted) !important;
  font-size: 0.8rem !important; font-weight: 600 !important;
  font-family: var(--sans) !important; padding: 0.4rem 0.9rem !important;
}
.stTabs [aria-selected="true"] { background: var(--card) !important; color: var(--cyan) !important; }
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; padding: 0.8rem 0 0 !important; }

/* ══ MISC ════════════════════════════════════════════════ */
.stAlert { border-radius: 10px !important; border-left-width: 3px !important; background: var(--card) !important; }
details { background: var(--card2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
summary { color: var(--muted) !important; font-size: 0.82rem !important; }
hr { border-color: var(--border) !important; }
[data-testid="stMetricValue"] { font-family: var(--mono) !important; color: var(--cyan) !important; font-size: 1.4rem !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.72rem !important; }

.footer {
  text-align: center; padding: 1.5rem 0 0.5rem;
  border-top: 1px solid var(--border); margin-top: 1rem;
}
.ft-logo { font-size: 1rem; font-weight: 800; letter-spacing: -0.5px; color: var(--text); }
.ft-logo span { color: var(--cyan); }
.ft-sub { font-size: 0.72rem; color: var(--muted); margin-top: 0.3rem; font-family: var(--mono); }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  CLASS DATA
# ═══════════════════════════════════════════════════════════
CLASS_LABELS = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

CLASS_META = {
    'Adenocarcinoma': {
        'icon': '🔴', 'risk': 'HIGH', 'state': 'cancer', 'short': 'Adeno',
        'color': '#fb4f6b', 'freq': '40% of cases', 'survival': '~25% (5-yr)',
        'desc': 'Most common lung cancer (~40%). Originates in mucus-secreting gland cells, typically in peripheral lung regions. Often found in non-smokers.',
        'action': 'Immediate oncology referral required. PET-CT scan and tissue biopsy strongly recommended.',
    },
    'Large Cell Carcinoma': {
        'icon': '🟠', 'risk': 'VERY HIGH', 'state': 'cancer', 'short': 'LCC',
        'color': '#f97316', 'freq': '15% of cases', 'survival': '~12% (5-yr)',
        'desc': 'Fast-growing, aggressive cancer (~15%) that can appear anywhere in the lung. Tends to metastasize early and rapidly.',
        'action': 'Urgent multidisciplinary team review. Immediate staging and treatment plan essential.',
    },
    'Normal': {
        'icon': '🟢', 'risk': 'NONE', 'state': 'normal', 'short': 'Normal',
        'color': '#10d98a', 'freq': 'Healthy', 'survival': 'N/A',
        'desc': 'No malignant tissue detected. Lung architecture appears within expected parameters.',
        'action': 'Continue annual CT screening if high-risk (smoker, age 50+). Maintain a healthy lifestyle.',
    },
    'Squamous Cell Carcinoma': {
        'icon': '🟡', 'risk': 'HIGH', 'state': 'warn', 'short': 'SqCC',
        'color': '#fbbf24', 'freq': '30% of cases', 'survival': '~16% (5-yr)',
        'desc': 'Strongly linked to smoking (~30%). Develops in flat cells lining bronchial airways, usually centrally located.',
        'action': 'Pulmonologist referral and bronchoscopy advised. Immediate smoking cessation critical.',
    },
}

# ═══════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════
for k, v in [
    ('analyzed', False), ('prediction', None), ('confidence', None),
    ('predictions', None), ('history', []), ('scan_count', 0)
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        s = rt.InferenceSession('lung_cancer_model.onnx')
        return s, True
    except:
        return None, False

session, model_ok = load_model()
input_name = session.get_inputs()[0].name if model_ok else None

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="logo">Lung<span>AI</span></div>
        <div class="tagline">v3.0 · ONNX Runtime · CNN</div>
    </div>
    """, unsafe_allow_html=True)

    if model_ok:
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Place `lung_cancer_model.onnx` here")

    st.markdown('<div class="sb-heading">Cancer Classes</div>', unsafe_allow_html=True)
    for label in CLASS_LABELS:
        m = CLASS_META[label]
        st.markdown(f"""
        <div class="sb-type">
            <div class="dot" style="background:{m['color']};box-shadow:0 0 6px {m['color']}80;"></div>
            <div>
                <div class="name">{label}</div>
                <div class="risk">Risk: {m['risk']} · {m['freq']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sb-heading">How to Use</div>', unsafe_allow_html=True)
    for i, s in enumerate(["Upload a CT scan (JPG/PNG)", "Click Analyze Scan", "Review charts & prediction", "Consult a certified physician"], 1):
        st.markdown(f'<div class="sb-step"><div class="num">{i}</div><div class="txt">{s}</div></div>', unsafe_allow_html=True)

    if st.session_state['scan_count'] > 0:
        st.markdown('<div class="sb-heading">Session Stats</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Total Scans", st.session_state['scan_count'])
        c2.metric("Normal", sum(1 for h in st.session_state['history'] if h['pred'] == 'Normal'))

    if st.session_state['history']:
        st.markdown('<div class="sb-heading">Recent Scans</div>', unsafe_allow_html=True)
        for i, h in enumerate(reversed(st.session_state['history'][-5:])):
            idx = len(st.session_state['history']) - i
            m = CLASS_META[h['pred']]
            st.markdown(f"""
            <div class="hist-row">
                <div class="hnum">#{idx}</div>
                <div class="hname">{h['pred']}</div>
                <div class="hconf" style="color:{m['color']};">{h['conf']:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
    st.warning("⚕️ Educational use only. Not a substitute for clinical diagnosis.")
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    if st.button("↺  Reset Session", use_container_width=True):
        for k in ['analyzed', 'prediction', 'confidence', 'predictions']:
            st.session_state[k] = False if k == 'analyzed' else None
        st.rerun()

# ═══════════════════════════════════════════════════════════
#  STOP IF NO MODEL
# ═══════════════════════════════════════════════════════════
if not model_ok:
    st.error("⚠️ `lung_cancer_model.onnx` not found. Place it in the same folder as `app.py` and restart.")
    st.stop()

# ═══════════════════════════════════════════════════════════
#  HERO HEADER
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <div class="hero-grid"></div>
    <div class="hero-content">
        <div class="hero-badge">
            <div class="pulse"></div>
            AI-Powered · Live Analysis
        </div>
        <h1>NodeNet<span class="hl"> AI</span></h1>
        <p>Upload a CT scan to classify lung tissue using a trained CNN — powered by ONNX Runtime, no TensorFlow required.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stat-strip">
    <div class="stat-pill"><div class="icon">🏥</div><div><div class="val">4</div><div class="lbl">Cancer Classes</div></div></div>
    <div class="stat-pill"><div class="icon">🧠</div><div><div class="val">CNN</div><div class="lbl">Architecture</div></div></div>
    <div class="stat-pill"><div class="icon">⚡</div><div><div class="val">ONNX</div><div class="lbl">Runtime Engine</div></div></div>
    <div class="stat-pill"><div class="icon">🔬</div><div><div class="val">{st.session_state['scan_count']}</div><div class="lbl">Scans This Session</div></div></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  MAIN — 2 COLUMNS
# ═══════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1.15], gap="large")

# ────────────────────────────────────────────────────────
#  LEFT — Upload
# ────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="panel-title">📤 CT Scan Input</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CT scan", type=["jpg","jpeg","png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        img_pil     = Image.open(uploaded_file).convert('RGB')
        img_resized = img_pil.resize((150, 150))
        st.image(img_pil, caption="Uploaded CT Scan", use_column_width=True)

        st.markdown(f"""
        <div class="meta-strip">
            <div class="meta-chip">W: <span class="v">{img_pil.width}px</span></div>
            <div class="meta-chip">H: <span class="v">{img_pil.height}px</span></div>
            <div class="meta-chip">Mode: <span class="v">{img_pil.mode}</span></div>
            <div class="meta-chip">File: <span class="v">{uploaded_file.name}</span></div>
            <div class="meta-chip">Size: <span class="v">{uploaded_file.size // 1024} KB</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        if st.button("🔬  Analyze Scan", use_container_width=True):
            with st.spinner("Running CNN inference…"):
                time.sleep(1.0)
                arr  = np.array(img_resized, dtype=np.float32) / 255.0
                arr  = np.expand_dims(arr, axis=0)
                raw  = session.run(None, {input_name: arr})[0]
                pred = CLASS_LABELS[np.argmax(raw)]
                conf = float(np.max(raw)) * 100

                st.session_state.update({
                    'analyzed':   True,
                    'prediction': pred,
                    'confidence': conf,
                    'predictions': raw[0],
                    'scan_count': st.session_state['scan_count'] + 1,
                })
                st.session_state['history'].append({'pred': pred, 'conf': conf})
            st.rerun()
    else:
        st.markdown("""
        <div class="dropzone">
            <div class="dz-icon">🩻</div>
            <div class="dz-title">No image loaded</div>
            <div class="dz-sub">Drag & drop or click Browse<br>Supports JPG · JPEG · PNG</div>
        </div>
        """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────
#  RIGHT — Results
# ────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="panel-title">📊 Analysis Results</div>', unsafe_allow_html=True)

    if st.session_state['analyzed'] and st.session_state['prediction']:
        pred  = st.session_state['prediction']
        conf  = st.session_state['confidence']
        probs = st.session_state['predictions']
        meta  = CLASS_META[pred]
        clr   = meta['color']
        state = meta['state']

        # Result hero
        st.markdown(f"""
        <div class="result-hero state-{state}">
            <div class="rh-glow" style="background:radial-gradient(circle,{clr}33,transparent 70%);"></div>
            <div class="rh-eye">{meta['icon']}</div>
            <div class="rh-label">Prediction Result</div>
            <div class="rh-class" style="color:{clr};">{pred}</div>
            <div class="rh-conf" style="color:{clr};">{conf:.1f}<span class="rh-unit">%</span></div>
            <div class="rh-sub">Model confidence · CNN + ONNX</div>
        </div>
        """, unsafe_allow_html=True)

        # Risk chips
        st.markdown(f"""
        <div class="risk-row">
            <div class="risk-chip">Risk: <span class="val" style="color:{clr};">{meta['risk']}</span></div>
            <div class="risk-chip">Frequency: <span class="val">{meta['freq']}</span></div>
            <div class="risk-chip">5-yr Survival: <span class="val">{meta['survival']}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Alert
        if pred == 'Normal':
            st.success(f"✅ **{meta['action']}**")
        else:
            st.error(f"⚠️ **{meta['action']}**")

        st.markdown("")

        # ── Charts ──────────────────────────────────────
        CHART_COLORS = [CLASS_META[l]['color'] for l in CLASS_LABELS]
        probs_pct    = [float(p) * 100 for p in probs]

        t1, t2, t3, t4 = st.tabs(["📊 Probabilities", "🍩 Distribution", "📈 Confidence Gauge", "🗂 Data Table"])

        # Tab 1 — Bar chart
        with t1:
            fig = go.Figure()
            for label, pct, col in zip(CLASS_LABELS, probs_pct, CHART_COLORS):
                fig.add_trace(go.Bar(
                    x=[pct], y=[label], orientation='h',
                    marker=dict(color=col, opacity=1.0 if label == pred else 0.3, line=dict(width=0)),
                    text=f"<b>{pct:.1f}%</b>" if label == pred else f"{pct:.1f}%",
                    textposition='outside',
                    textfont=dict(color=col if label == pred else '#5a7080', family='JetBrains Mono', size=12),
                    showlegend=False, name=label
                ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#5a7080', family='Syne'),
                xaxis=dict(range=[0,115], showgrid=True, gridcolor='#1e2830',
                           zeroline=False, ticksuffix='%', color='#5a7080',
                           tickfont=dict(family='JetBrains Mono', size=10)),
                yaxis=dict(showgrid=False, color='#d8e8f0', automargin=True,
                           tickfont=dict(family='Syne', size=11)),
                margin=dict(l=0, r=55, t=10, b=10),
                height=215, bargap=0.35
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Tab 2 — Donut
        with t2:
            fig2 = go.Figure(go.Pie(
                labels=[CLASS_META[l]['short'] for l in CLASS_LABELS],
                values=[float(p) for p in probs],
                hole=0.65,
                marker=dict(colors=CHART_COLORS, line=dict(color='#080c10', width=3)),
                textinfo='label+percent',
                textfont=dict(color='#d8e8f0', family='Syne', size=11),
                pull=[0.08 if l == pred else 0 for l in CLASS_LABELS],
                rotation=90
            ))
            fig2.add_annotation(text=f"<b>{conf:.0f}%</b>", x=0.5, y=0.55,
                font=dict(size=24, color=clr, family='JetBrains Mono'), showarrow=False)
            fig2.add_annotation(text=meta['short'], x=0.5, y=0.42,
                font=dict(size=11, color='#5a7080', family='Syne'), showarrow=False)
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(font=dict(color='#5a7080', size=10, family='Syne'),
                    bgcolor='rgba(0,0,0,0)', borderwidth=0,
                    orientation='h', x=0.5, xanchor='center', y=-0.05),
                margin=dict(l=10, r=10, t=10, b=30), height=255
            )
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

        # Tab 3 — Gauge
        with t3:
            fig3 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=conf,
                delta=dict(reference=50, increasing=dict(color=clr),
                    decreasing=dict(color='#5a7080'),
                    font=dict(family='JetBrains Mono', size=13)),
                number=dict(suffix="%", font=dict(color=clr, size=30, family='JetBrains Mono')),
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor='#1e2830',
                        tickfont=dict(color='#5a7080', family='JetBrains Mono', size=10), dtick=25),
                    bar=dict(color=clr, thickness=0.25),
                    bgcolor='#131920', borderwidth=1, bordercolor='#1e2830',
                    steps=[
                        dict(range=[0,33],   color='#0e1318'),
                        dict(range=[33,66],  color='#131920'),
                        dict(range=[66,100], color='#192028'),
                    ],
                    threshold=dict(line=dict(color=clr, width=2), thickness=0.75, value=conf)
                ),
                domain=dict(x=[0,1], y=[0.1,1]),
                title=dict(text=f"Confidence — {pred}", font=dict(color='#5a7080', size=12, family='Syne'))
            ))
            fig3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#5a7080', family='Syne'),
                margin=dict(l=20, r=20, t=20, b=10), height=245
            )
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

        # Tab 4 — Table
        with t4:
            df = pd.DataFrame({
                '🏷 Class':       CLASS_LABELS,
                '📊 Probability': [f"{float(p)*100:.2f}%" for p in probs],
                '⚠️ Risk':        [CLASS_META[l]['risk'] for l in CLASS_LABELS],
                '📈 5-yr Surv.':  [CLASS_META[l]['survival'] for l in CLASS_LABELS],
                '✅ Result':      ['← Predicted' if l == pred else '—' for l in CLASS_LABELS],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

        # About
        with st.expander(f"📖 About {pred}"):
            st.markdown(f"**{meta['desc']}**")
            st.caption(f"Recommended action: {meta['action']}")

    else:
        st.markdown("""
        <div style="background:#131920; border:1px solid #1e2830; border-radius:16px;
                    padding:4rem 2rem; text-align:center; color:#5a7080;">
            <div style="font-size:3rem; margin-bottom:1rem; opacity:0.4;">📈</div>
            <div style="font-weight:700; font-size:1.05rem; color:#d8e8f0; margin-bottom:0.5rem;">Awaiting Scan</div>
            <div style="font-size:0.85rem; line-height:1.7;">
                Upload a CT scan on the left<br>and click
                <span style="color:#00e5c8; font-weight:700;">Analyze Scan</span> to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    <div class="ft-logo">Lung<span>AI</span></div>
    <div class="ft-sub">
        Powered by CNN · ONNX Runtime · Streamlit &nbsp;|&nbsp;
        Educational use only &nbsp;|&nbsp;
        Always consult a certified physician
    </div>
</div>
""", unsafe_allow_html=True)