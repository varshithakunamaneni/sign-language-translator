"""
SignSpeak - Indian Sign Language (ISL) to Text & Speech  |  Web app (Streamlit)

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (main file: app.py)
"""
import hashlib
import io
import json
import os
import sys

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from utils import create_hand_detector, detect_hands, draw_hands, extract_features  # noqa: E402

MODEL_FILE = os.path.join(ROOT, "models", "sign_model.joblib")
DATA_FILE = os.path.join(ROOT, "data", "landmarks.csv")
METRICS_FILE = os.path.join(ROOT, "assets", "metrics.json")
CM_FILE = os.path.join(ROOT, "assets", "confusion_matrix.png")
WEB_MODEL_FILE = os.path.join(ROOT, "models", "web_model.json")
LIVE_HTML_FILE = os.path.join(ROOT, "web", "live.html")
WORDS_MODEL_FILE = os.path.join(ROOT, "models", "words_web_model.json")
REPLY_HTML_FILE = os.path.join(ROOT, "web", "reply.html")
SIGNS_DIR = os.path.join(ROOT, "web", "signs")

LANGUAGES = {  # name -> (gTTS language code, translation code)
    "English": ("en", None),
    "Hindi (हिन्दी)": ("hi", "hi"),
    "Kannada (ಕನ್ನಡ)": ("kn", "kn"),
    "Telugu (తెలుగు)": ("te", "te"),
}

st.set_page_config(page_title="SignSpeak · ISL Translator", page_icon="🤟", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------- Accessibility settings (kept across reruns) ----------------
st.session_state.setdefault("large_text", False)
st.session_state.setdefault("high_contrast", False)
LARGE, HC = st.session_state.large_text, st.session_state.high_contrast

# ---------------- Look and feel ----------------
# Colours come from .streamlit/config.toml ([theme.light] = Day, [theme.dark] = Night).
# The CSS below uses see-through greys, so it looks right in BOTH modes.
FS = 1.22 if LARGE else 1.0
PLUM, MAUVE, PEACH, IVORY = "#502D55", "#935073", "#F6DBC0", "#F8F4E9"   # Violet Dusk

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Nunito:wght@400;600;700;800&display=swap');
  .stApp p, .stApp li, .stApp label, .stApp td, .stApp th, .stApp input, .stApp button,
  .stApp .stMarkdown, .stApp [data-testid="stCaptionContainer"] {{ font-family:'Nunito', system-ui, sans-serif; }}
  .stApp h1, .stApp h2, .stApp h3, .stApp h4, .serif {{ font-family:'Fraunces', Georgia, serif !important; font-weight:600 !important; letter-spacing:-0.01em; }}
  .stApp p, .stApp li {{ font-size:{17 * FS}px; line-height:1.6; }}
  [data-testid="stHeader"] {{ background:transparent; }}
  .block-container {{ padding-top:1.4rem; max-width:1200px; }}

  /* header */
  .brand {{ display:flex; align-items:center; gap:14px; margin:4px 0 6px; animation:rise .7s ease-out both; }}
  .brand .mark {{ width:52px; height:52px; border-radius:14px; background:{MAUVE}; display:flex;
                  align-items:center; justify-content:center; transform:rotate(-6deg); }}
  .brand .mark svg {{ width:30px; height:30px; stroke:#FFFFFF; fill:none; stroke-width:2; stroke-linecap:round;
                      stroke-linejoin:round; transform-origin:70% 90%; animation:wave 3s ease-in-out 1s 2; }}
  .brand h1 {{ margin:0 !important; padding:0 !important; font-size:{2.6 * FS}rem !important; line-height:1.05 !important; }}
  .lede {{ font-size:{1.25 * FS}rem !important; max-width:720px; margin:6px 0 0 !important; animation:rise .7s ease-out .1s both; }}
  .hl {{ background:linear-gradient(transparent 58%, {PEACH} 58%); padding:0 2px; border-radius:2px; }}
  body.ss-night .hl {{ background:linear-gradient(transparent 58%, rgba(147,80,115,.85) 58%); }}

  /* facts row, like a newspaper fact bar */
  .facts {{ display:flex; flex-wrap:wrap; margin:18px 0 22px; border-top:1px solid rgba(128,128,128,.28);
            border-bottom:1px solid rgba(128,128,128,.28); }}
  .fact {{ flex:1 1 160px; padding:12px 18px; border-left:1px solid rgba(128,128,128,.22); animation:rise .6s ease-out both; }}
  .fact:first-child {{ border-left:none; padding-left:4px; }}
  .fact:nth-child(2) {{ animation-delay:.12s; }} .fact:nth-child(3) {{ animation-delay:.24s; }} .fact:nth-child(4) {{ animation-delay:.36s; }}
  .fact b {{ display:block; font-family:'Fraunces', serif; font-weight:600; font-size:{1.9 * FS}rem; line-height:1.1; }}
  .fact span {{ opacity:.75; font-size:{0.95 * FS}rem; }}

  /* tabs */
  [role="tablist"] {{ gap:6px; }}
  [data-testid="stTab"] {{ padding:8px 18px !important; border-radius:10px 10px 0 0; }}
  [data-testid="stTab"] p {{ font-size:{1.12 * FS}rem !important; font-weight:700 !important; }}

  /* buttons */
  .stButton > button {{ border-radius:12px; min-height:48px; font-weight:700; font-size:{1.02 * FS}rem;
                        transition:transform .12s ease; }}
  .stButton > button:active {{ transform:scale(.97); }}
  .stButton > button[kind="primary"] {{ background:{MAUVE} !important; border-color:{MAUVE} !important; }}
  .stButton > button[kind="primary"], .stButton > button[kind="primary"] p {{ color:#FFFFFF !important; }}
  .stButton > button[kind="primary"]:hover {{ background:#A8618A !important; }}
  body.ss-night .stButton > button[kind="primary"] {{ background:{PEACH} !important; border-color:{PEACH} !important; }}
  body.ss-night .stButton > button[kind="primary"], body.ss-night .stButton > button[kind="primary"] p {{ color:#2A1A2E !important; }}
  body.ss-night .stButton > button[kind="primary"]:hover {{ background:#FBE6D1 !important; }}
  .brand h1, .fact b, .step h4, .stat .v {{ color:{PLUM}; }}
  body.ss-night .brand h1, body.ss-night .fact b, body.ss-night .step h4, body.ss-night .stat .v {{ color:{PEACH}; }}

  /* cards used in Photo / How it works / Results */
  .card, .step, .stat {{ background:rgba(128,128,128,.07); border:1px solid rgba(128,128,128,.22);
                         border-radius:14px; padding:16px 18px; animation:rise .6s ease-out both; }}
  .label {{ font-weight:700; opacity:.8; margin-bottom:6px; font-size:{0.98 * FS}rem; }}
  .bigletter {{ font-family:'Fraunces', serif; font-size:{5.5 * FS}rem; font-weight:700; line-height:1; }}
  .bigletter.low {{ opacity:.5; }}
  .textbox {{ font-family:'Nunito', sans-serif; font-weight:800; letter-spacing:.06em; font-size:{2.3 * FS}rem;
              padding:12px 16px; border-radius:12px; border:2px solid rgba(128,128,128,.3); min-height:72px;
              background:rgba(128,128,128,.06); word-break:break-all; }}
  .textbox .cur {{ color:{MAUVE}; }}
  .translated {{ font-size:{1.8 * FS}rem; font-weight:700; margin-top:6px; }}
  .step .n {{ width:34px; height:34px; border-radius:50%; background:{MAUVE}; color:#FFFFFF; display:flex;
              align-items:center; justify-content:center; font-weight:800; margin-bottom:10px; }}
  .step h4 {{ margin:0 0 6px; font-size:{1.15 * FS}rem; }}
  .step p {{ margin:0; opacity:.78; font-size:{0.98 * FS}rem; }}
  .step:nth-child(2) {{ animation-delay:.08s; }} .step:nth-child(3) {{ animation-delay:.16s; }}
  .step:nth-child(4) {{ animation-delay:.24s; }} .step:nth-child(5) {{ animation-delay:.32s; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin-bottom:18px; }}
  .stat .k {{ opacity:.75; font-weight:700; }}
  .stat .v {{ font-family:'Fraunces', serif; font-size:{1.9 * FS}rem; font-weight:600; }}
  .stat .s {{ opacity:.7; font-size:{0.9 * FS}rem; }}
  .footer {{ margin-top:30px; padding:16px 4px; border-top:1px solid rgba(128,128,128,.25); opacity:.75;
             font-size:{0.92 * FS}rem; }}
  [data-testid="stCameraInput"] {{ border-radius:14px; overflow:hidden; }}

  @keyframes rise {{ from {{ opacity:0; transform:translateY(16px); }} to {{ opacity:1; transform:none; }} }}
  @keyframes wave {{ 0%,60%,100% {{ transform:rotate(0); }} 10%,30% {{ transform:rotate(-16deg); }} 20%,40% {{ transform:rotate(12deg); }} }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; transition:none !important; }} }}
</style>
""", unsafe_allow_html=True)

if HC:   # high contrast: pure black and yellow, overrides Day / Night
    st.markdown("""<style>
      .stApp, [data-testid="stHeader"] { background:#000 !important; color:#fff !important; }
      .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp span { color:#fff !important; }
      .card, .step, .stat, .textbox { background:#000 !important; border-color:#fff !important; }
      .hl { background:linear-gradient(transparent 58%, #FFD400 58%); color:#000 !important; }
    </style>""", unsafe_allow_html=True)


# ---------------- Cached resources ----------------
@st.cache_resource(show_spinner="Loading hand detector…")
def get_detector():
    return create_hand_detector(video_mode=False)


@st.cache_resource(show_spinner="Loading sign model…")
def get_model():
    """Load the trained model; if it can't be loaded (library version mismatch on the
    server), retrain quickly from the dataset instead."""
    try:
        bundle = joblib.load(MODEL_FILE)
        return bundle["model"], bundle.get("model_name", "model")
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        df = pd.read_csv(DATA_FILE)
        X = df[[c for c in df.columns if c.startswith("f")]].to_numpy(dtype=np.float32)
        y = df["label"].astype(str).to_numpy(dtype=object)
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42).fit(X, y)
        return model, "Random Forest (retrained on server)"


@st.cache_data
def live_page(body_class, stamp=""):
    """Real-time page: our model weights + dictionary injected into web/live.html."""
    if not os.path.exists(WEB_MODEL_FILE):
        return None
    from translate import DICTIONARY
    with open(LIVE_HTML_FILE, encoding="utf-8") as f:
        html = f.read()
    with open(WEB_MODEL_FILE) as f:
        model_json = f.read()
    words_json = "null"                       # word-sign model is optional
    if os.path.exists(WORDS_MODEL_FILE):
        with open(WORDS_MODEL_FILE) as f:
            words_json = f.read()
    return (html.replace("__MODEL_JSON__", model_json)
                .replace("__WORDS_JSON__", words_json)
                .replace("__DICT_JSON__", json.dumps(DICTIONARY, ensure_ascii=False))
                .replace("__BODY_CLASS__", body_class))


@st.cache_data
def reply_page(body_class, signs_stamp):
    """Reply page: the ISL letter photos (web/signs/A.jpg ...) are built into the page."""
    import base64
    signs = {}
    if os.path.isdir(SIGNS_DIR):
        for name in sorted(os.listdir(SIGNS_DIR)):
            letter, ext = os.path.splitext(name)
            if ext.lower() in (".jpg", ".jpeg", ".png") and len(letter) == 1 and letter.isalpha():
                with open(os.path.join(SIGNS_DIR, name), "rb") as f:
                    mime = "png" if ext.lower() == ".png" else "jpeg"
                    signs[letter.upper()] = f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()
    with open(REPLY_HTML_FILE, encoding="utf-8") as f:
        html = f.read()
    return html.replace("__SIGNS_JSON__", json.dumps(signs)).replace("__BODY_CLASS__", body_class)


def signs_stamp():
    """Changes whenever sign photos are added/updated, so the page refreshes."""
    if not os.path.isdir(SIGNS_DIR):
        return ""
    return ",".join(f"{n}:{os.path.getmtime(os.path.join(SIGNS_DIR, n)):.0f}" for n in sorted(os.listdir(SIGNS_DIR)))


def load_metrics():
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------- Prediction (photo mode) ----------------
def predict(img_bgr):
    """Try the image as-is AND mirrored (browser cameras differ); keep the more confident."""
    detector = get_detector()
    model, _ = get_model()
    classes = list(model.classes_)
    best = None
    for img in (img_bgr, cv2.flip(img_bgr, 1)):
        result = detect_hands(detector, img)
        feats, n_hands = extract_features(result)
        if n_hands == 0:
            continue
        probs = model.predict_proba(feats.reshape(1, -1))[0]
        order = np.argsort(probs)[::-1]
        cand = {"letter": classes[order[0]], "conf": float(probs[order[0]]),
                "top": [(classes[i], float(probs[i])) for i in order[:3]],
                "annotated": draw_hands(img.copy(), result)}
        if best is None or cand["conf"] > best["conf"]:
            best = cand
    return best


# ---------------- Speech ----------------
def make_speech(text, lang_name):
    """Returns (mp3 bytes, the text actually spoken, how it was translated)."""
    from gtts import gTTS
    from translate import translate
    tts_code, tr_code = LANGUAGES[lang_name]
    spoken, method = text.strip(), None
    if tr_code:
        spoken, method = translate(spoken, tr_code)
    buf = io.BytesIO()
    gTTS(spoken, lang=tts_code, tld="co.in" if tts_code == "en" else "com").write_to_fp(buf)
    return buf.getvalue(), spoken, method


for key, default in {"sentence": "", "last_img": None, "last_pred": None}.items():
    st.session_state.setdefault(key, default)


def add_text(s):
    st.session_state.sentence += s


# ---------------- Header ----------------
def embed(html, height):
    """Show our own HTML (live translator, Day/Night switch). Uses st.iframe on new
    Streamlit versions and components.html on older ones."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=True)



metrics = load_metrics()
best_name = metrics.get("best_model")
test_acc = metrics.get("test_split", {}).get(best_name, {}).get("accuracy")
unseen = metrics.get("unseen_person_average")
n_samples = metrics.get("num_samples")

HAND_SVG = ('<svg viewBox="0 0 24 24"><path d="M8 13V5.5a1.5 1.5 0 0 1 3 0V12"/><path d="M11 5.5v-2a1.5 1.5 0 0 1 3 0V12"/>'
            '<path d="M14 5.5a1.5 1.5 0 0 1 3 0V12"/><path d="M17 7.5a1.5 1.5 0 0 1 3 0V16a6 6 0 0 1-6 6h-2a6 6 0 0 1-5-2.7'
            'L4.2 15a1.5 1.5 0 0 1 2.5-1.7L8 15"/></svg>')

# Day / Night buttons: they switch Streamlit's own theme (same as menu ⋮ → Light / Dark)
DAY_NIGHT = r"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700;800&display=swap');
  body { margin:0; font-family:'Nunito', sans-serif; background:transparent; }
  .sw { display:inline-flex; border:1px solid rgba(128,128,128,.4); border-radius:999px; padding:3px; gap:2px; }
  button { font:inherit; font-weight:800; border:0; background:transparent; color:var(--t); padding:8px 16px;
           border-radius:999px; cursor:pointer; display:flex; align-items:center; gap:6px; font-size:15px;
           transition:background .25s, color .25s; }
  button[aria-pressed="true"] { background:var(--a); color:var(--o); }
  button:focus-visible { outline:3px solid #935073; outline-offset:2px; }
  svg { width:18px; height:18px; stroke:currentColor; fill:none; stroke-width:2; stroke-linecap:round; }
</style>
<div class="sw" role="group" aria-label="Day or night colours">
  <button id="day" aria-pressed="false"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4l1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4m11.4-11.4l1.4-1.4"/></svg>Day</button>
  <button id="night" aria-pressed="false"><svg viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/></svg>Night</button>
</div>
<script>
const P = window.parent.document;
function isLight() {
  try { const m = getComputedStyle(P.querySelector('.stApp')).backgroundColor.match(/\d+/g).map(Number);
        return (0.299*m[0] + 0.587*m[1] + 0.114*m[2]) > 140; } catch (e) { return false; }
}
function sync() {
  const l = isLight();
  const r = document.documentElement.style;
  r.setProperty('--t', l ? '#2A1A2E' : '#F8F4E9');
  r.setProperty('--a', l ? '#935073' : '#F6DBC0');
  r.setProperty('--o', l ? '#FFFFFF' : '#2A1A2E');
  P.body.classList.toggle('ss-night', !l); P.body.classList.toggle('ss-day', l);
  document.getElementById('day').setAttribute('aria-pressed', l);
  document.getElementById('night').setAttribute('aria-pressed', !l);
}
function setTheme(name) {
  const btn = P.querySelector('[data-testid="stMainMenuButton"]'); if (!btn) return;
  if (btn.getAttribute('aria-expanded') !== 'true') btn.click();
  setTimeout(() => {
    const it = P.querySelector('[data-testid="stMainMenuItem-theme-' + name + '"]');
    if (it) it.click();
    setTimeout(() => { if (btn.getAttribute('aria-expanded') === 'true') btn.click(); sync(); }, 120);
  }, 150);
}
document.getElementById('day').onclick = () => setTheme('Light');
document.getElementById('night').onclick = () => setTheme('Dark');
sync(); setInterval(sync, 500);
</script>
"""

top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown(f"""
    <div class="brand"><div class="mark" aria-hidden="true">{HAND_SVG}</div><h1>SignSpeak</h1></div>
    <p class="lede">Sign in front of the camera. We'll write it down and
    <span class="hl">read it out loud</span>, in English, हिन्दी, ಕನ್ನಡ or తెలుగు.</p>
    """, unsafe_allow_html=True)
with top_r:
    embed(DAY_NIGHT, 52)
    st.toggle("Bigger text", key="large_text", help="Make all text bigger.")
    st.toggle("High contrast", key="high_contrast", help="Black and yellow, for low vision.")


def stat(k, v, s):
    return f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'


def fact(v, s):
    return f'<div class="fact"><b>{v}</b><span>{s}</span></div>'


st.markdown('<div class="facts">'
            + fact(f"{test_acc:.1%}" if test_acc else "–", "correct on test signs")
            + fact(f"{unseen:.0%}" if unseen else "–", "correct for a brand-new person")
            + fact(f"{n_samples:,}" if n_samples else "–", "hand signs we recorded")
            + fact("4", "languages it speaks")
            + "</div>", unsafe_allow_html=True)

tab_live, tab_reply, tab_photo, tab_how, tab_results = st.tabs(
    ["Translate", "Reply", "Photo mode", "How it works", "Results"])

# ---------------- Live ----------------
with tab_live:
    body_class = " ".join(c for c, on in (("large", LARGE), ("hc", HC)) if on)
    stamp = "".join(f"{os.path.getmtime(f):.0f}" for f in (WEB_MODEL_FILE, WORDS_MODEL_FILE, LIVE_HTML_FILE)
                    if os.path.exists(f))
    page = live_page(body_class, stamp)
    if page is None:
        st.warning("Live mode needs models/web_model.json. Run: python src/export_web_model.py")
    else:
        has_phrases = "Quick phrases" in page          # taller page when quick phrases are included
        embed(page, (1450 if has_phrases else 920) if LARGE else (1180 if has_phrases else 820))

# ---------------- Reply (hearing person -> deaf person) ----------------
with tab_reply:
    body_class = " ".join(c for c, on in (("large", LARGE), ("hc", HC)) if on)
    embed(reply_page(body_class, signs_stamp() + f"{os.path.getmtime(REPLY_HTML_FILE):.0f}"), 1250 if LARGE else 1050)

# ---------------- Photo mode ----------------
with tab_photo:
    st.caption("A backup mode that works on any device: take one photo per sign. "
               "Speech here works in all 4 languages.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        source = st.radio("Input", ["Camera", "Upload a photo"], horizontal=True,
                          label_visibility="collapsed")
        if source == "Camera":
            img_file = st.camera_input("Show one ISL sign, then take a photo")
        else:
            img_file = st.file_uploader("Upload a photo of an ISL sign", type=["jpg", "jpeg", "png"])
        o1, o2 = st.columns(2)
        auto_add = o1.toggle("Auto-add letter", value=True)
        min_conf = o2.slider("Min confidence", 0.0, 1.0, 0.5, 0.05)

    with right:
        if img_file is not None:
            data = img_file.getvalue()
            digest = hashlib.md5(data).hexdigest()
            if digest != st.session_state.last_img:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                with st.spinner("Recognising sign…"):
                    st.session_state.last_pred = predict(img)
                st.session_state.last_img = digest
                p = st.session_state.last_pred
                if auto_add and p and p["conf"] >= min_conf:
                    add_text(p["letter"])
            pred = st.session_state.last_pred
            if pred is None:
                st.error("⚠ No hand detected. Move closer, add light, and keep both hands in the frame.")
            else:
                good = pred["conf"] >= min_conf
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f'<div class="card"><div class="label">Detected</div>'
                                f'<div class="bigletter {"" if good else "low"}">{pred["letter"]}</div>'
                                f'<div>{"✓" if good else "⚠ Low confidence ·"} {pred["conf"]:.0%}</div></div>',
                                unsafe_allow_html=True)
                with c2:
                    st.image(cv2.cvtColor(pred["annotated"], cv2.COLOR_BGR2RGB),
                             caption="Detected hand landmarks", width="stretch")
                st.markdown('<div class="label" style="margin-top:12px">Top guesses</div>',
                            unsafe_allow_html=True)
                for letter, p in pred["top"]:
                    st.progress(p, text=f"{letter} — {p:.0%}")
                if not auto_add:
                    st.button(f"➕ Add '{pred['letter']}'", on_click=add_text, args=(pred["letter"],))
        else:
            st.markdown('<div class="card" style="text-align:center;padding:40px 20px">'
                        '<div class="label">Take a photo of a sign to start</div>'
                        '<div style="opacity:.7">The letter will show up here.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="label" style="margin-top:18px">Your words</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="textbox" role="log" aria-live="polite">'
                f'{st.session_state.sentence or "&nbsp;"}<span class="cur">▍</span></div>',
                unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    b1.button("Space", on_click=add_text, args=(" ",), width="stretch")
    if b2.button("Delete", width="stretch"):
        st.session_state.sentence = st.session_state.sentence[:-1]
        st.rerun()
    if b3.button("Clear", width="stretch"):
        st.session_state.sentence = ""
        st.rerun()
    s1, s2 = st.columns([2, 1])
    lang = s1.selectbox("Speak in", list(LANGUAGES.keys()))
    s2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    speak_clicked = s2.button("Speak", type="primary", width="stretch")
    if speak_clicked:
        if not st.session_state.sentence.strip():
            st.warning("⚠ Nothing to speak yet: add some letters first.")
        else:
            try:
                with st.spinner("Generating speech…"):
                    audio, spoken, method = make_speech(st.session_state.sentence, lang)
                if spoken.strip().lower() != st.session_state.sentence.strip().lower():
                    st.markdown(f'<div class="translated">{spoken}</div>', unsafe_allow_html=True)
                    st.caption(f"Translated using: {method}")
                st.audio(audio, format="audio/mp3", autoplay=True)
            except Exception as e:
                st.error(f"⚠ Speech needs an internet connection ({e}).")

# ---------------- How it works ----------------
with tab_how:
    st.markdown("### The problem")
    st.markdown("India has a large deaf and hard-of-hearing community (estimates range from about "
                "**1.3 million to 18 million** people) but only **a few hundred certified ISL interpreters**. "
                "Everyday conversations at a hospital, bank, shop or classroom become hard. "
                "SignSpeak turns any camera into an instant ISL interpreter.")
    st.markdown("### How SignSpeak works")
    steps = [
        ("1", "See the hands", "The camera finds 21 points on each hand (MediaPipe)."),
        ("2", "Measure the shape", "We turn the hand shape into numbers, wherever the hand is."),
        ("3", "Recognise the sign", "Our model, trained on signs we recorded, picks the letter."),
        ("4", "Check it is steady", "A letter is added only after you hold it for about 1 second."),
        ("5", "Say it out loud", "The words are spoken in English, Hindi, Kannada or Telugu."),
    ]
    cols = st.columns(5)
    for col, (n, t, d) in zip(cols, steps):
        col.markdown(f'<div class="step"><div class="n">{n}</div><h4>{t}</h4><p>{d}</p></div>',
                     unsafe_allow_html=True)
    st.markdown("### Limitations and responsible use")
    st.markdown("- **Not a certified interpreter.** It can make mistakes. For medical, legal or emergency situations, "
                "use a professional ISL interpreter or call **112**.\n"
                "- **What it knows:** the ISL alphabet A–Z and 14 everyday words. Many ISL signs are not covered yet.\n"
                "- **New signers:** accuracy is lower for people the model has never seen (about 75% per letter, "
                "versus 99.8% on our test set). Smoothing and the progress ring help, and a wrong letter can be deleted.\n"
                "- **Conditions matter:** poor light, cluttered backgrounds, fast movements or hands leaving the frame reduce accuracy. "
                "Regional ISL variations may not be recognised.\n"
                "- **Explainable:** the app shows its top-3 guesses with confidence (Translate → Show details, and Photo mode), "
                "and the Results tab shows exactly where the model gets confused.\n"
                "- **Privacy:** live mode runs entirely in your browser; video is never uploaded or stored. "
                "Only hand-point coordinates are used. Our training data contains no names or face images.")
    st.markdown("### Designed for accessibility")
    st.markdown("- **Clear contrast** and big buttons with icons and short words\n"
                "- **Bigger text** and **high contrast** modes (top right)\n"
                "- Every status uses **icons and words**, never colour alone\n"
                "- A **visual flash** confirms each added letter, since deaf users can't rely on sound\n"
                "- **Keyboard shortcuts** and screen-reader labels on every control\n"
                "- **Privacy**: live mode runs on your device; only hand points are used, nothing is stored")

# ---------------- Results ----------------
with tab_results:
    st.markdown('<div class="stats">'
                + stat("Test accuracy", f"{test_acc:.1%}" if test_acc else "–", "held-out 20% split")
                + stat("Unseen signer", f"{unseen:.1%}" if unseen else "–", "person never seen in training")
                + stat("Training samples", f"{n_samples:,}" if n_samples else "–", "recorded by our team")
                + stat("Languages", "4", "English · Hindi · Kannada · Telugu")
                + "</div>", unsafe_allow_html=True)
    if metrics:
        st.markdown("### Model comparison")
        table = pd.DataFrame(metrics.get("test_split", {})).T
        table = table.rename(columns={"accuracy": "Accuracy", "macro_f1": "Macro F1",
                                      "train_time_s": "Train time (s)",
                                      "inference_ms_per_sample": "Inference (ms/sample)"})
        for col in ("Accuracy", "Macro F1"):
            if col in table:
                table[col] = table[col].map(lambda v: f"{v:.2%}")
        st.dataframe(table, width="stretch")
        per = metrics.get("unseen_person_accuracy") or {}
        if per:
            st.markdown("### Unseen-signer test")
            st.caption("Train on everyone except one person, then test on that person. "
                       "This is the honest measure of how well it works for a new user.")
            for i, (name, acc) in enumerate(per.items(), 1):
                st.progress(acc, text=f"Signer {i} — {acc:.1%}")
    if os.path.exists(CM_FILE):
        st.markdown("### Confusion matrix")
        st.image(CM_FILE, caption="Held-out test set: a darker diagonal means better accuracy",
                 width="stretch")

st.markdown('<div class="footer"><b>SignSpeak</b> · Problem statement FT-04 · Neural Storm 2026 · '
            'Built with Python, MediaPipe, scikit-learn and Streamlit<br>'
            '<b>Please note:</b> SignSpeak is a student prototype that helps with everyday communication. '
            'It can make mistakes and is <b>not a replacement for a certified ISL interpreter</b>. '
            'In a medical or other emergency, call 112 and do not rely on this app alone.</div>',
            unsafe_allow_html=True)
