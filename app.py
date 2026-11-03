"""
SignSpeak - Indian Sign Language (ISL) to Text & Speech  |  Web app (Streamlit)

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (main file: app.py)

Flow: take a photo of a sign -> MediaPipe finds hand landmarks -> our trained
model predicts the letter -> letters build words -> words are spoken aloud
in English, Hindi, Kannada or Telugu.
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

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from utils import create_hand_detector, detect_hands, draw_hands, extract_features  # noqa: E402

MODEL_FILE = os.path.join(ROOT, "models", "sign_model.joblib")
DATA_FILE = os.path.join(ROOT, "data", "landmarks.csv")
METRICS_FILE = os.path.join(ROOT, "assets", "metrics.json")
CM_FILE = os.path.join(ROOT, "assets", "confusion_matrix.png")
WEB_MODEL_FILE = os.path.join(ROOT, "models", "web_model.json")
LIVE_HTML_FILE = os.path.join(ROOT, "web", "live.html")

LANGUAGES = {  # name -> (gTTS language code, translation code)
    "English": ("en", None),
    "Hindi (हिन्दी)": ("hi", "hi"),
    "Kannada (ಕನ್ನಡ)": ("kn", "kn"),
    "Telugu (తెలుగు)": ("te", "te"),
}

st.set_page_config(page_title="SignSpeak - ISL Translator", page_icon="🤟", layout="wide")


# ---------------- Cached resources ----------------
@st.cache_resource(show_spinner="Loading hand detector...")
def get_detector():
    return create_hand_detector(video_mode=False)


@st.cache_resource(show_spinner="Loading sign model...")
def get_model():
    """Load the trained model. If it can't be loaded (e.g. different library
    versions on the server), retrain quickly from the dataset instead."""
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
def live_page():
    """Build the real-time page: our model weights + dictionary injected into web/live.html."""
    if not os.path.exists(WEB_MODEL_FILE):
        return None
    from translate import DICTIONARY
    with open(LIVE_HTML_FILE, encoding="utf-8") as f:
        html = f.read()
    with open(WEB_MODEL_FILE) as f:
        model_json = f.read()
    return (html.replace("__MODEL_JSON__", model_json)
                .replace("__DICT_JSON__", json.dumps(DICTIONARY, ensure_ascii=False)))


def load_metrics():
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------- Prediction ----------------
def predict(img_bgr):
    """Try the image as-is AND mirrored (browser cameras differ), keep the more
    confident result. Returns dict or None if no hand is found."""
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
        cand = {
            "letter": classes[order[0]],
            "conf": float(probs[order[0]]),
            "top": [(classes[i], float(probs[i])) for i in order[:3]],
            "annotated": draw_hands(img.copy(), result),
            "hands": n_hands,
        }
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
        if method == "untranslated":
            st.info("No translation found for this text - speaking it as is.")
    buf = io.BytesIO()
    gTTS(spoken, lang=tts_code, tld="co.in" if tts_code == "en" else "com").write_to_fp(buf)
    return buf.getvalue(), spoken, method


# ---------------- Session state ----------------
for key, default in {"sentence": "", "last_img": None, "last_pred": None}.items():
    st.session_state.setdefault(key, default)


def add_text(s):
    st.session_state.sentence += s


# ---------------- UI ----------------
st.title("🤟 SignSpeak")
st.markdown("#### Indian Sign Language → Text → Speech, powered by AI")
st.caption("Problem Statement FT-04 · Sign Language to Text/Speech Translator · Neural Storm 2026")

with st.sidebar:
    st.header("⚙️ Settings")
    lang = st.selectbox("Speak in", list(LANGUAGES.keys()))
    auto_add = st.toggle("Auto-add detected letter", value=True,
                         help="Add each recognised letter to the text automatically.")
    min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    st.header("📊 Model")
    metrics = load_metrics()
    _, model_name = get_model()
    st.write(f"**Model:** {model_name}")
    if metrics:
        best = metrics.get("best_model")
        acc = metrics.get("test_split", {}).get(best, {}).get("accuracy")
        if acc is not None:
            st.metric("Test accuracy", f"{acc:.1%}")
        if metrics.get("unseen_person_average") is not None:
            st.metric("Unseen-signer accuracy", f"{metrics['unseen_person_average']:.1%}")
        st.write(f"**Signs:** {metrics.get('num_signs')} (ISL A–Z)  \n"
                 f"**Training samples:** {metrics.get('num_samples'):,}")

tab_live, tab_translate, tab_how, tab_results = st.tabs(
    ["🔴 Live translation", "📷 Photo mode", "💡 How it works", "📈 Results"])

with tab_live:
    page = live_page()
    if page is None:
        st.warning("Live mode needs models/web_model.json - run: python src/export_web_model.py")
    else:
        st.caption("Real-time: hand tracking and our trained model run directly in your browser "
                   "(fast, private, works on phones). Click **Start camera** and allow access.")
        import streamlit.components.v1 as components
        components.html(page, height=760, scrolling=True)

with tab_translate:
    left, right = st.columns([1, 1], gap="large")

    with left:
        source = st.radio("Input", ["📷 Camera", "🖼️ Upload image"], horizontal=True,
                          label_visibility="collapsed")
        if source == "📷 Camera":
            img_file = st.camera_input("Show one ISL sign and take a photo")
        else:
            img_file = st.file_uploader("Upload a photo of an ISL sign",
                                        type=["jpg", "jpeg", "png"])
        st.caption("Tips: plain background, good light, only the signer's hands in view.")

    with right:
        if img_file is not None:
            data = img_file.getvalue()
            digest = hashlib.md5(data).hexdigest()
            is_new = digest != st.session_state.last_img

            if is_new:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                with st.spinner("Recognising sign..."):
                    st.session_state.last_pred = predict(img)
                st.session_state.last_img = digest
                pred = st.session_state.last_pred
                if auto_add and pred and pred["conf"] >= min_conf:
                    add_text(pred["letter"])

            pred = st.session_state.last_pred
            if pred is None:
                st.error("No hand detected. Move closer, improve lighting and try again.")
            else:
                c1, c2 = st.columns([1, 2])
                with c1:
                    color = "green" if pred["conf"] >= min_conf else "orange"
                    st.markdown(f"<h1 style='font-size:96px;margin:0;color:{color}'>"
                                f"{pred['letter']}</h1>", unsafe_allow_html=True)
                    st.write(f"Confidence: **{pred['conf']:.0%}**")
                    if pred["conf"] < min_conf:
                        st.warning("Low confidence - not added.")
                with c2:
                    st.image(cv2.cvtColor(pred["annotated"], cv2.COLOR_BGR2RGB),
                             caption="Detected hand landmarks", width="stretch")
                st.write("**Top predictions**")
                for letter, p in pred["top"]:
                    st.progress(p, text=f"{letter} — {p:.0%}")
                if not auto_add:
                    st.button(f"➕ Add '{pred['letter']}'", on_click=add_text, args=(pred["letter"],))
        else:
            st.info("👈 Take a photo of a sign to start.")

    st.divider()
    st.subheader("📝 Your text")
    st.markdown(f"<div style='font-size:42px;font-family:monospace;padding:12px 18px;"
                f"border-radius:10px;background:rgba(128,128,128,0.15);min-height:70px'>"
                f"{st.session_state.sentence or '&nbsp;'}<span style='opacity:.4'>_</span></div>",
                unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    b1.button("␣ Space", on_click=add_text, args=(" ",), width="stretch")
    if b2.button("⌫ Delete", width="stretch"):
        st.session_state.sentence = st.session_state.sentence[:-1]
        st.rerun()
    if b3.button("🗑️ Clear", width="stretch"):
        st.session_state.sentence = ""
        st.rerun()
    speak_clicked = b4.button("🔊 Speak", type="primary", width="stretch")

    if speak_clicked:
        if not st.session_state.sentence.strip():
            st.warning("Nothing to speak yet - add some letters first.")
        else:
            try:
                with st.spinner("Generating speech..."):
                    audio, spoken, method = make_speech(st.session_state.sentence, lang)
                if spoken.strip().lower() != st.session_state.sentence.strip().lower():
                    st.markdown(f"<div style='font-size:32px'>{lang}: <b>{spoken}</b></div>",
                                unsafe_allow_html=True)
                    st.caption(f"Translated using: {method}")
                st.audio(audio, format="audio/mp3", autoplay=True)
            except Exception as e:
                st.error(f"Speech failed (needs internet): {e}")

with tab_how:
    st.markdown("""
### The problem
Millions of deaf and hard-of-hearing people in India use **Indian Sign Language (ISL)**,
but very few hearing people understand it and certified interpreters are scarce.
Everyday conversations - at a hospital, bank or classroom - become difficult.

### Our solution
SignSpeak turns ISL signs into **text and speech** using a normal camera, with output in
**English, Hindi, Kannada and Telugu**.

### Pipeline
1. **Hand tracking** - Google MediaPipe finds 21 3-D landmarks on each hand (two-hand support for ISL).
2. **Feature engineering** - landmarks are made position- and size-independent (wrist-relative, scaled) → 126 features.
3. **Classification** - a Random Forest / Neural Network trained on our own recorded ISL dataset.
4. **Smoothing** (desktop app) - a letter is accepted only after it is predicted steadily, which removes flicker.
5. **Text → speech** - Google Text-to-Speech, with translation into Indian languages.
""")

with tab_results:
    metrics = load_metrics()
    if metrics:
        cols = st.columns(3)
        best = metrics.get("best_model")
        acc = metrics.get("test_split", {}).get(best, {}).get("accuracy")
        cols[0].metric("Test accuracy", f"{acc:.1%}" if acc is not None else "-")
        un = metrics.get("unseen_person_average")
        cols[1].metric("Unseen-signer accuracy", f"{un:.1%}" if un is not None else "-")
        cols[2].metric("Training samples", f"{metrics.get('num_samples', 0):,}")
        st.write("**Model comparison**")
        st.dataframe(pd.DataFrame(metrics.get("test_split", {})).T, width="stretch")
    if os.path.exists(CM_FILE):
        st.image(CM_FILE, caption="Confusion matrix on the held-out test set")
