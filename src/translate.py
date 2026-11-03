"""
Reliable English -> Hindi / Kannada / Telugu translation for spoken output.

Order of attempts (so the demo never breaks):
  1. Built-in offline dictionary of common words & phrases (no internet needed)
  2. Google Translate (free, may be rate-limited)
  3. MyMemory Translate (free backup service)
  4. Word-by-word offline dictionary, keeping unknown words in English
"""
from functools import lru_cache

# lang -> phrase/word -> translation
DICTIONARY = {
    "hi": {
        "hello": "नमस्ते", "hi": "नमस्ते", "thank you": "धन्यवाद", "thanks": "धन्यवाद",
        "help": "मदद", "help me": "मेरी मदद करो", "water": "पानी", "food": "खाना",
        "doctor": "डॉक्टर", "yes": "हाँ", "no": "नहीं", "please": "कृपया",
        "sorry": "माफ़ कीजिए", "good": "अच्छा", "good morning": "सुप्रभात",
        "god": "भगवान", "victory": "विजय", "love": "प्यार", "friend": "दोस्त",
        "mother": "माँ", "father": "पिता", "home": "घर", "school": "स्कूल",
        "hospital": "अस्पताल", "pain": "दर्द", "emergency": "आपातकाल",
        "stop": "रुको", "go": "जाओ", "come": "आओ", "happy": "खुश", "india": "भारत",
        "ok": "ठीक है", "okay": "ठीक है", "name": "नाम", "i": "मैं", "you": "आप",
        "my": "मेरा", "need": "ज़रूरत", "bye": "अलविदा", "welcome": "स्वागत है",
    },
    "kn": {
        "hello": "ನಮಸ್ಕಾರ", "hi": "ನಮಸ್ಕಾರ", "thank you": "ಧನ್ಯವಾದಗಳು", "thanks": "ಧನ್ಯವಾದಗಳು",
        "help": "ಸಹಾಯ", "help me": "ನನಗೆ ಸಹಾಯ ಮಾಡಿ", "water": "ನೀರು", "food": "ಊಟ",
        "doctor": "ವೈದ್ಯರು", "yes": "ಹೌದು", "no": "ಇಲ್ಲ", "please": "ದಯವಿಟ್ಟು",
        "sorry": "ಕ್ಷಮಿಸಿ", "good": "ಒಳ್ಳೆಯದು", "good morning": "ಶುಭೋದಯ",
        "god": "ದೇವರು", "victory": "ಗೆಲುವು", "love": "ಪ್ರೀತಿ", "friend": "ಸ್ನೇಹಿತ",
        "mother": "ಅಮ್ಮ", "father": "ಅಪ್ಪ", "home": "ಮನೆ", "school": "ಶಾಲೆ",
        "hospital": "ಆಸ್ಪತ್ರೆ", "pain": "ನೋವು", "emergency": "ತುರ್ತು",
        "stop": "ನಿಲ್ಲಿಸಿ", "go": "ಹೋಗು", "come": "ಬನ್ನಿ", "happy": "ಸಂತೋಷ", "india": "ಭಾರತ",
        "ok": "ಸರಿ", "okay": "ಸರಿ", "name": "ಹೆಸರು", "i": "ನಾನು", "you": "ನೀವು",
        "my": "ನನ್ನ", "need": "ಬೇಕು", "bye": "ಹೋಗಿ ಬನ್ನಿ", "welcome": "ಸ್ವಾಗತ",
    },
    "te": {
        "hello": "నమస్కారం", "hi": "నమస్కారం", "thank you": "ధన్యవాదాలు", "thanks": "ధన్యవాదాలు",
        "help": "సహాయం", "help me": "నాకు సహాయం చేయండి", "water": "నీరు", "food": "ఆహారం",
        "doctor": "డాక్టర్", "yes": "అవును", "no": "లేదు", "please": "దయచేసి",
        "sorry": "క్షమించండి", "good": "మంచి", "good morning": "శుభోదయం",
        "god": "దేవుడు", "victory": "విజయం", "love": "ప్రేమ", "friend": "స్నేహితుడు",
        "mother": "అమ్మ", "father": "నాన్న", "home": "ఇల్లు", "school": "పాఠశాల",
        "hospital": "ఆసుపత్రి", "pain": "నొప్పి", "emergency": "అత్యవసరం",
        "stop": "ఆపండి", "go": "వెళ్ళు", "come": "రండి", "happy": "సంతోషం", "india": "భారతదేశం",
        "ok": "సరే", "okay": "సరే", "name": "పేరు", "i": "నేను", "you": "మీరు",
        "my": "నా", "need": "కావాలి", "bye": "వెళ్ళొస్తాను", "welcome": "స్వాగతం",
    },
}

MYMEMORY_NAMES = {"hi": "hindi", "kn": "kannada", "te": "telugu"}


@lru_cache(maxsize=512)
def _online(text, lang):
    """Try free online translators. Returns translation or None."""
    try:
        from deep_translator import GoogleTranslator
        out = GoogleTranslator(source="en", target=lang).translate(text)
        if out and out.strip().lower() != text:
            return out
    except Exception:
        pass
    try:
        from deep_translator import MyMemoryTranslator
        out = MyMemoryTranslator(source="english", target=MYMEMORY_NAMES[lang]).translate(text)
        if out and out.strip().lower() != text and "MYMEMORY WARNING" not in out.upper():
            return out
    except Exception:
        pass
    return None


def translate(text, lang):
    """Returns (translated_text, method) where method describes what was used."""
    text = " ".join(text.lower().split())
    if not text or lang not in DICTIONARY:
        return text, "none"
    d = DICTIONARY[lang]

    if text in d:                                   # 1) whole phrase in dictionary
        return d[text], "offline dictionary"

    out = _online(text, lang)                       # 2-3) online services
    if out:
        return out, "online translator"

    words = text.split()                            # 4) word by word, unknown stays English
    translated = [d.get(w, w) for w in words]
    if any(t != w for t, w in zip(translated, words)):
        return " ".join(translated), "offline dictionary (partial)"
    return text, "untranslated"


if __name__ == "__main__":
    for phrase in ["victory", "god", "thank you", "help me", "hello friend"]:
        for lang in ("hi", "kn", "te"):
            print(lang, phrase, "->", translate(phrase, lang))
