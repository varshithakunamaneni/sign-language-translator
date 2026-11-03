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
        "toilet": "शौचालय", "where": "कहाँ", "deaf": "बधिर", "understand": "समझ", "learn": "सीखना", "family": "परिवार",
        # quick phrases
        "i need a doctor": "मुझे डॉक्टर की ज़रूरत है", "please help me": "कृपया मेरी मदद कीजिए",
        "please call my family": "कृपया मेरे परिवार को फ़ोन कीजिए",
        "i cannot hear, please write it down": "मुझे सुनाई नहीं देता, कृपया लिखकर बताइए",
        "i am in pain": "मुझे दर्द हो रहा है", "i need water": "मुझे पानी चाहिए",
        "where is the toilet?": "शौचालय कहाँ है?", "thank you very much": "बहुत-बहुत धन्यवाद",
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
        "toilet": "ಶೌಚಾಲಯ", "where": "ಎಲ್ಲಿ", "deaf": "ಕಿವುಡ", "understand": "ಅರ್ಥ", "learn": "ಕಲಿಯಿರಿ", "family": "ಕುಟುಂಬ",
        # quick phrases
        "i need a doctor": "ನನಗೆ ವೈದ್ಯರು ಬೇಕು", "please help me": "ದಯವಿಟ್ಟು ನನಗೆ ಸಹಾಯ ಮಾಡಿ",
        "please call my family": "ದಯವಿಟ್ಟು ನನ್ನ ಕುಟುಂಬಕ್ಕೆ ಕರೆ ಮಾಡಿ",
        "i cannot hear, please write it down": "ನನಗೆ ಕಿವಿ ಕೇಳಿಸುವುದಿಲ್ಲ, ದಯವಿಟ್ಟು ಬರೆದು ತೋರಿಸಿ",
        "i am in pain": "ನನಗೆ ನೋವಾಗುತ್ತಿದೆ", "i need water": "ನನಗೆ ನೀರು ಬೇಕು",
        "where is the toilet?": "ಶೌಚಾಲಯ ಎಲ್ಲಿದೆ?", "thank you very much": "ತುಂಬಾ ಧನ್ಯವಾದಗಳು",
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
        "toilet": "మరుగుదొడ్డి", "where": "ఎక్కడ", "deaf": "చెవిటి", "understand": "అర్థం", "learn": "నేర్చుకోండి", "family": "కుటుంబం",
        # quick phrases
        "i need a doctor": "నాకు డాక్టర్ కావాలి", "please help me": "దయచేసి నాకు సహాయం చేయండి",
        "please call my family": "దయచేసి నా కుటుంబానికి ఫోన్ చేయండి",
        "i cannot hear, please write it down": "నాకు వినిపించదు, దయచేసి రాసి చూపించండి",
        "i am in pain": "నాకు నొప్పిగా ఉంది", "i need water": "నాకు నీళ్ళు కావాలి",
        "where is the toilet?": "మరుగుదొడ్డి ఎక్కడ ఉంది?", "thank you very much": "చాలా ధన్యవాదాలు",
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

    words = text.split()                            # 4) longest known phrase first, unknown stays English
    out, i = [], 0
    while i < len(words):
        for n in range(min(4, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + n])
            if phrase in d:
                out.append(d[phrase]); i += n
                break
        else:
            out.append(words[i]); i += 1
    if out != words:
        return " ".join(out), "offline dictionary (partial)"
    return text, "untranslated"


if __name__ == "__main__":
    for phrase in ["victory", "god", "thank you", "help me", "hello friend"]:
        for lang in ("hi", "kn", "te"):
            print(lang, phrase, "->", translate(phrase, lang))
