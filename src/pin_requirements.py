"""
Pin the exact library versions installed on THIS computer into requirements.txt,
so judges (and Streamlit Cloud) install the same versions we tested with.

Usage (from the project root, with the venv active):
    python src/pin_requirements.py
"""
import os
from importlib.metadata import PackageNotFoundError, version

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ["mediapipe", "opencv-python", "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn",
            "joblib", "pyttsx3", "gTTS", "streamlit", "deep-translator", "pytest"]

lines = []
for p in PACKAGES:
    try:
        lines.append(f"{p}=={version(p)}")
    except PackageNotFoundError:
        print(f"  {p} is not installed - run: pip install {p}")
        lines.append(p)
with open(os.path.join(ROOT, "requirements.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("requirements.txt now contains:\n" + "\n".join(lines))
