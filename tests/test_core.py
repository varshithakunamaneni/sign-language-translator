"""
Automated tests for SignSpeak's core logic (no camera or internet needed).

Run from the project root:
    python -m pytest -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import utils                                  # noqa: E402
from translate import translate               # noqa: E402
import train                                  # noqa: E402
import train_words                            # noqa: E402


# ---------- fake MediaPipe results ----------
class LM:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class Cat:
    def __init__(self, name):
        self.category_name = name


class Result:
    def __init__(self, hands):            # hands: list of (label, 21x3 array)
        self.hand_landmarks = [[LM(*p) for p in pts] for _, pts in hands]
        self.handedness = [[Cat(label)] for label, _ in hands]


def random_hand(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.3, 0.7, (21, 3))


# ---------- feature extraction ----------
def test_feature_vector_shape_and_empty():
    feats, n = utils.extract_features(Result([]))
    assert feats.shape == (126,) and n == 0 and not feats.any()


def test_features_ignore_hand_position_and_size():
    """Moving or resizing the hand must not change the features."""
    pts = random_hand()
    moved = (pts - pts[0]) * 1.7 + np.array([0.1, -0.05, 0.02]) + pts[0]
    a, _ = utils.extract_features(Result([("Right", pts)]))
    b, _ = utils.extract_features(Result([("Right", moved)]))
    assert np.allclose(a, b, atol=1e-5)


def test_left_and_right_hands_use_separate_slots():
    feats, n = utils.extract_features(Result([("Left", random_hand(1))]))
    assert n == 1 and feats[:63].any() and not feats[63:].any()


def test_motion_features_add_position():
    feats, _ = utils.extract_motion_features(Result([("Right", random_hand(2))]))
    assert feats.shape == (utils.NUM_MOTION_FEATURES,)
    assert feats[129:132].any() and not feats[126:129].any()   # right-hand position filled, left empty


def test_resample_clip_length():
    frames = [np.full(4, i, np.float32) for i in range(37)]
    out = utils.resample_clip(np.linspace(0, 1.5, 37), frames, n=10)
    assert out.shape == (40,) and out[0] == 0 and out[-1] == 36


# ---------- augmentation ----------
def test_letter_mirror_twice_is_identity():
    x = np.random.default_rng(3).normal(size=(5, 126)).astype(np.float32)
    assert np.allclose(train.mirror(train.mirror(x)), x)


def test_word_mirror_twice_is_identity():
    x = np.random.default_rng(4).normal(size=(3, utils.WORD_FRAMES * utils.NUM_MOTION_FEATURES)).astype(np.float32)
    assert np.allclose(train_words.mirror(train_words.mirror(x)), x)


def test_augment_keeps_labels_aligned():
    x = np.ones((4, 126), np.float32)
    y = np.array(list("ABCD"), dtype=object)
    xa, ya = train.augment(x, y)
    assert len(xa) == len(ya) and set(ya) == set(y)


# ---------- translation ----------
@pytest.mark.parametrize("lang, text, expected", [
    ("hi", "thank you", "धन्यवाद"),
    ("kn", "water", "ನೀರು"),
    ("te", "doctor", "డాక్టర్"),
])
def test_offline_dictionary(lang, text, expected):
    assert translate(text, lang)[0] == expected


def test_phrase_inside_sentence_is_translated_as_a_whole():
    out, _ = translate("hello thank you", "hi")
    assert out == "नमस्ते धन्यवाद"


# ---------- browser model == Python model ----------
def test_exported_network_matches_sklearn():
    """The website runs our exported weights in JavaScript with the same maths as
    benchmark.mlp_forward. Check that maths equals scikit-learn's own predictions."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from benchmark import mlp_forward

    rng = np.random.default_rng(5)
    X = rng.normal(size=(120, 126)); y = rng.integers(0, 3, 120)
    sc = StandardScaler().fit(X)
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=300, random_state=0).fit(sc.transform(X), y)
    model = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
             "layers": [{"W": W.flatten().tolist(), "b": b.tolist(), "n_in": W.shape[0], "n_out": W.shape[1]}
                        for W, b in zip(mlp.coefs_, mlp.intercepts_)]}
    for row in X[:10]:
        assert np.allclose(mlp_forward(model, row), mlp.predict_proba(sc.transform(row[None]))[0], atol=1e-6)
