"""
=============================================================
VoiceGuard Pro — Dummy Model Generator (132 features)
=============================================================
Run:  python dummy_model.py
Then:  streamlit run app.py
=============================================================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

N_FEATURES = 132  # Must match app.py and calibrate_demo.py


def generate_synthetic_data(n_per_class=500, seed=42):
    """
    Synthetic data encoding known acoustic differences:
    - Real speech: higher MFCC variance, more delta energy, lower spectral flatness
    - AI speech: lower variance, less delta energy, higher spectral flatness
    """
    rng = np.random.RandomState(seed)

    # ── Real voice (label 0) ──
    real = np.zeros((n_per_class, N_FEATURES))
    # MFCC means (0-39): natural vocal tract
    real[:, 0] = rng.normal(-320, 30, n_per_class)
    for i in range(1, 40):
        real[:, i] = rng.normal(0, max(25 - i * 0.5, 3), n_per_class)
    # MFCC stds (40-79): HIGH variance = natural
    for i in range(40, 80):
        real[:, i] = rng.normal(45, 12, n_per_class)
    # Delta MFCC means (80-119): MORE temporal change = natural
    for i in range(80, 120):
        real[:, i] = rng.normal(0, 5, n_per_class)
    # Spectral features (120-131)
    real[:, 120] = rng.normal(2200, 400, n_per_class)    # centroid mean
    real[:, 121] = rng.normal(600, 150, n_per_class)     # centroid std (HIGH)
    real[:, 122] = rng.normal(2800, 500, n_per_class)    # bandwidth mean
    real[:, 123] = rng.normal(500, 120, n_per_class)     # bandwidth std
    real[:, 124] = rng.normal(0.06, 0.025, n_per_class)  # flatness mean (LOW)
    real[:, 125] = rng.normal(0.08, 0.03, n_per_class)   # flatness std (HIGH)
    real[:, 126] = rng.normal(4500, 800, n_per_class)    # rolloff mean
    real[:, 127] = rng.normal(900, 200, n_per_class)     # rolloff std
    real[:, 128] = rng.normal(0.07, 0.02, n_per_class)   # zcr mean
    real[:, 129] = rng.normal(0.04, 0.015, n_per_class)  # zcr std (HIGH)
    real[:, 130] = rng.normal(0.05, 0.015, n_per_class)  # rms mean
    real[:, 131] = rng.normal(0.06, 0.02, n_per_class)   # rms std (HIGH)

    # ── AI / Fake voice (label 1) ──
    fake = np.zeros((n_per_class, N_FEATURES))
    fake[:, 0] = rng.normal(-240, 22, n_per_class)
    for i in range(1, 40):
        fake[:, i] = rng.normal(3, max(20 - i * 0.4, 4), n_per_class)
    # MFCC stds: LOW variance = artificial uniformity
    for i in range(40, 80):
        fake[:, i] = rng.normal(20, 7, n_per_class)
    # Delta MFCC: LESS temporal change = smoothed
    for i in range(80, 120):
        fake[:, i] = rng.normal(0, 2, n_per_class)
    # Spectral features — vocoder signatures
    fake[:, 120] = rng.normal(2600, 300, n_per_class)    # centroid higher
    fake[:, 121] = rng.normal(300, 80, n_per_class)      # centroid std LOW
    fake[:, 122] = rng.normal(3200, 400, n_per_class)    # bandwidth wider
    fake[:, 123] = rng.normal(250, 60, n_per_class)      # bandwidth std LOW
    fake[:, 124] = rng.normal(0.18, 0.04, n_per_class)   # flatness HIGH
    fake[:, 125] = rng.normal(0.04, 0.015, n_per_class)  # flatness std LOW
    fake[:, 126] = rng.normal(5200, 600, n_per_class)    # rolloff higher
    fake[:, 127] = rng.normal(400, 100, n_per_class)     # rolloff std LOW
    fake[:, 128] = rng.normal(0.06, 0.015, n_per_class)  # zcr mean
    fake[:, 129] = rng.normal(0.015, 0.006, n_per_class) # zcr std LOW
    fake[:, 130] = rng.normal(0.06, 0.012, n_per_class)  # rms mean
    fake[:, 131] = rng.normal(0.02, 0.008, n_per_class)  # rms std LOW

    X = np.vstack([real, fake])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def main():
    print("=" * 60)
    print("  VoiceGuard Pro — Dummy Model (132 features)")
    print("=" * 60)

    X, y = generate_synthetic_data(500)
    print(f"\n  Dataset: {X.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"  Test Accuracy: {acc*100:.1f}%\n")
    print(classification_report(y_te, clf.predict(X_te),
                                target_names=["Real", "Fake"]))

    joblib.dump(clf, "voiceguard_model.pkl")
    print("  ✅  Saved → voiceguard_model.pkl")
    print("\n  ⚠️  This is a DUMMY model for UI testing only!")
    print("  For real detection, run:  python calibrate_demo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()