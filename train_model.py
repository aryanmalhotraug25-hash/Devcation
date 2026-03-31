"""
=============================================================
VoiceGuard Pro — Dataset Training Pipeline (132 features)
=============================================================
Expects:  dataset/real/*.wav  and  dataset/fake/*.wav
Run:      python train_model.py
=============================================================
"""

import os
import sys
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

DATASET_DIR = "dataset"
REAL_DIR    = os.path.join(DATASET_DIR, "real")
FAKE_DIR    = os.path.join(DATASET_DIR, "fake")
MODEL_PATH  = "voiceguard_model.pkl"
SR          = 22050
N_MFCC      = 40
SUPPORTED   = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def extract_features(filepath):
    """Extract 132 features from audio file."""
    try:
        y, sr = librosa.load(filepath, sr=SR, duration=15)
        if len(y) < sr * 0.5:
            return None

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std  = np.std(mfccs, axis=1)

        delta = librosa.feature.delta(mfccs)
        delta_mean = np.mean(delta, axis=1)

        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spec_flat = librosa.feature.spectral_flatness(y=y)[0]
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zcr       = librosa.feature.zero_crossing_rate(y)[0]
        rms       = librosa.feature.rms(y=y)[0]

        spectral = np.array([
            np.mean(spec_cent), np.std(spec_cent),
            np.mean(spec_bw),   np.std(spec_bw),
            np.mean(spec_flat), np.std(spec_flat),
            np.mean(spec_roll), np.std(spec_roll),
            np.mean(zcr),       np.std(zcr),
            np.mean(rms),       np.std(rms),
        ])

        return np.concatenate([mfcc_mean, mfcc_std, delta_mean, spectral])
    except Exception as e:
        print(f"    ❌ {filepath}: {e}")
        return None


def main():
    print("=" * 60)
    print("  VoiceGuard Pro — Training Pipeline (132 features)")
    print("=" * 60)

    for d in (REAL_DIR, FAKE_DIR):
        if not os.path.isdir(d):
            print(f"  ❌ Missing: {d}")
            sys.exit(1)

    X, y = [], []
    for label, folder, tag in [(0, REAL_DIR, "REAL"), (1, FAKE_DIR, "FAKE")]:
        files = [f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED)]
        print(f"\n  {tag}: {len(files)} files")
        for i, f in enumerate(files, 1):
            feat = extract_features(os.path.join(folder, f))
            if feat is not None and len(feat) == 132:
                X.append(feat)
                y.append(label)
            if i % 20 == 0:
                print(f"    Processed {i}/{len(files)}")

    X, y = np.array(X), np.array(y)
    print(f"\n  Total: {len(X)} samples | Real: {np.sum(y==0)} | Fake: {np.sum(y==1)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=3,
        random_state=42, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"\n  🎯 Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_te, clf.predict(X_te),
                                target_names=["Real", "Fake"]))

    joblib.dump(clf, MODEL_PATH)
    print(f"  ✅ Saved → {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()