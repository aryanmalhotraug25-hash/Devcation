"""
=============================================================
VoiceGuard Pro v7.0 — Dataset Training Pipeline (193 features)
=============================================================
Expects:  dataset/real/*.wav  and  dataset/fake/*.wav
Run:      python train_model.py
=============================================================
"""

import os
import sys
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

DATASET_DIR = "dataset"
REAL_DIR    = os.path.join(DATASET_DIR, "real")
FAKE_DIR    = os.path.join(DATASET_DIR, "fake")
MODEL_PATH  = "voiceguard_model.pkl"
SR          = 22050
N_MFCC      = 40
N_FEATURES  = 193
SUPPORTED   = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm")


def extract_features(y, sr, n_mfcc=N_MFCC):
    """Extract 193 features — must match app.py exactly."""
    if len(y) < sr * 0.3:
        return None

    # MFCCs + deltas
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta1 = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)

    # Spectral features
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sf = librosa.feature.spectral_flatness(y=y)[0]
    sro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zc = librosa.feature.zero_crossing_rate(y)[0]
    rm = librosa.feature.rms(y=y)[0]

    spectral = np.array([
        np.mean(sc), np.std(sc),
        np.mean(sb), np.std(sb),
        np.mean(sf), np.std(sf),
        np.mean(sro), np.std(sro),
        np.mean(zc), np.std(zc),
        np.mean(rm), np.std(rm),
    ], dtype=np.float32)

    # HNR
    try:
        harmonic, percussive = librosa.effects.hpss(y)
        hnr_ratio = np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-8)
        hnr_std = np.std(np.abs(harmonic)) / (np.std(np.abs(percussive)) + 1e-8)
    except:
        hnr_ratio, hnr_std = 1.0, 1.0

    # Spectral contrast
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        contrast_mean = np.mean(contrast, axis=1)
    except:
        contrast_mean = np.zeros(7, dtype=np.float32)

    # Chroma
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
    except:
        chroma_mean = np.zeros(12, dtype=np.float32)

    feat = np.concatenate([
        np.mean(mfccs, axis=1),     # 40
        np.std(mfccs, axis=1),      # 40
        np.mean(delta1, axis=1),    # 40
        np.mean(delta2, axis=1),    # 40
        spectral,                    # 12
        np.array([hnr_ratio, hnr_std], dtype=np.float32),  # 2
        contrast_mean,               # 7
        chroma_mean,                 # 12
    ]).astype(np.float32)            # Total: 193

    if len(feat) != N_FEATURES:
        return None
    return feat


def extract_from_file(filepath, max_dur=15):
    """Load file and extract features."""
    try:
        y, sr = librosa.load(filepath, sr=SR, duration=max_dur)
        return extract_features(y, sr)
    except Exception as e:
        print(f"    ❌ {os.path.basename(filepath)}: {e}")
        return None


def augment_and_extract(y, sr, label):
    """Generate augmented samples from a single audio segment."""
    samples = []

    # Original
    feat = extract_features(y, sr)
    if feat is not None:
        samples.append((feat, label))

    # --- Noise augmentation ---
    for noise_level in [0.001, 0.003, 0.006, 0.01]:
        noisy = y + np.random.normal(0, noise_level, len(y)).astype(np.float32)
        feat = extract_features(noisy, sr)
        if feat is not None:
            samples.append((feat, label))

    # --- Volume augmentation ---
    for gain in [0.6, 0.8, 1.3, 1.6]:
        loud = np.clip(y * gain, -1.0, 1.0).astype(np.float32)
        feat = extract_features(loud, sr)
        if feat is not None:
            samples.append((feat, label))

    # --- Pitch shift ---
    for n_steps in [-2, -1, 1, 2]:
        try:
            shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            feat = extract_features(shifted, sr)
            if feat is not None:
                samples.append((feat, label))
        except:
            pass

    # --- Time stretch ---
    for rate in [0.9, 1.1]:
        try:
            stretched = librosa.effects.time_stretch(y, rate=rate)
            min_len = int(sr * 0.5)
            if len(stretched) > min_len:
                feat = extract_features(stretched[:len(y)], sr) if len(stretched) >= len(y) else extract_features(stretched, sr)
                if feat is not None:
                    samples.append((feat, label))
        except:
            pass

    return samples


def main():
    print("=" * 65)
    print("  VoiceGuard Pro v7.0 — Training Pipeline (193 features)")
    print("=" * 65)

    for d in (REAL_DIR, FAKE_DIR):
        if not os.path.isdir(d):
            print(f"  ❌ Missing: {d}")
            print(f"     Create it and add .wav files")
            sys.exit(1)

    X, y = [], []
    stats = {"real_files": 0, "fake_files": 0, "real_samples": 0, "fake_samples": 0}

    for label, folder, tag in [(0, REAL_DIR, "REAL"), (1, FAKE_DIR, "FAKE")]:
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED)])
        print(f"\n  {tag}: {len(files)} files in {folder}")
        stats[f"{tag.lower()}_files"] = len(files)

        for i, f in enumerate(files, 1):
            filepath = os.path.join(folder, f)
            try:
                y_audio, sr = librosa.load(filepath, sr=SR, duration=30)
            except Exception as e:
                print(f"    ❌ Could not load {f}: {e}")
                continue

            if len(y_audio) < sr * 0.5:
                print(f"    ⚠️ {f} too short, skipping")
                continue

            # --- Segment into 3-second windows ---
            seg_len = int(3.0 * sr)
            hop_len = int(1.0 * sr)  # 1s hop for overlap

            segments = []
            for start in range(0, len(y_audio) - seg_len + 1, hop_len):
                segments.append(y_audio[start:start + seg_len])

            # Also use full audio
            if len(y_audio) >= seg_len:
                segments.append(y_audio[:int(15 * sr)])

            for seg in segments:
                aug_samples = augment_and_extract(seg, sr, label)
                for feat, lbl in aug_samples:
                    X.append(feat)
                    y.append(lbl)

            if i % 10 == 0 or i == len(files):
                count = sum(1 for lbl in y if lbl == label)
                print(f"    Processed {i}/{len(files)} → {count} samples so far")

    X = np.array(X)
    y = np.array(y)

    n_real = int(np.sum(y == 0))
    n_fake = int(np.sum(y == 1))
    print(f"\n  {'='*50}")
    print(f"  Total: {len(X)} samples | Real: {n_real} | Fake: {n_fake}")
    print(f"  Features per sample: {X.shape[1]}")
    print(f"  {'='*50}")

    if len(X) < 20:
        print("  ❌ Too few samples. Add more audio files!")
        sys.exit(1)

    # --- Split ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Try GradientBoosting first, fall back to RandomForest ---
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        print("\n  Using: GradientBoostingClassifier")
        clf = GradientBoostingClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            max_features='sqrt',
            random_state=42,
        )
    except:
        from sklearn.ensemble import RandomForestClassifier
        print("\n  Using: RandomForestClassifier (fallback)")
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )

    # --- Cross-validation ---
    print("\n  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # --- Train final model ---
    print("\n  Training final model...")
    clf.fit(X_tr, y_tr)

    # --- Evaluate ---
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print(f"\n  🎯 Test Accuracy: {acc*100:.2f}%\n")
    print(classification_report(
        y_te, y_pred, target_names=["Real (Human)", "Fake (AI)"]
    ))

    cm = confusion_matrix(y_te, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                Predicted Real  Predicted Fake")
    print(f"  Actual Real       {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  Actual Fake       {cm[1][0]:>5}          {cm[1][1]:>5}")

    # --- Feature importance (top 15) ---
    if hasattr(clf, 'feature_importances_'):
        imp = clf.feature_importances_
        top_idx = np.argsort(imp)[::-1][:15]

        feature_names = (
            [f"mfcc_mean_{i}" for i in range(40)] +
            [f"mfcc_std_{i}" for i in range(40)] +
            [f"delta1_mean_{i}" for i in range(40)] +
            [f"delta2_mean_{i}" for i in range(40)] +
            ["spec_cent_m", "spec_cent_s", "spec_bw_m", "spec_bw_s",
             "spec_flat_m", "spec_flat_s", "spec_roll_m", "spec_roll_s",
             "zcr_m", "zcr_s", "rms_m", "rms_s"] +
            ["hnr_ratio", "hnr_std"] +
            [f"contrast_{i}" for i in range(7)] +
            [f"chroma_{i}" for i in range(12)]
        )

        print(f"\n  📊 Top 15 Most Important Features:")
        for rank, idx in enumerate(top_idx, 1):
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"    {rank:>2}. {name:<20s}  importance: {imp[idx]:.4f}")

    # --- Save ---
    joblib.dump(clf, MODEL_PATH)
    print(f"\n  ✅ Saved → {MODEL_PATH}")
    print(f"  📏 Model expects {clf.n_features_in_} features")
    print("=" * 65)


if __name__ == "__main__":
    main()