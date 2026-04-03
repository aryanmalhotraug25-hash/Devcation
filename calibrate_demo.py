"""
=============================================================
VoiceGuard Pro — Demo Calibration Script (FFmpeg-safe)
=============================================================
Handles ANY audio format: mp3, m4a, webm, ogg, flac, wav, etc.

Run:  python calibrate_demo.py
=============================================================
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
import joblib

# ──────────────────────── Config ────────────────────────────
REAL_PATH  = "demo_audio/real.wav"
FAKE_PATH  = "demo_audio/fake_scam.wav"
MODEL_PATH = "voiceguard_model.pkl"
SR         = 22050
SEG_LEN    = 2.0
SEG_HOP    = 0.5
N_MFCC     = 40
N_FEATURES = 132


# ──────────────────── Safe Audio Loader ─────────────────────
def safe_load_audio(filepath, sr=22050, duration=30):
    """
    Try multiple methods to load any audio file:
      1. librosa direct (works for proper WAV)
      2. soundfile (works for proper WAV/FLAC)
      3. FFmpeg conversion to temp WAV then load
      4. pydub conversion then load
    """
    # Method 1: Direct librosa load
    try:
        y, rate = librosa.load(filepath, sr=sr, duration=duration)
        if len(y) > 0:
            print(f"    ✅ Loaded via librosa directly")
            return y, rate
    except Exception:
        pass

    # Method 2: soundfile
    try:
        import soundfile as sf
        data, rate = sf.read(filepath)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)  # stereo to mono
        if sr and sr != rate:
            data = librosa.resample(data, orig_sr=rate, target_sr=sr)
            rate = sr
        if duration:
            max_samples = int(duration * rate)
            data = data[:max_samples]
        if len(data) > 0:
            print(f"    ✅ Loaded via soundfile")
            return data.astype(np.float32), rate
    except Exception:
        pass

    # Method 3: FFmpeg conversion
    try:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix="_converted.wav", prefix="vg_"
        )
        tmp_path = tmp.name
        tmp.close()

        cmd = [
            "ffmpeg", "-y",
            "-i", filepath,
            "-ac", "1",            # mono
            "-ar", str(sr),        # target sample rate
            "-sample_fmt", "s16",  # 16-bit PCM
            "-t", str(duration),   # max duration
            tmp_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )

        if result.returncode == 0 and os.path.isfile(tmp_path):
            y, rate = librosa.load(tmp_path, sr=sr, duration=duration)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            if len(y) > 0:
                print(f"    ✅ Loaded via FFmpeg conversion")
                return y, rate

        # Cleanup on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    except FileNotFoundError:
        print("    ⚠  FFmpeg not found in PATH")
    except subprocess.TimeoutExpired:
        print("    ⚠  FFmpeg timed out")
    except Exception as e:
        print(f"    ⚠  FFmpeg error: {e}")

    # Method 4: pydub
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(filepath)
        audio = audio.set_channels(1).set_frame_rate(sr)

        if duration:
            audio = audio[:int(duration * 1000)]

        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2 ** 15)  # normalize 16-bit to [-1, 1]

        if len(samples) > 0:
            print(f"    ✅ Loaded via pydub")
            return samples, sr
    except ImportError:
        print("    ⚠  pydub not installed")
    except Exception as e:
        print(f"    ⚠  pydub error: {e}")

    return None, None


# ──────────────────── Feature Extraction ────────────────────
def extract_features(y, sr, n_mfcc=N_MFCC):
    """Extract 132-dimensional feature vector."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
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


# ──────────────────── Segmentation ──────────────────────────
def segment_audio(y, sr, seg_len=SEG_LEN, hop=SEG_HOP):
    seg_samples = int(seg_len * sr)
    hop_samples = int(hop * sr)
    segments = []
    for start in range(0, len(y) - seg_samples + 1, hop_samples):
        segments.append(y[start : start + seg_samples])
    return segments


# ──────────────────── Augmentation ──────────────────────────
def augment_segment(y, sr):
    results = [y.copy()]

    # Noise variants
    results.append(y + np.random.normal(0, 0.003, len(y)).astype(np.float32))
    results.append(y + np.random.normal(0, 0.008, len(y)).astype(np.float32))

    # Volume variants
    results.append(y * 1.4)
    results.append(y * 0.7)

    # Time stretch
    try:
        y_fast = librosa.effects.time_stretch(y, rate=1.1)
        if len(y_fast) >= len(y):
            results.append(y_fast[:len(y)])
        else:
            padded = np.zeros(len(y), dtype=np.float32)
            padded[:len(y_fast)] = y_fast
            results.append(padded)
    except Exception:
        pass

    try:
        y_slow = librosa.effects.time_stretch(y, rate=0.9)
        results.append(y_slow[:len(y)])
    except Exception:
        pass

    # Pitch shift
    try:
        results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5))
    except Exception:
        pass

    try:
        results.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5))
    except Exception:
        pass

    return results


# ──────────────────── Diagnose Audio File ───────────────────
def diagnose_file(filepath):
    """Print info about an audio file for debugging."""
    print(f"\n  📋 Diagnosing: {filepath}")
    print(f"     File size: {os.path.getsize(filepath):,} bytes")

    # Read first 16 bytes to check actual format
    with open(filepath, "rb") as f:
        header = f.read(16)

    # Check magic bytes
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        print("     Format: Genuine WAV (RIFF/WAVE)")
    elif header[:4] == b"RIFF":
        print("     Format: RIFF container (may not be standard WAV)")
    elif header[:3] == b"ID3" or header[:2] == b"\xff\xfb":
        print("     Format: MP3 (renamed to .wav!)")
    elif header[:4] == b"fLaC":
        print("     Format: FLAC (renamed to .wav!)")
    elif header[:4] == b"OggS":
        print("     Format: OGG (renamed to .wav!)")
    elif header[:4] == b"\x1aE\xdf\xa3":
        print("     Format: WebM/MKV (renamed to .wav!)")
    elif header[4:8] == b"ftyp":
        print("     Format: MP4/M4A (renamed to .wav!)")
    else:
        hex_preview = " ".join(f"{b:02x}" for b in header[:12])
        print(f"     Format: Unknown — header bytes: {hex_preview}")

    return header


# ──────────────────── Convert to Real WAV ───────────────────
def convert_to_real_wav(input_path, output_path=None):
    """
    Convert any audio file to a genuine 16-bit PCM WAV
    using FFmpeg or pydub.
    """
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + "_converted.wav"

    # Try FFmpeg first
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", "22050",
            "-sample_fmt", "s16",
            output_path,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
        if result.returncode == 0 and os.path.isfile(output_path):
            print(f"    ✅ Converted via FFmpeg → {output_path}")
            return output_path
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Try pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)
        audio.export(output_path, format="wav")
        print(f"    ✅ Converted via pydub → {output_path}")
        return output_path
    except Exception:
        pass

    return None


# ──────────────────── Main ──────────────────────────────────
def main():
    print("=" * 60)
    print("  VoiceGuard Pro — Demo Calibration (FFmpeg-safe)")
    print("=" * 60)

    # 1. Check files exist
    print("\n[1/5] Checking demo audio files …")
    for tag, path in [("REAL", REAL_PATH), ("FAKE", FAKE_PATH)]:
        if not os.path.isfile(path):
            print(f"  ❌  Missing: {path}")
            sys.exit(1)
        print(f"  ✅  {tag}: {path}")

    # 2. Diagnose and load
    print("\n[2/5] Loading & segmenting audio …")
    X_all, y_all = [], []

    for label, path, tag in [(0, REAL_PATH, "REAL"), (1, FAKE_PATH, "FAKE")]:
        header = diagnose_file(path)

        # If not genuine WAV, try converting first
        is_genuine_wav = (header[:4] == b"RIFF" and header[8:12] == b"WAVE")

        load_path = path
        converted_path = None

        if not is_genuine_wav:
            print(f"     ⚠  Not a genuine WAV — attempting conversion…")
            converted_path = convert_to_real_wav(path)
            if converted_path:
                load_path = converted_path
            else:
                print("     Trying direct load anyway…")

        # Load audio
        y_audio, sr = safe_load_audio(load_path, sr=SR, duration=30)

        if y_audio is None or len(y_audio) < SR * 0.5:
            print(f"\n  ❌  Could not load {tag} audio!")
            print(f"     Troubleshooting:")
            print(f"     1. Install FFmpeg: winget install Gyan.FFmpeg")
            print(f"     2. Install pydub:  pip install pydub")
            print(f"     3. Convert manually: Use Audacity to export as")
            print(f"        'WAV (Microsoft) signed 16-bit PCM'")
            sys.exit(1)

        duration = len(y_audio) / sr
        print(f"  {tag}: {duration:.1f}s loaded successfully")

        # Segment
        segments = segment_audio(y_audio, sr)
        print(f"  {tag}: {len(segments)} segments")

        # Augment and extract features
        seg_count = 0
        for seg in segments:
            augmented = augment_segment(seg, sr)
            for aug_y in augmented:
                try:
                    feat = extract_features(aug_y, sr)
                    if feat is not None and len(feat) == N_FEATURES:
                        X_all.append(feat)
                        y_all.append(label)
                        seg_count += 1
                except Exception:
                    continue

        print(f"  {tag}: {seg_count} total samples (with augmentation)")

        # Cleanup converted file
        if converted_path and os.path.isfile(converted_path):
            try:
                os.unlink(converted_path)
            except OSError:
                pass

    X = np.array(X_all)
    y = np.array(y_all)
    print(f"\n  Total: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Real: {np.sum(y == 0)} | Fake: {np.sum(y == 1)}")

    if len(X) < 10:
        print("❌  Too few samples. Use audio files at least 5 seconds long.")
        sys.exit(1)

    # 3. Cross-validate
    print("\n[3/5] Cross-validating …")
    clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight='balanced',  
    random_state=42,
    n_jobs=-1,
    )

    n_splits = min(5, min(np.sum(y == 0), np.sum(y == 1)))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"  CV Accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
    else:
        print("  (Skipping CV — too few per class)")

    # 4. Train final
    print("\n[4/5] Training final model …")
    clf.fit(X, y)

    y_pred = clf.predict(X)
    print("\n  Training Report:")
    print(classification_report(y, y_pred, target_names=["Real", "Fake"]))

    # Top features
    imp = clf.feature_importances_
    labels = (
        [f"MFCC-{i+1}_mean" for i in range(40)] +
        [f"MFCC-{i+1}_std" for i in range(40)] +
        [f"ΔMFCC-{i+1}_mean" for i in range(40)] +
        ["SpecCent_μ", "SpecCent_σ", "SpecBW_μ", "SpecBW_σ",
         "SpecFlat_μ", "SpecFlat_σ", "SpecRoll_μ", "SpecRoll_σ",
         "ZCR_μ", "ZCR_σ", "RMS_μ", "RMS_σ"]
    )
    top_idx = np.argsort(imp)[::-1][:15]
    print("  Top-15 Features:")
    for rank, idx in enumerate(top_idx, 1):
        bar = "█" * int(imp[idx] * 300)
        print(f"    {rank:>2}. {labels[idx]:<18} {imp[idx]:.4f}  {bar}")

    # 5. Verify
    print("\n[5/5] Verifying on demo files …")
    for tag, path in [("REAL", REAL_PATH), ("FAKE", FAKE_PATH)]:
        y_audio, sr = safe_load_audio(path, sr=SR, duration=15)
        if y_audio is None:
            print(f"  {tag}: Could not reload for verification")
            continue
        feat = extract_features(y_audio, sr)
        proba = clf.predict_proba(feat.reshape(1, -1))[0]
        classes = list(clf.classes_)
        fake_idx = classes.index(1) if 1 in classes else 1
        ai_pct = proba[fake_idx] * 100
        verdict = "🔴 FAKE" if ai_pct > 50 else "🟢 REAL"
        print(f"  {tag:>4}: AI={ai_pct:5.1f}%  →  {verdict}")

    joblib.dump(clf, MODEL_PATH)
    print(f"\n  ✅  Model saved → {MODEL_PATH}")
    print("\n" + "=" * 60)
    print("  Done! Now run:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()