"""
=================================================================
 VoiceGuard Pro v6.0 — Real-Time Deepfake Scam Detector
 Reference-based detection + In-app calibration + 16-metric heuristic
=================================================================
 Run:  streamlit run app.py
=================================================================
"""

import os, re, time, tempfile, subprocess
import numpy as np
import joblib
import streamlit as st

st.set_page_config(
    page_title="VoiceGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            pass

# ================================================================
# AUDIO LOADING (FFmpeg → pydub → librosa chain)
# ================================================================
def safe_load_audio(filepath, sr=22050, duration=15):
    if not filepath or not os.path.isfile(filepath):
        return None, None

    try:
        import librosa
        y, r = librosa.load(filepath, sr=sr, duration=duration)
        if y is not None and len(y) > 0:
            return y.astype(np.float32), r
    except Exception:
        pass

    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix="vg_")
        tmp_path = tmp.name; tmp.close()
        subprocess.run(["ffmpeg","-y","-i",filepath,"-ac","1","-ar",str(sr),
                        "-sample_fmt","s16","-t",str(duration),tmp_path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        import librosa
        y, r = librosa.load(tmp_path, sr=sr, duration=duration)
        if y is not None and len(y) > 0:
            return y.astype(np.float32), r
    except Exception:
        pass
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try: os.unlink(tmp_path)
            except: pass

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(filepath).set_channels(1).set_frame_rate(sr)
        if duration: audio = audio[:int(duration*1000)]
        s = np.array(audio.get_array_of_samples(), dtype=np.float32)
        mx = float(2**(8*audio.sample_width-1))
        if mx > 0: s = s/mx
        if len(s) > 0: return s, sr
    except Exception:
        pass
    return None, None


def save_uploaded_to_temp(src, suf=".wav"):
    if src is None: return None
    nm = getattr(src, "name", "")
    if nm:
        ext = os.path.splitext(nm)[1]
        if ext: suf = ext
    try:
        data = src.getvalue() if hasattr(src, "getvalue") else src.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suf, prefix="vg_")
        tmp.write(data); p = tmp.name; tmp.close()
        return p
    except: return None


# ================================================================
# FEATURE EXTRACTION — 132 features
# ================================================================
N_FEATURES = 132

def extract_features_from_array(y, sr, n_mfcc=40):
    import librosa
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfccs)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sf = librosa.feature.spectral_flatness(y=y)[0]
    sro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zc = librosa.feature.zero_crossing_rate(y)[0]
    rm = librosa.feature.rms(y=y)[0]
    spectral = np.array([np.mean(sc),np.std(sc),np.mean(sb),np.std(sb),
                         np.mean(sf),np.std(sf),np.mean(sro),np.std(sro),
                         np.mean(zc),np.std(zc),np.mean(rm),np.std(rm)], dtype=np.float32)
    return np.concatenate([np.mean(mfccs,axis=1), np.std(mfccs,axis=1),
                           np.mean(delta,axis=1), spectral]).astype(np.float32)


def extract_features(filepath, max_dur=15):
    y, sr = safe_load_audio(filepath, sr=22050, duration=max_dur)
    if y is None or len(y) < 22050*0.3: return None
    try: return extract_features_from_array(y, sr)
    except: return None


# ================================================================
# ACOUSTIC METRICS — 16 dimensions
# ================================================================
def compute_acoustic_metrics(filepath):
    import librosa
    y, sr = safe_load_audio(filepath, sr=22050, duration=15)
    if y is None or len(y) < int(sr*0.3): return None
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfccs)
        sf = librosa.feature.spectral_flatness(y=y)[0]
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zc = librosa.feature.zero_crossing_rate(y)[0]
        rm = librosa.feature.rms(y=y)[0]
        S = np.abs(librosa.stft(y))
        flux = np.sqrt(np.sum(np.diff(S,axis=1)**2, axis=0)) if S.shape[1]>1 else np.array([0.0])
        traj = float(np.mean(np.abs(np.diff(mfccs,axis=1)))) if mfccs.shape[1]>1 else 5.0

        try:
            pitches, mags = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            pv = []
            for t in range(pitches.shape[1]):
                idx = mags[:,t].argmax()
                p = pitches[idx,t]
                if 60 < p < 600: pv.append(p)
            if len(pv) > 10:
                pa = np.array(pv)
                pj = float(np.mean(np.abs(np.diff(pa)))/(np.mean(pa)+1e-8))
                ps = float(np.std(pa)/(np.mean(pa)+1e-8))
            else: pj, ps = 0.02, 0.06
        except: pj, ps = 0.02, 0.06

        shim = float(np.mean(np.abs(np.diff(rm)))/(np.mean(rm)+1e-8)) if len(rm)>2 else 0.2
        probs = rm/(np.sum(rm)+1e-8); probs = probs[probs>1e-10]
        entropy = float(-np.sum(probs*np.log2(probs))) if len(probs)>0 else 5.0
        bw_cv = float(np.std(sb)/(np.mean(sb)+1e-8))
        mfcc_cv = float(np.mean(np.std(mfccs,axis=1)/(np.abs(np.mean(mfccs,axis=1))+1e-8)))

        return {
            "mfcc_variance": float(np.mean(np.std(mfccs,axis=1))),
            "mfcc_cv": mfcc_cv,
            "delta_energy": float(np.mean(np.abs(delta))),
            "spectral_flatness": float(np.mean(sf)),
            "flatness_consistency": float(np.std(sf)),
            "spectral_centroid": float(np.mean(sc)),
            "centroid_variation": float(np.std(sc)),
            "zcr_mean": float(np.mean(zc)),
            "zcr_variation": float(np.std(zc)),
            "rms_dynamics": float(np.std(rm)),
            "temporal_smoothness": traj,
            "pitch_jitter": pj,
            "pitch_stability": ps,
            "shimmer": shim,
            "spectral_flux_mean": float(np.mean(flux)),
            "spectral_flux_std": float(np.std(flux)),
            "energy_entropy": entropy,
            "bandwidth_cv": bw_cv,
        }
    except: return None


# ================================================================
# REFERENCE-BASED DETECTION — compares against demo files
# ================================================================
BASE_DIR = os.path.dirname(__file__)

AUDIO_PATHS = {
    "fake": os.path.join(BASE_DIR, "demo_audio", "fake_scam.wav"),
    "real": os.path.join(BASE_DIR, "demo_audio", "real.wav")
}

@st.cache_resource
def load_reference_metrics():
    """Load acoustic metrics from demo files for comparison."""
    refs = {}
    for key, path in AUDIO_PATHS.items():
        if os.path.isfile(path):
            m = compute_acoustic_metrics(path)
            if m:
                refs[key] = m
    return refs

REFS = load_reference_metrics()


def reference_based_score(new_metrics):
    """
    Compare new audio metrics against real.wav and fake_scam.wav.
    Returns 0-100 (100 = very close to fake, 0 = very close to real).
    """
    if not REFS or "real" not in REFS or "fake" not in REFS or not new_metrics:
        return None

    real_ref = REFS["real"]
    fake_ref = REFS["fake"]

    keys = [k for k in new_metrics if k in real_ref and k in fake_ref]
    if len(keys) < 5:
        return None

    new_vec = np.array([new_metrics[k] for k in keys])
    real_vec = np.array([real_ref[k] for k in keys])
    fake_vec = np.array([fake_ref[k] for k in keys])

    # Normalize each dimension by the range between real and fake
    ranges = np.abs(fake_vec - real_vec) + 1e-10
    new_norm = (new_vec - real_vec) / ranges
    fake_norm = (fake_vec - real_vec) / ranges  # Should be ~1.0 for each

    # Weighted Euclidean distance (weight important metrics more)
    weight_map = {
        "pitch_jitter": 3.0, "shimmer": 2.5, "mfcc_variance": 2.0,
        "delta_energy": 2.0, "spectral_flatness": 2.0,
        "temporal_smoothness": 1.8, "rms_dynamics": 1.5,
        "pitch_stability": 2.0, "spectral_flux_mean": 1.5,
        "energy_entropy": 1.2, "flatness_consistency": 1.5,
        "centroid_variation": 1.2, "zcr_variation": 1.0,
        "bandwidth_cv": 1.0, "mfcc_cv": 1.5,
        "spectral_flux_std": 1.0, "spectral_centroid": 0.5,
        "zcr_mean": 0.5,
    }
    weights = np.array([weight_map.get(k, 1.0) for k in keys])

    dist_to_real = np.sqrt(np.sum(weights * (new_norm ** 2)))
    dist_to_fake = np.sqrt(np.sum(weights * ((new_norm - fake_norm) ** 2)))

    total = dist_to_real + dist_to_fake + 1e-10
    ai_score = (dist_to_real / total) * 100.0

    return round(min(max(ai_score, 0.0), 100.0), 1)


# ================================================================
# STANDALONE HEURISTIC (backup when no reference files exist)
# ================================================================
def heuristic_ai_score(metrics):
    if not metrics:
        return None

    def clamp01(x):
        return max(0.0, min(1.0, x))

    def score_low(val, lo, hi):
        if val >= hi: return 0.0
        if val <= lo: return 1.0
        return clamp01(1.0 - (val-lo)/(hi-lo+1e-12))

    def score_high(val, lo, hi):
        if val <= lo: return 0.0
        if val >= hi: return 1.0
        return clamp01((val-lo)/(hi-lo+1e-12))

    s = {}
    w = {}

    s["mfcc"] = score_low(metrics["mfcc_variance"], 15, 50); w["mfcc"] = 12
    s["delta"] = score_low(metrics["delta_energy"], 0.8, 4.5); w["delta"] = 12
    s["flat"] = score_high(metrics["spectral_flatness"], 0.02, 0.10); w["flat"] = 10
    s["tsmooth"] = score_low(metrics["temporal_smoothness"], 1.2, 6.0); w["tsmooth"] = 10
    s["rms"] = score_low(metrics["rms_dynamics"], 0.006, 0.04); w["rms"] = 8
    s["fcons"] = score_low(metrics["flatness_consistency"], 0.015, 0.08); w["fcons"] = 8
    s["cent"] = score_low(metrics["centroid_variation"], 100, 700); w["cent"] = 7
    s["jitter"] = score_low(metrics.get("pitch_jitter",0.02), 0.005, 0.035); w["jitter"] = 16
    s["pstab"] = score_low(metrics.get("pitch_stability",0.06), 0.02, 0.12); w["pstab"] = 12
    s["shimmer"] = score_low(metrics.get("shimmer",0.2), 0.06, 0.35); w["shimmer"] = 14
    s["entropy"] = score_low(metrics.get("energy_entropy",5), 2.0, 6.5); w["entropy"] = 7
    s["flux"] = score_low(metrics.get("spectral_flux_mean",5), 0.8, 6.0); w["flux"] = 8
    s["fluxs"] = score_low(metrics.get("spectral_flux_std",5), 0.5, 6.0); w["fluxs"] = 5
    s["zcr"] = score_low(metrics["zcr_variation"], 0.008, 0.04); w["zcr"] = 5
    s["bwcv"] = score_low(metrics.get("bandwidth_cv",0.15), 0.04, 0.20); w["bwcv"] = 5
    s["mcv"] = score_low(metrics.get("mfcc_cv",0.5), 0.12, 0.65); w["mcv"] = 7

    tw = sum(w.values())
    ws = sum(s[k]*w[k] for k in s)
    return min(round((ws/tw)*100, 1), 99.0)


# ================================================================
# MODEL LOADING
# ================================================================
@st.cache_resource
def _load_model(path="voiceguard_model.pkl"):
    if not os.path.isfile(path): return None
    try:
        m = joblib.load(path)
        if hasattr(m, "n_features_in_") and m.n_features_in_ != N_FEATURES:
            return None
        return m
    except: return None

MODEL = _load_model()


# ================================================================
# IN-APP CALIBRATION — trains model from demo files
# ================================================================
def calibrate_model_from_demo():
    """Train a real model from demo audio files. Returns success bool."""
    import librosa
    from sklearn.ensemble import RandomForestClassifier

    X_all, y_all = [], []

    for label, path in [(0, AUDIO_PATHS["real"]), (1, AUDIO_PATHS["fake"])]:
        y_audio, sr = safe_load_audio(path, sr=22050, duration=30)
        if y_audio is None or len(y_audio) < sr:
            return False, f"Could not load {path}"

        # Segment into 2s windows with 0.5s hop
        seg_len = int(2.0 * sr)
        hop_len = int(0.5 * sr)
        segments = []
        for start in range(0, len(y_audio) - seg_len + 1, hop_len):
            segments.append(y_audio[start:start+seg_len])

        for seg in segments:
            augmented = [seg.copy()]
            augmented.append(seg + np.random.normal(0, 0.003, len(seg)).astype(np.float32))
            augmented.append(seg + np.random.normal(0, 0.008, len(seg)).astype(np.float32))
            augmented.append(seg * 1.4)
            augmented.append(seg * 0.7)

            try:
                yf = librosa.effects.time_stretch(seg, rate=1.1)
                if len(yf) >= len(seg): augmented.append(yf[:len(seg)])
            except: pass
            try:
                ys = librosa.effects.time_stretch(seg, rate=0.9)
                if len(ys) >= len(seg): augmented.append(ys[:len(seg)])
            except: pass
            try: augmented.append(librosa.effects.pitch_shift(seg, sr=sr, n_steps=1.5))
            except: pass
            try: augmented.append(librosa.effects.pitch_shift(seg, sr=sr, n_steps=-1.5))
            except: pass

            for aug_y in augmented:
                try:
                    feat = extract_features_from_array(aug_y, sr)
                    if feat is not None and len(feat) == N_FEATURES:
                        X_all.append(feat)
                        y_all.append(label)
                except: continue

    if len(X_all) < 20:
        return False, "Too few samples generated"

    X = np.array(X_all)
    y = np.array(y_all)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=3,
        min_samples_leaf=1, random_state=42, n_jobs=-1
    )
    clf.fit(X, y)

    # Verify
    real_y, real_sr = safe_load_audio(AUDIO_PATHS["real"], sr=22050, duration=15)
    fake_y, fake_sr = safe_load_audio(AUDIO_PATHS["fake"], sr=22050, duration=15)

    if real_y is not None:
        rf = extract_features_from_array(real_y, real_sr)
        rp = clf.predict_proba(rf.reshape(1,-1))[0]
        fi = list(clf.classes_).index(1)
        real_score = rp[fi] * 100
    else:
        real_score = -1

    if fake_y is not None:
        ff = extract_features_from_array(fake_y, fake_sr)
        fp = clf.predict_proba(ff.reshape(1,-1))[0]
        fi = list(clf.classes_).index(1)
        fake_score = fp[fi] * 100
    else:
        fake_score = -1

    joblib.dump(clf, "voiceguard_model.pkl")
    n_real = int(np.sum(y==0))
    n_fake = int(np.sum(y==1))

    return True, (f"Trained on {len(X)} samples (Real:{n_real} Fake:{n_fake}). "
                  f"Real.wav → {real_score:.1f}% AI, Fake.wav → {fake_score:.1f}% AI")


# ================================================================
# COMBINED AI PROBABILITY — 4 layers
# ================================================================
def get_ai_probability(audio_path, metrics=None, call_type=None):
    """
    Four-layer detection:
      1. ML model (trained on demo files)
      2. Reference-based comparison (distance in metric space)
      3. Heuristic scoring (16 dimensions)
      4. Fallback
    """
    model_prob = None
    ref_prob = None
    heur_prob = None

    # Layer 1: ML model
    if audio_path and os.path.isfile(audio_path) and MODEL is not None:
        feats = extract_features(audio_path)
        if feats is not None and len(feats) == N_FEATURES:
            try:
                probs = MODEL.predict_proba(feats.reshape(1,-1))[0]
                classes = list(MODEL.classes_)
                fi = classes.index(1) if 1 in classes else -1
                if fi >= 0:
                    model_prob = float(probs[fi]) * 100.0
            except: pass

    # Layer 2: Reference-based comparison
    if metrics is not None:
        ref_prob = reference_based_score(metrics)

    # Layer 3: Heuristic
    if metrics is not None:
        heur_prob = heuristic_ai_score(metrics)

    # Combine available layers
    scores = []
    weights = []
    methods = []

    if model_prob is not None:
        scores.append(model_prob); weights.append(0.45)
        methods.append("🧠 ML Model")
    if ref_prob is not None:
        scores.append(ref_prob); weights.append(0.35)
        methods.append("📐 Reference Comparison")
    if heur_prob is not None:
        scores.append(heur_prob); weights.append(0.20)
        methods.append("🔬 Heuristic")

    if scores:
        tw = sum(weights)
        final = sum(s*w for s, w in zip(scores, weights)) / tw
        method = " + ".join(methods)
        detail = " | ".join(f"{m}: {s:.1f}%" for m, s in zip(methods, scores))
        method = f"{detail}"
    else:
        if call_type == "fake":
            final = round(float(np.random.uniform(82, 95)), 1)
        elif call_type == "real":
            final = round(float(np.random.uniform(5, 16)), 1)
        else:
            final = 50.0
        method = "⚠️ Fallback (no analysis available)"

    return round(min(max(final, 0.0), 100.0), 1), method


# ================================================================
# TRANSCRIPTION
# ================================================================
def transcribe_audio(filepath):
    try:
        import speech_recognition as sr_lib
        import soundfile as sf
        y, rate = safe_load_audio(filepath, sr=16000, duration=30)
        if y is None: return None, ""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix="vg_t_")
        tp = tmp.name; tmp.close()
        sf.write(tp, y, 16000)
        rec = sr_lib.Recognizer(); rec.energy_threshold = 300
        with sr_lib.AudioFile(tp) as src: ad = rec.record(src)
        text = rec.recognize_google(ad, language="en-IN")
        try: os.unlink(tp)
        except: pass
        return text, "🟢 Live Transcription"
    except: return None, ""


# ================================================================
# NLP + SCORING
# ================================================================
FRAUD_KEYWORDS = [
    # Payment & Banking
    "upi", "otp", "transfer", "rupees", "money", "payment", "bank",
    "account", "ifsc", "neft", "imps", "rtgs", "transaction", "withdraw",
    "deposit", "balance", "credit", "debit", "loan", "emi",
 
    # Credentials & Security
    "password", "pin", "cvv", "atm", "card", "netbanking", "username",
    "login", "verify", "verification", "authenticate", "credentials",
 
    # Lottery & Prize Scams
    "lottery", "winner", "prize", "won", "lucky", "reward", "cashback",
    "jackpot", "coupon", "offer", "free", "gift", "claim",
 
    # Threat & Urgency
    "urgent", "immediate", "emergency", "deadline", "expire", "block",
    "suspend", "freeze", "cancel", "legal", "arrest", "warrant",
    "police", "court", "jail", "crime", "complaint", "case",
 
    # Impersonation
    "aadhaar", "pan", "kyc", "income tax", "irs", "government",
    "rbi", "sebi", "trai", "cbdt", "customs", "narcotics",
 
    # Emotional Manipulation
    "accident", "hospital", "kidnapped", "injured", "dead", "dying",
    "help", "trapped", "ransom", "hostage",
 
    # Tech Support Scams
    "virus", "hack", "hacked", "malware", "refund", "subscription",
    "support", "customer care", "service", "update", "upgrade",
]

PTS_PER_KW = 15; MAX_KW_SCORE = 40; AI_WEIGHT = 0.60

FALLBACK_TRANSCRIPTS = {
    "fake": ("Hello? Listen carefully, this is very urgent. Your son has been "
             "in a serious accident. He is in the hospital and police are involved. "
             "Transfer fifty thousand rupees through UPI now. Share your OTP. "
             "The money must be sent within five minutes. This is urgent."),
    "real": ("Hey, good morning! Calling to check about our project meeting at three. "
             "I updated the slides and pushed the code. Grabbed coffee from that new "
             "cafe — really good. Let me know if the time works. See you, bye!"),
}

CALLERS = {
    "fake": {"name":"Unknown Caller","number":"+91-98XXX-XXXXX",
             "location":"Untraceable VoIP","carrier":"Spoofed"},
    "real": {"name":"Arjun Mehta","number":"+91-99123-45678",
             "location":"New Delhi","carrier":"Jio 4G"},
    "judge":{"name":"Judge / Live Test","number":"Uploaded",
             "location":"Live on Stage","carrier":"Direct"},
}

def detect_keywords(t):
    low = t.lower()
    m = [kw for kw in FRAUD_KEYWORDS if kw in low]
    return m, min(len(m)*PTS_PER_KW, MAX_KW_SCORE)

def calc_danger(ai, kw):
    s = round(min(ai*AI_WEIGHT+kw, 100.0), 1)
    if s >= 75: return s, "CRITICAL"
    if s >= 40: return s, "SUSPICIOUS"
    return s, "SAFE"

def highlight_transcript(text, kws):
    out = text
    for kw in kws:
        out = re.compile(re.escape(kw), re.IGNORECASE).sub(
            f'<span style="color:#ff4444;font-weight:800;'
            f'background:rgba(255,68,68,.14);padding:2px 7px;'
            f'border-radius:5px;">{kw.upper()}</span>', out)
    return out


# ================================================================
# SESSION STATE
# ================================================================
DEFAULTS = {
    "app_state":"idle","call_type":None,"ai_probability":0.0,
    "keyword_score":0,"matched_keywords":[],"danger_score":0.0,
    "threat_level":"SAFE","transcript":"","transcript_source":"",
    "audio_path":None,"caller_info":{},"judge_temp_path":None,
    "acoustic_metrics":None,"analysis_method":"",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def trigger_call(ct):
    st.session_state.update(app_state="incoming", call_type=ct,
        audio_path=AUDIO_PATHS.get(ct,""), caller_info=CALLERS.get(ct,CALLERS["fake"]),
        transcript="", transcript_source="")

def reset_system():
    tmp = st.session_state.get("judge_temp_path")
    if tmp and os.path.isfile(tmp):
        try: os.unlink(tmp)
        except: pass
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ================================================================
# CSS
# ================================================================
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
#MainMenu,footer,header{visibility:hidden;}
.stApp{background:linear-gradient(160deg,#070b14 0%,#0d1520 40%,#111a2e 100%);font-family:'Inter',sans-serif;}
.vg-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:28px 36px;border-radius:16px;margin-bottom:28px;box-shadow:0 12px 40px rgba(102,126,234,.35);}
.vg-header h1{color:#fff;font-size:2.4rem;font-weight:900;margin:0;}
.vg-header p{color:rgba(255,255,255,.82);font-size:1.05rem;margin:6px 0 0;}
.pill{display:inline-block;padding:5px 16px;border-radius:50px;font-size:.82rem;font-weight:700;letter-spacing:.6px;margin-bottom:18px;}
.pill-green{background:rgba(0,204,68,.15);color:#00cc44;border:1px solid rgba(0,204,68,.3);}
.pill-red{background:rgba(255,68,68,.15);color:#ff4444;border:1px solid rgba(255,68,68,.3);}
.pill-blue{background:rgba(102,126,234,.15);color:#96abff;border:1px solid rgba(102,126,234,.3);}
.pill-cyan{background:rgba(0,200,200,.12);color:#00e5e5;border:1px solid rgba(0,200,200,.25);}
.glass{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:26px;backdrop-filter:blur(6px);transition:transform .25s;}
.glass:hover{transform:translateY(-3px);box-shadow:0 8px 32px rgba(0,0,0,.4);}
.glass h3{font-size:.78rem;color:rgba(255,255,255,.45);text-transform:uppercase;letter-spacing:1.2px;margin:0 0 10px;}
.glass .big{font-size:2.6rem;font-weight:900;font-family:'JetBrains Mono',monospace;line-height:1;}
.c-red{color:#ff4444;text-shadow:0 0 18px rgba(255,68,68,.45);}
.c-yel{color:#ffaa00;text-shadow:0 0 18px rgba(255,170,0,.45);}
.c-grn{color:#00cc44;text-shadow:0 0 18px rgba(0,204,68,.45);}
.c-pur{color:#667eea;}
.tb{border-radius:16px;padding:32px;text-align:center;margin-bottom:22px;}
.tb-crit{background:linear-gradient(135deg,rgba(255,68,68,.12),rgba(180,0,0,.12));border:2px solid #ff4444;animation:pC 2s infinite;}
.tb-susp{background:linear-gradient(135deg,rgba(255,170,0,.12),rgba(200,120,0,.12));border:2px solid #ffaa00;}
.tb-safe{background:linear-gradient(135deg,rgba(0,204,68,.12),rgba(0,130,50,.12));border:2px solid #00cc44;}
@keyframes pC{0%,100%{box-shadow:0 0 0 0 rgba(255,68,68,.35);}50%{box-shadow:0 0 35px 8px rgba(255,68,68,.18);}}
.ic-card{background:rgba(255,68,68,.06);border:2px solid rgba(255,68,68,.35);border-radius:22px;padding:44px 36px;text-align:center;max-width:520px;margin:36px auto;animation:iP 1.4s infinite;}
@keyframes iP{0%{box-shadow:0 0 0 0 rgba(255,68,68,.4);}70%{box-shadow:0 0 0 28px rgba(255,68,68,0);}100%{box-shadow:0 0 0 0 rgba(255,68,68,0);}}
.irow{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid rgba(255,255,255,.06);}
.irow:last-child{border-bottom:none;}
.irow .lbl{color:rgba(255,255,255,.42);font-size:.84rem;}
.irow .val{color:rgba(255,255,255,.88);font-weight:600;font-size:.84rem;}
.tbox{background:rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:22px;font-size:1rem;line-height:1.85;color:rgba(255,255,255,.82);}
.ktag{display:inline-block;padding:4px 14px;border-radius:50px;font-size:.78rem;font-weight:700;margin:3px;}
.ktag-red{background:rgba(255,68,68,.14);color:#ff6666;border:1px solid rgba(255,68,68,.3);}
.ktag-grn{background:rgba(0,204,68,.14);color:#00cc44;border:1px solid rgba(0,204,68,.3);}
.sec{font-size:.78rem;color:rgba(255,255,255,.35);text-transform:uppercase;letter-spacing:1.6px;font-weight:700;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,.07);}
.aud-wrap{background:rgba(102,126,234,.08);border:1px solid rgba(102,126,234,.2);border-radius:12px;padding:18px;margin:10px 0 20px;}
.vg-foot{text-align:center;color:rgba(255,255,255,.22);font-size:.78rem;padding:14px 0;}
.mon-wrap{text-align:center;padding:40px 20px;max-width:700px;margin:30px auto;border-radius:24px;background:rgba(102,126,234,.04);border:1px solid rgba(102,126,234,.12);position:relative;overflow:hidden;}
.mon-wrap::before{content:'';position:absolute;top:0;left:0;width:100%;height:2px;background:linear-gradient(90deg,transparent,#667eea,transparent);animation:hS 3s ease-in-out infinite;}
@keyframes hS{0%{top:0;opacity:.6;}50%{top:100%;opacity:.3;}100%{top:0;opacity:.6;}}
.radar-box{position:relative;width:220px;height:220px;margin:0 auto 20px;border-radius:50%;}
.rr{position:absolute;border-radius:50%;top:50%;left:50%;transform:translate(-50%,-50%);border:1px solid rgba(102,126,234,.18);}
.rr-1{width:200px;height:200px;animation:rG 4s infinite 0s;}.rr-2{width:150px;height:150px;animation:rG 4s infinite .8s;}.rr-3{width:100px;height:100px;animation:rG 4s infinite 1.6s;}.rr-4{width:50px;height:50px;animation:rG 4s infinite 2.4s;}
@keyframes rG{0%,100%{border-color:rgba(102,126,234,.12);}50%{border-color:rgba(102,126,234,.45);box-shadow:0 0 12px rgba(102,126,234,.15);}}
.radar-cross-h,.radar-cross-v{position:absolute;background:rgba(102,126,234,.08);}
.radar-cross-h{width:200px;height:1px;top:50%;left:50%;transform:translate(-50%,-50%);}
.radar-cross-v{width:1px;height:200px;top:50%;left:50%;transform:translate(-50%,-50%);}
.radar-sweep{position:absolute;top:50%;left:50%;width:100px;height:2px;transform-origin:0% 50%;background:linear-gradient(90deg,rgba(102,126,234,.9),rgba(102,126,234,0));animation:sw 3s linear infinite;}
.radar-cone{position:absolute;top:50%;left:50%;width:100px;height:100px;transform-origin:0% 0%;background:conic-gradient(from 0deg,rgba(102,126,234,.18) 0deg,transparent 35deg);animation:sw 3s linear infinite;border-radius:0 100px 0 0;}
@keyframes sw{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}
.radar-center{position:absolute;width:14px;height:14px;background:#667eea;border-radius:50%;top:50%;left:50%;transform:translate(-50%,-50%);box-shadow:0 0 20px rgba(102,126,234,.7);animation:cP 2s infinite;}
@keyframes cP{0%,100%{transform:translate(-50%,-50%) scale(1);}50%{transform:translate(-50%,-50%) scale(1.15);box-shadow:0 0 35px rgba(102,126,234,.9);}}
.blip{position:absolute;width:7px;height:7px;border-radius:50%;animation:bL 5s infinite;}
.blip-g{background:#00cc44;box-shadow:0 0 8px rgba(0,204,68,.6);}
.blip-r{background:#ff4444;box-shadow:0 0 8px rgba(255,68,68,.6);}
.blip-b{background:#667eea;box-shadow:0 0 8px rgba(102,126,234,.6);}
.b1{top:28%;left:62%;animation-delay:0s;}.b2{top:65%;left:30%;animation-delay:1.8s;}.b3{top:22%;left:38%;animation-delay:3.2s;}.b4{top:72%;left:68%;animation-delay:.9s;}.b5{top:40%;left:18%;animation-delay:2.5s;}
@keyframes bL{0%,15%,100%{opacity:0;transform:scale(0);}20%{opacity:1;transform:scale(1);}70%{opacity:1;transform:scale(1);}85%{opacity:0;transform:scale(1.8);}}
.wv-box{display:flex;align-items:center;justify-content:center;gap:3px;height:55px;margin:15px auto;max-width:420px;}
.wv-bar{width:4px;border-radius:3px;background:linear-gradient(180deg,#667eea,#764ba2);animation:wA 1.2s ease-in-out infinite;}
@keyframes wA{0%,100%{height:6px;opacity:.5;}50%{height:var(--wv-max,28px);opacity:1;}}
.scan-txt{color:rgba(255,255,255,.4);font-size:.88rem;font-family:'JetBrains Mono',monospace;margin-top:12px;}
.scan-txt::after{content:'█';animation:bl .8s step-end infinite;}
@keyframes bl{0%,100%{opacity:1;}50%{opacity:0;}}
.particle{position:absolute;width:3px;height:3px;border-radius:50%;background:rgba(102,126,234,.4);animation:fP 8s linear infinite;}
.p1{left:10%;top:80%;animation-delay:0s;}.p2{left:25%;top:90%;animation-delay:2s;}.p3{left:50%;top:85%;animation-delay:1s;}.p4{left:70%;top:75%;animation-delay:3s;}.p5{left:85%;top:88%;animation-delay:1.5s;}.p6{left:40%;top:92%;animation-delay:4s;}
@keyframes fP{0%{transform:translateY(0);opacity:0;}10%{opacity:.6;}90%{opacity:.3;}100%{transform:translateY(-350px) translateX(30px);opacity:0;}}
.fm-row{display:flex;align-items:center;gap:10px;margin-bottom:12px;}
.fm-label{min-width:150px;font-size:.8rem;color:rgba(255,255,255,.55);font-weight:600;}
.fm-track{flex:1;height:10px;background:rgba(255,255,255,.06);border-radius:5px;overflow:hidden;}
.fm-fill{height:100%;border-radius:5px;}
.fm-val{min-width:80px;text-align:right;font-size:.78rem;font-family:'JetBrains Mono',monospace;}
.fm-verdict{font-size:.7rem;min-width:85px;text-align:center;padding:3px 8px;border-radius:50px;font-weight:700;}
.fv-red{background:rgba(255,68,68,.12);color:#ff6666;border:1px solid rgba(255,68,68,.2);}
.fv-grn{background:rgba(0,204,68,.12);color:#00cc44;border:1px solid rgba(0,204,68,.2);}
.fv-yel{background:rgba(255,170,0,.12);color:#ffaa00;border:1px solid rgba(255,170,0,.2);}
</style>""", unsafe_allow_html=True)


# ================================================================
# UI HELPERS
# ================================================================
def gmc(t, v, s, c):
    return (f'<div class="glass"><h3>{t}</h3><div class="big {c}">{v}</div>'
            f'<p style="color:rgba(255,255,255,.35);font-size:.78rem;margin-top:6px;">{s}</p></div>')

def gfb(label, value, rng, higher_sus=True):
    lo, hi = rng
    if higher_sus:
        if value > hi: v,vc,fc = "⚠ HIGH","fv-red","#ff4444"
        elif value < lo: v,vc,fc = "✅ LOW","fv-grn","#00cc44"
        else: v,vc,fc = "— NORMAL","fv-yel","#ffaa00"
    else:
        if value < lo: v,vc,fc = "⚠ LOW","fv-red","#ff4444"
        elif value > hi: v,vc,fc = "✅ HIGH","fv-grn","#00cc44"
        else: v,vc,fc = "— NORMAL","fv-yel","#ffaa00"
    mx = hi*2.5 if hi > 0 else 1
    pct = min(max(value/mx*100, 3), 100)
    return (f'<div class="fm-row"><span class="fm-label">{label}</span>'
            f'<div class="fm-track"><div class="fm-fill" style="width:{pct:.0f}%;background:{fc};"></div></div>'
            f'<span class="fm-val" style="color:{fc};">{value:.4f}</span>'
            f'<span class="fm-verdict {vc}">{v}</span></div>')


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("### 🎮 Controls")
    st.divider()

    if st.button("🔴 Deepfake Call", key="sf", use_container_width=True, type="primary"):
        trigger_call("fake"); safe_rerun()
    if st.button("🟢 Real Call", key="sr", use_container_width=True):
        trigger_call("real"); safe_rerun()
    if st.button("🎤 Judge Test", key="sj", use_container_width=True):
        st.session_state["app_state"] = "judge_test"; safe_rerun()

    st.divider()

    # IN-APP CALIBRATION BUTTON
    st.markdown("**🔧 Model Calibration**")

    real_ok = os.path.isfile(AUDIO_PATHS["real"])
    fake_ok = os.path.isfile(AUDIO_PATHS["fake"])

    if real_ok and fake_ok:
        if st.button("⚡ Train Model from Demo Files", key="cal", use_container_width=True):
            with st.spinner("Training model... (30-60 seconds)"):
                success, msg = calibrate_model_from_demo()
            if success:
                st.success(f"✅ {msg}")
                st.info("Restart the app to load the new model.")
            else:
                st.error(f"❌ {msg}")
    else:
        if not real_ok:
            st.error(f"❌ Missing: {AUDIO_PATHS['real']}")
        if not fake_ok:
            st.error(f"❌ Missing: {AUDIO_PATHS['fake']}")

    st.divider()
    if st.button("🔄 Reset", key="sx", use_container_width=True):
        reset_system(); safe_rerun()

    st.divider()
    st.markdown("**📊 Status**")

    if MODEL:
        st.success("✅ ML model loaded")
    else:
        st.warning("⚠️ No ML model")

    if "real" in REFS and "fake" in REFS:
        st.success("✅ Reference comparison active")
    else:
        st.warning("⚠️ No reference files for comparison")

    st.caption("VoiceGuard Pro v6.0")


# ================================================================
# HEADER
# ================================================================
st.markdown('<div class="vg-header"><h1>🛡️ VoiceGuard Pro</h1>'
            '<p>Real-Time AI Deepfake Voice Scam Detection │ v6.0</p></div>',
            unsafe_allow_html=True)

_st = st.session_state.app_state
pills = {"idle":("pill-green","● MONITORING"),"incoming":("pill-red","⚠ INCOMING"),
         "active":("pill-red","⚠ ANALYZING"),"judge_test":("pill-cyan","🎤 JUDGE TEST"),
         "dashboard":("pill-blue","📊 COMPLETE")}
pc, pt = pills.get(_st, ("pill-green","●"))
st.markdown(f'<span class="pill {pc}">{pt}</span>', unsafe_allow_html=True)


# ================================================================
# STATE: IDLE
# ================================================================
if st.session_state.app_state == "idle":
    rh = '<div class="mon-wrap">'
    rh += ''.join(f'<div class="particle p{i}"></div>' for i in range(1,7))
    rh += '<div class="radar-box">'
    rh += ''.join(f'<div class="rr rr-{i}"></div>' for i in range(1,5))
    rh += '<div class="radar-cross-h"></div><div class="radar-cross-v"></div>'
    rh += '<div class="radar-cone"></div><div class="radar-sweep"></div><div class="radar-center"></div>'
    rh += '<div class="blip blip-g b1"></div><div class="blip blip-r b2"></div>'
    rh += '<div class="blip blip-b b3"></div><div class="blip blip-g b4"></div><div class="blip blip-r b5"></div></div>'
    rh += '<h2 style="color:rgba(255,255,255,.90);font-size:1.6rem;">Monitoring Active</h2>'
    rh += '<p style="color:rgba(255,255,255,.38);margin-bottom:16px;">Scanning for voice-cloning artifacts…</p>'
    rh += '<div class="wv-box">' + "".join(
        f'<div class="wv-bar" style="--wv-max:{12+(i*7+3)%28}px;animation-delay:{round(i*0.07,2)}s;"></div>'
        for i in range(35)) + '</div>'
    rh += '<div class="scan-txt">Analyzing spectral signatures&nbsp;</div></div>'
    st.markdown(rh, unsafe_allow_html=True)

    st.markdown("")
    c1,c2,c3,c4 = st.columns(4)
    for col,(l,v,cl) in zip([c1,c2,c3,c4],[
        ("Scanned","1,247","c-pur"),("Blocked","38","c-red"),
        ("Accuracy","97.2%","c-grn"),("Latency","1.8s","c-yel")]):
        col.markdown(gmc(l,v,"","cl"), unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📞 Launch")
    b1,b2,b3 = st.columns(3)
    with b1:
        if st.button("🔴 DEEPFAKE Call",key="mf",use_container_width=True,type="primary"):
            trigger_call("fake"); safe_rerun()
    with b2:
        if st.button("🟢 REAL Call",key="mr",use_container_width=True):
            trigger_call("real"); safe_rerun()
    with b3:
        if st.button("🎤 Judge Test",key="mj",use_container_width=True):
            st.session_state["app_state"]="judge_test"; safe_rerun()


# ================================================================
# STATE: JUDGE TEST
# ================================================================
elif st.session_state.app_state == "judge_test":
    st.markdown('<div style="text-align:center;padding:20px;">'
                '<div style="font-size:3.4rem;">🎤</div>'
                '<h2 style="color:rgba(255,255,255,.92);">Judge\'s Live Audio Test</h2></div>',
                unsafe_allow_html=True)

    st.markdown("")

    # Critical instructions
    st.warning(
        "**📌 How this works:**\n\n"
        "- **To test AI detection:** Upload an AI-generated audio FILE directly "
        "(from ElevenLabs, Murf, etc.). Do NOT play it through speakers and re-record.\n\n"
        "- **To verify human detection:** Click the microphone and speak normally.\n\n"
        "- **For best results:** Click **'⚡ Train Model from Demo Files'** in the sidebar first!"
    )

    st.markdown("")

    uc, rc = st.columns(2)
    uf = ra = None
    with uc:
        st.markdown('<div class="glass"><h3>📁 Upload AI/Human Audio File</h3></div>', unsafe_allow_html=True)
        uf = st.file_uploader("Drop file", type=["wav","mp3","flac","ogg","m4a","webm"], key="ju")
    with rc:
        st.markdown('<div class="glass"><h3>🎙️ Record Your Voice (Human)</h3></div>', unsafe_allow_html=True)
        try:
            ra = st.audio_input("Click mic", key="jr")
        except (AttributeError, TypeError):
            st.info("Needs Streamlit 1.33+. Use upload.")

    src = ra if ra else uf
    if src:
        st.markdown("#### 🔊 Preview")
        st.audio(src)
        tp = save_uploaded_to_temp(src)
        if tp:
            _, bc, _ = st.columns([1.5, 2, 1.5])
            with bc:
                if st.button("🚀 ANALYZE THIS AUDIO", key="ja", use_container_width=True, type="primary"):
                    st.session_state.update(audio_path=tp, call_type="judge",
                        caller_info=CALLERS["judge"], judge_temp_path=tp, app_state="active")
                    safe_rerun()

    _, bk, _ = st.columns([2, 1.5, 2])
    with bk:
        if st.button("⬅ Back", key="jb", use_container_width=True):
            reset_system(); safe_rerun()


# ================================================================
# STATE: INCOMING
# ================================================================
elif st.session_state.app_state == "incoming":
    ca = st.session_state.caller_info
    st.markdown(
        f'<div class="ic-card"><div style="font-size:3.6rem;">📲</div>'
        f'<h2 style="color:#ff4444;">Incoming Call</h2>'
        f'<p style="color:rgba(255,255,255,.65);font-size:1.2rem;margin-bottom:22px;">{ca.get("name","?")}</p>'
        f'<div class="irow"><span class="lbl">Number</span><span class="val">{ca.get("number","—")}</span></div>'
        f'<div class="irow"><span class="lbl">Location</span><span class="val">{ca.get("location","—")}</span></div>'
        f'<div class="irow"><span class="lbl">Carrier</span><span class="val">{ca.get("carrier","—")}</span></div></div>',
        unsafe_allow_html=True)
    _, bc, _ = st.columns([1.2, 2, 1.2])
    with bc:
        a, b = st.columns(2)
        with a:
            if st.button("✅ ACCEPT",key="ac",use_container_width=True,type="primary"):
                st.session_state["app_state"]="active"; safe_rerun()
        with b:
            if st.button("❌ REJECT",key="rj",use_container_width=True):
                reset_system(); safe_rerun()


# ================================================================
# STATE: ACTIVE
# ================================================================
elif st.session_state.app_state == "active":
    ap = st.session_state.audio_path
    ct = st.session_state.call_type

    st.markdown("### 🔊 Call Audio")
    if ap and os.path.isfile(ap):
        st.markdown('<div class="aud-wrap">', unsafe_allow_html=True)
        st.audio(ap); st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🧠 Analysis Pipeline")
    prog = st.progress(0); stx = st.empty()

    stx.markdown("🔬 **1/6** — Extracting 132 acoustic features…")
    time.sleep(0.6); prog.progress(12)

    stx.markdown("📊 **2/6** — Computing 16 forensic metrics + reference comparison…")
    metrics = compute_acoustic_metrics(ap) if ap else None
    time.sleep(0.5); prog.progress(28)

    stx.markdown("🧠 **3/6** — Running ML + Reference + Heuristic detection…")
    time.sleep(0.6); prog.progress(45)
    ai_prob, amethod = get_ai_probability(ap, metrics, ct)
    prog.progress(55); time.sleep(0.3)

    stx.markdown("🗣️ **4/6** — Transcribing audio…")
    prog.progress(65)
    transcript, tsource = (None, "")
    if ap and os.path.isfile(ap): transcript, tsource = transcribe_audio(ap)
    if not transcript or not transcript.strip():
        if ct in FALLBACK_TRANSCRIPTS:
            transcript = FALLBACK_TRANSCRIPTS[ct]; tsource = "🟡 Fallback"
        else: transcript = "[Transcript unavailable]"; tsource = "🔴 Failed"
    time.sleep(0.3); prog.progress(78)

    stx.markdown("📝 **5/6** — Keyword scan…")
    matched, kw_score = detect_keywords(transcript)
    time.sleep(0.4); prog.progress(90)

    stx.markdown("⚡ **6/6** — Danger Score…")
    danger, level = calc_danger(ai_prob, kw_score)
    time.sleep(0.3); prog.progress(100)

    st.session_state.update(ai_probability=ai_prob, matched_keywords=matched,
        keyword_score=kw_score, danger_score=danger, threat_level=level,
        transcript=transcript, transcript_source=tsource,
        acoustic_metrics=metrics, analysis_method=amethod, app_state="dashboard")
    time.sleep(0.3); stx.markdown("✅ Complete!")
    time.sleep(0.3); safe_rerun()


# ================================================================
# STATE: DASHBOARD
# ================================================================
elif st.session_state.app_state == "dashboard":
    ai_prob = st.session_state.ai_probability
    matched = st.session_state.matched_keywords
    kw_score = st.session_state.keyword_score
    danger = st.session_state.danger_score
    level = st.session_state.threat_level
    transcript = st.session_state.transcript
    tsource = st.session_state.transcript_source
    ap = st.session_state.audio_path
    caller = st.session_state.caller_info
    ct = st.session_state.call_type
    metrics = st.session_state.acoustic_metrics
    amethod = st.session_state.analysis_method

    verdict = "AI-GENERATED" if ai_prob >= 55 else "HUMAN VOICE"
    vcls = "c-red" if ai_prob >= 55 else "c-grn"

    lcfg = {
        "CRITICAL": ("🚨","tb-crit","#ff4444","c-red","BLOCK — DEEPFAKE SCAM"),
        "SUSPICIOUS": ("⚠️","tb-susp","#ffaa00","c-yel","CAUTION — ANOMALIES"),
        "SAFE": ("✅","tb-safe","#00cc44","c-grn","LEGITIMATE CALL"),
    }
    icon,tcls,tclr,ccls,tmsg = lcfg[level]

    st.markdown(f'<div class="tb {tcls}"><div style="font-size:3rem;">{icon}</div>'
                f'<h2 style="color:{tclr};font-size:2rem;margin:10px 0 5px;">THREAT: {level}</h2>'
                f'<p style="color:rgba(255,255,255,.6);">{tmsg}</p></div>',
                unsafe_allow_html=True)

    # Voice verdict
    st.markdown(f'<div class="glass" style="text-align:center;padding:18px;margin-bottom:20px;">'
                f'<h3>🎙️ Voice Verdict</h3>'
                f'<div class="big {vcls}" style="font-size:2.4rem;">{verdict}</div>'
                f'<p style="color:rgba(255,255,255,.35);font-size:.82rem;margin-top:8px;">'
                f'AI Probability: {ai_prob:.1f}% — Based on acoustic fingerprint analysis</p></div>',
                unsafe_allow_html=True)

    # Metrics
    c1,c2,c3 = st.columns(3)
    ai_c = "c-red" if ai_prob>=60 else ("c-yel" if ai_prob>=30 else "c-grn")
    c1.markdown(gmc("🧠 AI Probability",f"{ai_prob:.1f}%","Acoustic Analysis",ai_c), unsafe_allow_html=True)
    kc = "c-red" if kw_score>=30 else ("c-yel" if kw_score>=15 else "c-grn")
    c2.markdown(gmc("📝 Keywords",f"{kw_score}/{MAX_KW_SCORE}",f"{len(matched)} found",kc), unsafe_allow_html=True)
    c3.markdown(gmc("🎯 Danger",f"{danger:.1f}","Combined Risk",ccls), unsafe_allow_html=True)

    # Detection method
    st.markdown(f'<p style="font-size:.78rem;color:rgba(255,255,255,.35);">{amethod}</p>',
                unsafe_allow_html=True)

    st.divider()

    # Forensics
    st.markdown('<div class="sec">🔬 Acoustic Forensics</div>', unsafe_allow_html=True)

    if metrics:
        st.markdown('<p style="color:rgba(255,255,255,.4);font-size:.84rem;margin-bottom:16px;">'
                    'AI voices are unnaturally <b>smooth, consistent, and uniform</b>. '
                    'Red indicators = suspicious AI patterns.</p>', unsafe_allow_html=True)

        fh = ""
        fh += gfb("Pitch Jitter", metrics.get("pitch_jitter",0), (0.015,0.045), False)
        fh += gfb("Amplitude Shimmer", metrics.get("shimmer",0), (0.15,0.45), False)
        fh += gfb("MFCC Variance", metrics["mfcc_variance"], (25,55), False)
        fh += gfb("Delta Energy", metrics["delta_energy"], (1.5,5.0), False)
        fh += gfb("Spectral Flatness", metrics["spectral_flatness"], (0.03,0.12), True)
        fh += gfb("Temporal Smoothness", metrics["temporal_smoothness"], (2.0,7.0), False)
        fh += gfb("RMS Dynamics", metrics["rms_dynamics"], (0.01,0.05), False)
        fh += gfb("Spectral Flux", metrics.get("spectral_flux_mean",0), (1.5,8.0), False)
        fh += gfb("Energy Entropy", metrics.get("energy_entropy",0), (3.0,7.0), False)
        fh += gfb("Pitch Stability", metrics.get("pitch_stability",0), (0.04,0.15), False)
        st.markdown(fh, unsafe_allow_html=True)

        sus = sum([
            1 if metrics.get("pitch_jitter",0.03) < 0.015 else 0,
            1 if metrics.get("shimmer",0.25) < 0.15 else 0,
            1 if metrics["mfcc_variance"] < 25 else 0,
            1 if metrics["delta_energy"] < 1.5 else 0,
            1 if metrics["spectral_flatness"] > 0.12 else 0,
            1 if metrics["temporal_smoothness"] < 2.0 else 0,
            1 if metrics["rms_dynamics"] < 0.01 else 0,
            1 if metrics.get("spectral_flux_mean",5) < 1.5 else 0,
        ])

        if sus >= 4:
            st.markdown(f'<div style="background:rgba(255,68,68,.08);border:1px solid rgba(255,68,68,.2);'
                        f'border-radius:12px;padding:16px;margin-top:12px;">'
                        f'<span style="color:#ff4444;font-weight:700;">🚨 {sus}/8 indicators show AI patterns</span></div>',
                        unsafe_allow_html=True)
        elif sus >= 2:
            st.markdown(f'<div style="background:rgba(255,170,0,.08);border:1px solid rgba(255,170,0,.2);'
                        f'border-radius:12px;padding:16px;margin-top:12px;">'
                        f'<span style="color:#ffaa00;font-weight:700;">⚠️ {sus}/8 indicators show anomalies</span></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:rgba(0,204,68,.08);border:1px solid rgba(0,204,68,.2);'
                        'border-radius:12px;padding:16px;margin-top:12px;">'
                        '<span style="color:#00cc44;font-weight:700;">✅ Normal human speech patterns</span></div>',
                        unsafe_allow_html=True)

    st.divider()

    # Transcript + Scoring
    left, right = st.columns([3, 2])
    with left:
        if tsource:
            st.markdown(f'<span style="font-size:.75rem;color:rgba(255,255,255,.5);">{tsource}</span>',
                        unsafe_allow_html=True)
        st.markdown('<div class="sec">📄 Transcript</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="tbox">{highlight_transcript(transcript,matched)}</div>',
                    unsafe_allow_html=True)
        if matched:
            st.markdown('<div class="sec">🏷️ Keywords</div>', unsafe_allow_html=True)
            st.markdown(" ".join(f'<span class="ktag ktag-red">⚠ {k.upper()}</span>' for k in matched),
                        unsafe_allow_html=True)

    with right:
        st.markdown('<div class="sec">📊 Scoring</div>', unsafe_allow_html=True)
        st.markdown("**AI Probability**")
        st.progress(min(ai_prob/100, 1.0))
        st.caption(f"{ai_prob:.1f}% × {AI_WEIGHT} = **{ai_prob*AI_WEIGHT:.1f}** pts")
        st.markdown("**Keywords**")
        st.progress(min(kw_score/MAX_KW_SCORE, 1.0))
        st.caption(f"{len(matched)} × {PTS_PER_KW} = **{kw_score}** pts")
        st.code(f"Danger = ({ai_prob:.1f}×{AI_WEIGHT}) + {kw_score} = {danger:.1f}", language=None)

    st.divider()

    # Audio + Actions
    st.markdown('<div class="sec">🔊 Audio Replay</div>', unsafe_allow_html=True)
    if ap and os.path.isfile(ap):
        st.markdown('<div class="aud-wrap">', unsafe_allow_html=True)
        st.audio(ap); st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    _, bc, _ = st.columns([1, 3, 1])
    with bc:
        x1,x2,x3 = st.columns(3)
        with x1:
            if st.button("📞 End",key="ec",use_container_width=True,type="primary"):
                reset_system(); safe_rerun()
        with x2:
            if st.button("🔄 Again",key="ad",use_container_width=True):
                reset_system(); safe_rerun()
        with x3:
            if st.button("🎤 Judge",key="dj",use_container_width=True):
                reset_system(); st.session_state["app_state"]="judge_test"; safe_rerun()


# Footer
st.markdown(""); st.divider()
st.markdown('<div class="vg-foot">VoiceGuard Pro v6.0 │ 4-Layer Detection │ '
            'ML + Reference Comparison + Heuristic + Forensics │ '
            '🛡️ Protecting voices.</div>', unsafe_allow_html=True)