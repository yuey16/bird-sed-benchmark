"""
Experiment 3: Off-the-shelf Speech VAD + BEATs Linear Target Classifier

Approach:
  1. Extract BEATs frame-level features (768-dim, ~50 frames per 10s clip)
  2. Interpolate BEATs features to match label resolution 
  3. Compute Silero VAD frame-level speech indicator as an extra feature
  4. Concatenate: [BEATs_768 | vad_flag_1] = 769-dim per frame
  5. Train a logistic regression on frame-level features vs GT labels
  6. Evaluate frame-level SED on test set
"""

import os
import sys
import numpy as np
import torch
import torchaudio
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from BEATs import BEATs, BEATsConfig
from data_utils import (
    load_data, get_audio_and_labels, compute_sed_metrics,
    FRAME_SHIFT_MS
)

BEATS_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "BEATs_iter3_plus_AS2M.pt")
SILERO_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_beats_model():
    ckpt = torch.load(BEATS_CKPT, map_location="cpu")
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(DEVICE)
    return model


def load_silero_vad():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    return model, utils[0]


def get_beats_frame_features(beats_model, audio, sr):
    """Extract BEATs features and interpolate to 20ms frame grid."""
    wav = torch.from_numpy(audio).float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
    wav = wav.to(DEVICE)

    with torch.no_grad():
        feats, _ = beats_model.extract_features(wav)
    feats = feats.squeeze(0).cpu().numpy()  # (T_beats, 768)
    return feats


def interpolate_features(feats, target_len):
    """Linearly interpolate (T_beats, D) features to (target_len, D)."""
    n_beats, d = feats.shape
    if n_beats == target_len:
        return feats
    x_orig = np.linspace(0, 1, n_beats)
    x_target = np.linspace(0, 1, target_len)
    interp_fn = interp1d(x_orig, feats, axis=0, kind="linear", fill_value="extrapolate")
    return interp_fn(x_target)


def get_silero_frame_vad(vad_model, get_speech_ts, audio, sr, n_frames):
    """Get frame-level VAD flags from Silero."""
    wav = torch.from_numpy(audio).float()
    if wav.dim() > 1:
        wav = wav.mean(dim=0)
    if sr != SILERO_SR:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SILERO_SR)
    if wav.abs().max() > 1.0:
        wav = wav / wav.abs().max()

    timestamps = get_speech_ts(wav, vad_model, sampling_rate=SILERO_SR)

    vad_flags = np.zeros(n_frames, dtype=np.float32)
    frame_dur_samples = int(SILERO_SR * FRAME_SHIFT_MS / 1000)
    for ts in timestamps:
        f_start = max(0, min(ts["start"] // frame_dur_samples, n_frames))
        f_end = max(0, min(ts["end"] // frame_dur_samples, n_frames))
        vad_flags[f_start:f_end] = 1.0
    return vad_flags


def extract_features(beats_model, vad_model, get_speech_ts, dataset):
    """Extract frame-level feature matrix and labels for the whole dataset."""
    all_feats, all_labels = [], []

    for i, ex in enumerate(dataset):
        wav, sr, labels = get_audio_and_labels(ex)
        n_frames = len(labels)

        beats_feats = get_beats_frame_features(beats_model, wav, sr)
        beats_interp = interpolate_features(beats_feats, n_frames)

        vad_flags = get_silero_frame_vad(vad_model, get_speech_ts, wav, sr, n_frames)

        combined = np.column_stack([beats_interp, vad_flags])  # (n_frames, 769)
        all_feats.append(combined)
        all_labels.append(labels)

        if (i + 1) % 100 == 0:
            print(f"    Extracted {i+1}/{len(dataset)} clips...")

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"    Total frames: {len(y)}, Feature dim: {X.shape[1]}")
    print(f"    Positive rate: {y.mean():.4f}")
    return X, y


def main():
    print("=" * 60)
    print("EXP 3: Speech VAD + BEATs Linear Target Classifier")
    print("=" * 60)

    train_ds, test_ds = load_data()

    print("\nLoading BEATs model...")
    beats_model = load_beats_model()
    print(f"BEATs loaded on {DEVICE}")

    print("Loading Silero VAD...")
    vad_model, get_speech_ts = load_silero_vad()
    print("Silero VAD loaded.")

    print("\n--- Extracting train features ---")
    X_train, y_train = extract_features(beats_model, vad_model, get_speech_ts, train_ds)

    print("\n--- Extracting test features ---")
    X_test, y_test = extract_features(beats_model, vad_model, get_speech_ts, test_ds)

    print("\n--- Training linear classifier ---")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42, solver="lbfgs")
    clf.fit(X_train_sc, y_train)
    print("  Classifier trained.")

    print("\n--- Train set results ---")
    train_preds = clf.predict(X_train_sc)
    compute_sed_metrics(y_train, train_preds)

    print("\n--- Test set results ---")
    test_preds = clf.predict(X_test_sc)
    metrics = compute_sed_metrics(y_test, test_preds)

    print(f"\n{'=' * 60}")
    print(f"FINAL -- VAD+BEATs Linear Test F1: {metrics['f1']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
