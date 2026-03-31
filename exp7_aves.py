"""
Experiment 7: AVES — Animal Vocalization Encoder + Linear Classifier

Approach:
  - AVES (esp_aves2_sl_beats_bio) is a BEATs model fine-tuned on bioacoustics data
  - Extract frame-level embeddings (768-dim, ~496 frames per 10s clip)
  - Interpolate to label resolution 
  - Train logistic regression on frame-level features
  - Compare with Exp 3 (general BEATs) to see if domain-specific pretraining helps
"""

import os
import sys
import numpy as np
import torch
import torchaudio
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_utils import load_data, get_audio_and_labels, compute_sed_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_aves_model():
    import avex
    model = avex.load_model("esp_aves2_sl_beats_bio")
    model.register_hooks_for_layers(["last_layer"])
    model.eval()
    model.to(DEVICE)
    return model


def get_aves_frame_features(model, audio, sr):
    """Extract AVES frame-level embeddings, interpolate to target resolution."""
    wav = torch.from_numpy(audio).float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
    wav = wav.to(DEVICE)

    with torch.no_grad():
        emb = model.extract_embeddings(wav)  # (1, T_aves, 768)
    feats = emb.squeeze(0).cpu().numpy()  # (T_aves, 768)
    return feats


def interpolate_features(feats, target_len):
    n_src, d = feats.shape
    if n_src == target_len:
        return feats
    x_orig = np.linspace(0, 1, n_src)
    x_target = np.linspace(0, 1, target_len)
    interp_fn = interp1d(x_orig, feats, axis=0, kind="linear", fill_value="extrapolate")
    return interp_fn(x_target)


def extract_features(model, dataset):
    all_feats, all_labels = [], []
    for i, ex in enumerate(dataset):
        wav, sr, labels = get_audio_and_labels(ex)
        n_frames = len(labels)
        feats = get_aves_frame_features(model, wav, sr)
        feats_interp = interpolate_features(feats, n_frames)
        all_feats.append(feats_interp)
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
    print("EXP 7: AVES (Animal Vocalization Encoder) + Linear Classifier")
    print("=" * 60)

    train_ds, test_ds = load_data()

    print(f"\nLoading AVES model (esp_aves2_sl_beats_bio) on {DEVICE}...")
    model = load_aves_model()
    print("AVES loaded.")

    print("\n--- Extracting train features ---")
    X_train, y_train = extract_features(model, train_ds)

    print("\n--- Extracting test features ---")
    X_test, y_test = extract_features(model, test_ds)

    print("\n--- Training linear classifier ---")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42, solver="lbfgs")
    clf.fit(X_train_sc, y_train)
    print("  Classifier trained.")

    print("\n--- Train set results ---")
    compute_sed_metrics(y_train, clf.predict(X_train_sc))

    print("\n--- Test set results ---")
    metrics = compute_sed_metrics(y_test, clf.predict(X_test_sc))

    print(f"\n{'=' * 60}")
    print(f"FINAL -- AVES Linear Test F1: {metrics['f1']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
