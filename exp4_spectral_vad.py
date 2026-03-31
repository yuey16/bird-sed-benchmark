"""
Experiment 4: Spectral-based VAD (signal processing baselines)

Three sub-methods:
  4a. Spectral Flux — rate of spectral change between adjacent frames
  4b. Spectral Entropy — low entropy = tonal bird call, high entropy = noise
  4c. Multi-band Energy — logistic regression on 3-band energy features
"""

import numpy as np
import librosa
from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from data_utils import load_data, get_audio_and_labels, compute_sed_metrics

FRAME_MS = 20
N_FFT = 2048


def compute_stft_frames(audio, sr):
    """Compute magnitude spectrogram aligned to 20ms frames."""
    hop = int(sr * FRAME_MS / 1000)
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=hop))
    return S, hop


# ── 4a: Spectral Flux ──

def spectral_flux(S):
    """Frame-level spectral flux (L2 norm of spectral difference)."""
    diff = np.diff(S, axis=1)
    flux = np.linalg.norm(diff, axis=0)
    flux = np.concatenate([[0.0], flux])
    return flux


def run_spectral_flux(train_ds, test_ds):
    print("\n  === 4a: Spectral Flux ===")

    train_feats, train_labels = [], []
    for ex in train_ds:
        wav, sr, labels = get_audio_and_labels(ex)
        S, _ = compute_stft_frames(wav, sr)
        flux = spectral_flux(S)
        n = min(len(flux), len(labels))
        train_feats.append(flux[:n])
        train_labels.append(labels[:n])
    train_feats = np.concatenate(train_feats)
    train_labels = np.concatenate(train_labels)

    lo, hi = np.percentile(train_feats, 1), np.percentile(train_feats, 99)
    best_f1, best_t = -1, lo
    for t in np.linspace(lo, hi, 200):
        preds = (train_feats > t).astype(int)
        f1 = f1_score(train_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print(f"    Best threshold: {best_t:.6f} (train F1={best_f1:.4f})")

    print("\n    --- Train ---")
    compute_sed_metrics(train_labels, (train_feats > best_t).astype(int))

    test_preds_all, test_labels_all = [], []
    for ex in test_ds:
        wav, sr, labels = get_audio_and_labels(ex)
        S, _ = compute_stft_frames(wav, sr)
        flux = spectral_flux(S)
        n = min(len(flux), len(labels))
        test_preds_all.append((flux[:n] > best_t).astype(int))
        test_labels_all.append(labels[:n])

    print("\n    --- Test ---")
    m = compute_sed_metrics(np.concatenate(test_labels_all),
                            np.concatenate(test_preds_all))
    return m


# ── 4b: Spectral Entropy ──

def spectral_entropy(S):
    """Per-frame spectral entropy. Low = tonal, high = noise.
    Returns 1 - normalized_entropy so that high values indicate tonal content."""
    S_power = S ** 2 + 1e-12
    S_norm = S_power / S_power.sum(axis=0, keepdims=True)
    max_ent = np.log2(S.shape[0])
    ent = np.zeros(S.shape[1])
    for i in range(S.shape[1]):
        h = scipy_entropy(S_norm[:, i], base=2)
        ent[i] = h if np.isfinite(h) else max_ent
    inv_ent = 1.0 - ent / max_ent
    return inv_ent


def run_spectral_entropy(train_ds, test_ds):
    print("\n  === 4b: Spectral Entropy (inverted) ===")

    train_feats, train_labels = [], []
    for ex in train_ds:
        wav, sr, labels = get_audio_and_labels(ex)
        S, _ = compute_stft_frames(wav, sr)
        se = spectral_entropy(S)
        n = min(len(se), len(labels))
        train_feats.append(se[:n])
        train_labels.append(labels[:n])
    train_feats = np.concatenate(train_feats)
    train_labels = np.concatenate(train_labels)

    lo, hi = np.percentile(train_feats, 1), np.percentile(train_feats, 99)
    best_f1, best_t = -1, lo
    for t in np.linspace(lo, hi, 200):
        preds = (train_feats > t).astype(int)
        f1 = f1_score(train_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print(f"    Best threshold: {best_t:.6f} (train F1={best_f1:.4f})")

    print("\n    --- Train ---")
    compute_sed_metrics(train_labels, (train_feats > best_t).astype(int))

    test_preds_all, test_labels_all = [], []
    for ex in test_ds:
        wav, sr, labels = get_audio_and_labels(ex)
        S, _ = compute_stft_frames(wav, sr)
        se = spectral_entropy(S)
        n = min(len(se), len(labels))
        test_preds_all.append((se[:n] > best_t).astype(int))
        test_labels_all.append(labels[:n])

    print("\n    --- Test ---")
    m = compute_sed_metrics(np.concatenate(test_labels_all),
                            np.concatenate(test_preds_all))
    return m


# ── 4c: Multi-band Energy ──

def multiband_energy(S, sr, hop):
    """Compute energy in 3 frequency bands: 0-2k, 2-4k, 4-8k Hz."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    bands = [(0, 2000), (2000, 4000), (4000, 8000)]
    band_energies = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        band_e = (S[mask, :] ** 2).sum(axis=0)
        band_energies.append(band_e)
    return np.stack(band_energies, axis=1)  # (n_frames, 3)


def run_multiband_energy(train_ds, test_ds):
    print("\n  === 4c: Multi-band Energy (LogReg) ===")

    train_feats, train_labels = [], []
    for ex in train_ds:
        wav, sr, labels = get_audio_and_labels(ex)
        S, hop = compute_stft_frames(wav, sr)
        mb = multiband_energy(S, sr, hop)
        n = min(len(mb), len(labels))
        train_feats.append(mb[:n])
        train_labels.append(labels[:n])
    X_train = np.concatenate(train_feats)
    y_train = np.concatenate(train_labels)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(X_train_sc, y_train)

    print("\n    --- Train ---")
    compute_sed_metrics(y_train, clf.predict(X_train_sc))

    test_feats, test_labels = [], []
    for ex in test_ds:
        wav, sr, labels = get_audio_and_labels(ex)
        S, hop = compute_stft_frames(wav, sr)
        mb = multiband_energy(S, sr, hop)
        n = min(len(mb), len(labels))
        test_feats.append(mb[:n])
        test_labels.append(labels[:n])
    X_test = np.concatenate(test_feats)
    y_test = np.concatenate(test_labels)
    X_test_sc = scaler.transform(X_test)

    print("\n    --- Test ---")
    m = compute_sed_metrics(y_test, clf.predict(X_test_sc))
    return m


def main():
    print("=" * 60)
    print("EXPERIMENT 4: Spectral-based VAD")
    print("=" * 60)

    train_ds, test_ds = load_data()

    m_flux = run_spectral_flux(train_ds, test_ds)
    m_entropy = run_spectral_entropy(train_ds, test_ds)
    m_multiband = run_multiband_energy(train_ds, test_ds)

    print(f"\n{'=' * 60}")
    print(f"FINAL -- Spectral Flux    Test F1: {m_flux['f1']:.4f}")
    print(f"FINAL -- Spectral Entropy Test F1: {m_entropy['f1']:.4f}")
    print(f"FINAL -- Multi-band Energy Test F1: {m_multiband['f1']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
