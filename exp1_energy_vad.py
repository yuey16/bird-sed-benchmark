"""
Experiment 1: Energy-based VAD for Sound Event Detection

"""

import numpy as np
import librosa
from data_utils import load_data, get_audio_and_labels, compute_sed_metrics

FRAME_MS = 20
HOP_LENGTH_AT_16K = int(16000 * FRAME_MS / 1000)  # 320 samples


def energy_vad_frames(audio, sr, hop_length=None):
    """Compute per-frame RMS energy aligned to 20ms frames."""
    if hop_length is None:
        hop_length = int(sr * FRAME_MS / 1000)
    frame_length = hop_length * 2
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return rms


def collect_energies_and_labels(dataset):
    """Gather all per-frame energies and GT labels across the dataset."""
    all_energies = []
    all_labels = []
    for ex in dataset:
        wav, sr, labels = get_audio_and_labels(ex)
        rms = energy_vad_frames(wav, sr)
        n = min(len(rms), len(labels))
        all_energies.append(rms[:n])
        all_labels.append(labels[:n])
    return np.concatenate(all_energies), np.concatenate(all_labels)


def find_best_threshold(energies, labels):
    """Sweep absolute energy thresholds to find best frame-level F1."""
    from sklearn.metrics import f1_score
    lo, hi = np.percentile(energies, 1), np.percentile(energies, 99)
    best_f1, best_t = -1, lo
    for t in np.linspace(lo, hi, 200):
        preds = (energies > t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print(f"  Best energy threshold: {best_t:.6f} (train F1={best_f1:.4f})")
    return best_t


def evaluate(dataset, threshold):
    all_preds, all_labels = [], []
    for ex in dataset:
        wav, sr, labels = get_audio_and_labels(ex)
        rms = energy_vad_frames(wav, sr)
        n = min(len(rms), len(labels))
        preds = (rms[:n] > threshold).astype(int)
        all_preds.append(preds)
        all_labels.append(labels[:n])
    return compute_sed_metrics(np.concatenate(all_labels), np.concatenate(all_preds))


def main():
    print("=" * 60)
    print("EXPERIMENT 1: Energy-based VAD")
    print("=" * 60)

    train_ds, test_ds = load_data()

    print("\n--- Computing train energies ---")
    train_e, train_l = collect_energies_and_labels(train_ds)
    print(f"  Total train frames: {len(train_e)}")

    print("\n--- Finding optimal threshold ---")
    threshold = find_best_threshold(train_e, train_l)

    print("\n--- Train set results ---")
    evaluate(train_ds, threshold)

    print("\n--- Test set results ---")
    metrics = evaluate(test_ds, threshold)

    print(f"\n{'=' * 60}")
    print(f"FINAL -- Energy-VAD Test F1: {metrics['f1']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
