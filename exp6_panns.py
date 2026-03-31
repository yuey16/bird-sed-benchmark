"""
Experiment 6: PANNs CNN14 — general audio SED, zero-shot + threshold-tuned

Approach:
  - PANNs CNN14 trained on AudioSet gives frame-level probabilities for 527 classes
  - Aggregate bird-related class probs per frame
  - Zero-shot: fixed threshold (0.5), then tuned threshold on train set
  - Frame resolution: ~100ms per PANNs frame, interpolated to 20ms label grid
"""

import numpy as np
import torch
import librosa
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score
from panns_inference import SoundEventDetection
from data_utils import load_data, get_audio_and_labels, compute_sed_metrics

PANNS_SR = 32000

BIRD_CLASS_INDICES = [111, 112, 113, 117, 119, 121]
# 111: Bird
# 112: Bird vocalization, bird call, bird song
# 113: Chirp, tweet
# 117: Crow
# 119: Owl
# 121: Bird flight, flapping wings


def load_panns(device="cuda"):
    sed = SoundEventDetection(checkpoint_path=None, device=device)
    return sed


def panns_bird_frame_scores(sed_model, audio, sr, n_frames):
    """Run PANNs SED, extract bird class probs, interpolate to label resolution."""
    if sr != PANNS_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=PANNS_SR)

    audio_input = audio[None, :]  # (1, samples)
    framewise = sed_model.inference(audio_input)  # (1, T_panns, 527)
    framewise = framewise[0]  # (T_panns, 527)

    bird_probs = framewise[:, BIRD_CLASS_INDICES].max(axis=1)  # (T_panns,)

    t_panns = len(bird_probs)
    if t_panns == 1:
        return np.full(n_frames, bird_probs[0])

    x_orig = np.linspace(0, 1, t_panns)
    x_target = np.linspace(0, 1, n_frames)
    interp_fn = interp1d(x_orig, bird_probs, kind="linear", fill_value="extrapolate")
    return interp_fn(x_target).astype(np.float32)


def collect_scores(sed_model, dataset):
    all_scores, all_labels = [], []
    for i, ex in enumerate(dataset):
        wav, sr, labels = get_audio_and_labels(ex)
        n_frames = len(labels)
        scores = panns_bird_frame_scores(sed_model, wav, sr, n_frames)
        all_scores.append(scores)
        all_labels.append(labels)
        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(dataset)} clips...")
    return np.concatenate(all_scores), np.concatenate(all_labels)


def find_best_threshold(scores, labels):
    best_f1, best_t = -1, 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print(f"  Best threshold: {best_t:.2f} (train F1={best_f1:.4f})")
    return best_t


def main():
    print("=" * 60)
    print("EXPERIMENT 6: PANNs CNN14 (general audio SED)")
    print("=" * 60)

    train_ds, test_ds = load_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading PANNs CNN14 on {device}...")
    sed_model = load_panns(device=device)
    print("PANNs loaded.")

    print("\n--- Collecting train scores ---")
    train_scores, train_labels = collect_scores(sed_model, train_ds)

    # Zero-shot evaluation (threshold=0.5)
    print("\n--- Zero-shot (threshold=0.5) Train ---")
    m_zs_train = compute_sed_metrics(train_labels, (train_scores >= 0.5).astype(int))

    print("\n--- Tuning threshold on train ---")
    best_t = find_best_threshold(train_scores, train_labels)

    print("\n--- Tuned Train results ---")
    compute_sed_metrics(train_labels, (train_scores >= best_t).astype(int))

    print("\n--- Collecting test scores ---")
    test_scores, test_labels = collect_scores(sed_model, test_ds)

    print("\n--- Zero-shot (threshold=0.5) Test ---")
    m_zs = compute_sed_metrics(test_labels, (test_scores >= 0.5).astype(int))

    print("\n--- Tuned Test results ---")
    m_tuned = compute_sed_metrics(test_labels, (test_scores >= best_t).astype(int))

    print(f"\n{'=' * 60}")
    print(f"FINAL -- PANNs Zero-shot Test F1:  {m_zs['f1']:.4f}")
    print(f"FINAL -- PANNs Tuned     Test F1:  {m_tuned['f1']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
