"""
Experiment 5: BirdNET — domain-matched pre-trained bird call detector

Approach:
  - BirdNET analyzes audio in 3-second windows, outputs species confidence
  - If any bird species detected above a confidence threshold, mark those
    frames as positive
  - Sweep threshold on train set, evaluate on test
  - Runs on CPU (TFLite backend)
"""

import os
import tempfile
import numpy as np
import soundfile as sf
from sklearn.metrics import f1_score
from data_utils import load_data, get_audio_and_labels, compute_sed_metrics, FRAME_SHIFT_MS

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def init_birdnet():
    from birdnetlib.analyzer import Analyzer
    analyzer = Analyzer()
    return analyzer


def birdnet_to_frame_scores(analyzer, audio, sr, n_frames):
    """Run BirdNET on audio, return per-frame max confidence scores."""
    frame_scores = np.zeros(n_frames, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio, sr)

        from birdnetlib import Recording
        recording = Recording(
            analyzer,
            f.name,
            min_conf=0.01,
        )
        recording.analyze()

        for det in recording.detections:
            start_s = det["start_time"]
            end_s = det["end_time"]
            conf = det["confidence"]

            frame_dur = FRAME_SHIFT_MS / 1000.0
            f_start = int(start_s / frame_dur)
            f_end = int(end_s / frame_dur)
            f_start = max(0, min(f_start, n_frames))
            f_end = max(0, min(f_end, n_frames))
            frame_scores[f_start:f_end] = np.maximum(frame_scores[f_start:f_end], conf)

    return frame_scores


def collect_scores(analyzer, dataset):
    all_scores, all_labels = [], []
    for i, ex in enumerate(dataset):
        wav, sr, labels = get_audio_and_labels(ex)
        n_frames = len(labels)
        scores = birdnet_to_frame_scores(analyzer, wav, sr, n_frames)
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
    print(f"  Best confidence threshold: {best_t:.2f} (train F1={best_f1:.4f})")
    return best_t


def main():
    print("=" * 60)
    print("EXPERIMENT 5: BirdNET (domain-matched bird detector)")
    print("=" * 60)

    train_ds, test_ds = load_data()

    print("\nLoading BirdNET analyzer...")
    analyzer = init_birdnet()
    print("BirdNET loaded.")

    print("\n--- Collecting train scores ---")
    train_scores, train_labels = collect_scores(analyzer, train_ds)
    print(f"  Total train frames: {len(train_labels)}")

    print("\n--- Finding optimal threshold ---")
    best_t = find_best_threshold(train_scores, train_labels)

    print("\n--- Train set results ---")
    compute_sed_metrics(train_labels, (train_scores >= best_t).astype(int))

    print("\n--- Collecting test scores ---")
    test_scores, test_labels = collect_scores(analyzer, test_ds)

    print("\n--- Test set results ---")
    metrics = compute_sed_metrics(test_labels, (test_scores >= best_t).astype(int))

    print(f"\n{'=' * 60}")
    print(f"FINAL -- BirdNET Test F1: {metrics['f1']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
