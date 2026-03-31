"""
Experiment 2: Off-the-shelf Speech VAD (Silero VAD) for Sound Event Detection

Approach:
  - Run Silero VAD to get speech timestamps for each cliP
  - Evaluate frame-level detection against GT bird-call labels
  - Note: Silero is a *speech* VAD, so this tests whether speech VAD
    can transfer to detecting bird vocalizations (likely limited)
"""

import numpy as np
import torch
import torchaudio
from data_utils import (
    load_data, get_audio_and_labels, compute_sed_metrics,
    FRAME_SHIFT_MS
)

SILERO_SR = 16000


def load_silero_vad():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def silero_to_frame_preds(model, get_speech_ts, audio, sr, n_frames):
    """Run Silero VAD and convert output to frame-level binary predictions."""
    wav = torch.from_numpy(audio).float()
    if wav.dim() > 1:
        wav = wav.mean(dim=0)
    if sr != SILERO_SR:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SILERO_SR)
    if wav.abs().max() > 1.0:
        wav = wav / wav.abs().max()

    timestamps = get_speech_ts(wav, model, sampling_rate=SILERO_SR)

    frame_preds = np.zeros(n_frames, dtype=np.int32)
    frame_dur_samples = int(SILERO_SR * FRAME_SHIFT_MS / 1000)
    for ts in timestamps:
        f_start = ts["start"] // frame_dur_samples
        f_end = ts["end"] // frame_dur_samples
        f_start = max(0, min(f_start, n_frames))
        f_end = max(0, min(f_end, n_frames))
        frame_preds[f_start:f_end] = 1

    return frame_preds


def evaluate(model, get_speech_ts, dataset):
    all_preds, all_labels = [], []
    for i, ex in enumerate(dataset):
        wav, sr, labels = get_audio_and_labels(ex)
        n_frames = len(labels)
        preds = silero_to_frame_preds(model, get_speech_ts, wav, sr, n_frames)
        all_preds.append(preds)
        all_labels.append(labels)
        if (i + 1) % 200 == 0:
            print(f"    Processed {i+1}/{len(dataset)} clips...")

    return compute_sed_metrics(np.concatenate(all_labels), np.concatenate(all_preds))


def main():
    print("=" * 60)
    print("EXPERIMENT 2: Off-the-shelf Speech VAD (Silero VAD)")
    print("=" * 60)

    train_ds, test_ds = load_data()

    print("\nLoading Silero VAD model...")
    model, get_speech_ts = load_silero_vad()
    print("Silero VAD loaded.")

    print("\n--- Train set results ---")
    evaluate(model, get_speech_ts, train_ds)

    print("\n--- Test set results ---")
    metrics = evaluate(model, get_speech_ts, test_ds)

    print(f"\n{'=' * 60}")
    print(f"FINAL -- Silero VAD Test F1: {metrics['f1']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
