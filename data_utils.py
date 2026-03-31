"""Shared data loading utilities for bird-call sound event detection."""

import numpy as np
from datasets import load_dataset

DATASET_ID = "Liangjingyong1/data_final"
FRAME_SHIFT_MS = 20  # each label frame = 20ms


def load_data():
    """Load dataset. Returns (train_ds, test_ds, sr).
    Auto-splits into 80/20 since dataset has only a 'train' split.
    """
    ds = load_dataset(DATASET_ID, trust_remote_code=True)
    print(f"Splits: {list(ds.keys())}")
    full = ds["train"]
    print(f"Total samples: {len(full)}")
    print(f"Features: {list(full.features.keys())}")
    print(f"Task: {full[0]['task']}")

    split = full.train_test_split(test_size=0.2, seed=42)
    print(f"Train: {len(split['train'])}, Test: {len(split['test'])}")
    return split["train"], split["test"]


def get_audio_and_labels(example):
    """Extract audio array, sample rate, and frame-level binary labels."""
    audio = example["audio"]
    wav = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    labels = np.array(example["annotations"]["labels"], dtype=np.int32)
    return wav, sr, labels


def timestamps_to_frame_labels(timestamps_sec, total_frames, sr=16000):
    """Convert list of [start, end] second-intervals to frame-level binary labels."""
    frame_labels = np.zeros(total_frames, dtype=np.int32)
    frame_dur = FRAME_SHIFT_MS / 1000.0
    for start, end in timestamps_sec:
        f_start = int(start / frame_dur)
        f_end = int(end / frame_dur)
        f_start = max(0, min(f_start, total_frames))
        f_end = max(0, min(f_end, total_frames))
        frame_labels[f_start:f_end] = 1
    return frame_labels


def compute_sed_metrics(all_labels, all_preds):
    """Compute frame-level SED metrics given concatenated labels and preds."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix
    )
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"  Frame-level metrics:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    pos_rate = np.mean(all_labels)
    pred_rate = np.mean(all_preds)
    print(f"    Label positive rate: {pos_rate:.4f}")
    print(f"    Pred  positive rate: {pred_rate:.4f}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
