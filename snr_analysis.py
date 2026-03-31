"""For each clip, uses frame-level binary labels to segment the waveform into
"signal" (bird-call active) and "noise" (background) regions, then computes:
  SNR_clip = 10 * log10( mean_power_signal / mean_power_noise )

Produces per-clip statistics, histogram bins, and overall summary.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import load_data, get_audio_and_labels, FRAME_SHIFT_MS


def clip_snr(wav, sr, labels):
    """Compute SNR for one clip using frame-level labels.
    Returns SNR in dB, signal RMS, noise RMS, and fraction of active frames.
    Returns None for SNR if signal or noise region is empty."""
    frame_samples = int(sr * FRAME_SHIFT_MS / 1000)
    n_frames = len(labels)

    sig_samples = []
    noise_samples = []

    for i in range(n_frames):
        start = i * frame_samples
        end = min(start + frame_samples, len(wav))
        if start >= len(wav):
            break
        chunk = wav[start:end]
        if labels[i] == 1:
            sig_samples.append(chunk)
        else:
            noise_samples.append(chunk)

    active_ratio = np.mean(labels)

    if len(sig_samples) == 0 or len(noise_samples) == 0:
        return None, 0.0, 0.0, active_ratio

    sig_all = np.concatenate(sig_samples)
    noise_all = np.concatenate(noise_samples)

    sig_power = np.mean(sig_all ** 2)
    noise_power = np.mean(noise_all ** 2)

    sig_rms = np.sqrt(sig_power)
    noise_rms = np.sqrt(noise_power)

    if noise_power < 1e-20:
        return None, sig_rms, noise_rms, active_ratio

    snr_db = 10 * np.log10(sig_power / noise_power)
    return snr_db, sig_rms, noise_rms, active_ratio


def analyse_dataset(dataset, name="Dataset"):
    """Run SNR analysis on a dataset split."""
    snrs = []
    sig_rms_list = []
    noise_rms_list = []
    active_ratios = []
    skipped = 0
    all_silent = 0
    no_bird = 0

    for i, ex in enumerate(dataset):
        wav, sr, labels = get_audio_and_labels(ex)

        if labels.sum() == 0:
            no_bird += 1
            continue
        if labels.sum() == len(labels):
            all_silent += 1
            continue

        snr_db, s_rms, n_rms, act_ratio = clip_snr(wav, sr, labels)

        if snr_db is None:
            skipped += 1
            continue

        snrs.append(snr_db)
        sig_rms_list.append(s_rms)
        noise_rms_list.append(n_rms)
        active_ratios.append(act_ratio)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(dataset)}...")

    snrs = np.array(snrs)
    sig_rms_list = np.array(sig_rms_list)
    noise_rms_list = np.array(noise_rms_list)
    active_ratios = np.array(active_ratios)

    print(f"\n{'=' * 60}")
    print(f"SNR Analysis: {name}")
    print(f"{'=' * 60}")
    print(f"  Total clips: {len(dataset)}")
    print(f"  Clips with valid SNR: {len(snrs)}")
    print(f"  Clips with no bird call (all background): {no_bird}")
    print(f"  Clips fully active (no background): {all_silent}")
    print(f"  Skipped (computation issue): {skipped}")

    if len(snrs) == 0:
        print("  No valid SNR values computed.")
        return

    print(f"\n  --- SNR Distribution (dB) ---")
    print(f"    Mean:   {snrs.mean():.2f} dB")
    print(f"    Median: {np.median(snrs):.2f} dB")
    print(f"    Std:    {snrs.std():.2f} dB")
    print(f"    Min:    {snrs.min():.2f} dB")
    print(f"    Max:    {snrs.max():.2f} dB")
    print(f"    25th %%: {np.percentile(snrs, 25):.2f} dB")
    print(f"    75th %%: {np.percentile(snrs, 75):.2f} dB")

    print(f"\n  --- SNR Histogram ---")
    bins = [(-np.inf, -10), (-10, -5), (-5, 0), (0, 5), (5, 10),
            (10, 15), (15, 20), (20, 30), (30, np.inf)]
    for lo, hi in bins:
        count = np.sum((snrs >= lo) & (snrs < hi))
        pct = 100.0 * count / len(snrs)
        if lo == -np.inf:
            label = f"     < {hi:3.0f} dB"
        elif hi == np.inf:
            label = f"  >= {lo:3.0f}     dB"
        else:
            label = f"  {lo:4.0f} to {hi:3.0f} dB"
        bar = "#" * int(pct / 2)
        print(f"    {label}: {count:5d} ({pct:5.1f}%)  {bar}")

    print(f"\n  --- Signal vs Noise RMS ---")
    print(f"    Signal RMS  -- mean: {sig_rms_list.mean():.6f}, "
          f"median: {np.median(sig_rms_list):.6f}")
    print(f"    Noise  RMS  -- mean: {noise_rms_list.mean():.6f}, "
          f"median: {np.median(noise_rms_list):.6f}")
    print(f"    Ratio (sig/noise RMS) -- mean: "
          f"{(sig_rms_list / (noise_rms_list + 1e-20)).mean():.2f}")

    print(f"\n  --- Active Ratio (fraction of frames with bird call) ---")
    print(f"    Mean:   {active_ratios.mean():.4f}")
    print(f"    Median: {np.median(active_ratios):.4f}")
    print(f"    Min:    {active_ratios.min():.4f}")
    print(f"    Max:    {active_ratios.max():.4f}")

    print(f"\n  --- SNR by Active Ratio Quartile ---")
    for q, (lo_q, hi_q) in enumerate([(0, 25), (25, 50), (50, 75), (75, 100)]):
        lo_v = np.percentile(active_ratios, lo_q)
        hi_v = np.percentile(active_ratios, hi_q)
        if q == 3:
            mask = (active_ratios >= lo_v) & (active_ratios <= hi_v)
        else:
            mask = (active_ratios >= lo_v) & (active_ratios < hi_v)
        if mask.sum() > 0:
            print(f"    Q{q+1} (active {lo_v:.2f}-{hi_v:.2f}): "
                  f"mean SNR={snrs[mask].mean():.2f} dB, n={mask.sum()}")

    return {
        "snrs": snrs,
        "sig_rms": sig_rms_list,
        "noise_rms": noise_rms_list,
        "active_ratios": active_ratios,
    }


def main():
    train_ds, test_ds = load_data()

    print("\n--- Analysing FULL dataset (train + test) ---")
    from datasets import concatenate_datasets
    full_ds = concatenate_datasets([train_ds, test_ds])
    full_results = analyse_dataset(full_ds, "Full Dataset")

    print("\n\n--- Analysing TRAIN split ---")
    train_results = analyse_dataset(train_ds, "Train Split")

    print("\n\n--- Analysing TEST split ---")
    test_results = analyse_dataset(test_ds, "Test Split")

    print(f"\n{'=' * 60}")
    print("SNR ANALYSIS COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
