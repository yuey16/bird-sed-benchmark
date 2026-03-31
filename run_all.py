"""
Run experiments on the Liangjingyong1/data_final dataset.

Usage:
    python run_all.py          # run all 7
    python run_all.py 1        # run only exp 1
    python run_all.py 4 5 6 7  # run only the new experiments
"""

import sys
import time
import importlib


EXPERIMENTS = {
    "1": ("exp1_energy_vad", "Energy-VAD"),
    "2": ("exp2_silero_vad", "Off-the-shelf Speech VAD (Silero)"),
    "3": ("exp3_vad_beats_classifier", "Speech VAD + BEATs Linear Classifier"),
    "4": ("exp4_spectral_vad", "Spectral-based VAD (Flux/Entropy/Multi-band)"),
    "5": ("exp5_birdnet", "BirdNET (domain-matched bird detector)"),
    "6": ("exp6_panns", "PANNs CNN14 (general audio SED)"),
    "7": ("exp7_aves", "AVES (animal vocalization encoder + linear)"),
    "8": ("exp8_dynamic_threshold_dnn", "DNN + Dynamic Threshold (Xia et al. 2017)"),
    "9": ("exp9_temporal_regularisation", "BEATs + Temporal Regularisation (Linear/CRNN + TV/Hysteresis)"),
}


def run_experiment(exp_id):
    module_name, description = EXPERIMENTS[exp_id]
    print("\n" + "#" * 70)
    print(f"# Experiment {exp_id}: {description}")
    print("#" * 70 + "\n")

    t0 = time.time()
    try:
        mod = importlib.import_module(module_name)
        mod.main()
    except Exception as e:
        print(f"\n*** Exp {exp_id} FAILED: {e} ***")
        import traceback
        traceback.print_exc()
    elapsed = time.time() - t0
    print(f"\n[Exp {exp_id} completed in {elapsed:.1f}s]")
    return elapsed


def main():
    if len(sys.argv) > 1:
        to_run = [a for a in sys.argv[1:] if a in EXPERIMENTS]
    else:
        to_run = list(EXPERIMENTS.keys())

    if not to_run:
        print(f"Usage: python run_all.py [{'|'.join(EXPERIMENTS.keys())}]")
        return

    print("=" * 70)
    print("RUNNING EXPERIMENTS:", ", ".join(
        f"Exp {e} ({EXPERIMENTS[e][1]})" for e in to_run
    ))
    print("=" * 70)

    results = {}
    for exp_id in to_run:
        results[exp_id] = run_experiment(exp_id)

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for exp_id, elapsed in results.items():
        _, desc = EXPERIMENTS[exp_id]
        print(f"  Exp {exp_id} ({desc}): {elapsed:.1f}s")


if __name__ == "__main__":
    main()
