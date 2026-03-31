
## Experiments

| # | Method | Script |
|---|--------|--------|
| 1 | Energy-based VAD | `exp1_energy_vad.py` |
| 2 | Silero VAD (speech) | `exp2_silero_vad.py` |
| 3 | Silero VAD + BEATs Linear Classifier | `exp3_vad_beats_classifier.py` |
| 4 | Spectral VAD (Flux / Entropy / Multi-band) | `exp4_spectral_vad.py` |
| 5 | BirdNET (domain-matched bird detector) | `exp5_birdnet.py` |
| 6 | PANNs CNN14 (general audio SED) | `exp6_panns.py` |
| 7 | AVES (animal vocalization encoder + linear) | `exp7_aves.py` |
| 8 | DNN + Dynamic Threshold (Xia et al. 2017) | `exp8_dynamic_threshold_dnn.py` |
| 9a-d | BEATs + Linear/CRNN + TV/Hysteresis | `exp9_temporal_regularisation.py` |

## Additional Analysis

- **SNR Analysis**: `snr_analysis.py` -- per-clip signal-to-noise ratio analysis using frame-level labels.

## Setup

### Prerequisites

```bash
pip install torch torchaudio datasets librosa scikit-learn scipy
pip install birdnetlib panns-inference avex
```

### BEATs Checkpoint

Download the BEATs checkpoint (required for Exp 3, 7, 9):

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Bencr/beats-checkpoints', 'BEATs_iter3_plus_AS2M.pt', repo_type='dataset', local_dir='.')"
```

### HuggingFace Token

The dataset is gated. Set your HF token:

```bash
export HF_TOKEN=your_token_here
```

## Running

```bash
# Run all experiments
python run_all.py

# Run specific experiments
python run_all.py 1 3 9

# Run SNR analysis
python snr_analysis.py
```

## Shared Utilities

- `data_utils.py` -- dataset loading, audio/label extraction, SED metrics
- `BEATs.py`, `backbone.py`, `modules.py`, `quantizer.py` -- Microsoft BEATs model
- `run_all.py` -- experiment runner/orchestrator
